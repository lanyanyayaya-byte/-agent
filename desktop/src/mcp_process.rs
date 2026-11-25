//! Host-level supervision and identity-verified cleanup for managed stdio MCP process groups.
//!
//! When Python Sidecar spawns a local stdio MCP server in its own POSIX process group
//! (`start_new_session=True`), Sidecar atomically records kernel-backed process identity under
//! `<app_data_dir>/runtime/mcp-procs/<instance_id>/`.
//!
//! If Sidecar exits abnormally or is terminated with `SIGKILL`, the Rust host inspects only
//! that exited Sidecar instance's registry files, verifies live kernel process identity
//! (PID, PGID, UID, microsecond start timestamp, canonical executable path) via macOS
//! `libproc`, and only then terminates confirmed managed process groups.
//!
//! Known capability boundaries (honestly documented, no separate daemon introduced):
//! - If the Rust host process itself is also `SIGKILL`'d, no supervisor remains to run cleanup;
//!   on subsequent launch stale registry files are deleted without blindly signaling PIDs.
//! - If an external stdio program intentionally calls `setsid()` / `setpgid()` to detach a
//!   background daemon into a new process group, `killpg(pgid)` cannot reach the detached tree.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::diagnostics::HostLog;

const SIGTERM_WAIT: Duration = Duration::from_millis(800);
const SIGKILL_WAIT: Duration = Duration::from_millis(500);
const POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ReapReport {
    pub(crate) reaped_groups: usize,
    pub(crate) already_exited: usize,
    pub(crate) skipped_mismatch: usize,
    pub(crate) pending_race_markers: usize,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ProcessRegistryEntry {
    Pending {
        #[allow(dead_code)]
        version: u8,
        instance_id: String,
        sidecar_pid: u32,
        server_id: String,
        generation: u64,
        #[allow(dead_code)]
        executable_path: String,
        #[allow(dead_code)]
        started_at_ms: u64,
    },
    Active {
        #[allow(dead_code)]
        version: u8,
        instance_id: String,
        sidecar_pid: u32,
        server_id: String,
        generation: u64,
        pid: u32,
        pgid: u32,
        uid: u32,
        start_tvsec: Option<u64>,
        start_tvusec: Option<u64>,
        executable_path: String,
        #[allow(dead_code)]
        registered_at_ms: u64,
    },
}

fn registry_root(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("runtime").join("mcp-procs")
}

fn instance_registry_dir(app_data_dir: &Path, instance_id: Uuid) -> PathBuf {
    registry_root(app_data_dir).join(instance_id.to_string())
}

fn is_regular_dir(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| !meta.file_type().is_symlink() && meta.is_dir())
}

fn is_regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| !meta.file_type().is_symlink() && meta.is_file())
}

/// Remove expired registry directories from prior host/sidecar runs WITHOUT signaling any PID.
pub(crate) fn cleanup_stale_registries(
    app_data_dir: &Path,
    keep_instance_id: Option<Uuid>,
    log: &HostLog,
) -> usize {
    let root = registry_root(app_data_dir);
    if !is_regular_dir(&root) {
        return 0;
    }
    let keep_name = keep_instance_id.map(|id| id.to_string());
    let mut removed_files = 0usize;
    let Ok(entries) = fs::read_dir(&root) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(ref keep) = keep_name {
            if entry.file_name().to_string_lossy() == *keep {
                continue;
            }
        }
        if !is_regular_dir(&path) {
            continue;
        }
        if let Ok(files) = fs::read_dir(&path) {
            for file in files.flatten() {
                let fpath = file.path();
                if is_regular_file(&fpath) && fs::remove_file(&fpath).is_ok() {
                    removed_files += 1;
                }
            }
        }
        let _ = fs::remove_dir(&path);
    }
    if removed_files > 0 {
        log.record(
            "mcp_stale_registry_cleaned",
            json!({ "removed_files": removed_files }),
        );
    }
    removed_files
}

/// Reap any surviving managed stdio MCP process groups belonging to the exited Sidecar instance.
pub(crate) async fn reap_instance_processes(
    app_data_dir: &Path,
    instance_id: Uuid,
    sidecar_pid: Option<u32>,
    log: &HostLog,
) -> ReapReport {
    let dir = instance_registry_dir(app_data_dir, instance_id);
    if !is_regular_dir(&dir) {
        return ReapReport::default();
    }

    let mut report = ReapReport::default();
    let expected_instance = instance_id.to_string();
    let Ok(entries) = fs::read_dir(&dir) else {
        return report;
    };

    let files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| is_regular_file(path))
        .collect();

    for path in files {
        let raw = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => {
                let _ = fs::remove_file(&path);
                continue;
            }
        };
        let entry: ProcessRegistryEntry = match serde_json::from_slice(&raw) {
            Ok(parsed) => parsed,
            Err(_) => {
                let _ = fs::remove_file(&path);
                continue;
            }
        };

        match entry {
            ProcessRegistryEntry::Pending {
                instance_id: rec_inst,
                sidecar_pid: rec_spid,
                server_id,
                generation,
                ..
            } => {
                if rec_inst == expected_instance && sidecar_pid.is_none_or(|p| p == rec_spid) {
                    report.pending_race_markers += 1;
                    log.record(
                        "mcp_orphan_race_pending_detected",
                        json!({
                            "instance_id": instance_id,
                            "server_id": server_id,
                            "generation": generation,
                        }),
                    );
                }
                let _ = fs::remove_file(&path);
            }
            ProcessRegistryEntry::Active {
                instance_id: rec_inst,
                sidecar_pid: rec_spid,
                server_id,
                generation,
                pid,
                pgid,
                uid,
                start_tvsec,
                start_tvusec,
                executable_path,
                ..
            } => {
                if rec_inst != expected_instance
                    || sidecar_pid.is_some_and(|expected| expected != rec_spid)
                {
                    report.skipped_mismatch += 1;
                    log.record(
                        "mcp_reap_identity_mismatch",
                        json!({
                            "instance_id": instance_id,
                            "server_id": server_id,
                            "reason": "owner_mismatch",
                        }),
                    );
                    let _ = fs::remove_file(&path);
                    continue;
                }

                match verify_and_reap_group(
                    pid,
                    pgid,
                    uid,
                    start_tvsec,
                    start_tvusec,
                    &executable_path,
                )
                .await
                {
                    ReapOutcome::Reaped(stage) => {
                        report.reaped_groups += 1;
                        log.record(
                            "mcp_orphan_reaped",
                            json!({
                                "instance_id": instance_id,
                                "server_id": server_id,
                                "generation": generation,
                                "pid": pid,
                                "pgid": pgid,
                                "stage": stage,
                            }),
                        );
                    }
                    ReapOutcome::AlreadyExited => {
                        report.already_exited += 1;
                    }
                    ReapOutcome::IdentityMismatch(reason) => {
                        report.skipped_mismatch += 1;
                        log.record(
                            "mcp_reap_identity_mismatch",
                            json!({
                                "instance_id": instance_id,
                                "server_id": server_id,
                                "generation": generation,
                                "pid": pid,
                                "pgid": pgid,
                                "reason": reason,
                            }),
                        );
                    }
                }
                let _ = fs::remove_file(&path);
            }
        }
    }

    let _ = fs::remove_dir(&dir);
    report
}

enum ReapOutcome {
    Reaped(&'static str),
    AlreadyExited,
    IdentityMismatch(&'static str),
}

#[cfg(target_os = "macos")]
async fn verify_and_reap_group(
    pid: u32,
    pgid: u32,
    expected_uid: u32,
    expected_start_tvsec: Option<u64>,
    expected_start_tvusec: Option<u64>,
    expected_executable_path: &str,
) -> ReapOutcome {
    let current_uid = unsafe { libc::getuid() };
    let host_pgid = unsafe { libc::getpgrp() } as u32;
    if pid <= 1 || pgid <= 1 || pgid == host_pgid || expected_uid != current_uid {
        return ReapOutcome::IdentityMismatch("invalid_ids_or_uid");
    }

    let (Some(rec_sec), Some(rec_usec)) = (expected_start_tvsec, expected_start_tvusec) else {
        // Without kernel start timestamp, refuse to kill to prevent PID/PGID reuse false kills.
        return ReapOutcome::IdentityMismatch("missing_kernel_start_time");
    };

    if let Some(live) = darwin::query_bsdinfo(pid) {
        if live.uid != current_uid || live.pgid != pgid {
            return ReapOutcome::IdentityMismatch("uid_or_pgid_changed");
        }
        if live.start_tvsec != rec_sec || live.start_tvusec != rec_usec {
            return ReapOutcome::IdentityMismatch("start_time_mismatch");
        }
        let Some(live_exe) = darwin::query_pidpath(pid) else {
            return ReapOutcome::IdentityMismatch("exe_path_unavailable");
        };
        if !same_canonical_path(&live_exe, Path::new(expected_executable_path)) {
            return ReapOutcome::IdentityMismatch("exe_path_mismatch");
        }
        return terminate_pgid(pgid).await;
    }

    // Leader PID has already exited; check if any grandchildren remain in the same PGID.
    let members = darwin::list_pgid_members(pgid);
    if members.is_empty() {
        return ReapOutcome::AlreadyExited;
    }

    let now_sec = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    for member_pid in members {
        let Some(info) = darwin::query_bsdinfo(member_pid) else {
            continue;
        };
        if info.uid != current_uid || info.pgid != pgid {
            return ReapOutcome::IdentityMismatch("group_member_uid_or_pgid_mismatch");
        }
        // Any member must have started at or after the managed leader's creation time.
        if info.start_tvsec < rec_sec
            || (info.start_tvsec == rec_sec && info.start_tvusec < rec_usec)
            || info.start_tvsec > now_sec.saturating_add(2)
        {
            return ReapOutcome::IdentityMismatch("group_member_start_time_out_of_window");
        }
    }

    terminate_pgid(pgid).await
}

#[cfg(not(target_os = "macos"))]
async fn verify_and_reap_group(
    _pid: u32,
    _pgid: u32,
    _expected_uid: u32,
    _expected_start_tvsec: Option<u64>,
    _expected_start_tvusec: Option<u64>,
    _expected_executable_path: &str,
) -> ReapOutcome {
    ReapOutcome::IdentityMismatch("unsupported_platform")
}

#[cfg(unix)]
async fn terminate_pgid(pgid: u32) -> ReapOutcome {
    let pgid_i32 = pgid as libc::pid_t;
    let term_rc = unsafe { libc::killpg(pgid_i32, libc::SIGTERM) };
    if term_rc != 0 {
        let err = std::io::Error::last_os_error().raw_os_error();
        if err == Some(libc::ESRCH) {
            return ReapOutcome::AlreadyExited;
        }
    }

    if wait_pgid_gone(pgid_i32, SIGTERM_WAIT).await {
        return ReapOutcome::Reaped("sigterm");
    }

    let _ = unsafe { libc::killpg(pgid_i32, libc::SIGKILL) };
    let _ = wait_pgid_gone(pgid_i32, SIGKILL_WAIT).await;
    ReapOutcome::Reaped("sigkill")
}

#[cfg(unix)]
async fn wait_pgid_gone(pgid: libc::pid_t, max_wait: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + max_wait;
    loop {
        let rc = unsafe { libc::killpg(pgid, 0) };
        if rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

#[cfg(target_os = "macos")]
fn same_canonical_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

#[cfg(target_os = "macos")]
mod darwin {
    use std::ffi::{c_int, c_void, OsStr};
    use std::os::unix::ffi::OsStrExt;
    use std::path::PathBuf;

    const PROC_PIDTBSDINFO: c_int = 3;
    const PROC_BSDINFO_SIZE: c_int = 136;
    const PROC_PIDPATHINFO_MAXSIZE: u32 = 4096;

    extern "C" {
        fn proc_pidinfo(
            pid: c_int,
            flavor: c_int,
            arg: u64,
            buffer: *mut c_void,
            buffersize: c_int,
        ) -> c_int;
        fn proc_pidpath(pid: c_int, buffer: *mut c_void, buffersize: u32) -> c_int;
        fn proc_listpgrppids(pgrpid: libc::pid_t, buffer: *mut c_void, buffersize: c_int) -> c_int;
    }

    pub(super) struct BsdProcInfo {
        pub(super) uid: u32,
        pub(super) pgid: u32,
        pub(super) start_tvsec: u64,
        pub(super) start_tvusec: u64,
    }

    pub(super) fn query_bsdinfo(pid: u32) -> Option<BsdProcInfo> {
        let mut buf = [0_u8; PROC_BSDINFO_SIZE as usize];
        let size = unsafe {
            proc_pidinfo(
                pid as c_int,
                PROC_PIDTBSDINFO,
                0,
                buf.as_mut_ptr().cast::<c_void>(),
                PROC_BSDINFO_SIZE,
            )
        };
        if size != PROC_BSDINFO_SIZE {
            return None;
        }
        let live_pid = u32::from_ne_bytes(buf[12..16].try_into().ok()?);
        if live_pid != pid {
            return None;
        }
        let uid = u32::from_ne_bytes(buf[20..24].try_into().ok()?);
        let pgid = u32::from_ne_bytes(buf[100..104].try_into().ok()?);
        let start_tvsec = u64::from_ne_bytes(buf[120..128].try_into().ok()?);
        let start_tvusec = u64::from_ne_bytes(buf[128..136].try_into().ok()?);
        Some(BsdProcInfo {
            uid,
            pgid,
            start_tvsec,
            start_tvusec,
        })
    }

    pub(super) fn query_pidpath(pid: u32) -> Option<PathBuf> {
        let mut buf = vec![0_u8; PROC_PIDPATHINFO_MAXSIZE as usize];
        let len = unsafe {
            proc_pidpath(
                pid as c_int,
                buf.as_mut_ptr().cast::<c_void>(),
                PROC_PIDPATHINFO_MAXSIZE,
            )
        };
        if len <= 0 {
            return None;
        }
        let bytes = &buf[..len as usize];
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        if end == 0 {
            return None;
        }
        Some(PathBuf::from(OsStr::from_bytes(&bytes[..end])))
    }

    pub(super) fn list_pgid_members(pgid: u32) -> Vec<u32> {
        let bytes_needed = unsafe { proc_listpgrppids(pgid as libc::pid_t, std::ptr::null_mut(), 0) };
        if bytes_needed <= 0 {
            return Vec::new();
        }
        let capacity = (bytes_needed as usize / std::mem::size_of::<libc::pid_t>()) + 16;
        let mut pids = vec![0 as libc::pid_t; capacity];
        let filled = unsafe {
            proc_listpgrppids(
                pgid as libc::pid_t,
                pids.as_mut_ptr().cast::<c_void>(),
                (pids.len() * std::mem::size_of::<libc::pid_t>()) as c_int,
            )
        };
        if filled <= 0 {
            return Vec::new();
        }
        let count = filled as usize / std::mem::size_of::<libc::pid_t>();
        pids.into_iter()
            .take(count)
            .filter(|&pid| pid > 1)
            .map(|pid| pid as u32)
            .collect()
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::process::{Command as StdCommand, Stdio};
    use tempfile::tempdir;

    fn pid_alive(pid: u32) -> bool {
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }

    fn spawn_stubborn_group_process() -> std::process::Child {
        // Spawns a process in its own process group that ignores SIGTERM (requires SIGKILL).
        let mut cmd = StdCommand::new("/bin/sh");
        cmd.args(["-c", "trap '' TERM; sleep 60"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        cmd.spawn().expect("spawn stubborn child")
    }

    fn random_uuid() -> Uuid {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes).unwrap();
        Uuid::from_bytes(bytes)
    }

    #[tokio::test]
    async fn reaps_verified_stubborn_process_group_and_skips_stale_or_mismatched_pid() {
        let data = tempdir().unwrap();
        let log = HostLog::new(data.path());
        let instance_id = random_uuid();
        let old_instance = random_uuid();
        let sidecar_pid = 424242_u32;

        // 1. Spawn a real stubborn process (ignores SIGTERM) and register its exact kernel identity.
        let mut managed = spawn_stubborn_group_process();
        let managed_pid = managed.id();
        tokio::time::sleep(Duration::from_millis(80)).await;
        let managed_info = darwin::query_bsdinfo(managed_pid).expect("managed bsdinfo");
        let managed_exe = darwin::query_pidpath(managed_pid).expect("managed pidpath");

        // 2. Spawn a second unrelated process and place a MISMATCHED record (simulating PID reuse)
        //    plus a STALE record under an old instance_id. Neither may be killed!
        let mut bystander = spawn_stubborn_group_process();
        let bystander_pid = bystander.id();
        tokio::time::sleep(Duration::from_millis(80)).await;
        let bystander_info = darwin::query_bsdinfo(bystander_pid).expect("bystander bsdinfo");
        let bystander_exe = darwin::query_pidpath(bystander_pid).expect("bystander pidpath");

        let active_dir = instance_registry_dir(data.path(), instance_id);
        fs::create_dir_all(&active_dir).unwrap();
        fs::write(
            active_dir.join(format!("proc-srv1-g1-{managed_pid}.json")),
            serde_json::to_vec(&json!({
                "kind": "active",
                "version": 1,
                "instance_id": instance_id.to_string(),
                "sidecar_pid": sidecar_pid,
                "server_id": "srv1",
                "generation": 1,
                "pid": managed_pid,
                "pgid": managed_info.pgid,
                "uid": managed_info.uid,
                "start_tvsec": managed_info.start_tvsec,
                "start_tvusec": managed_info.start_tvusec,
                "executable_path": managed_exe.to_string_lossy(),
                "registered_at_ms": 1,
            }))
            .unwrap(),
        )
        .unwrap();

        // Wrong start_tvsec -> simulates PID reuse
        fs::write(
            active_dir.join(format!("proc-reuse-g1-{bystander_pid}.json")),
            serde_json::to_vec(&json!({
                "kind": "active",
                "version": 1,
                "instance_id": instance_id.to_string(),
                "sidecar_pid": sidecar_pid,
                "server_id": "srv-reuse",
                "generation": 1,
                "pid": bystander_pid,
                "pgid": bystander_info.pgid,
                "uid": bystander_info.uid,
                "start_tvsec": bystander_info.start_tvsec.saturating_sub(120),
                "start_tvusec": bystander_info.start_tvusec,
                "executable_path": bystander_exe.to_string_lossy(),
                "registered_at_ms": 1,
            }))
            .unwrap(),
        )
        .unwrap();

        // Stale old instance registry file pointing at bystander -> cleanup must NOT kill bystander
        let stale_dir = instance_registry_dir(data.path(), old_instance);
        fs::create_dir_all(&stale_dir).unwrap();
        fs::write(
            stale_dir.join(format!("proc-stale-g1-{bystander_pid}.json")),
            serde_json::to_vec(&json!({
                "kind": "active",
                "version": 1,
                "instance_id": old_instance.to_string(),
                "sidecar_pid": 999999,
                "server_id": "srv-stale",
                "generation": 1,
                "pid": bystander_pid,
                "pgid": bystander_info.pgid,
                "uid": bystander_info.uid,
                "start_tvsec": bystander_info.start_tvsec,
                "start_tvusec": bystander_info.start_tvusec,
                "executable_path": bystander_exe.to_string_lossy(),
                "registered_at_ms": 1,
            }))
            .unwrap(),
        )
        .unwrap();

        // Stale cleanup removes old directory without killing bystander.
        let cleaned = cleanup_stale_registries(data.path(), Some(instance_id), &log);
        assert_eq!(cleaned, 1);
        assert!(!stale_dir.exists());
        assert!(pid_alive(bystander_pid), "stale cleanup must never kill live processes");

        // Reaping active instance kills verified stubborn process group, skips PID-reuse bystander.
        let report = reap_instance_processes(data.path(), instance_id, Some(sidecar_pid), &log).await;
        assert_eq!(report.reaped_groups, 1);
        assert_eq!(report.skipped_mismatch, 1);
        assert!(!active_dir.exists());

        let _ = managed.wait();
        assert!(!pid_alive(managed_pid), "verified stubborn MCP process must be reaped");
        assert!(pid_alive(bystander_pid), "mismatched PID reuse bystander must NOT be killed");

        // Cleanup bystander cleanly
        let _ = unsafe { libc::killpg(bystander_info.pgid as libc::pid_t, libc::SIGKILL) };
        let _ = bystander.wait();
    }
}
