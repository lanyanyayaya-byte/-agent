use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

const MAX_LOG_BYTES: u64 = 1_048_576;
pub(crate) const LOG_DIRECTORY_OPEN_FAILED: &str = "LOG_DIRECTORY_OPEN_FAILED";

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenLogDirectoryRequest {}

#[derive(Clone)]
pub(crate) struct HostLog(Arc<Mutex<PathBuf>>);

impl HostLog {
    pub(crate) fn new(app_data_dir: &Path) -> Self {
        Self(Arc::new(Mutex::new(app_data_dir.join("logs"))))
    }

    pub(crate) fn record(&self, event: &'static str, fields: Value) {
        let mut entry = json!({
            "event": event,
            "ts_ms": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
            "pid": std::process::id(),
        });
        if let (Some(entry), Value::Object(fields)) = (entry.as_object_mut(), fields) {
            entry.extend(fields);
        }
        if let Ok(dir) = self.0.lock() {
            if write_entry(&dir, entry).is_err() {
                eprintln!("KnowledgeAgent: could not write host diagnostics.");
            }
        }
    }

    pub(crate) fn open_directory(&self) -> Result<PathBuf, &'static str> {
        self.open_directory_with(open_in_system_file_manager)
    }

    pub(crate) fn open_directory_with(
        &self,
        opener: impl FnOnce(&Path) -> bool,
    ) -> Result<PathBuf, &'static str> {
        let dir = self
            .0
            .lock()
            .map_err(|_| LOG_DIRECTORY_OPEN_FAILED)?
            .clone();
        if ensure_log_dir(&dir).is_err() {
            return Err(LOG_DIRECTORY_OPEN_FAILED);
        }
        if !opener(&dir) {
            self.record("log_directory_open_error", json!({}));
            return Err(LOG_DIRECTORY_OPEN_FAILED);
        }
        self.record("log_directory_opened", json!({}));
        Ok(dir)
    }
}

#[tauri::command]
pub(crate) async fn open_log_directory(
    log: tauri::State<'_, HostLog>,
    #[allow(unused_variables)] request: Option<OpenLogDirectoryRequest>,
) -> Result<(), &'static str> {
    log.open_directory().map(|_| ())
}

fn open_in_system_file_manager(dir: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        Command::new("/usr/bin/open")
            .arg("--")
            .arg(dir)
            .status()
            .is_ok_and(|status| status.success())
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer.exe")
            .arg(dir)
            .status()
            .is_ok_and(|status| status.success())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(dir)
            .status()
            .is_ok_and(|status| status.success())
    }
}

fn ensure_log_dir(dir: &Path) -> io::Result<()> {
    if !dir.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "log directory must be an absolute path",
        ));
    }
    if let Ok(existing) = fs::symlink_metadata(dir) {
        if existing.file_type().is_symlink() || !existing.file_type().is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "log path is not a regular directory",
            ));
        }
    } else {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(dir)?;
    }
    let metadata = fs::symlink_metadata(dir)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "log path is not a regular directory",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn write_entry(dir: &Path, entry: Value) -> io::Result<()> {
    ensure_log_dir(dir)?;
    let path = dir.join("host.jsonl");
    let mut line = serde_json::to_vec(&entry)?;
    line.push(b'\n');
    if fs::metadata(&path).is_ok_and(|file| file.len() + line.len() as u64 > MAX_LOG_BYTES) {
        fs::rename(&path, dir.join("host.jsonl.1"))?;
    }
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)?.write_all(&line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::from_value;
    use tempfile::tempdir;

    #[test]
    fn open_log_directory_ipc_rejects_arbitrary_path_arguments() {
        assert!(from_value::<OpenLogDirectoryRequest>(json!({})).is_ok());
        for (field, value) in [
            ("path", json!("/etc")),
            ("dir", json!("/Users")),
            ("target", json!("../../secret")),
        ] {
            assert!(from_value::<OpenLogDirectoryRequest>(json!({ field: value })).is_err());
        }
    }

    #[test]
    fn open_directory_validates_fixed_log_dir_rejects_symlinks_and_rotates_logs() {
        let data = tempdir().unwrap();
        let log = HostLog::new(data.path());
        let expected_dir = data.path().join("logs");

        let mut opened = None;
        let result = log.open_directory_with(|dir| {
            opened = Some(dir.to_path_buf());
            true
        });
        assert_eq!(result.unwrap(), expected_dir);
        assert_eq!(opened, Some(expected_dir.clone()));

        let failed = log.open_directory_with(|_| false);
        assert_eq!(failed.err(), Some(LOG_DIRECTORY_OPEN_FAILED));

        let host_log = fs::read_to_string(expected_dir.join("host.jsonl")).unwrap();
        assert!(host_log.contains("log_directory_opened"));
        assert!(host_log.contains("log_directory_open_error"));

        // Verify log rotation when exceeding 1 MiB.
        let large_value = "x".repeat(600_000);
        log.record("rotation_probe_1", json!({ "padding": large_value }));
        log.record("rotation_probe_2", json!({ "padding": large_value }));
        assert!(expected_dir.join("host.jsonl.1").is_file());
        assert!(expected_dir.join("host.jsonl").is_file());

        // Replace logs directory with a symlink or file and ensure open_directory_with rejects it.
        #[cfg(unix)]
        {
            let symlink_root = tempdir().unwrap();
            let outside = tempdir().unwrap();
            let symlink_logs = symlink_root.path().join("logs");
            std::os::unix::fs::symlink(outside.path(), &symlink_logs).unwrap();
            let symlink_host_log = HostLog::new(symlink_root.path());
            let mut called = false;
            assert_eq!(
                symlink_host_log.open_directory_with(|_| {
                    called = true;
                    true
                }),
                Err(LOG_DIRECTORY_OPEN_FAILED)
            );
            assert!(!called);
        }
    }
}
