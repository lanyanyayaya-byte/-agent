//! Privileged Tauri Host IPC commands for Workspace attachments, Artifacts, and Runtime preparation.
//!
//! WebView never supplies arbitrary filesystem paths:
//! - Attachment import selects files via native `NSOpenPanel` and forwards to Host API.
//! - Artifact Open / Reveal / Export resolves the file path exclusively from Host API.
//! - Runtime preparation forwards user-confirmed dependency requests via Host Token.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{State, WebviewWindow};
use uuid::Uuid;

use crate::{
    file_picker::{self, NativePicker, Selection},
    sidecar::Sidecar,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImportWorkspaceFileRequest {
    session_id: Uuid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImportDroppedFilesRequest {
    session_id: Uuid,
    paths: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeleteWorkspaceFileRequest {
    session_id: Uuid,
    file_id: Uuid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactRefRequest {
    session_id: Uuid,
    artifact_id: Uuid,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimePrepareRequest {
    #[serde(default)]
    python_packages: Vec<String>,
    #[serde(default)]
    install_mermaid_cli: bool,
    #[serde(default = "default_true")]
    ensure_python_venv: bool,
    timeout_seconds: Option<f64>,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct ArtifactHostInfo {
    filename: String,
    absolute_path: String,
    file_exists: bool,
}

#[derive(Serialize)]
pub(crate) struct ExportArtifactResult {
    exported: bool,
    filename: String,
}

fn open_path(path: &Path, reveal: bool) -> bool {
    #[cfg(target_os = "macos")]
    {
        let mut cmd = Command::new("/usr/bin/open");
        if reveal {
            cmd.arg("-R");
        }
        cmd.arg("--")
            .arg(path)
            .status()
            .is_ok_and(|status| status.success())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (path, reveal);
        false
    }
}

async fn resolve_artifact(
    sidecar: &Sidecar,
    session_id: Uuid,
    artifact_id: Uuid,
) -> Result<(ArtifactHostInfo, PathBuf), String> {
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let raw = host.get_artifact_host_path(session_id, artifact_id).await?;
    let info: ArtifactHostInfo =
        serde_json::from_value(raw).map_err(|_| "WORKSPACE_INVALID_RESPONSE".to_owned())?;
    let path = PathBuf::from(&info.absolute_path);
    if !info.file_exists || !path.is_file() {
        return Err("WORKSPACE_ARTIFACT_MISSING".to_owned());
    }
    Ok((info, path))
}

#[tauri::command]
pub(crate) async fn select_and_import_workspace_file(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    picker: State<'_, NativePicker>,
    request: ImportWorkspaceFileRequest,
) -> Result<Option<Value>, String> {
    if window.label() != "main" {
        return Err("WORKSPACE_FORBIDDEN".to_owned());
    }
    let selected = {
        let _panel = picker
            .0
            .try_lock()
            .map_err(|_| "WORKSPACE_PICKER_BUSY".to_owned())?;
        file_picker::pick(window, Selection::WorkspaceFile, "WORKSPACE_PICKER_FAILED")
            .await
            .map_err(str::to_owned)?
    };
    let Some(path) = selected else {
        return Ok(None);
    };
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.import_workspace_file(request.session_id, &path)
        .await
        .map(Some)
}

#[tauri::command]
pub(crate) async fn select_and_import_workspace_files(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    picker: State<'_, NativePicker>,
    request: ImportWorkspaceFileRequest,
) -> Result<Vec<Value>, String> {
    if window.label() != "main" {
        return Err("WORKSPACE_FORBIDDEN".to_owned());
    }
    let paths = {
        let _panel = picker
            .0
            .try_lock()
            .map_err(|_| "WORKSPACE_PICKER_BUSY".to_owned())?;
        file_picker::pick_multiple_workspace_files(window, "WORKSPACE_PICKER_FAILED")
            .await
            .map_err(str::to_owned)?
    };
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let mut imported = Vec::with_capacity(paths.len());
    for path in paths {
        let item = host.import_workspace_file(request.session_id, &path).await?;
        imported.push(item);
    }
    Ok(imported)
}

#[tauri::command]
pub(crate) async fn import_dropped_workspace_files(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: ImportDroppedFilesRequest,
) -> Result<Vec<Value>, String> {
    if window.label() != "main" {
        return Err("WORKSPACE_FORBIDDEN".to_owned());
    }
    if request.paths.is_empty() {
        return Ok(Vec::new());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let mut imported = Vec::new();
    for raw in request.paths.into_iter().take(20) {
        let path = PathBuf::from(&raw);
        if path.is_absolute() && path.is_file() {
            let item = host.import_workspace_file(request.session_id, &raw).await?;
            imported.push(item);
        }
    }
    Ok(imported)
}

#[tauri::command]
pub(crate) async fn delete_workspace_file(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: DeleteWorkspaceFileRequest,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("WORKSPACE_FORBIDDEN".to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.delete_workspace_file(request.session_id, request.file_id)
        .await
}

#[tauri::command]
pub(crate) async fn open_artifact(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: ArtifactRefRequest,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("WORKSPACE_FORBIDDEN".to_owned());
    }
    let (_info, path) = resolve_artifact(&sidecar, request.session_id, request.artifact_id).await?;
    if !open_path(&path, false) {
        return Err("ARTIFACT_OPEN_FAILED".to_owned());
    }
    Ok(())
}

#[tauri::command]
pub(crate) async fn reveal_artifact_in_finder(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: ArtifactRefRequest,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("WORKSPACE_FORBIDDEN".to_owned());
    }
    let (_info, path) = resolve_artifact(&sidecar, request.session_id, request.artifact_id).await?;
    if !open_path(&path, true) {
        return Err("ARTIFACT_REVEAL_FAILED".to_owned());
    }
    Ok(())
}

#[tauri::command]
pub(crate) async fn export_artifact(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    picker: State<'_, NativePicker>,
    request: ArtifactRefRequest,
) -> Result<Option<ExportArtifactResult>, String> {
    if window.label() != "main" {
        return Err("WORKSPACE_FORBIDDEN".to_owned());
    }
    let (info, src_path) =
        resolve_artifact(&sidecar, request.session_id, request.artifact_id).await?;
    let destination = {
        let _panel = picker
            .0
            .try_lock()
            .map_err(|_| "WORKSPACE_PICKER_BUSY".to_owned())?;
        file_picker::pick_save_destination(window, info.filename.clone(), "ARTIFACT_SAVE_FAILED")
            .await
            .map_err(str::to_owned)?
    };
    let Some(dest_str) = destination else {
        return Ok(None);
    };
    let dest_path = PathBuf::from(dest_str);
    tauri::async_runtime::spawn_blocking(move || fs::copy(&src_path, &dest_path))
        .await
        .map_err(|_| "ARTIFACT_EXPORT_FAILED".to_owned())?
        .map_err(|_| "ARTIFACT_EXPORT_FAILED".to_owned())?;
    Ok(Some(ExportArtifactResult {
        exported: true,
        filename: info.filename,
    }))
}

#[tauri::command]
pub(crate) async fn runtime_prepare(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: RuntimePrepareRequest,
) -> Result<Value, String> {
    if window.label() != "main" {
        return Err("RUNTIME_FORBIDDEN".to_owned());
    }
    let mut body = json!({
        "python_packages": request.python_packages,
        "install_mermaid_cli": request.install_mermaid_cli,
        "ensure_python_venv": request.ensure_python_venv,
    });
    if let Some(timeout) = request.timeout_seconds {
        body["timeout_seconds"] = json!(timeout);
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.prepare_runtime(&body).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::from_value;

    #[test]
    fn workspace_ipc_rejects_arbitrary_paths() {
        let sid = "d2ab1e6d-3f55-485e-bfab-14ece5424c0b";
        let aid = "bf644ca9-bd5f-4cd9-af10-15603dbccf1e";
        assert!(from_value::<ImportWorkspaceFileRequest>(json!({"session_id": sid})).is_ok());
        assert!(
            from_value::<ImportWorkspaceFileRequest>(json!({"session_id": sid, "path": "/etc/passwd"}))
                .is_err()
        );
        assert!(
            from_value::<ArtifactRefRequest>(json!({"session_id": sid, "artifact_id": aid})).is_ok()
        );
        assert!(
            from_value::<ArtifactRefRequest>(json!({
                "session_id": sid,
                "artifact_id": aid,
                "absolute_path": "/etc/passwd"
            }))
            .is_err()
        );
    }
}
