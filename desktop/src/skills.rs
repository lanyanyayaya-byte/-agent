//! Privileged Tauri Host IPC commands for Agent Skills management (GitHub, local folder, ZIP).

use serde::Deserialize;
use serde_json::{json, Value};
use tauri::{State, WebviewWindow};

use crate::{
    file_picker::{self, NativePicker, Selection},
    sidecar::Sidecar,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreviewGitHubSkillsRequest {
    url: String,
    #[serde(default)]
    ref_name: Option<String>,
    #[serde(default)]
    subpath: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PickLocalSkillSourceRequest {
    mode: String, // "directory" | "zip"
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InstallSkillsRequest {
    preview_id: String,
    selected_relative_paths: Vec<String>,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default = "default_true")]
    replace_existing: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SkillIdRequest {
    skill_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SetSkillEnabledRequest {
    skill_id: String,
    enabled: bool,
}

fn default_true() -> bool {
    true
}

fn valid_skill_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

#[tauri::command]
pub(crate) async fn skills_preview_github(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: PreviewGitHubSkillsRequest,
) -> Result<Value, String> {
    if window.label() != "main" {
        return Err("SKILLS_FORBIDDEN".to_owned());
    }
    let mut body = json!({
        "source_kind": "github",
        "source": request.url,
    });
    if let Some(ref_val) = request.ref_name.filter(|s| !s.trim().is_empty()) {
        body["ref"] = json!(ref_val);
    }
    if let Some(sub) = request.subpath.filter(|s| !s.trim().is_empty()) {
        body["subpath"] = json!(sub);
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.skills_import_preview(&body).await
}

#[tauri::command]
pub(crate) async fn skills_pick_and_preview_local(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    picker: State<'_, NativePicker>,
    request: PickLocalSkillSourceRequest,
) -> Result<Option<Value>, String> {
    if window.label() != "main" {
        return Err("SKILLS_FORBIDDEN".to_owned());
    }
    let (selection, source_kind) = match request.mode.as_str() {
        "directory" => (Selection::SkillDirectory, "local_dir"),
        "zip" => (Selection::SkillZip, "local_zip"),
        _ => return Err("SKILL_INVALID_SOURCE".to_owned()),
    };
    let selected = {
        let _panel = picker
            .0
            .try_lock()
            .map_err(|_| "SKILLS_PICKER_BUSY".to_owned())?;
        file_picker::pick(window, selection, "SKILLS_PICKER_FAILED")
            .await
            .map_err(str::to_owned)?
    };
    let Some(path) = selected else {
        return Ok(None);
    };
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let body = json!({
        "source_kind": source_kind,
        "source": path,
    });
    host.skills_import_preview(&body).await.map(Some)
}

#[tauri::command]
pub(crate) async fn skills_install_from_preview(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: InstallSkillsRequest,
) -> Result<Value, String> {
    if window.label() != "main" {
        return Err("SKILLS_FORBIDDEN".to_owned());
    }
    let body = json!({
        "preview_id": request.preview_id,
        "selected_relative_paths": request.selected_relative_paths,
        "enabled": request.enabled,
        "replace_existing": request.replace_existing,
    });
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.skills_install(&body).await
}

#[tauri::command]
pub(crate) async fn skills_set_enabled(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: SetSkillEnabledRequest,
) -> Result<Value, String> {
    if window.label() != "main" {
        return Err("SKILLS_FORBIDDEN".to_owned());
    }
    if !valid_skill_id(&request.skill_id) {
        return Err("SKILL_NOT_FOUND".to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.skills_set_enabled(&request.skill_id, request.enabled)
        .await
}

#[tauri::command]
pub(crate) async fn skills_update(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: SkillIdRequest,
) -> Result<Value, String> {
    if window.label() != "main" {
        return Err("SKILLS_FORBIDDEN".to_owned());
    }
    if !valid_skill_id(&request.skill_id) {
        return Err("SKILL_NOT_FOUND".to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.skills_update(&request.skill_id).await
}

#[tauri::command]
pub(crate) async fn skills_prepare(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: SkillIdRequest,
) -> Result<Value, String> {
    if window.label() != "main" {
        return Err("SKILLS_FORBIDDEN".to_owned());
    }
    if !valid_skill_id(&request.skill_id) {
        return Err("SKILL_NOT_FOUND".to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.skills_prepare(&request.skill_id).await
}

#[tauri::command]
pub(crate) async fn skills_delete(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    request: SkillIdRequest,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("SKILLS_FORBIDDEN".to_owned());
    }
    if !valid_skill_id(&request.skill_id) {
        return Err("SKILL_NOT_FOUND".to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.skills_delete(&request.skill_id).await
}
