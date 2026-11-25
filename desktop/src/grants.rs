use serde::{Deserialize, Serialize};
use tauri::{State, WebviewWindow};
use uuid::Uuid;

use crate::{
    file_picker::{self, NativePicker, Selection},
    sidecar::Sidecar,
};

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GrantScope {
    File,
    Directory,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Read,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GrantView {
    pub(crate) grant_id: Uuid,
    pub(crate) session_id: Uuid,
    pub(crate) scope: GrantScope,
    display_name: String,
    operations: [Operation; 1],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelectGrant {
    session_id: Uuid,
    scope: GrantScope,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RevokeGrant {
    session_id: Uuid,
    grant_id: Uuid,
}

#[tauri::command]
pub(crate) async fn select_file_grant(
    window: WebviewWindow,
    sidecar: State<'_, Sidecar>,
    picker: State<'_, NativePicker>,
    request: SelectGrant,
) -> Result<Option<GrantView>, &'static str> {
    let _open_panel = picker.0.try_lock().map_err(|_| "GRANT_PICKER_BUSY")?;
    let selection = match request.scope {
        GrantScope::File => Selection::GrantFile,
        GrantScope::Directory => Selection::GrantDirectory,
    };
    sidecar
        .select_grant(
            request.session_id,
            request.scope,
            file_picker::pick(window, selection, "GRANT_PICKER_FAILED"),
        )
        .await
}

#[tauri::command]
pub(crate) async fn revoke_file_grant(
    sidecar: State<'_, Sidecar>,
    request: RevokeGrant,
) -> Result<(), &'static str> {
    sidecar
        .revoke_grant(request.session_id, request.grant_id)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{from_value, json};

    #[test]
    fn ipc_rejects_frontend_supplied_paths_and_grant_fields() {
        let session_id = "d2ab1e6d-3f55-485e-bfab-14ece5424c0b";
        let grant_id = "bf644ca9-bd5f-4cd9-af10-15603dbccf1e";
        let mut selection = json!({"session_id": session_id, "scope": "directory"});
        assert!(from_value::<SelectGrant>(selection.clone()).is_ok());
        for (field, value) in [
            ("path", json!("/private/forged")),
            ("grant_id", json!(grant_id)),
            ("operations", json!(["write"])),
            ("lifetime", json!("persistent")),
        ] {
            selection[field] = value;
            assert!(from_value::<SelectGrant>(selection.clone()).is_err());
            selection.as_object_mut().unwrap().remove(field);
        }
        let mut revoke = json!({"session_id": session_id, "grant_id": grant_id});
        assert!(from_value::<RevokeGrant>(revoke.clone()).is_ok());
        revoke["path"] = json!("/private/forged");
        assert!(from_value::<RevokeGrant>(revoke).is_err());
    }
}
