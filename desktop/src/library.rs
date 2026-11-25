use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(target_os = "macos")]
mod staging;

#[cfg(target_os = "macos")]
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ImportAction {
    Select { document_id: Option<String> },
    Retry { import_id: Uuid },
    CancelCopy {},
}

#[derive(Debug, Serialize)]
pub(crate) struct ImportError {
    code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    import_id: Option<Uuid>,
}

impl From<&'static str> for ImportError {
    fn from(code: &'static str) -> Self {
        Self {
            code,
            import_id: None,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImportResult {
    pub(crate) import_id: Uuid,
    pub(crate) document_id: String,
    pub(crate) document_version_id: String,
    pub(crate) outcome: ImportOutcome,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ImportOutcome {
    Imported,
    Duplicate,
    Updated,
    Unchanged,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Serialize)]
pub(crate) struct HostImport {
    pub(crate) import_id: Uuid,
    pub(crate) filename: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) document_id: Option<String>,
}

#[cfg(target_os = "macos")]
pub(crate) fn valid_document_id(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(target_os = "macos")]
mod native {
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex as SyncMutex,
    };

    use tauri::{State, WebviewWindow};
    use tokio::sync::{watch, Mutex};

    use super::{
        staging::StagedFile, valid_document_id, HostImport, ImportAction, ImportError, ImportResult,
    };
    use crate::{
        file_picker::{self, NativePicker, Selection},
        sidecar::{HostImportError, Sidecar},
    };
    use uuid::Uuid;

    pub(crate) struct LibraryImporter {
        app_data_dir: PathBuf,
        pending: Mutex<Option<PendingImport>>,
        copying: SyncMutex<Option<Arc<CopyControl>>>,
    }

    struct PendingImport {
        request: HostImport,
        staged: Arc<StagedFile>,
    }

    enum Preparation {
        Selected(PathBuf, Option<String>),
        Retry(HostImport, Arc<StagedFile>),
    }

    impl LibraryImporter {
        pub(crate) fn new(app_data_dir: PathBuf) -> Self {
            Self {
                app_data_dir,
                pending: Mutex::new(None),
                copying: SyncMutex::new(None),
            }
        }

        fn begin_copy(&self, closing: watch::Receiver<bool>) -> CopyGuard<'_> {
            let control = Arc::new(CopyControl {
                cancelled: AtomicBool::new(false),
                closing,
            });
            *self.copying.lock().unwrap() = Some(control.clone());
            CopyGuard {
                active: &self.copying,
                control,
            }
        }

        fn cancel_copy(&self) -> Result<(), ImportError> {
            let active = self.copying.lock().unwrap();
            let control = active.as_ref().ok_or("IMPORT_NOT_COPYING")?;
            control.cancelled.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    pub(super) struct CopyControl {
        pub(super) cancelled: AtomicBool,
        pub(super) closing: watch::Receiver<bool>,
    }

    impl CopyControl {
        pub(super) fn check(&self) -> Result<(), &'static str> {
            if self.cancelled.load(Ordering::Relaxed) || *self.closing.borrow() {
                return Err("IMPORT_CANCELLED");
            }
            Ok(())
        }
    }

    struct CopyGuard<'a> {
        active: &'a SyncMutex<Option<Arc<CopyControl>>>,
        control: Arc<CopyControl>,
    }

    impl CopyGuard<'_> {
        fn finish(&self) -> Result<(), &'static str> {
            // Serialize cancellation with the handoff to HTTP.
            let mut active = self.active.lock().unwrap();
            *active = None;
            self.control.check()
        }
    }

    impl Drop for CopyGuard<'_> {
        fn drop(&mut self) {
            *self.active.lock().unwrap() = None;
            self.control.cancelled.store(true, Ordering::Relaxed);
        }
    }

    fn retry_error(code: &'static str, import_id: Uuid) -> ImportError {
        ImportError {
            code,
            import_id: Some(import_id),
        }
    }

    #[tauri::command]
    pub(crate) async fn select_and_import_document(
        window: WebviewWindow,
        sidecar: State<'_, Sidecar>,
        importer: State<'_, LibraryImporter>,
        picker: State<'_, NativePicker>,
        request: ImportAction,
    ) -> Result<Option<ImportResult>, ImportError> {
        if window.label() != "main" {
            return Err("IMPORT_FORBIDDEN".into());
        }
        if matches!(request, ImportAction::CancelCopy {}) {
            importer.cancel_copy()?;
            return Ok(None);
        }
        let mut pending = importer.pending.try_lock().map_err(|_| "IMPORT_BUSY")?;
        let preparation = match request {
            ImportAction::Select { document_id } => {
                if let Some(previous) = pending.as_ref() {
                    return Err(retry_error(
                        "IMPORT_RETRY_REQUIRED",
                        previous.request.import_id,
                    ));
                }
                if document_id
                    .as_deref()
                    .is_some_and(|id| !valid_document_id(id))
                {
                    return Err("VALIDATION_ERROR".into());
                }
                let selected = {
                    let _panel = picker.0.try_lock().map_err(|_| "IMPORT_PICKER_BUSY")?;
                    file_picker::pick(window, Selection::Textbook, "IMPORT_PICKER_FAILED").await?
                };
                let Some(path) = selected else {
                    return Ok(None);
                };
                Preparation::Selected(PathBuf::from(path), document_id)
            }
            ImportAction::Retry { import_id } => {
                let previous = pending
                    .as_ref()
                    .filter(|entry| entry.request.import_id == import_id)
                    .ok_or("IMPORT_RETRY_NOT_FOUND")?;
                Preparation::Retry(previous.request.clone(), previous.staged.clone())
            }
            ImportAction::CancelCopy {} => unreachable!(),
        };
        let retry_id = match &preparation {
            Preparation::Selected(..) => None,
            Preparation::Retry(request, _) => Some(request.import_id),
        };
        let prepare_error = |code| ImportError {
            code,
            import_id: retry_id,
        };
        let host = sidecar.import_connection().await.map_err(prepare_error)?;
        let instance = host.instance_id();
        let app_data = importer.app_data_dir.clone();
        let copy = importer.begin_copy(sidecar.closing_signal());
        let control = copy.control.clone();
        let (request, staged) = tauri::async_runtime::spawn_blocking(move || match preparation {
            Preparation::Selected(path, document_id) => {
                StagedFile::prepare(&app_data, instance, &path, document_id, &control)
            }
            Preparation::Retry(request, previous) => {
                let staged =
                    previous.for_instance(&app_data, instance, request.import_id, &control)?;
                Ok((request, staged))
            }
        })
        .await
        .map_err(|_| prepare_error("IMPORT_COPY_FAILED"))?
        .map_err(prepare_error)?;
        copy.finish().map_err(prepare_error)?;
        drop(copy);

        // Store the receipt key and the snapshot before the first await that can
        // leave the outcome unknown, including a dropped IPC future.
        let import_id = request.import_id;
        *pending = Some(PendingImport {
            request: request.clone(),
            staged,
        });
        match host.import_document(&request).await {
            Ok(result) => {
                *pending = None;
                Ok(Some(result))
            }
            Err(HostImportError::Rejected(code)) => {
                *pending = None;
                Err(code.into())
            }
            Err(HostImportError::Uncertain(code)) => Err(retry_error(code, import_id)),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::{from_value, json};

        #[test]
        fn import_ipc_uses_only_selection_retry_id_or_cancel() {
            let id = "bf644ca9-bd5f-4cd9-af10-15603dbccf1e";
            for request in [
                json!({"action": "select"}),
                json!({"action": "select", "document_id": "a".repeat(32)}),
                json!({"action": "retry", "import_id": id}),
                json!({"action": "cancel_copy"}),
            ] {
                assert!(from_value::<ImportAction>(request.clone()).is_ok());
                for field in ["path", "filename", "grant_id", "instance_id"] {
                    let mut forged = request.clone();
                    forged[field] = json!("/private/forged.md");
                    assert!(from_value::<ImportAction>(forged).is_err(), "{field}");
                }
            }
            assert!(
                from_value::<ImportAction>(json!({"action": "select", "import_id": id})).is_err()
            );
            assert!(from_value::<ImportAction>(
                json!({"action": "retry", "import_id": id, "document_id": "a".repeat(32)})
            )
            .is_err());
        }

        #[test]
        fn copy_can_be_cancelled_until_it_is_handed_off() {
            let importer = LibraryImporter::new(PathBuf::new());
            let (closing, signal) = watch::channel(false);
            let copy = importer.begin_copy(signal.clone());
            importer.cancel_copy().unwrap();
            assert_eq!(copy.control.check(), Err("IMPORT_CANCELLED"));
            assert_eq!(copy.finish(), Err("IMPORT_CANCELLED"));
            drop(copy);

            let copy = importer.begin_copy(signal.clone());
            copy.finish().unwrap();
            assert_eq!(
                importer.cancel_copy().unwrap_err().code,
                "IMPORT_NOT_COPYING"
            );
            drop(copy);

            let copy = importer.begin_copy(signal);
            closing.send(true).unwrap();
            assert_eq!(copy.control.check(), Err("IMPORT_CANCELLED"));
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) use native::{
    __cmd__select_and_import_document, __tauri_command_name_select_and_import_document,
};
#[cfg(target_os = "macos")]
pub(crate) use native::{select_and_import_document, LibraryImporter};

#[cfg(not(target_os = "macos"))]
#[tauri::command]
pub(crate) async fn select_and_import_document(
    window: tauri::WebviewWindow,
) -> Result<Option<ImportResult>, ImportError> {
    if window.label() != "main" {
        return Err("IMPORT_FORBIDDEN".into());
    }
    Err("IMPORT_UNSUPPORTED_PLATFORM".into())
}
