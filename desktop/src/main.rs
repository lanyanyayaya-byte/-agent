#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod diagnostics;
mod file_picker;
mod grants;
mod keychain;
mod library;
mod sidecar;

use serde::Serialize;
use tauri::{Manager, RunEvent};
use zeroize::Zeroizing;

#[derive(Clone, Serialize)]
struct BackendConnection {
    base_url: String,
    ui_token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    siliconflow_restore_error: Option<&'static str>,
}

#[tauri::command]
async fn get_backend_connection(
    sidecar: tauri::State<'_, sidecar::Sidecar>,
) -> Result<BackendConnection, &'static str> {
    sidecar.connection().await
}

#[tauri::command]
async fn save_provider_credential(
    sidecar: tauri::State<'_, sidecar::Sidecar>,
    provider: String,
    api_key: String,
) -> Result<sidecar::ModelConfiguration, &'static str> {
    sidecar
        .save_provider_credential(&provider, Zeroizing::new(api_key))
        .await
}

#[tauri::command]
async fn delete_provider_credential(
    sidecar: tauri::State<'_, sidecar::Sidecar>,
    provider: String,
) -> Result<sidecar::ModelConfiguration, &'static str> {
    sidecar.delete_provider_credential(&provider).await
}

#[tauri::command]
async fn test_siliconflow_connection(
    sidecar: tauri::State<'_, sidecar::Sidecar>,
) -> Result<sidecar::EmbeddingConnection, &'static str> {
    sidecar.test_siliconflow_connection().await
}

fn main() {
    let builder = tauri::Builder::default();
    #[cfg(target_os = "macos")]
    let builder = builder
        .menu(|app| {
            use tauri::menu::{Menu, MenuItem};
            let menu = Menu::default(app)?;
            for item in menu.items()? {
                if let Some(submenu) = item.as_submenu() {
                    if submenu.text()? == "View" {
                        submenu.insert(
                            &MenuItem::with_id(
                                app,
                                "reload-chat",
                                "重新加载界面",
                                true,
                                Some("Cmd+R"),
                            )?,
                            0,
                        )?;
                        break;
                    }
                }
            }
            Ok(menu)
        })
        .on_menu_event(|app, event| {
            if event.id() == "reload-chat" {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.reload();
                }
            }
        });
    builder
        .manage(file_picker::NativePicker::default())
        .invoke_handler(tauri::generate_handler![
            get_backend_connection,
            save_provider_credential,
            delete_provider_credential,
            test_siliconflow_connection,
            diagnostics::open_log_directory,
            grants::select_file_grant,
            grants::revoke_file_grant,
            library::select_and_import_document
        ])
        .setup(|app| {
            let app_data_dir = app.path().app_data_dir()?;
            app.manage(diagnostics::HostLog::new(&app_data_dir));
            #[cfg(target_os = "macos")]
            app.manage(library::LibraryImporter::new(app_data_dir.clone()));
            app.manage(sidecar::Sidecar::launch(app.handle(), app_data_dir)?);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("could not start KnowledgeAgent")
        .run(|app, event| match event {
            RunEvent::ExitRequested { api, code, .. } => {
                let sidecar = app.state::<sidecar::Sidecar>();
                if sidecar.is_stopped() {
                    return;
                }
                api.prevent_exit();
                if sidecar.request_shutdown() {
                    let app = app.clone();
                    let sidecar = sidecar.inner().clone();
                    tauri::async_runtime::spawn(async move {
                        sidecar.wait_stopped().await;
                        app.exit(code.unwrap_or(0));
                    });
                }
            }
            #[cfg(target_os = "macos")]
            RunEvent::Exit => {
                // Cocoa can send applicationWillTerminate directly, including from Cmd+Q.
                let sidecar = app.state::<sidecar::Sidecar>().inner().clone();
                if !sidecar.is_stopped() {
                    sidecar.request_shutdown();
                    tauri::async_runtime::block_on(sidecar.wait_stopped());
                }
            }
            _ => {}
        });
}
