fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "get_backend_connection",
            "save_provider_credential",
            "delete_provider_credential",
            "test_siliconflow_connection",
            "open_log_directory",
            "select_file_grant",
            "revoke_file_grant",
            "select_and_import_document",
        ]),
    ))
    .expect("could not build the Tauri app");
}
