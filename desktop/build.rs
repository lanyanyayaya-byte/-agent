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
            "mcp_save_server",
            "mcp_import_servers_json",
            "mcp_trust_server",
            "mcp_revoke_trust",
            "mcp_delete_server",
            "mcp_connect_server",
            "mcp_disconnect_server",
            "mcp_refresh_tools",
            "mcp_set_server_enabled",
            "mcp_set_tool_permission",
        ]),
    ))
    .expect("could not build the Tauri app");
}
