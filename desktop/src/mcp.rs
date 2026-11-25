//! Privileged MCP management IPC commands, JSON import auditing, Keychain/SQLite
//! compensation flows, and macOS native `NSAlert` trust confirmation dialog.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::WebviewWindow;
use tokio::sync::Mutex;
use zeroize::Zeroizing;

use crate::{
    keychain::{
        is_valid_mcp_server_id, validate_mcp_secrets, KeychainError, McpCredentialStore,
        MCP_CREDENTIAL_REJECTED,
    },
    sidecar::{HostConnection, Sidecar},
};

pub(crate) const MCP_TRUST_CANCELLED: &str = "MCP_TRUST_CANCELLED";
pub(crate) const MCP_TRUST_DIALOG_FAILED: &str = "MCP_TRUST_DIALOG_FAILED";
pub(crate) const MCP_CREDENTIAL_REPLAY_FAILED: &str = "MCP_CREDENTIAL_REPLAY_FAILED";
pub(crate) const MCP_KEYCHAIN_CLEANUP_FAILED: &str = "MCP_KEYCHAIN_CLEANUP_FAILED";
pub(crate) const MCP_INVALID_REQUEST: &str = "MCP_INVALID_CONFIG";

const MAX_IMPORT_JSON_BYTES: usize = 256 * 1024;
const MAX_IMPORT_SERVERS: usize = 16;

#[derive(Default)]
pub(crate) struct NativeTrustPrompt(pub(crate) Mutex<()>);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ConfigEntryRequest {
    Plain { value: String },
    Secret { secret_key: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SaveMcpServerRequest {
    pub(crate) server_id: Option<String>,
    pub(crate) name: String,
    pub(crate) transport: String,
    pub(crate) command: Option<String>,
    #[serde(default)]
    pub(crate) args: Vec<String>,
    pub(crate) cwd: Option<String>,
    #[serde(default)]
    pub(crate) clear_cwd: bool,
    pub(crate) url: Option<String>,
    #[serde(default)]
    pub(crate) allow_private_network: bool,
    #[serde(default)]
    pub(crate) env_config: BTreeMap<String, ConfigEntryRequest>,
    #[serde(default)]
    pub(crate) header_config: BTreeMap<String, ConfigEntryRequest>,
    /// Optional plaintext secret values to write/rotate in Keychain `{identifier}.mcp-credentials`.
    /// Never persisted to SQLite or logged.
    #[serde(default)]
    pub(crate) secret_values: BTreeMap<String, String>,
    /// Secret keys to remove from Keychain for an updated server.
    #[serde(default)]
    pub(crate) remove_secret_keys: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImportMcpServersRequest {
    pub(crate) raw_json: String,
}

#[derive(Serialize)]
pub(crate) struct ImportMcpServersResult {
    pub(crate) servers: Vec<Value>,
    pub(crate) warnings: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrustMcpServerRequest {
    pub(crate) server_id: String,
    #[serde(default)]
    pub(crate) enable: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ServerIdRequest {
    pub(crate) server_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SetServerEnabledRequest {
    pub(crate) server_id: String,
    pub(crate) enabled: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SetToolPermissionRequest {
    pub(crate) server_id: String,
    pub(crate) tool_id: String,
    pub(crate) permission_state: String,
    pub(crate) expected_definition_hash: Option<String>,
    pub(crate) expected_config_revision: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TrustPreview {
    pub(crate) server_id: String,
    pub(crate) name: String,
    pub(crate) transport: String,
    pub(crate) command: Option<String>,
    pub(crate) resolved_command: Option<String>,
    pub(crate) executable_exists: Option<bool>,
    #[serde(default)]
    pub(crate) args: Vec<String>,
    pub(crate) cwd: Option<String>,
    pub(crate) url: Option<String>,
    #[serde(default)]
    pub(crate) private_network_target: bool,
    #[serde(default)]
    pub(crate) high_risk_command: bool,
    #[serde(default)]
    pub(crate) high_risk_reasons: Vec<String>,
    #[serde(default)]
    pub(crate) env_config: BTreeMap<String, Value>,
    #[serde(default)]
    pub(crate) header_config: BTreeMap<String, Value>,
    pub(crate) config_revision: u64,
    pub(crate) current_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrustDialogContent {
    pub(crate) title: String,
    pub(crate) body: String,
    pub(crate) critical: bool,
}

/// Build the strictly redacted confirmation copy shown in the native macOS `NSAlert`.
/// Never includes secret values; binds user approval visually to config_revision and fingerprint.
pub(crate) fn format_trust_dialog(preview: &TrustPreview) -> TrustDialogContent {
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("服务名称：{}", preview.name));

    if preview.transport == "stdio" {
        lines.push("传输方式：stdio（本地外部可执行程序）".to_owned());
        let exe_display = preview
            .resolved_command
            .as_deref()
            .or(preview.command.as_deref())
            .unwrap_or("(未指定)");
        lines.push(format!("实际可执行路径：{exe_display}"));
        if preview.executable_exists == Some(false) {
            lines.push("状态提示：当前路径下未找到可执行程序。".to_owned());
        }
        if !preview.args.is_empty() {
            lines.push(format!("脱敏启动参数：{}", preview.args.join(" ")));
        }
        if let Some(ref cwd) = preview.cwd {
            lines.push(format!("工作目录：{cwd}"));
        }
        if !preview.env_config.is_empty() {
            let keys: Vec<&str> = preview.env_config.keys().map(String::as_str).collect();
            lines.push(format!("环境变量键（秘钥已脱敏）：{}", keys.join(", ")));
        }
    } else {
        lines.push("传输方式：Streamable HTTP（网络 MCP 服务）".to_owned());
        lines.push(format!(
            "服务地址：{}",
            preview.url.as_deref().unwrap_or("(未指定)")
        ));
        if !preview.header_config.is_empty() {
            let keys: Vec<&str> = preview.header_config.keys().map(String::as_str).collect();
            lines.push(format!("请求头键（凭据已脱敏）：{}", keys.join(", ")));
        }
    }

    lines.push(String::new());
    lines.push("安全风险提示：".to_owned());
    if preview.transport == "stdio" {
        lines.push("• 本地 stdio 服务不具备操作系统级文件沙箱，将以当前用户权限运行。".to_owned());
        lines.push("• 仅校验路径无法保证同一路径下的程序内容在后续未被替换。".to_owned());
        if preview.high_risk_command {
            lines.push("• 【高风险入口警告】检测到动态包执行器或脚本解释器，可能在运行时下载或执行外部代码：".to_owned());
            for reason in &preview.high_risk_reasons {
                lines.push(format!("  - {reason}"));
            }
        }
    } else if preview.private_network_target {
        lines.push("• 【内网地址提示】目标指向私有内网地址，请确认该服务受信任。".to_owned());
    }
    lines.push("• 外部 MCP 工具可能产生数据外发或外部副作用；信任服务后仍需逐个工具单独授权。".to_owned());

    let short_fp: String = preview.current_fingerprint.chars().take(12).collect();
    lines.push(String::new());
    lines.push(format!(
        "绑定校验：配置版本 r{} · 指纹 {}…",
        preview.config_revision, short_fp
    ));

    let title = if preview.high_risk_command {
        format!("高风险确认：是否信任本地 MCP 服务「{}」？", preview.name)
    } else {
        format!("确认信任 MCP 服务「{}」？", preview.name)
    };

    TrustDialogContent {
        title,
        body: lines.join("\n"),
        critical: preview.high_risk_command,
    }
}

#[cfg(target_os = "macos")]
async fn confirm_trust_in_native_alert(
    window: WebviewWindow,
    content: TrustDialogContent,
) -> Result<bool, &'static str> {
    use std::cell::Cell;

    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{
        NSAlert, NSAlertSecondButtonReturn, NSAlertStyle, NSModalResponse, NSWindow,
    };
    use objc2_foundation::NSString;
    use tokio::sync::oneshot;

    let (reply, result) = oneshot::channel();
    let parent = window.clone();
    window
        .run_on_main_thread(move || {
            let (Some(main_thread), Ok(parent_ptr)) = (MainThreadMarker::new(), parent.ns_window())
            else {
                let _ = reply.send(Err(MCP_TRUST_DIALOG_FAILED));
                return;
            };
            let Some(parent_win) = (unsafe { parent_ptr.cast::<NSWindow>().as_ref() }) else {
                let _ = reply.send(Err(MCP_TRUST_DIALOG_FAILED));
                return;
            };

            let alert = NSAlert::new(main_thread);
            alert.setAlertStyle(if content.critical {
                NSAlertStyle::Critical
            } else {
                NSAlertStyle::Warning
            });
            alert.setMessageText(&NSString::from_str(&content.title));
            alert.setInformativeText(&NSString::from_str(&content.body));
            // First button (default Enter) is Cancel so accidental confirmation cannot trust a server.
            alert.addButtonWithTitle(&NSString::from_str("取消"));
            alert.addButtonWithTitle(&NSString::from_str("确认信任"));

            let reply = Cell::new(Some(reply));
            let completion = RcBlock::new(move |response: NSModalResponse| {
                let confirmed = response == NSAlertSecondButtonReturn;
                if let Some(tx) = reply.take() {
                    let _ = tx.send(Ok(confirmed));
                }
            });
            alert.beginSheetModalForWindow_completionHandler(parent_win, Some(&completion));
        })
        .map_err(|_| MCP_TRUST_DIALOG_FAILED)?;

    result.await.map_err(|_| MCP_TRUST_DIALOG_FAILED)?
}

#[cfg(not(target_os = "macos"))]
async fn confirm_trust_in_native_alert(
    _window: WebviewWindow,
    _content: TrustDialogContent,
) -> Result<bool, &'static str> {
    Err(MCP_TRUST_DIALOG_FAILED)
}

fn declared_secret_keys(req: &SaveMcpServerRequest) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    let map = if req.transport == "stdio" {
        &req.env_config
    } else {
        &req.header_config
    };
    for (k, entry) in map {
        if matches!(entry, ConfigEntryRequest::Secret { .. }) {
            keys.insert(k.clone());
        }
    }
    keys
}

fn build_server_payload(req: &SaveMcpServerRequest, is_update: bool) -> Value {
    let mut payload = json!({
        "name": req.name,
        "transport": req.transport,
        "command": req.command,
        "args": req.args,
        "cwd": req.cwd,
        "url": req.url,
        "allow_private_network": req.allow_private_network,
        "env_config": req.env_config,
        "header_config": req.header_config,
    });
    if is_update {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("clear_cwd".to_owned(), Value::Bool(req.clear_cwd));
        }
    }
    payload
}

/// Core save/update orchestration across Keychain (`{id}.mcp-credentials`), SQLite, and Sidecar memory.
/// Explicitly implements failure compensation without pretending cross-system atomicity exists.
pub(crate) async fn save_server_with_store(
    host: &HostConnection,
    mcp_credentials: &Arc<McpCredentialStore>,
    req: SaveMcpServerRequest,
) -> Result<Value, String> {
    let declared = declared_secret_keys(&req);
    let mut incoming_secrets: BTreeMap<String, Zeroizing<String>> = BTreeMap::new();
    for (k, v) in &req.secret_values {
        if !declared.contains(k) {
            return Err(MCP_CREDENTIAL_REJECTED.to_owned());
        }
        incoming_secrets.insert(k.clone(), Zeroizing::new(v.clone()));
    }
    if !incoming_secrets.is_empty() {
        validate_mcp_secrets(&incoming_secrets).map_err(|_| MCP_CREDENTIAL_REJECTED.to_owned())?;
    }

    let _guard = mcp_credentials.lock().await;

    if let Some(ref server_id) = req.server_id {
        if !is_valid_mcp_server_id(server_id) {
            return Err(MCP_INVALID_REQUEST.to_owned());
        }
        let previous = match mcp_credentials.load(server_id) {
            Ok(map) => map,
            Err(KeychainError::Corrupted) => None,
            Err(error) => {
                host.log().record(
                    "mcp_keychain_read_error",
                    json!({
                        "instance_id": host.host_instance_id(),
                        "service": mcp_credentials.service(),
                        "server_id": server_id,
                        "reason": error.diagnostic_reason(),
                        "os_status": error.os_status(),
                    }),
                );
                return Err(error.ipc_code().to_owned());
            }
        };

        // Merge existing secrets kept by declared keys, remove deleted keys, apply newly entered secrets.
        let remove_set: BTreeSet<&str> = req.remove_secret_keys.iter().map(String::as_str).collect();
        let mut target_secrets: BTreeMap<String, Zeroizing<String>> = previous
            .clone()
            .unwrap_or_default()
            .into_iter()
            .filter(|(k, _)| declared.contains(k) && !remove_set.contains(k.as_str()))
            .collect();
        for (k, v) in incoming_secrets {
            target_secrets.insert(k, v);
        }

        let keychain_changed = previous.as_ref() != Some(&target_secrets)
            && !(previous.is_none() && target_secrets.is_empty());

        if keychain_changed {
            if let Err(error) = mcp_credentials.save(server_id, &target_secrets) {
                host.log().record(
                    "mcp_keychain_write_error",
                    json!({
                        "instance_id": host.host_instance_id(),
                        "service": mcp_credentials.service(),
                        "server_id": server_id,
                        "reason": error.diagnostic_reason(),
                        "os_status": error.os_status(),
                    }),
                );
                return Err(error.ipc_code().to_owned());
            }
        }

        let payload = build_server_payload(&req, true);
        let updated = match host.update_mcp_server(server_id, &payload).await {
            Ok(view) => view,
            Err(db_err) => {
                // Compensation: DB save failed -> roll back Keychain to prior snapshot.
                if keychain_changed {
                    let rollback_ok = match previous {
                        Some(ref old) => mcp_credentials.save(server_id, old).is_ok(),
                        None => mcp_credentials.delete(server_id).is_ok(),
                    };
                    host.log().record(
                        "mcp_keychain_rollback",
                        json!({
                            "instance_id": host.host_instance_id(),
                            "server_id": server_id,
                            "rollback_ok": rollback_ok,
                            "trigger": db_err,
                        }),
                    );
                }
                return Err(db_err);
            }
        };

        // Replay updated secrets into Sidecar memory. If this fails, do NOT roll back Keychain
        // because Keychain now matches SQLite; the server remains untrusted/disconnected until replayed.
        let replay_result = if target_secrets.is_empty() {
            host.delete_mcp_credentials(server_id).await
        } else {
            host.put_mcp_credentials(server_id, &target_secrets, "update")
                .await
                .map(|_| ())
        };
        if let Err(code) = replay_result {
            host.log().record(
                "mcp_credential_replay_error",
                json!({
                    "instance_id": host.host_instance_id(),
                    "server_id": server_id,
                    "code": code,
                }),
            );
            return Err(MCP_CREDENTIAL_REPLAY_FAILED.to_owned());
        }

        // Return refreshed view reflecting updated credentials_complete status.
        Ok(host.get_mcp_server(server_id).await.unwrap_or(updated))
    } else {
        // Creating a new server: create untrusted+disabled row first to obtain canonical server_id.
        let payload = build_server_payload(&req, false);
        let created = host.create_mcp_server(&payload).await?;
        let new_id = created
            .get("server_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "MCP_INVALID_RESPONSE".to_owned())?
            .to_owned();

        if !incoming_secrets.is_empty() {
            if let Err(error) = mcp_credentials.save(&new_id, &incoming_secrets) {
                host.log().record(
                    "mcp_keychain_write_error",
                    json!({
                        "instance_id": host.host_instance_id(),
                        "service": mcp_credentials.service(),
                        "server_id": new_id,
                        "reason": error.diagnostic_reason(),
                        "os_status": error.os_status(),
                    }),
                );
                // Compensation: Keychain write failed -> delete newly created untrusted DB row.
                let db_rollback_ok = host.delete_mcp_server(&new_id).await.is_ok();
                host.log().record(
                    "mcp_db_rollback_after_keychain_error",
                    json!({
                        "instance_id": host.host_instance_id(),
                        "server_id": new_id,
                        "rollback_ok": db_rollback_ok,
                    }),
                );
                return Err(error.ipc_code().to_owned());
            }

            if let Err(code) = host
                .put_mcp_credentials(&new_id, &incoming_secrets, "update")
                .await
            {
                host.log().record(
                    "mcp_credential_replay_error",
                    json!({
                        "instance_id": host.host_instance_id(),
                        "server_id": new_id,
                        "code": code,
                    }),
                );
                return Err(MCP_CREDENTIAL_REPLAY_FAILED.to_owned());
            }
            return Ok(host.get_mcp_server(&new_id).await.unwrap_or(created));
        }

        Ok(created)
    }
}

pub(crate) async fn delete_server_with_store(
    host: &HostConnection,
    mcp_credentials: &Arc<McpCredentialStore>,
    server_id: &str,
) -> Result<(), String> {
    if !is_valid_mcp_server_id(server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    let _guard = mcp_credentials.lock().await;
    host.delete_mcp_server(server_id).await?;
    if let Err(error) = mcp_credentials.delete(server_id) {
        host.log().record(
            "mcp_keychain_delete_error",
            json!({
                "instance_id": host.host_instance_id(),
                "service": mcp_credentials.service(),
                "server_id": server_id,
                "reason": error.diagnostic_reason(),
                "os_status": error.os_status(),
            }),
        );
        return Err(MCP_KEYCHAIN_CLEANUP_FAILED.to_owned());
    }
    Ok(())
}

pub(crate) async fn connect_server_with_store(
    host: &HostConnection,
    mcp_credentials: &Arc<McpCredentialStore>,
    server_id: &str,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    // Self-heal in-memory credentials from Keychain before connecting if missing in memory.
    if let Ok(view) = host.get_mcp_server(server_id).await {
        let missing = view
            .get("missing_secret_keys")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !missing.is_empty() {
            let loaded = {
                let _guard = mcp_credentials.lock().await;
                mcp_credentials.load(server_id)
            };
            if let Ok(Some(all_secrets)) = loaded {
                let declared_keys: BTreeSet<String> = {
                    let field = if view.get("transport").and_then(Value::as_str) == Some("stdio") {
                        "env_config"
                    } else {
                        "header_config"
                    };
                    view.get(field)
                        .and_then(Value::as_object)
                        .map(|obj| {
                            obj.iter()
                                .filter(|(_, v)| v.get("kind").and_then(Value::as_str) == Some("secret"))
                                .map(|(k, _)| k.clone())
                                .collect()
                        })
                        .unwrap_or_default()
                };
                let filtered: BTreeMap<String, Zeroizing<String>> = all_secrets
                    .into_iter()
                    .filter(|(k, _)| declared_keys.contains(k))
                    .collect();
                if !filtered.is_empty() {
                    let _ = host.put_mcp_credentials(server_id, &filtered, "restore").await;
                }
            }
        }
    }
    host.connect_mcp_server(server_id).await
}

fn is_suspected_secret_key(key: &str) -> bool {
    let upper = key.trim().to_ascii_uppercase();
    [
        "KEY",
        "TOKEN",
        "SECRET",
        "AUTH",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "COOKIE",
        "PRIVATE",
    ]
    .iter()
    .any(|part| upper.contains(part))
}

fn is_strict_secret_header(key: &str) -> bool {
    matches!(
        key.trim().to_ascii_lowercase().as_str(),
        "authorization"
            | "proxy-authorization"
            | "x-api-key"
            | "api-key"
            | "x-auth-token"
            | "x-access-token"
            | "cookie"
    )
}

fn looks_like_secret_value(val: &str) -> bool {
    let lower = val.trim().to_ascii_lowercase();
    lower.starts_with("bearer ")
        || lower.starts_with("basic ")
        || val.contains("sk-")
        || val.contains("ghp_")
        || val.contains("github_pat_")
        || val.starts_with("eyJ")
}

pub(crate) struct ParsedImportDraft {
    pub(crate) request: SaveMcpServerRequest,
    pub(crate) warnings: Vec<String>,
}

/// Strictly parse `{"mcpServers": {...}}` JSON into untrusted draft requests.
/// Separates suspected secrets into `secret_values` (for Keychain) and rejects unknown fields
/// or unsupported legacy `sse` transport. Never executes external commands.
pub(crate) fn parse_import_json(raw_json: &str) -> Result<Vec<ParsedImportDraft>, &'static str> {
    if raw_json.trim().is_empty() || raw_json.len() > MAX_IMPORT_JSON_BYTES {
        return Err(MCP_INVALID_REQUEST);
    }
    let root: Value = serde_json::from_str(raw_json).map_err(|_| MCP_INVALID_REQUEST)?;
    let root_obj = root.as_object().ok_or(MCP_INVALID_REQUEST)?;
    if root_obj.len() != 1 || !root_obj.contains_key("mcpServers") {
        return Err(MCP_INVALID_REQUEST);
    }
    let servers = root_obj
        .get("mcpServers")
        .and_then(Value::as_object)
        .ok_or(MCP_INVALID_REQUEST)?;
    if servers.is_empty() || servers.len() > MAX_IMPORT_SERVERS {
        return Err(MCP_INVALID_REQUEST);
    }

    let mut drafts = Vec::new();
    let mut seen_names = BTreeSet::new();
    for (raw_name, entry_val) in servers {
        let name = raw_name.trim();
        if name.is_empty() || name.len() > 100 || !seen_names.insert(name.to_ascii_lowercase()) {
            return Err(MCP_INVALID_REQUEST);
        }
        let entry = entry_val.as_object().ok_or(MCP_INVALID_REQUEST)?;
        let explicit_transport = entry
            .get("transport")
            .or_else(|| entry.get("type"))
            .and_then(Value::as_str);

        let transport = match explicit_transport {
            Some(t) => match t.trim().to_ascii_lowercase().as_str() {
                "stdio" => "stdio",
                "http" | "streamable-http" | "streamable_http" => "streamable_http",
                _ => return Err(MCP_INVALID_REQUEST),
            },
            None => {
                if entry.contains_key("command") && !entry.contains_key("url") {
                    "stdio"
                } else if entry.contains_key("url") && !entry.contains_key("command") {
                    "streamable_http"
                } else {
                    return Err(MCP_INVALID_REQUEST);
                }
            }
        };

        let mut warnings = Vec::new();
        let mut env_config = BTreeMap::new();
        let mut header_config = BTreeMap::new();
        let mut secret_values = BTreeMap::new();

        if transport == "stdio" {
            for key in entry.keys() {
                if !matches!(
                    key.as_str(),
                    "command" | "args" | "cwd" | "env" | "transport" | "type"
                ) {
                    return Err(MCP_INVALID_REQUEST);
                }
            }
            let command = entry
                .get("command")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or(MCP_INVALID_REQUEST)?
                .to_owned();
            let args: Vec<String> = match entry.get("args") {
                None => Vec::new(),
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|v| v.as_str().map(ToOwned::to_owned))
                    .collect::<Option<Vec<_>>>()
                    .ok_or(MCP_INVALID_REQUEST)?,
                Some(_) => return Err(MCP_INVALID_REQUEST),
            };
            let cwd = match entry.get("cwd") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(_) => return Err(MCP_INVALID_REQUEST),
            };
            if let Some(env_val) = entry.get("env") {
                let env_obj = env_val.as_object().ok_or(MCP_INVALID_REQUEST)?;
                for (k, v) in env_obj {
                    let s = v.as_str().ok_or(MCP_INVALID_REQUEST)?;
                    if is_suspected_secret_key(k) || looks_like_secret_value(s) {
                        env_config.insert(
                            k.clone(),
                            ConfigEntryRequest::Secret {
                                secret_key: k.clone(),
                            },
                        );
                        secret_values.insert(k.clone(), s.to_owned());
                    } else {
                        env_config.insert(
                            k.clone(),
                            ConfigEntryRequest::Plain {
                                value: s.to_owned(),
                            },
                        );
                    }
                }
            }
            if !secret_values.is_empty() {
                warnings.push(format!(
                    "服务「{name}」检测到 {} 个敏感环境变量，已隔离存入 macOS 钥匙串。",
                    secret_values.len()
                ));
            }
            drafts.push(ParsedImportDraft {
                request: SaveMcpServerRequest {
                    server_id: None,
                    name: name.to_owned(),
                    transport: "stdio".to_owned(),
                    command: Some(command),
                    args,
                    cwd,
                    clear_cwd: false,
                    url: None,
                    allow_private_network: false,
                    env_config,
                    header_config,
                    secret_values,
                    remove_secret_keys: Vec::new(),
                },
                warnings,
            });
        } else {
            for key in entry.keys() {
                if !matches!(
                    key.as_str(),
                    "url"
                        | "headers"
                        | "transport"
                        | "type"
                        | "allowPrivateNetwork"
                        | "allow_private_network"
                ) {
                    return Err(MCP_INVALID_REQUEST);
                }
            }
            let url = entry
                .get("url")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or(MCP_INVALID_REQUEST)?
                .to_owned();
            let allow_private_network = entry
                .get("allowPrivateNetwork")
                .or_else(|| entry.get("allow_private_network"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let Some(hdr_val) = entry.get("headers") {
                let hdr_obj = hdr_val.as_object().ok_or(MCP_INVALID_REQUEST)?;
                for (k, v) in hdr_obj {
                    let s = v.as_str().ok_or(MCP_INVALID_REQUEST)?;
                    if is_strict_secret_header(k)
                        || is_suspected_secret_key(k)
                        || looks_like_secret_value(s)
                    {
                        header_config.insert(
                            k.clone(),
                            ConfigEntryRequest::Secret {
                                secret_key: k.clone(),
                            },
                        );
                        secret_values.insert(k.clone(), s.to_owned());
                    } else {
                        header_config.insert(
                            k.clone(),
                            ConfigEntryRequest::Plain {
                                value: s.to_owned(),
                            },
                        );
                    }
                }
            }
            if !secret_values.is_empty() {
                warnings.push(format!(
                    "服务「{name}」检测到 {} 个敏感 Header，已隔离存入 macOS 钥匙串。",
                    secret_values.len()
                ));
            }
            drafts.push(ParsedImportDraft {
                request: SaveMcpServerRequest {
                    server_id: None,
                    name: name.to_owned(),
                    transport: "streamable_http".to_owned(),
                    command: None,
                    args: Vec::new(),
                    cwd: None,
                    clear_cwd: false,
                    url: Some(url),
                    allow_private_network,
                    env_config,
                    header_config,
                    secret_values,
                    remove_secret_keys: Vec::new(),
                },
                warnings,
            });
        }
    }

    Ok(drafts)
}

// ---------------------------------------------------------------------------
// Tauri Privileged IPC Commands (allowlisted in capabilities/main-connection.json)
// ---------------------------------------------------------------------------

#[tauri::command]
pub(crate) async fn mcp_save_server(
    sidecar: tauri::State<'_, Sidecar>,
    request: SaveMcpServerRequest,
) -> Result<Value, String> {
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let creds = sidecar.mcp_credentials();
    save_server_with_store(&host, &creds, request).await
}

#[tauri::command]
pub(crate) async fn mcp_import_servers_json(
    sidecar: tauri::State<'_, Sidecar>,
    request: ImportMcpServersRequest,
) -> Result<ImportMcpServersResult, String> {
    let drafts = parse_import_json(&request.raw_json).map_err(str::to_owned)?;
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let creds = sidecar.mcp_credentials();

    let mut created_servers = Vec::new();
    let mut warnings = Vec::new();
    for draft in drafts {
        let mut draft_warnings = draft.warnings;
        let saved = save_server_with_store(&host, &creds, draft.request).await?;
        created_servers.push(saved);
        warnings.append(&mut draft_warnings);
    }
    Ok(ImportMcpServersResult {
        servers: created_servers,
        warnings,
    })
}

#[tauri::command]
pub(crate) async fn mcp_trust_server(
    window: WebviewWindow,
    prompt_lock: tauri::State<'_, NativeTrustPrompt>,
    sidecar: tauri::State<'_, Sidecar>,
    request: TrustMcpServerRequest,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(&request.server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    let _prompt_guard = prompt_lock.0.lock().await;
    // Pin to the same Sidecar connection across dialog display and trust submission.
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let raw_preview = host.get_mcp_server(&request.server_id).await?;
    let preview: TrustPreview =
        serde_json::from_value(raw_preview).map_err(|_| "MCP_INVALID_RESPONSE".to_owned())?;

    let dialog = format_trust_dialog(&preview);
    let confirmed = confirm_trust_in_native_alert(window, dialog)
        .await
        .map_err(str::to_owned)?;
    if !confirmed {
        return Err(MCP_TRUST_CANCELLED.to_owned());
    }
    if sidecar.is_closing() {
        return Err("BACKEND_SHUTTING_DOWN".to_owned());
    }
    host.trust_mcp_server(
        &preview.server_id,
        preview.config_revision,
        &preview.current_fingerprint,
        request.enable,
    )
    .await
}

pub(crate) async fn revoke_trust_with_host(
    host: &HostConnection,
    server_id: &str,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    host.revoke_mcp_server_trust(server_id).await
}

#[tauri::command]
pub(crate) async fn mcp_revoke_trust(
    sidecar: tauri::State<'_, Sidecar>,
    request: ServerIdRequest,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(&request.server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    revoke_trust_with_host(&host, &request.server_id).await
}

#[tauri::command]
pub(crate) async fn mcp_delete_server(
    sidecar: tauri::State<'_, Sidecar>,
    request: ServerIdRequest,
) -> Result<(), String> {
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let creds = sidecar.mcp_credentials();
    delete_server_with_store(&host, &creds, &request.server_id).await
}

#[tauri::command]
pub(crate) async fn mcp_connect_server(
    sidecar: tauri::State<'_, Sidecar>,
    request: ServerIdRequest,
) -> Result<Value, String> {
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    let creds = sidecar.mcp_credentials();
    connect_server_with_store(&host, &creds, &request.server_id).await
}

#[tauri::command]
pub(crate) async fn mcp_disconnect_server(
    sidecar: tauri::State<'_, Sidecar>,
    request: ServerIdRequest,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(&request.server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.disconnect_mcp_server(&request.server_id).await
}

#[tauri::command]
pub(crate) async fn mcp_refresh_tools(
    sidecar: tauri::State<'_, Sidecar>,
    request: ServerIdRequest,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(&request.server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.refresh_mcp_tools(&request.server_id).await
}

#[tauri::command]
pub(crate) async fn mcp_set_server_enabled(
    sidecar: tauri::State<'_, Sidecar>,
    request: SetServerEnabledRequest,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(&request.server_id) {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.update_mcp_server(&request.server_id, &json!({ "enabled": request.enabled }))
        .await
}

#[tauri::command]
pub(crate) async fn mcp_set_tool_permission(
    sidecar: tauri::State<'_, Sidecar>,
    request: SetToolPermissionRequest,
) -> Result<Value, String> {
    if !is_valid_mcp_server_id(&request.server_id) || request.tool_id.trim().is_empty() {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    if request.permission_state != "allowed" && request.permission_state != "denied" {
        return Err(MCP_INVALID_REQUEST.to_owned());
    }
    let host = sidecar.mcp_host_connection().await.map_err(str::to_owned)?;
    host.set_mcp_tool_permission(
        &request.server_id,
        &request.tool_id,
        &json!({
            "permission_state": request.permission_state,
            "expected_definition_hash": request.expected_definition_hash,
            "expected_config_revision": request.expected_config_revision,
        }),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_dialog_shows_required_disclosures_binding_and_zero_secrets() {
        let mut env_cfg = BTreeMap::new();
        env_cfg.insert(
            "API_KEY".to_owned(),
            json!({ "kind": "secret", "secret_key": "API_KEY", "configured": true }),
        );
        let preview = TrustPreview {
            server_id: "srv-1".to_owned(),
            name: "Dynamic NPX Tool".to_owned(),
            transport: "stdio".to_owned(),
            command: Some("npx".to_owned()),
            resolved_command: Some("/opt/homebrew/bin/npx".to_owned()),
            executable_exists: Some(true),
            args: vec!["-y".to_owned(), "@example/mcp".to_owned(), "--token=[REDACTED]".to_owned()],
            cwd: Some("/tmp".to_owned()),
            url: None,
            private_network_target: false,
            high_risk_command: true,
            high_risk_reasons: vec!["使用动态包执行器 npx，运行时可能自动下载并执行远程代码包。".to_owned()],
            env_config: env_cfg,
            header_config: BTreeMap::new(),
            config_revision: 3,
            current_fingerprint: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .to_owned(),
        };

        let dialog = format_trust_dialog(&preview);
        assert!(dialog.critical);
        assert!(dialog.title.contains("Dynamic NPX Tool"));
        assert!(dialog.body.contains("/opt/homebrew/bin/npx"));
        assert!(dialog.body.contains("--token=[REDACTED]"));
        assert!(dialog.body.contains("不具备操作系统级文件沙箱"));
        assert!(dialog.body.contains("仅校验路径无法保证同一路径下的程序内容在后续未被替换"));
        assert!(dialog.body.contains("数据外发或外部副作用"));
        assert!(dialog.body.contains("r3"));
        assert!(dialog.body.contains("abcdef012345"));
    }

    #[test]
    fn import_json_separates_secrets_rejects_unknown_fields_and_legacy_sse() {
        let valid = r#"{
            "mcpServers": {
                "local-reader": {
                    "command": "/usr/bin/python3",
                    "args": ["server.py"],
                    "env": {
                        "MODE": "readonly",
                        "GITHUB_TOKEN": "ghp_1234567890abcdef1234567890"
                    }
                },
                "remote-search": {
                    "url": "https://mcp.example.com/mcp",
                    "headers": {
                        "Authorization": "Bearer super-secret-jwt",
                        "X-Client": "knowledgeagent"
                    }
                }
            }
        }"#;
        let drafts = parse_import_json(valid).unwrap();
        assert_eq!(drafts.len(), 2);
        let stdio = drafts.iter().find(|d| d.request.name == "local-reader").unwrap();
        assert!(matches!(
            stdio.request.env_config.get("GITHUB_TOKEN"),
            Some(ConfigEntryRequest::Secret { secret_key }) if secret_key == "GITHUB_TOKEN"
        ));
        assert!(matches!(
            stdio.request.env_config.get("MODE"),
            Some(ConfigEntryRequest::Plain { value }) if value == "readonly"
        ));
        assert_eq!(
            stdio.request.secret_values.get("GITHUB_TOKEN").map(String::as_str),
            Some("ghp_1234567890abcdef1234567890")
        );

        let http = drafts.iter().find(|d| d.request.name == "remote-search").unwrap();
        assert!(matches!(
            http.request.header_config.get("Authorization"),
            Some(ConfigEntryRequest::Secret { secret_key }) if secret_key == "Authorization"
        ));
        assert!(http.request.secret_values.contains_key("Authorization"));

        // Reject unknown top-level field
        assert!(parse_import_json(r#"{"mcpServers": {}, "evil": true}"#).is_err());
        // Reject unknown server field
        assert!(parse_import_json(r#"{"mcpServers": {"a": {"command": "py", "autoStart": true}}}"#).is_err());
        // Reject legacy SSE
        assert!(parse_import_json(r#"{"mcpServers": {"a": {"type": "sse", "url": "https://x.test"}}}"#).is_err());
    }

    #[test]
    fn ipc_requests_reject_unknown_fields() {
        assert!(serde_json::from_value::<TrustMcpServerRequest>(json!({
            "server_id": "srv-1",
            "expected_fingerprint": "injected-from-webview"
        }))
        .is_err());
        assert!(serde_json::from_value::<SetServerEnabledRequest>(json!({
            "server_id": "srv-1",
            "enabled": true,
            "trust_state": "trusted"
        }))
        .is_err());
        assert!(serde_json::from_value::<ServerIdRequest>(json!({
            "server_id": "srv-1",
            "force": true
        }))
        .is_err());
    }
}
