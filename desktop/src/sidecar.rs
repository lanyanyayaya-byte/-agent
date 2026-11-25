use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::{Client, Method, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
#[cfg(all(not(all(dev, debug_assertions)), not(feature = "sidecar-onefile")))]
use tauri::Manager;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::timeout;
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::{
    diagnostics::HostLog,
    grants::{GrantScope, GrantView},
    keychain::{
        credential_account, validate_credential, CredentialStore, McpCredentialStore,
        CREDENTIAL_REJECTED, DEEPSEEK_PROVIDER, KEYCHAIN_ACCESS_FAILED,
        KEYCHAIN_CREDENTIAL_CORRUPTED, SILICONFLOW_PROVIDER,
    },
    mcp_process,
    BackendConnection,
};

#[cfg(target_os = "macos")]
use crate::library::{valid_document_id, HostImport, ImportResult};

#[cfg(test)]
mod tests;

const PROTOCOL_VERSION: u8 = 1;
const MAX_READY_BYTES: u64 = 4096;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(12);
const HOST_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
const MCP_MANAGEMENT_TIMEOUT: Duration = Duration::from_secs(10);
const MCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const EMBEDDING_TEST_TIMEOUT: Duration = Duration::from_secs(30);
#[cfg(target_os = "macos")]
const IMPORT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_PENDING_CONNECTIONS: usize = 32;
const UNAVAILABLE: &str = "BACKEND_SIDECAR_UNAVAILABLE";
const DATABASE_LOCKED: &str = "BACKEND_DATABASE_LOCKED";
const SHUTTING_DOWN: &str = "BACKEND_SHUTTING_DOWN";
const REPLAY_FAILED: &str = "MODEL_CREDENTIAL_REPLAY_FAILED";
const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
const SILICONFLOW_BASE_URL: &str = "https://api.siliconflow.cn/v1";
const EMBEDDING_MODEL: &str = "BAAI/bge-m3";
const EMBEDDING_DIMENSIONS: u16 = 1024;

// A SiliconFlow restore error belongs to this connection and must never block chat.
type CredentialReplay = Result<Option<&'static str>, &'static str>;

type ConnectionReply = oneshot::Sender<Result<SidecarConnection, &'static str>>;

#[derive(Clone)]
struct SidecarConnection {
    ui: BackendConnection,
    host: HostConnection,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelConfiguration {
    configured: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmbeddingConnection {
    model: String,
    dimensions: u16,
}

#[derive(Clone)]
pub(crate) struct Sidecar {
    connections: mpsc::Sender<ConnectionReply>,
    closing: watch::Sender<bool>,
    stopped: watch::Receiver<bool>,
    credentials: Arc<CredentialStore>,
    mcp_credentials: Arc<McpCredentialStore>,
}

impl Sidecar {
    pub(crate) fn launch(
        app: &tauri::AppHandle,
        app_data_dir: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(all(dev, debug_assertions))]
        let command = development_command();
        #[cfg(not(all(dev, debug_assertions)))]
        let command = bundled_command(app)?;
        let credentials = CredentialStore::for_identifier(&app.config().identifier);
        let mcp_credentials = McpCredentialStore::for_identifier(&app.config().identifier);
        Ok(Self::start_with_mcp(
            command,
            app_data_dir,
            credentials,
            mcp_credentials,
            reconfigure_after_ready,
        ))
    }

    #[cfg(test)]
    fn start<Fut>(
        command: Command,
        app_data_dir: PathBuf,
        credentials: Arc<CredentialStore>,
        reconfigure: impl Fn(HostConnection, Arc<CredentialStore>) -> Fut + Send + Sync + 'static,
    ) -> Self
    where
        Fut: Future<Output = CredentialReplay> + Send,
    {
        let mcp_service = format!("{}.mcp-test", credentials.service());
        let mcp_credentials =
            McpCredentialStore::with_backend(mcp_service, Arc::new(NoopMcpCredentialBackend));
        Self::start_with_mcp(
            command,
            app_data_dir,
            credentials,
            mcp_credentials,
            reconfigure,
        )
    }

    pub(crate) fn start_with_mcp<Fut>(
        mut command: Command,
        app_data_dir: PathBuf,
        credentials: Arc<CredentialStore>,
        mcp_credentials: Arc<McpCredentialStore>,
        reconfigure: impl Fn(HostConnection, Arc<CredentialStore>) -> Fut + Send + Sync + 'static,
    ) -> Self
    where
        Fut: Future<Output = CredentialReplay> + Send,
    {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        let (connections, requests) = mpsc::channel(MAX_PENDING_CONNECTIONS);
        let (closing, shutdown) = watch::channel(false);
        let (finished, stopped) = watch::channel(false);
        let replay_credentials = credentials.clone();
        let replay_mcp_credentials = mcp_credentials.clone();
        tauri::async_runtime::spawn(async move {
            supervise(
                command,
                app_data_dir,
                requests,
                shutdown,
                replay_mcp_credentials,
                move |host| reconfigure(host, replay_credentials.clone()),
            )
            .await;
            let _ = finished.send(true);
        });
        Self {
            connections,
            closing,
            stopped,
            credentials,
            mcp_credentials,
        }
    }

    pub(crate) async fn connection(&self) -> Result<BackendConnection, &'static str> {
        Ok(self.current_connection().await?.ui)
    }

    pub(crate) async fn mcp_host_connection(&self) -> Result<HostConnection, &'static str> {
        Ok(self.current_connection().await?.host)
    }

    pub(crate) fn mcp_credentials(&self) -> Arc<McpCredentialStore> {
        self.mcp_credentials.clone()
    }

    pub(crate) fn is_closing(&self) -> bool {
        *self.closing.borrow()
    }

    #[cfg(target_os = "macos")]
    pub(crate) async fn import_connection(&self) -> Result<HostConnection, &'static str> {
        Ok(self.current_connection().await?.host)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn closing_signal(&self) -> watch::Receiver<bool> {
        self.closing.subscribe()
    }

    pub(crate) async fn select_grant(
        &self,
        session_id: Uuid,
        scope: GrantScope,
        pick: impl Future<Output = Result<Option<String>, &'static str>>,
    ) -> Result<Option<GrantView>, &'static str> {
        // Keep the same instance across the dialog. A selection is never replayed to a restarted Sidecar.
        let connection = self.current_connection().await?;
        let Some(path) = pick.await? else {
            return Ok(None);
        };
        if *self.closing.borrow() {
            return Err(SHUTTING_DOWN);
        }
        connection
            .host
            .create_grant(session_id, scope, &path)
            .await
            .map(Some)
    }

    pub(crate) async fn revoke_grant(
        &self,
        session_id: Uuid,
        grant_id: Uuid,
    ) -> Result<(), &'static str> {
        self.current_connection()
            .await?
            .host
            .revoke_grant(session_id, grant_id)
            .await
    }

    pub(crate) async fn save_provider_credential(
        &self,
        provider: &str,
        api_key: Zeroizing<String>,
    ) -> Result<ModelConfiguration, &'static str> {
        validate_credential(provider, &api_key).map_err(|error| error.ipc_code())?;
        // Resolve a live connection before acquiring the lock used by the readiness hook.
        let connection = self.current_connection().await?;
        let _guard = self.credentials.lock().await;
        if *self.closing.borrow() {
            return Err(SHUTTING_DOWN);
        }
        let previous = match self.credentials.load(provider) {
            Ok(secret) => secret,
            Err(crate::keychain::KeychainError::Corrupted) => None,
            Err(error) => {
                connection.host.log.record(
                    "keychain_write_error",
                    json!({
                        "instance_id": connection.host.instance_id,
                        "service": self.credentials.service(),
                        "provider": provider,
                        "reason": error.diagnostic_reason(),
                        "os_status": error.os_status(),
                    }),
                );
                return Err(error.ipc_code());
            }
        };
        if let Err(error) = self.credentials.save(provider, &api_key) {
            connection.host.log.record(
                "keychain_write_error",
                json!({
                    "instance_id": connection.host.instance_id,
                    "service": self.credentials.service(),
                    "provider": provider,
                    "reason": error.diagnostic_reason(),
                    "os_status": error.os_status(),
                }),
            );
            return Err(error.ipc_code());
        }
        match connection.host.put_credential(provider, &api_key).await {
            Ok(result) => {
                connection.host.log.record(
                    "keychain_credential_saved",
                    json!({
                        "instance_id": connection.host.instance_id,
                        "service": self.credentials.service(),
                        "provider": provider,
                    }),
                );
                Ok(result)
            }
            Err(error) => {
                match previous {
                    Some(old) => {
                        let _ = self.credentials.save(provider, &old);
                    }
                    None => {
                        let _ = self.credentials.delete(provider);
                    }
                }
                Err(error)
            }
        }
    }

    pub(crate) async fn delete_provider_credential(
        &self,
        provider: &str,
    ) -> Result<ModelConfiguration, &'static str> {
        credential_account(provider).map_err(|error| error.ipc_code())?;
        let connection = self.current_connection().await?;
        let _guard = self.credentials.lock().await;
        if *self.closing.borrow() {
            return Err(SHUTTING_DOWN);
        }
        let previous = match self.credentials.load(provider) {
            Ok(secret) => secret,
            Err(crate::keychain::KeychainError::Corrupted) => None,
            Err(error) => {
                connection.host.log.record(
                    "keychain_delete_error",
                    json!({
                        "instance_id": connection.host.instance_id,
                        "service": self.credentials.service(),
                        "provider": provider,
                        "reason": error.diagnostic_reason(),
                        "os_status": error.os_status(),
                    }),
                );
                return Err(error.ipc_code());
            }
        };
        if let Err(error) = self.credentials.delete(provider) {
            connection.host.log.record(
                "keychain_delete_error",
                json!({
                    "instance_id": connection.host.instance_id,
                    "service": self.credentials.service(),
                    "provider": provider,
                    "reason": error.diagnostic_reason(),
                    "os_status": error.os_status(),
                }),
            );
            return Err(error.ipc_code());
        }
        match connection.host.delete_credential(provider).await {
            Ok(result) => {
                connection.host.log.record(
                    "keychain_credential_deleted",
                    json!({
                        "instance_id": connection.host.instance_id,
                        "service": self.credentials.service(),
                        "provider": provider,
                    }),
                );
                Ok(result)
            }
            Err(error) => {
                if let Some(old) = previous {
                    let _ = self.credentials.save(provider, &old);
                }
                Err(error)
            }
        }
    }

    pub(crate) async fn test_siliconflow_connection(
        &self,
    ) -> Result<EmbeddingConnection, &'static str> {
        self.current_connection()
            .await?
            .host
            .test_siliconflow_connection()
            .await
    }

    async fn current_connection(&self) -> Result<SidecarConnection, &'static str> {
        if *self.closing.borrow() {
            return Err(SHUTTING_DOWN);
        }
        let (reply, response) = oneshot::channel();
        self.connections
            .send(reply)
            .await
            .map_err(|_| UNAVAILABLE)?;
        let result = response.await.map_err(|_| UNAVAILABLE)?;
        if *self.closing.borrow() {
            return Err(SHUTTING_DOWN);
        }
        result
    }

    pub(crate) fn request_shutdown(&self) -> bool {
        self.closing.send_if_modified(|closing| {
            let first_request = !*closing;
            *closing = true;
            first_request
        })
    }

    pub(crate) fn is_stopped(&self) -> bool {
        *self.stopped.borrow() || self.stopped.has_changed().is_err()
    }

    pub(crate) async fn wait_stopped(mut self) {
        let _ = self.stopped.wait_for(|stopped| *stopped).await;
    }
}

#[cfg(any(all(dev, debug_assertions), test))]
fn development_command() -> Command {
    let backend = Path::new(env!("CARGO_MANIFEST_DIR")).join("../backend");
    let interpreter = if cfg!(windows) {
        ".venv/Scripts/python.exe"
    } else {
        ".venv/bin/python"
    };
    // Launch directly so the desktop owns the process and the stdin writer.
    let mut command = Command::new(backend.join(interpreter));
    command
        .current_dir(backend)
        .args(["-E", "-s", "-B", "-m", "app.sidecar"]);
    command
}

#[cfg(not(all(dev, debug_assertions)))]
fn bundled_command(_app: &tauri::AppHandle) -> Result<Command, Box<dyn std::error::Error>> {
    #[cfg(feature = "sidecar-onefile")]
    let directory = std::env::current_exe()?.with_file_name("");
    #[cfg(not(feature = "sidecar-onefile"))]
    let directory = _app.path().resource_dir()?.join("sidecar");

    let executable = directory.join("knowledgeagent-sidecar");
    let mut command = Command::new(executable);
    command.current_dir(directory);
    #[cfg(feature = "sidecar-onefile")]
    command.env("PYINSTALLER_RESET_ENVIRONMENT", "1");
    Ok(command)
}

async fn supervise<Fut>(
    mut command: Command,
    app_data_dir: PathBuf,
    mut requests: mpsc::Receiver<ConnectionReply>,
    mut closing: watch::Receiver<bool>,
    mcp_credentials: Arc<McpCredentialStore>,
    reconfigure: impl Fn(HostConnection) -> Fut,
) where
    Fut: Future<Output = CredentialReplay>,
{
    let mut pending = Vec::new();
    let log = HostLog::new(&app_data_dir);
    // Clean expired registry files from previous host launches without signaling any PID.
    mcp_process::cleanup_stale_registries(&app_data_dir, None, &log);
    while !*closing.borrow() {
        let running = start_sidecar(
            &mut command,
            &app_data_dir,
            &mut requests,
            &mut closing,
            &mut pending,
            &reconfigure,
            &log,
        )
        .await;
        match running {
            Ok(running) => {
                for reply in pending.drain(..) {
                    let _ = reply.send(Ok(running.connection.clone()));
                }
                running
                    .supervise(&app_data_dir, &mcp_credentials, &mut requests, &mut closing)
                    .await;
            }
            Err(reason) => {
                if reason != SHUTTING_DOWN {
                    log.record("sidecar_start_error", json!({"reason": reason}));
                    eprintln!("KnowledgeAgent Sidecar: {reason}");
                }
                for reply in pending.drain(..) {
                    let _ = reply.send(Err(match reason {
                        DATABASE_LOCKED
                        | REPLAY_FAILED
                        | KEYCHAIN_ACCESS_FAILED
                        | KEYCHAIN_CREDENTIAL_CORRUPTED => reason,
                        _ => UNAVAILABLE,
                    }));
                }
            }
        }
        if *closing.borrow() {
            return;
        }

        // Restart on the existing connection request, including the UI's Retry button.
        tokio::select! {
            biased;
            _ = closing.changed() => return,
            request = requests.recv() => match request {
                Some(reply) => pending.push(reply),
                None => return,
            },
        }
    }
}

async fn start_sidecar<Fut>(
    command: &mut Command,
    app_data_dir: &Path,
    requests: &mut mpsc::Receiver<ConnectionReply>,
    closing: &mut watch::Receiver<bool>,
    pending: &mut Vec<ConnectionReply>,
    reconfigure: &impl Fn(HostConnection) -> Fut,
    log: &HostLog,
) -> Result<RunningSidecar, &'static str>
where
    Fut: Future<Output = CredentialReplay>,
{
    let ui_token = random_token()?;
    let host_token = random_token()?;
    let mut child = command.spawn().map_err(|error| {
        log.record(
            "sidecar_spawn_error",
            json!({"os_error": error.raw_os_error()}),
        );
        "could not launch Python"
    })?;
    // Child::wait closes an untaken stdin, so keep the writer beside the child handle.
    let mut stdin = child.stdin.take().expect("sidecar stdin must be piped");
    let result = {
        let startup = timeout(
            STARTUP_TIMEOUT,
            handshake(
                &mut child,
                &mut stdin,
                app_data_dir,
                ui_token,
                host_token,
                reconfigure,
                log,
            ),
        );
        tokio::pin!(startup);
        loop {
            tokio::select! {
                biased;
                _ = closing.changed() => break Err(SHUTTING_DOWN),
                result = &mut startup => break result.unwrap_or(Err("startup timed out")),
                request = requests.recv(), if pending.len() < MAX_PENDING_CONNECTIONS => match request {
                    Some(reply) => pending.push(reply),
                    None => break Err(SHUTTING_DOWN),
                },
            }
        }
    };

    match result {
        Ok((host, ui)) => {
            log.record(
                "sidecar_connected",
                json!({"instance_id": host.instance_id, "child_pid": child.id()}),
            );
            eprintln!(
                "KnowledgeAgent Sidecar: ready (instance {}, {}).",
                host.instance_id, ui.base_url
            );
            Ok(RunningSidecar {
                child,
                stdin,
                connection: SidecarConnection { host, ui },
            })
        }
        Err(reason) => {
            drop(stdin);
            wait_or_terminate(&mut child).await;
            if reason != SHUTTING_DOWN && is_database_locked(app_data_dir) {
                Err(DATABASE_LOCKED)
            } else {
                Err(reason)
            }
        }
    }
}

fn is_database_locked(app_data_dir: &Path) -> bool {
    let lock_path = app_data_dir.join("knowledgeagent.lock");
    let Ok(file) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(lock_path)
    else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        let locked = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) != 0 };
        if !locked {
            unsafe { libc::flock(fd, libc::LOCK_UN) };
        }
        locked
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        false
    }
}

#[derive(Serialize)]
struct Bootstrap<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    protocol_version: u8,
    ui_token: &'a str,
    host_token: &'a str,
    app_data_dir: &'a Path,
    parent_pid: u32,
    restore_credentials: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadyFrame {
    #[serde(rename = "type")]
    kind: String,
    protocol_version: u8,
    instance_id: Uuid,
    port: u16,
}

async fn handshake<Fut>(
    child: &mut Child,
    stdin: &mut ChildStdin,
    app_data_dir: &Path,
    ui_token: String,
    host_token: String,
    reconfigure: &impl Fn(HostConnection) -> Fut,
    log: &HostLog,
) -> Result<(HostConnection, BackendConnection), &'static str>
where
    Fut: Future<Output = CredentialReplay>,
{
    let mut bootstrap = serde_json::to_vec(&Bootstrap {
        kind: "bootstrap",
        protocol_version: PROTOCOL_VERSION,
        ui_token: &ui_token,
        host_token: &host_token,
        app_data_dir,
        parent_pid: std::process::id(),
        restore_credentials: true,
    })
    .map_err(|_| "could not encode bootstrap")?;
    bootstrap.push(b'\n');
    stdin
        .write_all(&bootstrap)
        .await
        .map_err(|_| "could not send bootstrap")?;
    drop(bootstrap);

    let ready = read_ready(child.stdout.take().expect("sidecar stdout must be piped")).await?;
    require_alive(child)?;
    let base_url = format!("http://127.0.0.1:{}", ready.port);
    let host = HostConnection::new(base_url.clone(), host_token, ready.instance_id, log.clone())?;
    host.check_ready().await?;
    let siliconflow_restore_error = reconfigure(host.clone()).await?;
    host.signal_credential_restore_complete().await;
    require_alive(child)?;
    Ok((
        host,
        BackendConnection {
            base_url,
            ui_token,
            siliconflow_restore_error,
        },
    ))
}

fn random_token() -> Result<String, &'static str> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|_| "could not generate a random token")?;
    let mut token = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut token, "{byte:02x}").expect("writing to a string is infallible");
    }
    Ok(token)
}

async fn read_ready(stdout: impl AsyncRead + Unpin) -> Result<ReadyFrame, &'static str> {
    let mut read = BufReader::new(stdout.take(MAX_READY_BYTES));
    let mut line = Vec::new();
    read.read_until(b'\n', &mut line)
        .await
        .map_err(|_| "could not read ready frame")?;
    if !line.ends_with(b"\n") {
        return Err("missing or incomplete ready frame");
    }
    let ready: ReadyFrame = serde_json::from_slice(&line).map_err(|_| "invalid ready frame")?;
    if ready.kind != "ready"
        || ready.protocol_version != PROTOCOL_VERSION
        || ready.port == 0
        || ready.instance_id.is_nil()
    {
        return Err("invalid ready frame");
    }
    Ok(ready)
}

fn require_alive(child: &mut Child) -> Result<(), &'static str> {
    match child.try_wait() {
        Ok(None) => Ok(()),
        _ => Err("Python exited during startup"),
    }
}

#[derive(Clone)]
pub(crate) struct HostConnection {
    client: Client,
    base_url: String,
    host_token: String,
    instance_id: Uuid,
    log: HostLog,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
pub(crate) enum HostImportError {
    Rejected(&'static str),
    Uncertain(&'static str),
}

impl HostConnection {
    fn new(
        base_url: String,
        host_token: String,
        instance_id: Uuid,
        log: HostLog,
    ) -> Result<Self, &'static str> {
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(HOST_REQUEST_TIMEOUT)
            .build()
            .map_err(|_| "could not initialize host HTTP client")?;
        Ok(Self {
            client,
            base_url,
            host_token,
            instance_id,
            log,
        })
    }

    fn log_request(
        &self,
        event: &'static str,
        started: Instant,
        response: &reqwest::Result<Response>,
    ) {
        let response = response.as_ref().ok();
        let request_id = response
            .and_then(|reply| reply.headers().get("x-request-id"))
            .and_then(|header| header.to_str().ok())
            .and_then(|value| Uuid::parse_str(value).ok());
        self.log.record(event, json!({
            "instance_id": self.instance_id, "request_id": request_id,
            "status": response.map(|reply| reply.status().as_u16()), "elapsed_ms": started.elapsed().as_millis(),
        }));
    }

    async fn check_ready(&self) -> Result<(), &'static str> {
        #[derive(Deserialize)]
        struct ReadyResponse {
            status: String,
            instance_id: Uuid,
        }

        let error = "host ready check failed";
        let started = Instant::now();
        let response = self
            .client
            .get(format!("{}/api/v1/_host/ready", self.base_url))
            .bearer_auth(&self.host_token)
            .send()
            .await;
        self.log_request("host_ready_request", started, &response);
        let response = response.map_err(|_| error)?;
        if response.status() != StatusCode::OK {
            return Err(error);
        }
        let ready: ReadyResponse = response.json().await.map_err(|_| error)?;
        if ready.status != "ready" || ready.instance_id != self.instance_id {
            return Err(error);
        }
        Ok(())
    }

    async fn signal_credential_restore_complete(&self) {
        let started = Instant::now();
        let response = self
            .client
            .post(format!(
                "{}/api/v1/_host/model-credentials/restore-complete",
                self.base_url
            ))
            .bearer_auth(&self.host_token)
            .send()
            .await;
        self.log_request("host_credential_restore_complete", started, &response);
    }

    async fn request_shutdown(&self) -> bool {
        let started = Instant::now();
        let response = self
            .client
            .post(format!("{}/api/v1/_host/shutdown", self.base_url))
            .bearer_auth(&self.host_token)
            .send()
            .await;
        self.log_request("host_shutdown_request", started, &response);
        response.is_ok_and(|response| response.status() == StatusCode::NO_CONTENT)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn instance_id(&self) -> Uuid {
        self.instance_id
    }

    #[cfg(target_os = "macos")]
    pub(crate) async fn import_document(
        &self,
        request: &HostImport,
    ) -> Result<ImportResult, HostImportError> {
        self.send_import(request, IMPORT_REQUEST_TIMEOUT).await
    }

    #[cfg(target_os = "macos")]
    async fn send_import(
        &self,
        request: &HostImport,
        deadline: Duration,
    ) -> Result<ImportResult, HostImportError> {
        use HostImportError::{Rejected, Uncertain};

        fn transport_error(error: reqwest::Error) -> HostImportError {
            Uncertain(if error.is_timeout() {
                "IMPORT_REQUEST_TIMEOUT"
            } else if error.is_connect() {
                UNAVAILABLE
            } else {
                "IMPORT_RESPONSE_UNCERTAIN"
            })
        }

        #[derive(Deserialize)]
        struct ErrorResponse {
            error: ErrorCode,
        }

        #[derive(Deserialize)]
        struct ErrorCode {
            code: String,
        }

        let started = Instant::now();
        let response = self
            .client
            .post(format!("{}/_host/knowledge/imports", self.base_url))
            .timeout(deadline)
            .bearer_auth(&self.host_token)
            .json(request)
            .send()
            .await;
        self.log_request("host_knowledge_import", started, &response);
        let response = response.map_err(transport_error)?;
        let status = response.status();
        if status == StatusCode::OK {
            let result: ImportResult = response.json().await.map_err(transport_error)?;
            if result.import_id != request.import_id
                || !valid_document_id(&result.document_id)
                || !valid_document_id(&result.document_version_id)
                || request
                    .document_id
                    .as_ref()
                    .is_some_and(|id| id != &result.document_id)
            {
                return Err(Uncertain("IMPORT_RESPONSE_UNCERTAIN"));
            }
            return Ok(result);
        }
        if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
            return Err(Uncertain("IMPORT_HOST_AUTH_FAILED"));
        }
        let failure: ErrorResponse = response.json().await.map_err(transport_error)?;
        Err(match (status.as_u16(), failure.error.code.as_str()) {
            (404, "TEXTBOOK_NOT_FOUND") => Rejected("TEXTBOOK_NOT_FOUND"),
            (409, "TEXTBOOK_FORMAT_MISMATCH") => Rejected("TEXTBOOK_FORMAT_MISMATCH"),
            (409, "IMPORT_ID_CONFLICT") => Rejected("IMPORT_ID_CONFLICT"),
            (410, "TEXTBOOK_REMOVED") => Rejected("TEXTBOOK_REMOVED"),
            (413, "TEXTBOOK_TOO_LARGE") => Rejected("TEXTBOOK_TOO_LARGE"),
            (415, "INVALID_TEXTBOOK_FORMAT") => Rejected("INVALID_TEXTBOOK_FORMAT"),
            (415, "INVALID_IMPORT_SOURCE") => Rejected("INVALID_IMPORT_SOURCE"),
            (422, "VALIDATION_ERROR") => Rejected("VALIDATION_ERROR"),
            (404, "IMPORT_SOURCE_NOT_FOUND") => Uncertain("IMPORT_SOURCE_NOT_FOUND"),
            (409, "IMPORT_SOURCE_CHANGED") => Uncertain("IMPORT_SOURCE_CHANGED"),
            (503, "KNOWLEDGE_STORE_UNAVAILABLE") => Uncertain("KNOWLEDGE_STORE_UNAVAILABLE"),
            _ => Uncertain("IMPORT_RESPONSE_UNCERTAIN"),
        })
    }

    async fn create_grant(
        &self,
        session_id: Uuid,
        scope: GrantScope,
        path: &str,
    ) -> Result<GrantView, &'static str> {
        const FAILED: &str = "GRANT_CREATE_FAILED";
        let started = Instant::now();
        let response = self.client
            .post(format!("{}/api/v1/_host/sessions/{session_id}/grants", self.base_url))
            .bearer_auth(&self.host_token)
            .json(&json!({
                "scope": scope, "path": path, "operations": ["read"], "lifetime": "application_session"
            }))
            .send().await;
        self.log_request("host_grant_create", started, &response);
        let response = response.map_err(|_| FAILED)?;
        match response.status() {
            StatusCode::CREATED => {}
            StatusCode::BAD_REQUEST | StatusCode::FORBIDDEN => return Err("GRANT_PATH_REJECTED"),
            StatusCode::NOT_FOUND => return Err("GRANT_SESSION_NOT_FOUND"),
            _ => return Err(FAILED),
        }
        let grant: GrantView = response.json().await.map_err(|_| FAILED)?;
        if grant.grant_id.is_nil() || grant.session_id != session_id || grant.scope != scope {
            return Err(FAILED);
        }
        Ok(grant)
    }

    async fn revoke_grant(&self, session_id: Uuid, grant_id: Uuid) -> Result<(), &'static str> {
        const FAILED: &str = "GRANT_REVOKE_FAILED";
        let started = Instant::now();
        let response = self
            .client
            .delete(format!(
                "{}/api/v1/_host/grants/{grant_id}?session_id={session_id}",
                self.base_url
            ))
            .bearer_auth(&self.host_token)
            .send()
            .await;
        self.log_request("host_grant_revoke", started, &response);
        match response.map_err(|_| FAILED)?.status() {
            StatusCode::NO_CONTENT => Ok(()),
            StatusCode::NOT_FOUND => Err("GRANT_NOT_FOUND"),
            _ => Err(FAILED),
        }
    }

    async fn put_credential(
        &self,
        provider: &str,
        api_key: &str,
    ) -> Result<ModelConfiguration, &'static str> {
        #[derive(Serialize)]
        struct Body<'a> {
            api_key: &'a str,
            base_url: &'static str,
        }

        let base_url = match provider {
            DEEPSEEK_PROVIDER => DEEPSEEK_BASE_URL,
            SILICONFLOW_PROVIDER => SILICONFLOW_BASE_URL,
            _ => return Err(CREDENTIAL_REJECTED),
        };
        let started = Instant::now();
        let response = self
            .client
            .put(format!(
                "{}/api/v1/_host/model-credentials/{provider}",
                self.base_url
            ))
            .bearer_auth(&self.host_token)
            .json(&Body { api_key, base_url })
            .send()
            .await;
        self.log_request("host_credential_request", started, &response);
        let response = response.map_err(|_| CREDENTIAL_REJECTED)?;
        if response.status() != StatusCode::OK {
            return Err(CREDENTIAL_REJECTED);
        }
        let configured: ModelConfiguration =
            response.json().await.map_err(|_| CREDENTIAL_REJECTED)?;
        if !configured.configured {
            return Err(CREDENTIAL_REJECTED);
        }
        Ok(configured)
    }

    async fn delete_credential(&self, provider: &str) -> Result<ModelConfiguration, &'static str> {
        const FAILED: &str = "MODEL_CREDENTIAL_DELETE_FAILED";
        let started = Instant::now();
        let response = self
            .client
            .delete(format!(
                "{}/api/v1/_host/model-credentials/{provider}",
                self.base_url
            ))
            .bearer_auth(&self.host_token)
            .send()
            .await;
        self.log_request("host_credential_delete_request", started, &response);
        let response = response.map_err(|_| FAILED)?;
        if response.status() != StatusCode::OK {
            return Err(FAILED);
        }
        let configured: ModelConfiguration = response.json().await.map_err(|_| FAILED)?;
        if configured.configured {
            return Err(FAILED);
        }
        Ok(configured)
    }

    async fn test_siliconflow_connection(&self) -> Result<EmbeddingConnection, &'static str> {
        const FAILED: &str = "EMBEDDING_TEST_FAILED";
        let started = Instant::now();
        let response = self
            .client
            .post(format!(
                "{}/api/v1/_host/model-credentials/siliconflow/test",
                self.base_url
            ))
            .bearer_auth(&self.host_token)
            .timeout(EMBEDDING_TEST_TIMEOUT)
            .send()
            .await;
        self.log_request("host_embedding_test", started, &response);
        let response = response.map_err(|error| {
            if error.is_timeout() {
                "EMBEDDING_TIMEOUT"
            } else {
                FAILED
            }
        })?;
        let status = response.status();
        if status == StatusCode::OK {
            let result: EmbeddingConnection = response
                .json()
                .await
                .map_err(|_| "EMBEDDING_INVALID_RESPONSE")?;
            if result.model != EMBEDDING_MODEL || result.dimensions != EMBEDDING_DIMENSIONS {
                return Err("EMBEDDING_INVALID_RESPONSE");
            }
            return Ok(result);
        }

        #[derive(Deserialize)]
        struct TestErrorResponse {
            error: TestError,
        }
        #[derive(Deserialize)]
        struct TestError {
            code: String,
        }

        let failure: TestErrorResponse = response.json().await.map_err(|_| FAILED)?;
        Err(match (status.as_u16(), failure.error.code.as_str()) {
            (409, "EMBEDDING_NOT_CONFIGURED") => "EMBEDDING_NOT_CONFIGURED",
            (401, "EMBEDDING_AUTH_FAILED") => "EMBEDDING_AUTH_FAILED",
            (403, "EMBEDDING_FORBIDDEN") => "EMBEDDING_FORBIDDEN",
            (429, "EMBEDDING_RATE_LIMITED") => "EMBEDDING_RATE_LIMITED",
            (504, "EMBEDDING_TIMEOUT") => "EMBEDDING_TIMEOUT",
            (502, "EMBEDDING_NETWORK_ERROR") => "EMBEDDING_NETWORK_ERROR",
            (502, "EMBEDDING_SERVICE_ERROR") => "EMBEDDING_SERVICE_ERROR",
            (502, "EMBEDDING_INVALID_RESPONSE") => "EMBEDDING_INVALID_RESPONSE",
            _ => FAILED,
        })
    }

    pub(crate) fn log(&self) -> &HostLog {
        &self.log
    }

    pub(crate) fn host_instance_id(&self) -> Uuid {
        self.instance_id
    }

    async fn mcp_request_json(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        request_timeout: Duration,
        event: &'static str,
    ) -> Result<Value, String> {
        #[derive(Deserialize)]
        struct ApiErrEnvelope {
            error: ApiErrInner,
        }
        #[derive(Deserialize)]
        struct ApiErrInner {
            code: String,
        }

        let started = Instant::now();
        let mut builder = self
            .client
            .request(method, format!("{}{}", self.base_url, path))
            .timeout(request_timeout)
            .bearer_auth(&self.host_token);
        if let Some(payload) = body {
            builder = builder.json(payload);
        }
        let response = builder.send().await;
        self.log_request(event, started, &response);
        let response = response.map_err(|err| {
            if err.is_timeout() {
                "MCP_REQUEST_TIMEOUT".to_owned()
            } else {
                UNAVAILABLE.to_owned()
            }
        })?;
        let status = response.status();
        if status == StatusCode::NO_CONTENT {
            return Ok(Value::Null);
        }
        if status.is_success() {
            return response
                .json::<Value>()
                .await
                .map_err(|_| "MCP_INVALID_RESPONSE".to_owned());
        }
        if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
            if let Ok(parsed) = response.json::<ApiErrEnvelope>().await {
                if parsed.error.code.starts_with("MCP_") {
                    return Err(parsed.error.code);
                }
            }
            return Err("MCP_HOST_AUTH_FAILED".to_owned());
        }
        if let Ok(parsed) = response.json::<ApiErrEnvelope>().await {
            return Err(parsed.error.code);
        }
        Err("MCP_REQUEST_FAILED".to_owned())
    }

    pub(crate) async fn get_mcp_server(&self, server_id: &str) -> Result<Value, String> {
        self.mcp_request_json(
            Method::GET,
            &format!("/api/v1/_host/mcp/servers/{server_id}"),
            None,
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_get_server",
        )
        .await
    }

    pub(crate) async fn create_mcp_server(&self, body: &Value) -> Result<Value, String> {
        self.mcp_request_json(
            Method::POST,
            "/api/v1/_host/mcp/servers",
            Some(body),
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_create_server",
        )
        .await
    }

    pub(crate) async fn update_mcp_server(
        &self,
        server_id: &str,
        body: &Value,
    ) -> Result<Value, String> {
        self.mcp_request_json(
            Method::PATCH,
            &format!("/api/v1/_host/mcp/servers/{server_id}"),
            Some(body),
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_update_server",
        )
        .await
    }

    pub(crate) async fn trust_mcp_server(
        &self,
        server_id: &str,
        expected_config_revision: u64,
        expected_fingerprint: &str,
        enable: bool,
    ) -> Result<Value, String> {
        let body = json!({
            "expected_config_revision": expected_config_revision,
            "expected_fingerprint": expected_fingerprint,
            "enable": enable,
        });
        self.mcp_request_json(
            Method::POST,
            &format!("/api/v1/_host/mcp/servers/{server_id}/trust"),
            Some(&body),
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_trust_server",
        )
        .await
    }

    pub(crate) async fn revoke_mcp_server_trust(&self, server_id: &str) -> Result<Value, String> {
        self.mcp_request_json(
            Method::POST,
            &format!("/api/v1/_host/mcp/servers/{server_id}/revoke-trust"),
            None,
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_revoke_trust",
        )
        .await
    }

    pub(crate) async fn delete_mcp_server(&self, server_id: &str) -> Result<(), String> {
        self.mcp_request_json(
            Method::DELETE,
            &format!("/api/v1/_host/mcp/servers/{server_id}"),
            None,
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_delete_server",
        )
        .await
        .map(|_| ())
    }

    pub(crate) async fn connect_mcp_server(&self, server_id: &str) -> Result<Value, String> {
        self.mcp_request_json(
            Method::POST,
            &format!("/api/v1/_host/mcp/servers/{server_id}/connect"),
            None,
            MCP_CONNECT_TIMEOUT,
            "host_mcp_connect_server",
        )
        .await
    }

    pub(crate) async fn disconnect_mcp_server(&self, server_id: &str) -> Result<Value, String> {
        self.mcp_request_json(
            Method::POST,
            &format!("/api/v1/_host/mcp/servers/{server_id}/disconnect"),
            None,
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_disconnect_server",
        )
        .await
    }

    pub(crate) async fn refresh_mcp_tools(&self, server_id: &str) -> Result<Value, String> {
        self.mcp_request_json(
            Method::POST,
            &format!("/api/v1/_host/mcp/servers/{server_id}/refresh"),
            None,
            MCP_CONNECT_TIMEOUT,
            "host_mcp_refresh_tools",
        )
        .await
    }

    pub(crate) async fn set_mcp_tool_permission(
        &self,
        server_id: &str,
        tool_id: &str,
        body: &Value,
    ) -> Result<Value, String> {
        self.mcp_request_json(
            Method::PATCH,
            &format!("/api/v1/_host/mcp/servers/{server_id}/tools/{tool_id}"),
            Some(body),
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_set_tool_permission",
        )
        .await
    }

    pub(crate) async fn list_mcp_credential_refs(&self) -> Result<Vec<McpCredentialRef>, String> {
        let value = self
            .mcp_request_json(
                Method::GET,
                "/api/v1/_host/mcp/credential-refs",
                None,
                MCP_MANAGEMENT_TIMEOUT,
                "host_mcp_credential_refs",
            )
            .await?;
        serde_json::from_value(value).map_err(|_| "MCP_INVALID_RESPONSE".to_owned())
    }

    pub(crate) async fn put_mcp_credentials(
        &self,
        server_id: &str,
        secrets: &BTreeMap<String, Zeroizing<String>>,
        mode: &str,
    ) -> Result<Value, String> {
        let plain: BTreeMap<&str, &str> = secrets
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let body = json!({
            "secrets": plain,
            "mode": mode,
        });
        self.mcp_request_json(
            Method::PUT,
            &format!("/api/v1/_host/mcp/servers/{server_id}/credentials"),
            Some(&body),
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_put_credentials",
        )
        .await
    }

    pub(crate) async fn delete_mcp_credentials(&self, server_id: &str) -> Result<(), String> {
        self.mcp_request_json(
            Method::DELETE,
            &format!("/api/v1/_host/mcp/servers/{server_id}/credentials"),
            None,
            MCP_MANAGEMENT_TIMEOUT,
            "host_mcp_delete_credentials",
        )
        .await
        .map(|_| ())
    }

    pub(crate) async fn signal_mcp_restore_complete(&self) {
        let _ = self
            .mcp_request_json(
                Method::POST,
                "/api/v1/_host/mcp/restore-complete",
                None,
                MCP_MANAGEMENT_TIMEOUT,
                "host_mcp_restore_complete",
            )
            .await;
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpCredentialRef {
    pub(crate) server_id: String,
    #[allow(dead_code)]
    pub(crate) credential_ref: String,
    pub(crate) secret_keys: Vec<String>,
}

#[cfg(test)]
struct NoopMcpCredentialBackend;

#[cfg(test)]
impl crate::keychain::CredentialBackend for NoopMcpCredentialBackend {
    fn load(
        &self,
        _service: &str,
        _account: &str,
    ) -> Result<Option<Zeroizing<String>>, crate::keychain::KeychainError> {
        Ok(None)
    }
    fn save(
        &self,
        _service: &str,
        _account: &str,
        _secret: &str,
    ) -> Result<(), crate::keychain::KeychainError> {
        Ok(())
    }
    fn delete(
        &self,
        _service: &str,
        _account: &str,
    ) -> Result<(), crate::keychain::KeychainError> {
        Ok(())
    }
}

async fn restore_mcp_credentials_in_background(
    host: HostConnection,
    mcp_credentials: Arc<McpCredentialStore>,
    mut stop: watch::Receiver<bool>,
) {
    if *stop.borrow() {
        return;
    }
    let refs = tokio::select! {
        biased;
        _ = stop.changed() => return,
        res = host.list_mcp_credential_refs() => match res {
            Ok(list) => list,
            Err(code) => {
                host.log.record("mcp_restore_refs_error", json!({
                    "instance_id": host.instance_id,
                    "code": code,
                }));
                return;
            }
        }
    };

    for item in refs {
        if *stop.borrow() {
            return;
        }
        let loaded = {
            let _guard = mcp_credentials.lock().await;
            mcp_credentials.load(&item.server_id)
        };
        match loaded {
            Ok(Some(all_secrets)) => {
                // Only replay keys declared in SQLite for this server_id; ignore any orphaned keys.
                let filtered: BTreeMap<String, Zeroizing<String>> = all_secrets
                    .into_iter()
                    .filter(|(k, _)| item.secret_keys.iter().any(|declared| declared == k))
                    .collect();
                let missing_count = item
                    .secret_keys
                    .iter()
                    .filter(|k| !filtered.contains_key(*k))
                    .count();
                if missing_count > 0 {
                    host.log.record(
                        "mcp_keychain_restore_partial",
                        json!({
                            "instance_id": host.instance_id,
                            "server_id": item.server_id,
                            "missing_count": missing_count,
                        }),
                    );
                }
                if !filtered.is_empty() {
                    if let Err(code) = host
                        .put_mcp_credentials(&item.server_id, &filtered, "restore")
                        .await
                    {
                        host.log.record(
                            "mcp_keychain_restore_replay_error",
                            json!({
                                "instance_id": host.instance_id,
                                "server_id": item.server_id,
                                "code": code,
                            }),
                        );
                    }
                }
            }
            Ok(None) => {
                host.log.record(
                    "mcp_keychain_restore_missing",
                    json!({
                        "instance_id": host.instance_id,
                        "server_id": item.server_id,
                    }),
                );
            }
            Err(error) => {
                host.log.record(
                    "mcp_keychain_restore_read_error",
                    json!({
                        "instance_id": host.instance_id,
                        "server_id": item.server_id,
                        "service": mcp_credentials.service(),
                        "reason": error.diagnostic_reason(),
                        "os_status": error.os_status(),
                    }),
                );
            }
        }
    }

    if !*stop.borrow() {
        host.signal_mcp_restore_complete().await;
    }
}

async fn reconfigure_after_ready(
    host: HostConnection,
    credentials: Arc<CredentialStore>,
) -> CredentialReplay {
    let _guard = credentials.lock().await;
    replay_credential(&host, &credentials, DEEPSEEK_PROVIDER).await?;
    Ok(replay_credential(&host, &credentials, SILICONFLOW_PROVIDER)
        .await
        .err())
}

async fn replay_credential(
    host: &HostConnection,
    credentials: &CredentialStore,
    provider: &str,
) -> Result<(), &'static str> {
    match credentials.load(provider) {
        Ok(Some(key)) => {
            host.log.record(
                "keychain_credential_loaded",
                json!({
                    "instance_id": host.instance_id,
                    "service": credentials.service(),
                    "provider": provider,
                }),
            );
            if host.put_credential(provider, &key).await.is_err() {
                host.log.record(
                    "keychain_credential_replay_error",
                    json!({"instance_id": host.instance_id, "provider": provider, "reason": REPLAY_FAILED}),
                );
                return Err(REPLAY_FAILED);
            }
            Ok(())
        }
        Ok(None) => {
            host.log.record(
                "keychain_credential_absent",
                json!({
                    "instance_id": host.instance_id,
                    "service": credentials.service(),
                    "provider": provider,
                }),
            );
            Ok(())
        }
        Err(error) => {
            host.log.record(
                "keychain_read_error",
                json!({
                    "instance_id": host.instance_id,
                    "service": credentials.service(),
                    "provider": provider,
                    "reason": error.diagnostic_reason(),
                    "os_status": error.os_status(),
                }),
            );
            Err(error.ipc_code())
        }
    }
}

struct RunningSidecar {
    child: Child,
    stdin: ChildStdin,
    connection: SidecarConnection,
}

impl RunningSidecar {
    async fn supervise(
        mut self,
        app_data_dir: &Path,
        mcp_credentials: &Arc<McpCredentialStore>,
        requests: &mut mpsc::Receiver<ConnectionReply>,
        closing: &mut watch::Receiver<bool>,
    ) {
        let sidecar_pid = self.child.id();
        let instance_id = self.connection.host.instance_id;
        let log = self.connection.host.log.clone();

        // Spawn non-blocking background MCP credential recovery after Sidecar is ready.
        let (restore_cancel_tx, restore_cancel_rx) = watch::channel(false);
        let restore_task = tauri::async_runtime::spawn(restore_mcp_credentials_in_background(
            self.connection.host.clone(),
            mcp_credentials.clone(),
            restore_cancel_rx,
        ));

        loop {
            tokio::select! {
                biased;
                _ = closing.changed() => break,
                result = self.child.wait() => {
                    let _ = restore_cancel_tx.send(true);
                    restore_task.abort();
                    self.connection.host.log.record("sidecar_exited", json!({
                        "instance_id": instance_id,
                        "success": result.as_ref().is_ok_and(|status| status.success()),
                    }));
                    match result {
                        Ok(status) => eprintln!("KnowledgeAgent Sidecar: Python exited ({status})."),
                        Err(_) => {
                            drop(self.stdin);
                            wait_or_terminate(&mut self.child).await;
                        }
                    }
                    // Host-level supervision: reap any verified stdio MCP process groups left behind
                    // if Sidecar crashed or received SIGKILL.
                    let _ = mcp_process::reap_instance_processes(
                        app_data_dir,
                        instance_id,
                        sidecar_pid,
                        &log,
                    )
                    .await;
                    return;
                },
                request = requests.recv() => match request {
                    Some(reply) => { let _ = reply.send(Ok(self.connection.clone())); }
                    None => break,
                },
            }
        }

        let _ = restore_cancel_tx.send(true);
        restore_task.abort();

        if self.connection.host.request_shutdown().await {
            eprintln!("KnowledgeAgent Sidecar: host shutdown acknowledged.");
        } else {
            eprintln!("KnowledgeAgent Sidecar: host shutdown request failed, closing stdin.");
        }
        drop(self.stdin);
        wait_or_terminate(&mut self.child).await;
        // Post-shutdown safety check: reap any stubborn MCP child group if Sidecar had to be SIGKILL'd.
        let _ = mcp_process::reap_instance_processes(
            app_data_dir,
            instance_id,
            sidecar_pid,
            &log,
        )
        .await;
    }
}

async fn wait_or_terminate(child: &mut Child) {
    if let Ok(Ok(status)) = timeout(SHUTDOWN_TIMEOUT, child.wait()).await {
        eprintln!("KnowledgeAgent Sidecar: Python stopped ({status}).");
        return;
    }
    eprintln!("KnowledgeAgent Sidecar: terminating the supervised Python process.");
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // The spawned child leads its own group. This also covers the onefile worker.
        unsafe { libc::killpg(pid as libc::pid_t, libc::SIGKILL) };
    }
    let _ = child.start_kill();
    let _ = timeout(SHUTDOWN_TIMEOUT, child.wait()).await;
}
