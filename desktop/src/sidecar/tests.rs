use std::sync::Arc;

use tempfile::tempdir;
use tokio::sync::Semaphore;

use super::*;
use crate::keychain::{CredentialBackend, KeychainError};

struct TestKeychainGuard {
    store: Arc<CredentialStore>,
}

impl TestKeychainGuard {
    fn new() -> Self {
        let id = format!(
            "local.knowledgeagent.desktop.test.{}",
            random_token().unwrap()
        );
        Self {
            store: CredentialStore::for_identifier(&id),
        }
    }
}

impl Drop for TestKeychainGuard {
    fn drop(&mut self) {
        let _ = self.store.delete(DEEPSEEK_PROVIDER);
        let _ = self.store.delete(SILICONFLOW_PROVIDER);
    }
}

#[tokio::test]
async fn ready_frame_requires_a_supported_version_instance_and_port() {
    let valid = serde_json::json!({
        "type": "ready", "protocol_version": 1,
        "instance_id": "d2ab1e6d-3f55-485e-bfab-14ece5424c0b", "port": 54321,
    });
    let parse = |value: serde_json::Value| async move {
        let line = format!("{value}\n");
        read_ready(line.as_bytes()).await
    };

    assert_eq!(parse(valid.clone()).await.unwrap().port, 54321);
    for (field, value) in [
        ("type", serde_json::json!("bootstrap")),
        ("protocol_version", serde_json::json!(2)),
        ("instance_id", serde_json::json!("invalid")),
        ("instance_id", serde_json::json!(Uuid::nil())),
        ("port", serde_json::json!(0)),
        ("port", serde_json::json!(65536)),
        ("extra", serde_json::json!(true)),
    ] {
        let mut bad = valid.clone();
        bad[field] = value;
        assert!(parse(bad).await.is_err(), "accepted invalid {field}");
    }
    assert!(read_ready(valid.to_string().as_bytes()).await.is_err());
    let oversized = vec![b' '; MAX_READY_BYTES as usize + 1];
    assert!(read_ready(oversized.as_slice()).await.is_err());
}

#[tokio::test]
async fn mismatched_host_ready_identity_never_reaches_the_hook_or_ipc() {
    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    let command = python_script(
        r#"
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from uuid import uuid4

bootstrap = json.loads(sys.stdin.buffer.readline())

class WrongInstance(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/api/v1/_host/ready':
            self.send_error(404)
            return
        if self.headers.get('Authorization') != 'Bearer ' + bootstrap['host_token']:
            self.send_error(401)
            return
        body = json.dumps({'status': 'ready', 'instance_id': str(uuid4())}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

with HTTPServer(('127.0.0.1', 0), WrongInstance) as server:
    print(json.dumps({'type': 'ready', 'protocol_version': 1,
                      'instance_id': str(uuid4()), 'port': server.server_port}), flush=True)
    server.handle_request()
    sys.stdin.buffer.read()
"#,
    );
    let sidecar = Sidecar::start(
        command,
        data.path().to_owned(),
        keychain.store.clone(),
        |_, _| async {
            panic!("hook ran without a matching host ready response");
        },
    );
    assert_eq!(sidecar.connection().await.err(), Some(UNAVAILABLE));
    assert!(sidecar.request_shutdown());
    sidecar.wait_stopped().await;
}

#[cfg(unix)]
#[tokio::test]
async fn crash_retry_runs_the_hook_again_before_publishing_rotated_connection() {
    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    let pid_file = data.path().join("sidecar.pid");
    let digest_file = data.path().join("credential-sha256.txt");
    let mut command = python_script(
        r#"
import hashlib, os, sys
from pathlib import Path
from app.main import create_app
from app.services.models import ModelService
from app.sidecar import main

pid_path = Path(sys.argv[1])
digest_path = Path(sys.argv[2])
pid_path.write_text(str(os.getpid()))

class TrackingModels(ModelService):
    def configure_deepseek(self, key, base_url):
        super().configure_deepseek(key, base_url)
        digest_path.write_text(hashlib.sha256(key.get_secret_value().encode()).hexdigest())

    def clear_deepseek(self):
        super().clear_deepseek()
        digest_path.write_text("cleared")

def factory(*args, **kwargs):
    return create_app(*args, **kwargs, models=TrackingModels())

raise SystemExit(main(app_factory=factory))
"#,
    );
    command.arg(&pid_file).arg(&digest_file);

    let (hook_calls, mut hooks) = mpsc::channel(2);
    let reconfiguration = Arc::new(Semaphore::new(0));
    let gate = reconfiguration.clone();
    let app_data = data.path().join("app-data");
    let sidecar = Sidecar::start(
        command,
        app_data.clone(),
        keychain.store.clone(),
        move |host, credentials| {
            let gate = gate.clone();
            let hook_calls = hook_calls.clone();
            async move {
                assert!(host.check_ready().await.is_ok());
                let _ = hook_calls.send(host.clone()).await;
                gate.acquire().await.unwrap().forget();
                reconfigure_after_ready(host, credentials).await
            }
        },
    );

    let first_request = sidecar.connection();
    tokio::pin!(first_request);
    let first_host = tokio::select! {
        _ = &mut first_request => panic!("connection published before the hook completed"),
        call = hooks.recv() => call.unwrap(),
    };
    assert!(timeout(Duration::from_millis(25), &mut first_request)
        .await
        .is_err());
    reconfiguration.add_permits(1);
    let first = first_request.await.unwrap();
    let serialized = serde_json::to_value(&first).unwrap();
    assert_eq!(serialized.as_object().unwrap().len(), 2);
    assert!(serialized.get("base_url").is_some());
    assert!(serialized.get("ui_token").is_some());
    assert!(!serialized.to_string().contains(&first_host.host_token));

    // No credential in Keychain at first startup -> model_configured is false.
    let status_url = format!("{}/api/v1/status", first.base_url);
    let unconfigured = first_host
        .client
        .get(&status_url)
        .bearer_auth(&first.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(
        unconfigured.json::<serde_json::Value>().await.unwrap()["model_configured"],
        false
    );

    // Save initial credential, then rotate to a second credential in Keychain.
    let initial_key = Zeroizing::new(random_token().unwrap());
    let rotated_key = Zeroizing::new(random_token().unwrap());
    assert_ne!(initial_key.as_str(), rotated_key.as_str());
    assert!(
        sidecar
            .save_provider_credential("deepseek", initial_key.clone())
            .await
            .unwrap()
            .configured
    );
    let initial_digest = std::fs::read_to_string(&digest_file).unwrap();
    assert!(
        sidecar
            .save_provider_credential("deepseek", rotated_key.clone())
            .await
            .unwrap()
            .configured
    );
    let rotated_digest = std::fs::read_to_string(&digest_file).unwrap();
    assert_ne!(initial_digest, rotated_digest);
    assert_eq!(
        keychain.store.load("deepseek").unwrap(),
        Some(rotated_key.clone())
    );
    // Wipe the digest file before killing Sidecar to prove crash recovery replays the rotated key.
    std::fs::write(&digest_file, "").unwrap();

    let session = first_host
        .client
        .post(format!("{}/api/v1/sessions", first.base_url))
        .bearer_auth(&first.ui_token)
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(session.status(), StatusCode::CREATED);
    let session: serde_json::Value = session.json().await.unwrap();
    let session_id = Uuid::parse_str(session["session_id"].as_str().unwrap()).unwrap();
    let note = data.path().canonicalize().unwrap().join("p41-note.md");
    std::fs::write(&note, "fixture").unwrap();
    let path = note.to_str().unwrap().to_owned();
    let grant = sidecar
        .select_grant(session_id, GrantScope::File, async {
            Ok(Some(path.clone()))
        })
        .await
        .unwrap()
        .unwrap();
    let (dialog_open, wait_dialog) = oneshot::channel();
    let (confirm, wait_confirmation) = oneshot::channel();
    let pending_selection = tokio::spawn({
        let sidecar = sidecar.clone();
        async move {
            sidecar
                .select_grant(session_id, GrantScope::File, async {
                    dialog_open.send(()).unwrap();
                    wait_confirmation.await.unwrap();
                    Ok(Some(path))
                })
                .await
        }
    });
    wait_dialog.await.unwrap();

    let pid = std::fs::read_to_string(&pid_file)
        .unwrap()
        .parse::<u32>()
        .unwrap();
    assert!(Command::new("/bin/kill")
        .args(["-KILL", &pid.to_string()])
        .status()
        .await
        .unwrap()
        .success());
    timeout(Duration::from_secs(5), async {
        while first_host.check_ready().await.is_ok() {
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
        // Let the supervised wait reap the killed process before the next UI request.
        while Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .await
            .unwrap()
            .success()
        {
            tokio::time::sleep(Duration::from_millis(15)).await;
        }
    })
    .await
    .unwrap();

    let retry = sidecar.connection();
    tokio::pin!(retry);
    let second_host = tokio::select! {
        _ = &mut retry => panic!("retry published before the new hook completed"),
        call = hooks.recv() => call.unwrap(),
    };
    assert!(timeout(Duration::from_millis(25), &mut retry)
        .await
        .is_err());
    reconfiguration.add_permits(1);
    let second = retry.await.unwrap();
    assert_ne!(first_host.instance_id, second_host.instance_id);
    assert!(first_host.host_token != second_host.host_token);
    assert!(first.ui_token != second.ui_token && second.ui_token != second_host.host_token);
    assert_eq!(second.base_url, second_host.base_url);
    // Crash recovery must have replayed the rotated Keychain credential, not the initial one.
    assert_eq!(
        std::fs::read_to_string(&digest_file).unwrap(),
        rotated_digest
    );

    confirm.send(()).unwrap();
    assert_eq!(
        pending_selection.await.unwrap().err(),
        Some("GRANT_CREATE_FAILED")
    );
    let grants = second_host
        .client
        .get(format!(
            "{}/api/v1/sessions/{session_id}/grants",
            second.base_url
        ))
        .bearer_auth(&second.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(grants.status(), StatusCode::OK);
    assert_eq!(grants.json::<serde_json::Value>().await.unwrap(), json!([]));
    assert_eq!(
        sidecar.revoke_grant(session_id, grant.grant_id).await.err(),
        Some("GRANT_NOT_FOUND")
    );
    let status = format!("{}/api/v1/status", second.base_url);
    assert_eq!(
        second_host
            .client
            .get(&status)
            .bearer_auth(&first.ui_token)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED,
    );
    let response = second_host
        .client
        .get(&status)
        .bearer_auth(&second.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.json::<serde_json::Value>().await.unwrap()["model_configured"],
        true
    );

    // Delete the credential: Keychain item is removed and Sidecar transitions to unconfigured.
    assert_eq!(
        sidecar.delete_provider_credential("other").await.err(),
        Some(CREDENTIAL_REJECTED)
    );
    assert!(
        !sidecar
            .delete_provider_credential("deepseek")
            .await
            .unwrap()
            .configured
    );
    assert_eq!(keychain.store.load("deepseek").unwrap(), None);
    assert_eq!(std::fs::read_to_string(&digest_file).unwrap(), "cleared");
    let after_delete = second_host
        .client
        .get(&status)
        .bearer_auth(&second.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(
        after_delete.json::<serde_json::Value>().await.unwrap()["model_configured"],
        false
    );
    let denied_run = second_host
        .client
        .post(format!(
            "{}/api/v1/sessions/{session_id}/runs/stream",
            second.base_url
        ))
        .bearer_auth(&second.ui_token)
        .json(&json!({
            "client_request_id": "d2ab1e6d-3f55-485e-bfab-14ece5424c0b",
            "message": "删除后请求模型"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(denied_run.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        denied_run.json::<serde_json::Value>().await.unwrap()["error"]["code"],
        "MODEL_NOT_CONFIGURED"
    );

    assert!(sidecar.request_shutdown());
    assert!(!sidecar.request_shutdown());
    assert_eq!(sidecar.connection().await.err(), Some(SHUTTING_DOWN));
    sidecar.clone().wait_stopped().await;
    assert!(sidecar.is_stopped());
    assert!(second_host.check_ready().await.is_err());

    let host_logs = std::fs::read_to_string(app_data.join("logs/host.jsonl")).unwrap();
    let sidecar_logs = std::fs::read_to_string(app_data.join("logs/sidecar.jsonl")).unwrap();
    for secret in [
        initial_key.as_str(),
        rotated_key.as_str(),
        &first.ui_token,
        &first_host.host_token,
        &second.ui_token,
        &second_host.host_token,
    ] {
        assert!(!host_logs.contains(secret));
        assert!(!sidecar_logs.contains(secret));
    }
    assert!(host_logs.contains("keychain_credential_saved"));
    assert!(host_logs.contains("keychain_credential_loaded"));
    assert!(host_logs.contains("keychain_credential_deleted"));
}

fn python_script(script: &str) -> Command {
    let development = development_command();
    let template = development.as_std();
    let mut command = Command::new(template.get_program());
    command.current_dir(template.get_current_dir().unwrap());
    command.args(["-E", "-s", "-B", "-c", script]);
    command
}

#[tokio::test]
async fn rejected_replay_does_not_publish_a_connection() {
    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    keychain
        .store
        .save(DEEPSEEK_PROVIDER, &random_token().unwrap())
        .unwrap();
    let mut command = development_command();
    command.env("KNOWLEDGEAGENT_DEEPSEEK_MODEL", "");
    let sidecar = Sidecar::start(
        command,
        data.path().to_owned(),
        keychain.store.clone(),
        reconfigure_after_ready,
    );

    assert_eq!(sidecar.connection().await.err(), Some(REPLAY_FAILED));
    sidecar.request_shutdown();
    sidecar.wait_stopped().await;
}

struct FailingBackend(KeychainError);

impl CredentialBackend for FailingBackend {
    fn load(
        &self,
        _service: &str,
        _account: &str,
    ) -> Result<Option<Zeroizing<String>>, KeychainError> {
        Err(self.0)
    }

    fn save(&self, _service: &str, _account: &str, _secret: &str) -> Result<(), KeychainError> {
        Err(self.0)
    }

    fn delete(&self, _service: &str, _account: &str) -> Result<(), KeychainError> {
        Err(self.0)
    }
}

#[tokio::test]
async fn keychain_read_errors_are_reported_without_publishing_connection() {
    for (error, expected_code) in [
        (KeychainError::AccessDenied(-25293), KEYCHAIN_ACCESS_FAILED),
        (KeychainError::Corrupted, KEYCHAIN_CREDENTIAL_CORRUPTED),
    ] {
        let data = tempdir().unwrap();
        let store = CredentialStore::with_backend(
            "local.knowledgeagent.desktop.test.failing",
            Arc::new(FailingBackend(error)),
        );
        let sidecar = Sidecar::start(
            development_command(),
            data.path().to_owned(),
            store,
            reconfigure_after_ready,
        );
        assert_eq!(sidecar.connection().await.err(), Some(expected_code));
        sidecar.request_shutdown();
        sidecar.wait_stopped().await;
    }
}

#[tokio::test]
async fn full_restart_restores_from_real_keychain_and_write_delete_failures_do_not_desync() {
    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    let app_data = data.path().join("app-data");

    let first_sidecar = Sidecar::start(
        development_command(),
        app_data.clone(),
        keychain.store.clone(),
        reconfigure_after_ready,
    );
    let first_conn = first_sidecar.connection().await.unwrap();
    let key = Zeroizing::new(random_token().unwrap());
    let siliconflow = Zeroizing::new(random_token().unwrap());
    let rotated_siliconflow = Zeroizing::new(random_token().unwrap());
    assert!(
        first_sidecar
            .save_provider_credential(DEEPSEEK_PROVIDER, key.clone())
            .await
            .unwrap()
            .configured
    );
    for credential in [&siliconflow, &rotated_siliconflow] {
        assert!(
            first_sidecar
                .save_provider_credential(SILICONFLOW_PROVIDER, credential.clone())
                .await
                .unwrap()
                .configured
        );
    }
    assert_eq!(
        keychain.store.load(DEEPSEEK_PROVIDER).unwrap().as_ref(),
        Some(&key)
    );
    assert_eq!(
        keychain.store.load(SILICONFLOW_PROVIDER).unwrap().as_ref(),
        Some(&rotated_siliconflow)
    );
    assert!(first_sidecar.request_shutdown());
    first_sidecar.wait_stopped().await;

    // A fresh host instance using the same Keychain service must auto-configure the new Sidecar.
    let second_sidecar = Sidecar::start(
        development_command(),
        app_data.clone(),
        keychain.store.clone(),
        reconfigure_after_ready,
    );
    let second_conn = second_sidecar.connection().await.unwrap();
    assert_ne!(first_conn.base_url, second_conn.base_url);
    assert_ne!(first_conn.ui_token, second_conn.ui_token);
    assert!(second_conn.siliconflow_restore_error.is_none());
    let client = Client::builder().no_proxy().build().unwrap();
    let read_status = || async {
        client
            .get(format!("{}/api/v1/status", second_conn.base_url))
            .bearer_auth(&second_conn.ui_token)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json::<serde_json::Value>()
            .await
            .unwrap()
    };
    let status = read_status().await;
    assert_eq!(status["model_configured"], true);
    assert_eq!(status["siliconflow_configured"], true);
    let denied = client
        .post(format!(
            "{}/api/v1/_host/model-credentials/siliconflow/test",
            second_conn.base_url
        ))
        .bearer_auth(&second_conn.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    assert!(
        !second_sidecar
            .delete_provider_credential(SILICONFLOW_PROVIDER)
            .await
            .unwrap()
            .configured
    );
    assert!(keychain.store.load(SILICONFLOW_PROVIDER).unwrap().is_none());
    let status = read_status().await;
    assert_eq!(status["model_configured"], true);
    assert_eq!(status["siliconflow_configured"], false);
    assert_eq!(
        second_sidecar.test_siliconflow_connection().await.err(),
        Some("EMBEDDING_NOT_CONFIGURED")
    );
    assert!(second_sidecar.request_shutdown());
    second_sidecar.wait_stopped().await;

    // Keychain write/delete failures must return KEYCHAIN_ACCESS_FAILED and never falsely configure Sidecar.
    struct WriteFailBackend;
    impl CredentialBackend for WriteFailBackend {
        fn load(
            &self,
            _service: &str,
            _account: &str,
        ) -> Result<Option<Zeroizing<String>>, KeychainError> {
            Ok(None)
        }
        fn save(&self, _service: &str, _account: &str, _secret: &str) -> Result<(), KeychainError> {
            Err(KeychainError::AccessDenied(-25293))
        }
        fn delete(&self, _service: &str, _account: &str) -> Result<(), KeychainError> {
            Err(KeychainError::AccessDenied(-25293))
        }
    }

    let failing_store = CredentialStore::with_backend(
        "local.knowledgeagent.desktop.test.write-fail",
        Arc::new(WriteFailBackend),
    );
    let third_sidecar = Sidecar::start(
        development_command(),
        data.path().join("write-fail-data"),
        failing_store,
        reconfigure_after_ready,
    );
    let third_conn = third_sidecar.connection().await.unwrap();
    assert_eq!(
        third_sidecar
            .save_provider_credential("deepseek", Zeroizing::new(random_token().unwrap()))
            .await
            .err(),
        Some(KEYCHAIN_ACCESS_FAILED)
    );
    assert_eq!(
        third_sidecar
            .delete_provider_credential("deepseek")
            .await
            .err(),
        Some(KEYCHAIN_ACCESS_FAILED)
    );
    let status_after_failed_write = client
        .get(format!("{}/api/v1/status", third_conn.base_url))
        .bearer_auth(&third_conn.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(
        status_after_failed_write
            .json::<serde_json::Value>()
            .await
            .unwrap()["model_configured"],
        false
    );
    assert!(third_sidecar.request_shutdown());
    third_sidecar.wait_stopped().await;
}

#[cfg(unix)]
#[tokio::test]
async fn second_instance_on_same_data_dir_reports_database_locked_and_recovers_after_owner_exits() {
    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    let app_data = data.path().join("shared-app-data");

    let primary = Sidecar::start(
        development_command(),
        app_data.clone(),
        keychain.store.clone(),
        reconfigure_after_ready,
    );
    let primary_conn = primary.connection().await.unwrap();

    let contender = Sidecar::start(
        development_command(),
        app_data.clone(),
        keychain.store.clone(),
        reconfigure_after_ready,
    );
    assert_eq!(contender.connection().await.err(), Some(DATABASE_LOCKED));

    // Primary instance must remain healthy while contender is rejected.
    let client = Client::builder().no_proxy().build().unwrap();
    let status = client
        .get(format!("{}/api/v1/status", primary_conn.base_url))
        .bearer_auth(&primary_conn.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(status.status(), StatusCode::OK);

    assert!(primary.request_shutdown());
    primary.wait_stopped().await;

    // Once primary releases the database lock, contender can retry and connect cleanly.
    let recovered_conn = contender.connection().await.unwrap();
    let recovered_status = client
        .get(format!("{}/api/v1/status", recovered_conn.base_url))
        .bearer_auth(&recovered_conn.ui_token)
        .send()
        .await
        .unwrap();
    assert_eq!(recovered_status.status(), StatusCode::OK);
    assert!(contender.request_shutdown());
    contender.wait_stopped().await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn textbook_import_timeout_and_restart_replay_the_same_receipt() {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    let pause_flag = data.path().join("hold-import-response");
    std::fs::write(&pause_flag, b"").unwrap();
    let mut command = python_script(
        r#"
import sys, time
from pathlib import Path
from app.services.library import LibraryService
from app.sidecar import main

flag = Path(sys.argv[1])
original = LibraryService.import_textbook

def import_textbook(self, *args, **kwargs):
    receipt = original(self, *args, **kwargs)
    if flag.exists():
        flag.unlink()
        time.sleep(1.5)
    return receipt

LibraryService.import_textbook = import_textbook
raise SystemExit(main())
"#,
    );
    command.arg(&pause_flag);
    let sidecar = Sidecar::start(
        command,
        data.path().to_owned(),
        keychain.store.clone(),
        |_, _| async { Ok(None) },
    );
    let stage = |host: &HostConnection, import_id: Uuid| {
        let dir = data
            .path()
            .join("knowledge/temporary")
            .join(host.instance_id.to_string());
        assert_eq!(dir.metadata().unwrap().permissions().mode() & 0o777, 0o700);
        let path = dir.join(import_id.to_string());
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        file.write_all(b"# rust host textbook").unwrap();
        file.sync_all().unwrap();
        path
    };

    let first = sidecar.import_connection().await.unwrap();
    let request = HostImport {
        import_id: uuid::Builder::from_random_bytes([37; 16]).into_uuid(),
        filename: "教材.md".into(),
        document_id: None,
    };
    let staged = stage(&first, request.import_id);
    assert!(matches!(
        first
            .send_import(&request, Duration::from_millis(500))
            .await,
        Err(HostImportError::Uncertain("IMPORT_REQUEST_TIMEOUT"))
    ));
    assert!(!pause_flag.exists());
    assert!(!staged.exists());
    let receipt = first.import_document(&request).await.unwrap();
    assert_eq!(receipt.import_id, request.import_id);
    assert!(matches!(
        receipt.outcome,
        crate::library::ImportOutcome::Imported
    ));
    assert!(sidecar.request_shutdown());
    sidecar.wait_stopped().await;
    assert!(matches!(
        first.import_document(&request).await,
        Err(HostImportError::Uncertain(UNAVAILABLE))
    ));

    let sidecar = Sidecar::start(
        development_command(),
        data.path().to_owned(),
        keychain.store.clone(),
        |_, _| async { Ok(None) },
    );
    let restarted = sidecar.import_connection().await.unwrap();
    assert_ne!(first.instance_id, restarted.instance_id);
    stage(&restarted, request.import_id);
    let replay = restarted.import_document(&request).await.unwrap();
    assert_eq!(replay.import_id, receipt.import_id);
    assert_eq!(replay.document_id, receipt.document_id);
    assert_eq!(replay.document_version_id, receipt.document_version_id);
    assert!(matches!(
        replay.outcome,
        crate::library::ImportOutcome::Imported
    ));

    let mut missing = request;
    missing.import_id = uuid::Builder::from_random_bytes([42; 16]).into_uuid();
    missing.document_id = Some("b".repeat(32));
    stage(&restarted, missing.import_id);
    assert!(matches!(
        restarted.import_document(&missing).await,
        Err(HostImportError::Rejected("TEXTBOOK_NOT_FOUND"))
    ));
    assert!(sidecar.request_shutdown());
    sidecar.wait_stopped().await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn sidecar_sigkill_reaps_stubborn_stdio_mcp_process_group_and_normal_exit_cleans_up() {
    use std::collections::BTreeMap;
    use std::path::Path;

    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    let mcp_service = format!("local.knowledgeagent.desktop.test.{}.mcp-sigkill", random_token().unwrap());
    let mcp_creds = McpCredentialStore::with_backend(mcp_service, Arc::new(NoopMcpCredentialBackend));

    let backend_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../backend");
    let python_exe = backend_dir
        .join(".venv/bin/python")
        .canonicalize()
        .expect("backend venv python");
    let stubborn_script = backend_dir
        .join("tests/fixtures/mcp/stdio_test_server.py")
        .canonicalize()
        .expect("stdio_test_server.py");

    let sidecar = Sidecar::start_with_mcp(
        development_command(),
        data.path().to_owned(),
        keychain.store.clone(),
        mcp_creds.clone(),
        |_, _| async { Ok(None) },
    );
    let host = sidecar.mcp_host_connection().await.unwrap();
    let instance_id = host.host_instance_id();

    // Find Sidecar OS PID from host.jsonl
    let host_log_path = data.path().join("logs/host.jsonl");
    let sidecar_pid: u32 = std::fs::read_to_string(&host_log_path)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|ev| {
            ev.get("event").and_then(|v| v.as_str()) == Some("sidecar_connected")
                && ev.get("instance_id").and_then(|v| v.as_str()) == Some(&instance_id.to_string())
        })
        .and_then(|ev| ev.get("child_pid").and_then(|v| v.as_u64()))
        .expect("sidecar child_pid in host.jsonl") as u32;

    // Create, trust, and connect a stubborn stdio MCP server (ignores EOF & SIGTERM!)
    let created = host
        .create_mcp_server(&json!({
            "name": "StubbornStdio",
            "transport": "stdio",
            "command": python_exe.to_string_lossy(),
            "args": [stubborn_script.to_string_lossy(), "--ignore-eof-and-sigterm"],
            "env_config": BTreeMap::<String, serde_json::Value>::new(),
            "header_config": BTreeMap::<String, serde_json::Value>::new(),
        }))
        .await
        .unwrap();
    let server_id = created["server_id"].as_str().unwrap().to_owned();
    let rev = created["config_revision"].as_u64().unwrap();
    let fp = created["current_fingerprint"].as_str().unwrap().to_owned();
    host.trust_mcp_server(&server_id, rev, &fp, true)
        .await
        .unwrap();
    let connected = host.connect_mcp_server(&server_id).await.unwrap();
    assert_eq!(connected["runtime"]["state"].as_str(), Some("connected"));

    // Read the registered managed MCP child PID from the instance process registry
    let proc_dir = data
        .path()
        .join("runtime/mcp-procs")
        .join(instance_id.to_string());
    let mut mcp_child_pid: Option<u32> = None;
    for entry in std::fs::read_dir(&proc_dir).unwrap().flatten() {
        let raw = std::fs::read(entry.path()).unwrap();
        let val: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        if val.get("kind").and_then(|v| v.as_str()) == Some("active") {
            mcp_child_pid = val.get("pid").and_then(|v| v.as_u64()).map(|p| p as u32);
            break;
        }
    }
    let mcp_pid = mcp_child_pid.expect("active MCP process record must exist");
    assert_eq!(
        unsafe { libc::kill(mcp_pid as libc::pid_t, 0) },
        0,
        "stubborn MCP child must be alive before Sidecar SIGKILL"
    );

    // Send SIGKILL directly to the Sidecar Python process!
    assert_eq!(unsafe { libc::kill(sidecar_pid as libc::pid_t, libc::SIGKILL) }, 0);

    // Wait for Rust host supervisor to detect Sidecar death, reap the stubborn MCP child group,
    // log mcp_orphan_reaped, and remove the instance registry directory.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let rc = unsafe { libc::kill(mcp_pid as libc::pid_t, 0) };
        let child_gone = rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
        if child_gone && !proc_dir.exists() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "stubborn stdio MCP child (pid {mcp_pid}) or registry dir survived Sidecar SIGKILL!"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Verify host.jsonl recorded mcp_orphan_reaped and registry dir was cleaned
    let log_text = std::fs::read_to_string(&host_log_path).unwrap();
    assert!(
        log_text.contains("\"event\":\"mcp_orphan_reaped\""),
        "expected mcp_orphan_reaped event in host.jsonl"
    );
    assert!(!proc_dir.exists());

    // Now test normal restart + clean shutdown leaves 0 processes and 0 registry files
    let host2 = sidecar.mcp_host_connection().await.unwrap();
    let inst2 = host2.host_instance_id();
    assert_ne!(inst2, instance_id);
    // Auto-connect (via background mcp restore-complete) or explicit connect works on restarted Sidecar
    let conn2 = host2.connect_mcp_server(&server_id).await.unwrap();
    assert_eq!(conn2["runtime"]["state"].as_str(), Some("connected"));
    let proc_dir_2 = data
        .path()
        .join("runtime/mcp-procs")
        .join(inst2.to_string());
    let mut mcp_pid_2 = None;
    for entry in std::fs::read_dir(&proc_dir_2).unwrap().flatten() {
        let val: serde_json::Value =
            serde_json::from_slice(&std::fs::read(entry.path()).unwrap()).unwrap();
        if val.get("kind").and_then(|v| v.as_str()) == Some("active") {
            mcp_pid_2 = val.get("pid").and_then(|v| v.as_u64()).map(|p| p as u32);
        }
    }
    let mcp_pid_2 = mcp_pid_2.expect("restarted MCP child pid");

    assert!(sidecar.request_shutdown());
    sidecar.wait_stopped().await;

    assert_ne!(
        unsafe { libc::kill(mcp_pid_2 as libc::pid_t, 0) },
        0,
        "MCP child must exit cleanly on normal shutdown"
    );
    assert!(!proc_dir_2.exists());
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn mcp_keychain_sqlite_memory_partial_failure_compensation_and_startup_recovery() {
    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;
    use std::sync::Mutex as StdMutex;

    use crate::mcp::{
        connect_server_with_store, delete_server_with_store, save_server_with_store,
        ConfigEntryRequest, SaveMcpServerRequest, MCP_KEYCHAIN_CLEANUP_FAILED,
    };

    #[derive(Default)]
    struct FaultyKeychainBackend {
        items: StdMutex<HashMap<(String, String), Zeroizing<String>>>,
        fail_save: StdMutex<bool>,
        fail_delete: StdMutex<bool>,
    }

    impl CredentialBackend for FaultyKeychainBackend {
        fn load(
            &self,
            service: &str,
            account: &str,
        ) -> Result<Option<Zeroizing<String>>, KeychainError> {
            Ok(self
                .items
                .lock()
                .unwrap()
                .get(&(service.to_owned(), account.to_owned()))
                .cloned())
        }
        fn save(&self, service: &str, account: &str, secret: &str) -> Result<(), KeychainError> {
            if *self.fail_save.lock().unwrap() {
                return Err(KeychainError::Unavailable(-25291));
            }
            self.items.lock().unwrap().insert(
                (service.to_owned(), account.to_owned()),
                Zeroizing::new(secret.to_owned()),
            );
            Ok(())
        }
        fn delete(&self, service: &str, account: &str) -> Result<(), KeychainError> {
            if *self.fail_delete.lock().unwrap() {
                return Err(KeychainError::AccessDenied(-25293));
            }
            self.items
                .lock()
                .unwrap()
                .remove(&(service.to_owned(), account.to_owned()));
            Ok(())
        }
    }

    let data = tempdir().unwrap();
    let keychain = TestKeychainGuard::new();
    let faulty = Arc::new(FaultyKeychainBackend::default());
    let mcp_creds = McpCredentialStore::with_backend(
        "local.knowledgeagent.desktop.test.mcp-credentials",
        faulty.clone(),
    );

    let backend_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../backend");
    let python_exe = backend_dir
        .join(".venv/bin/python")
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let stdio_script = backend_dir
        .join("tests/fixtures/mcp/stdio_test_server.py")
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();

    let sidecar = Sidecar::start_with_mcp(
        development_command(),
        data.path().to_owned(),
        keychain.store.clone(),
        mcp_creds.clone(),
        |_, _| async { Ok(None) },
    );
    let host = sidecar.mcp_host_connection().await.unwrap();

    // Create Server A with secret in Keychain
    let secret_a_v1 = "mcp-secret-a-v1-never-leak-111";
    let mut env_a = BTreeMap::new();
    env_a.insert(
        "SERVER_TOKEN".to_owned(),
        ConfigEntryRequest::Secret {
            secret_key: "SERVER_TOKEN".to_owned(),
        },
    );
    let mut secrets_a = BTreeMap::new();
    secrets_a.insert("SERVER_TOKEN".to_owned(), secret_a_v1.to_owned());

    let srv_a = save_server_with_store(
        &host,
        &mcp_creds,
        SaveMcpServerRequest {
            server_id: None,
            name: "ServerAlpha".to_owned(),
            transport: "stdio".to_owned(),
            command: Some(python_exe.clone()),
            args: vec![stdio_script.clone()],
            cwd: None,
            clear_cwd: false,
            url: None,
            allow_private_network: false,
            env_config: env_a.clone(),
            header_config: BTreeMap::new(),
            secret_values: secrets_a,
            remove_secret_keys: Vec::new(),
        },
    )
    .await
    .unwrap();
    let id_a = srv_a["server_id"].as_str().unwrap().to_owned();
    assert_eq!(srv_a["credentials_complete"].as_bool(), Some(true));

    // Create Server B with its own secret
    let secret_b = "mcp-secret-b-v1-never-leak-222";
    let mut secrets_b = BTreeMap::new();
    secrets_b.insert("SERVER_TOKEN".to_owned(), secret_b.to_owned());
    let srv_b = save_server_with_store(
        &host,
        &mcp_creds,
        SaveMcpServerRequest {
            server_id: None,
            name: "ServerBeta".to_owned(),
            transport: "stdio".to_owned(),
            command: Some(python_exe.clone()),
            args: vec![stdio_script.clone()],
            cwd: None,
            clear_cwd: false,
            url: None,
            allow_private_network: false,
            env_config: env_a.clone(),
            header_config: BTreeMap::new(),
            secret_values: secrets_b,
            remove_secret_keys: Vec::new(),
        },
    )
    .await
    .unwrap();
    let id_b = srv_b["server_id"].as_str().unwrap().to_owned();

    // Scenario 1: Keychain write succeeds on update, but DB save fails (name conflict with ServerBeta)
    // -> Keychain MUST roll back to secret_a_v1!
    let mut secrets_a_v2 = BTreeMap::new();
    secrets_a_v2.insert("SERVER_TOKEN".to_owned(), "mcp-secret-a-v2-rolled-back".to_owned());
    let err = save_server_with_store(
        &host,
        &mcp_creds,
        SaveMcpServerRequest {
            server_id: Some(id_a.clone()),
            name: "ServerBeta".to_owned(), // collides with ServerBeta!
            transport: "stdio".to_owned(),
            command: Some(python_exe.clone()),
            args: vec![stdio_script.clone()],
            cwd: None,
            clear_cwd: false,
            url: None,
            allow_private_network: false,
            env_config: env_a.clone(),
            header_config: BTreeMap::new(),
            secret_values: secrets_a_v2,
            remove_secret_keys: Vec::new(),
        },
    )
    .await
    .unwrap_err();
    assert_eq!(err, "MCP_SERVER_NAME_CONFLICT");
    let stored_a = mcp_creds.load(&id_a).unwrap().unwrap();
    assert_eq!(stored_a.get("SERVER_TOKEN").map(|z| z.as_str()), Some(secret_a_v1));

    // Scenario 2: DB create succeeds, but Keychain write fails -> newly created DB row MUST roll back!
    *faulty.fail_save.lock().unwrap() = true;
    let mut secrets_c = BTreeMap::new();
    secrets_c.insert("SERVER_TOKEN".to_owned(), "mcp-secret-c-never-saved".to_owned());
    let create_err = save_server_with_store(
        &host,
        &mcp_creds,
        SaveMcpServerRequest {
            server_id: None,
            name: "ServerGamma".to_owned(),
            transport: "stdio".to_owned(),
            command: Some(python_exe.clone()),
            args: vec![stdio_script.clone()],
            cwd: None,
            clear_cwd: false,
            url: None,
            allow_private_network: false,
            env_config: env_a.clone(),
            header_config: BTreeMap::new(),
            secret_values: secrets_c,
            remove_secret_keys: Vec::new(),
        },
    )
    .await
    .unwrap_err();
    assert_eq!(create_err, KEYCHAIN_ACCESS_FAILED);
    *faulty.fail_save.lock().unwrap() = false;
    // ServerGamma must NOT remain in SQLite (refs only contain ServerAlpha & ServerBeta)
    let refs = host.list_mcp_credential_refs().await.unwrap();
    assert_eq!(refs.len(), 2);

    // Scenario 3: Trust both ServerAlpha and ServerBeta; clear ServerAlpha memory credentials
    // (simulating failed replay) -> connect_server_with_store self-heals by replaying from Keychain!
    host.trust_mcp_server(
        &id_a,
        srv_a["config_revision"].as_u64().unwrap(),
        srv_a["current_fingerprint"].as_str().unwrap(),
        true,
    )
    .await
    .unwrap();
    host.trust_mcp_server(
        &id_b,
        srv_b["config_revision"].as_u64().unwrap(),
        srv_b["current_fingerprint"].as_str().unwrap(),
        true,
    )
    .await
    .unwrap();
    host.delete_mcp_credentials(&id_a).await.unwrap();
    assert_eq!(
        host.get_mcp_server(&id_a).await.unwrap()["credentials_complete"].as_bool(),
        Some(false)
    );
    let healed = connect_server_with_store(&host, &mcp_creds, &id_a)
        .await
        .unwrap();
    assert_eq!(healed["runtime"]["state"].as_str(), Some("connected"));
    assert_eq!(healed["credentials_complete"].as_bool(), Some(true));

    // Scenario 4: Server delete succeeds in SQLite, but Keychain delete fails -> returns MCP_KEYCHAIN_CLEANUP_FAILED
    // while DB row is deleted and logs contain zero plaintext secrets.
    let mut secrets_del = BTreeMap::new();
    secrets_del.insert("SERVER_TOKEN".to_owned(), "mcp-secret-del-never-leak-333".to_owned());
    let srv_del = save_server_with_store(
        &host,
        &mcp_creds,
        SaveMcpServerRequest {
            server_id: None,
            name: "ServerDeleteMe".to_owned(),
            transport: "stdio".to_owned(),
            command: Some(python_exe.clone()),
            args: vec![stdio_script.clone()],
            cwd: None,
            clear_cwd: false,
            url: None,
            allow_private_network: false,
            env_config: env_a.clone(),
            header_config: BTreeMap::new(),
            secret_values: secrets_del,
            remove_secret_keys: Vec::new(),
        },
    )
    .await
    .unwrap();
    let id_del = srv_del["server_id"].as_str().unwrap().to_owned();
    *faulty.fail_delete.lock().unwrap() = true;
    let del_err = delete_server_with_store(&host, &mcp_creds, &id_del)
        .await
        .unwrap_err();
    assert_eq!(del_err, MCP_KEYCHAIN_CLEANUP_FAILED);
    *faulty.fail_delete.lock().unwrap() = false;
    // DB row is gone
    assert!(host.get_mcp_server(&id_del).await.is_err());

    // Scenario 5: Restart Sidecar with partial Keychain credentials missing (delete ServerBeta's Keychain entry).
    // Background restore must recover ServerAlpha -> connected, while ServerBeta stays disconnected with credentials_complete=false!
    mcp_creds.delete(&id_b).unwrap();
    assert!(sidecar.request_shutdown());
    sidecar.wait_stopped().await;

    let sidecar2 = Sidecar::start_with_mcp(
        development_command(),
        data.path().to_owned(),
        keychain.store.clone(),
        mcp_creds.clone(),
        |_, _| async { Ok(None) },
    );
    let host2 = sidecar2.mcp_host_connection().await.unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let a_view = host2.get_mcp_server(&id_a).await.unwrap();
        if a_view["runtime"]["state"].as_str() == Some("connected") {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "ServerAlpha did not auto-recover after restart: {a_view}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let b_view = host2.get_mcp_server(&id_b).await.unwrap();
    assert_eq!(b_view["credentials_complete"].as_bool(), Some(false));
    assert_eq!(b_view["runtime"]["state"].as_str(), Some("disconnected"));

    // Scenario 6: Host IPC revoke-trust validates server_id, disconnects active server,
    // resets trust_state to untrusted, and blocks reconnect until re-trusted.
    let bad_revoke = crate::mcp::revoke_trust_with_host(&host2, "../evil-id")
        .await
        .unwrap_err();
    assert_eq!(bad_revoke, crate::mcp::MCP_INVALID_REQUEST);

    let revoked = crate::mcp::revoke_trust_with_host(&host2, &id_a)
        .await
        .unwrap();
    assert_eq!(revoked["trust_state"].as_str(), Some("untrusted"));
    assert_eq!(revoked["trust_valid"].as_bool(), Some(false));
    assert_eq!(revoked["enabled"].as_bool(), Some(false));
    assert_eq!(revoked["runtime"]["state"].as_str(), Some("disconnected"));

    let reconnect_err = host2.connect_mcp_server(&id_a).await.unwrap_err();
    assert_eq!(reconnect_err, "MCP_SERVER_UNAVAILABLE");
    let reenable_err = host2
        .update_mcp_server(&id_a, &json!({ "enabled": true }))
        .await
        .unwrap_err();
    assert_eq!(reenable_err, "MCP_UNTRUSTED_SERVER");

    assert!(sidecar2.request_shutdown());
    sidecar2.wait_stopped().await;

    // Verify ZERO plaintext secrets leaked into host.jsonl or sidecar.jsonl!
    for log_file in ["logs/host.jsonl", "logs/sidecar.jsonl"] {
        let path = data.path().join(log_file);
        if path.exists() {
            let content = std::fs::read_to_string(path).unwrap();
            for secret in [
                secret_a_v1,
                secret_b,
                "mcp-secret-a-v2-rolled-back",
                "mcp-secret-c-never-saved",
                "mcp-secret-del-never-leak-333",
            ] {
                assert!(!content.contains(secret), "secret leaked into {log_file}");
            }
        }
    }
}
