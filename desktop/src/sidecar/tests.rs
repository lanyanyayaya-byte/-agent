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
