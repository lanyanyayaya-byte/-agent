use std::collections::BTreeMap;
use std::sync::Arc;

use tokio::sync::{Mutex, MutexGuard};
use zeroize::Zeroizing;

pub(crate) const DEEPSEEK_PROVIDER: &str = "deepseek";
pub(crate) const SILICONFLOW_PROVIDER: &str = "siliconflow";
pub(crate) const CREDENTIAL_REJECTED: &str = "MODEL_CREDENTIAL_REJECTED";
pub(crate) const MCP_CREDENTIAL_REJECTED: &str = "MCP_CREDENTIAL_REJECTED";
pub(crate) const KEYCHAIN_ACCESS_FAILED: &str = "KEYCHAIN_ACCESS_FAILED";
pub(crate) const KEYCHAIN_CREDENTIAL_CORRUPTED: &str = "KEYCHAIN_CREDENTIAL_CORRUPTED";

pub(crate) const MCP_MAX_SECRET_VALUE_BYTES: usize = 4096;
pub(crate) const MCP_MAX_SECRET_ENTRIES: usize = 64;
pub(crate) const MCP_MAX_BUNDLE_BYTES: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeychainError {
    Rejected,
    AccessDenied(i32),
    Corrupted,
    Unavailable(i32),
}

impl KeychainError {
    pub(crate) fn ipc_code(self) -> &'static str {
        match self {
            Self::Rejected => CREDENTIAL_REJECTED,
            Self::Corrupted => KEYCHAIN_CREDENTIAL_CORRUPTED,
            Self::AccessDenied(_) | Self::Unavailable(_) => KEYCHAIN_ACCESS_FAILED,
        }
    }

    pub(crate) fn diagnostic_reason(self) -> &'static str {
        match self {
            Self::Rejected => "invalid_credential",
            Self::AccessDenied(_) => "access_denied",
            Self::Corrupted => "corrupted",
            Self::Unavailable(_) => "keychain_unavailable",
        }
    }

    pub(crate) fn os_status(self) -> Option<i32> {
        match self {
            Self::AccessDenied(status) | Self::Unavailable(status) => Some(status),
            Self::Rejected | Self::Corrupted => None,
        }
    }
}

pub(crate) fn is_valid_secret(secret: &str) -> bool {
    !secret.is_empty() && secret.len() <= 512 && secret.bytes().all(|byte| byte.is_ascii_graphic())
}

pub(crate) fn validate_credential(
    provider: &str,
    secret: &str,
) -> Result<&'static str, KeychainError> {
    if !is_valid_secret(secret) {
        return Err(KeychainError::Rejected);
    }
    credential_account(provider)
}

pub(crate) fn credential_account(provider: &str) -> Result<&'static str, KeychainError> {
    match provider {
        DEEPSEEK_PROVIDER => Ok(DEEPSEEK_PROVIDER),
        SILICONFLOW_PROVIDER => Ok("siliconflow-embedding"),
        _ => Err(KeychainError::Rejected),
    }
}

pub(crate) trait CredentialBackend: Send + Sync {
    fn load(
        &self,
        service: &str,
        account: &str,
    ) -> Result<Option<Zeroizing<String>>, KeychainError>;
    fn save(&self, service: &str, account: &str, secret: &str) -> Result<(), KeychainError>;
    fn delete(&self, service: &str, account: &str) -> Result<(), KeychainError>;
}

pub(crate) struct CredentialStore {
    service: String,
    backend: Arc<dyn CredentialBackend>,
    lock: Mutex<()>,
}

impl CredentialStore {
    pub(crate) fn for_identifier(identifier: &str) -> Arc<Self> {
        let trimmed = identifier.trim();
        let base = if trimmed.is_empty() {
            "local.knowledgeagent.desktop"
        } else {
            trimmed
        };
        let service = std::env::var("KNOWLEDGEAGENT_KEYCHAIN_SERVICE")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("{base}.model-credentials"));
        Self::with_backend(service, Arc::new(PlatformKeychain))
    }

    pub(crate) fn with_backend(
        service: impl Into<String>,
        backend: Arc<dyn CredentialBackend>,
    ) -> Arc<Self> {
        Arc::new(Self {
            service: service.into(),
            backend,
            lock: Mutex::new(()),
        })
    }

    pub(crate) fn service(&self) -> &str {
        &self.service
    }

    pub(crate) async fn lock(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().await
    }

    pub(crate) fn load(&self, provider: &str) -> Result<Option<Zeroizing<String>>, KeychainError> {
        let account = credential_account(provider)?;
        let Some(secret) = self.backend.load(&self.service, account)? else {
            return Ok(None);
        };
        if !is_valid_secret(&secret) {
            return Err(KeychainError::Corrupted);
        }
        Ok(Some(secret))
    }

    pub(crate) fn save(&self, provider: &str, secret: &str) -> Result<(), KeychainError> {
        let account = validate_credential(provider, secret)?;
        self.backend.save(&self.service, account, secret)?;
        match self.load(provider)? {
            Some(stored) if stored.as_str() == secret => Ok(()),
            _ => Err(KeychainError::Unavailable(-1)),
        }
    }

    pub(crate) fn delete(&self, provider: &str) -> Result<(), KeychainError> {
        self.backend
            .delete(&self.service, credential_account(provider)?)
    }
}

pub(crate) fn is_valid_mcp_server_id(server_id: &str) -> bool {
    let bytes = server_id.as_bytes();
    if bytes.is_empty() || bytes.len() > 128 {
        return false;
    }
    if !bytes[0].is_ascii_alphanumeric() {
        return false;
    }
    bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || matches!(*b, b'.' | b'_' | b'-'))
}

pub(crate) fn mcp_credential_account(server_id: &str) -> Result<String, KeychainError> {
    if !is_valid_mcp_server_id(server_id) {
        return Err(KeychainError::Rejected);
    }
    Ok(format!("mcp-server:{server_id}"))
}

pub(crate) fn is_valid_mcp_secret_key(key: &str) -> bool {
    let bytes = key.as_bytes();
    if bytes.is_empty() || bytes.len() > 128 {
        return false;
    }
    bytes.iter().all(|b| {
        b.is_ascii_alphanumeric()
            || matches!(
                *b,
                b'!' | b'#'
                    | b'$'
                    | b'%'
                    | b'&'
                    | b'\''
                    | b'*'
                    | b'+'
                    | b'-'
                    | b'.'
                    | b'^'
                    | b'_'
                    | b'`'
                    | b'|'
                    | b'~'
            )
    })
}

pub(crate) fn is_valid_mcp_secret_value(value: &str) -> bool {
    if value.is_empty() || value.len() > MCP_MAX_SECRET_VALUE_BYTES {
        return false;
    }
    // Allow printable UTF-8 including spaces (e.g. "Bearer xxx"), reject NUL, CR, LF and ASCII controls.
    !value
        .bytes()
        .any(|b| b == 0 || b == b'\r' || b == b'\n' || b < 0x20 || b == 0x7f)
}

pub(crate) fn validate_mcp_secrets(
    secrets: &BTreeMap<String, Zeroizing<String>>,
) -> Result<(), KeychainError> {
    if secrets.len() > MCP_MAX_SECRET_ENTRIES {
        return Err(KeychainError::Rejected);
    }
    for (key, value) in secrets {
        if !is_valid_mcp_secret_key(key) || !is_valid_mcp_secret_value(value) {
            return Err(KeychainError::Rejected);
        }
    }
    Ok(())
}

/// Physically isolated Keychain store for MCP server credentials under `{identifier}.mcp-credentials`.
/// Never reads, modifies, or shares namespace with `{identifier}.model-credentials`.
pub(crate) struct McpCredentialStore {
    service: String,
    backend: Arc<dyn CredentialBackend>,
    lock: Mutex<()>,
}

impl McpCredentialStore {
    pub(crate) fn for_identifier(identifier: &str) -> Arc<Self> {
        let trimmed = identifier.trim();
        let base = if trimmed.is_empty() {
            "local.knowledgeagent.desktop"
        } else {
            trimmed
        };
        let service = std::env::var("KNOWLEDGEAGENT_MCP_KEYCHAIN_SERVICE")
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("{base}.mcp-credentials"));
        Self::with_backend(service, Arc::new(PlatformKeychain))
    }

    pub(crate) fn with_backend(
        service: impl Into<String>,
        backend: Arc<dyn CredentialBackend>,
    ) -> Arc<Self> {
        Arc::new(Self {
            service: service.into(),
            backend,
            lock: Mutex::new(()),
        })
    }

    pub(crate) fn service(&self) -> &str {
        &self.service
    }

    pub(crate) async fn lock(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().await
    }

    pub(crate) fn load(
        &self,
        server_id: &str,
    ) -> Result<Option<BTreeMap<String, Zeroizing<String>>>, KeychainError> {
        let account = mcp_credential_account(server_id)?;
        let Some(raw) = self.backend.load(&self.service, &account)? else {
            return Ok(None);
        };
        if raw.len() > MCP_MAX_BUNDLE_BYTES {
            return Err(KeychainError::Corrupted);
        }
        let parsed: BTreeMap<String, String> =
            serde_json::from_str(&raw).map_err(|_| KeychainError::Corrupted)?;
        let mut secrets = BTreeMap::new();
        for (key, value) in parsed {
            if !is_valid_mcp_secret_key(&key) || !is_valid_mcp_secret_value(&value) {
                return Err(KeychainError::Corrupted);
            }
            secrets.insert(key, Zeroizing::new(value));
        }
        if secrets.len() > MCP_MAX_SECRET_ENTRIES {
            return Err(KeychainError::Corrupted);
        }
        Ok(Some(secrets))
    }

    pub(crate) fn save(
        &self,
        server_id: &str,
        secrets: &BTreeMap<String, Zeroizing<String>>,
    ) -> Result<(), KeychainError> {
        let account = mcp_credential_account(server_id)?;
        if secrets.is_empty() {
            return self.backend.delete(&self.service, &account);
        }
        validate_mcp_secrets(secrets)?;
        let plain_map: BTreeMap<&str, &str> = secrets
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let encoded = Zeroizing::new(
            serde_json::to_string(&plain_map).map_err(|_| KeychainError::Rejected)?,
        );
        if encoded.len() > MCP_MAX_BUNDLE_BYTES {
            return Err(KeychainError::Rejected);
        }
        self.backend.save(&self.service, &account, &encoded)?;
        match self.load(server_id)? {
            Some(stored) if stored == *secrets => Ok(()),
            _ => Err(KeychainError::Unavailable(-1)),
        }
    }

    pub(crate) fn delete(&self, server_id: &str) -> Result<(), KeychainError> {
        let account = mcp_credential_account(server_id)?;
        self.backend.delete(&self.service, &account)
    }
}

struct PlatformKeychain;

#[cfg(target_os = "macos")]
impl CredentialBackend for PlatformKeychain {
    fn load(
        &self,
        service: &str,
        account: &str,
    ) -> Result<Option<Zeroizing<String>>, KeychainError> {
        macos::load(service, account)
    }

    fn save(&self, service: &str, account: &str, secret: &str) -> Result<(), KeychainError> {
        macos::save(service, account, secret)
    }

    fn delete(&self, service: &str, account: &str) -> Result<(), KeychainError> {
        macos::delete(service, account)
    }
}

#[cfg(not(target_os = "macos"))]
impl CredentialBackend for PlatformKeychain {
    fn load(
        &self,
        _service: &str,
        _account: &str,
    ) -> Result<Option<Zeroizing<String>>, KeychainError> {
        Err(KeychainError::Unavailable(-1))
    }

    fn save(&self, _service: &str, _account: &str, _secret: &str) -> Result<(), KeychainError> {
        Err(KeychainError::Unavailable(-1))
    }

    fn delete(&self, _service: &str, _account: &str) -> Result<(), KeychainError> {
        Err(KeychainError::Unavailable(-1))
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};

    use zeroize::Zeroizing;

    use super::KeychainError;

    type OSStatus = i32;
    type SecKeychainRef = *mut c_void;
    type SecKeychainItemRef = *mut c_void;
    type CFTypeRef = *const c_void;

    const ERR_SEC_SUCCESS: OSStatus = 0;
    const ERR_SEC_USER_CANCELED: OSStatus = -128;
    const ERR_SEC_AUTH_FAILED: OSStatus = -25293;
    const ERR_SEC_DUPLICATE_ITEM: OSStatus = -25299;
    const ERR_SEC_ITEM_NOT_FOUND: OSStatus = -25300;
    const ERR_SEC_INTERACTION_NOT_ALLOWED: OSStatus = -25308;

    #[link(name = "Security", kind = "framework")]
    extern "C" {
        fn SecKeychainFindGenericPassword(
            keychain_or_array: CFTypeRef,
            service_name_length: u32,
            service_name: *const u8,
            account_name_length: u32,
            account_name: *const u8,
            password_length: *mut u32,
            password_data: *mut *mut c_void,
            item_ref: *mut SecKeychainItemRef,
        ) -> OSStatus;

        fn SecKeychainAddGenericPassword(
            keychain: SecKeychainRef,
            service_name_length: u32,
            service_name: *const u8,
            account_name_length: u32,
            account_name: *const u8,
            password_length: u32,
            password_data: *const u8,
            item_ref: *mut SecKeychainItemRef,
        ) -> OSStatus;

        fn SecKeychainItemModifyAttributesAndData(
            item_ref: SecKeychainItemRef,
            attr_list: *const c_void,
            length: u32,
            data: *const u8,
        ) -> OSStatus;

        fn SecKeychainItemDelete(item_ref: SecKeychainItemRef) -> OSStatus;

        fn SecKeychainItemFreeContent(attr_list: *mut c_void, data: *mut c_void) -> OSStatus;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: CFTypeRef);
    }

    struct ItemGuard(SecKeychainItemRef);

    impl Drop for ItemGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0.cast()) };
            }
        }
    }

    fn map_status(status: OSStatus) -> KeychainError {
        match status {
            ERR_SEC_AUTH_FAILED | ERR_SEC_INTERACTION_NOT_ALLOWED | ERR_SEC_USER_CANCELED => {
                KeychainError::AccessDenied(status)
            }
            _ => KeychainError::Unavailable(status),
        }
    }

    fn find_item(service: &str, account: &str) -> Result<Option<ItemGuard>, KeychainError> {
        let mut item: SecKeychainItemRef = null_mut();
        let status = unsafe {
            SecKeychainFindGenericPassword(
                null(),
                service.len() as u32,
                service.as_ptr(),
                account.len() as u32,
                account.as_ptr(),
                null_mut(),
                null_mut(),
                &mut item,
            )
        };
        match status {
            ERR_SEC_SUCCESS => {
                if item.is_null() {
                    Err(KeychainError::Unavailable(-1))
                } else {
                    Ok(Some(ItemGuard(item)))
                }
            }
            ERR_SEC_ITEM_NOT_FOUND => Ok(None),
            other => Err(map_status(other)),
        }
    }

    pub(super) fn load(
        service: &str,
        account: &str,
    ) -> Result<Option<Zeroizing<String>>, KeychainError> {
        let mut length: u32 = 0;
        let mut data: *mut c_void = null_mut();
        let status = unsafe {
            SecKeychainFindGenericPassword(
                null(),
                service.len() as u32,
                service.as_ptr(),
                account.len() as u32,
                account.as_ptr(),
                &mut length,
                &mut data,
                null_mut(),
            )
        };
        match status {
            ERR_SEC_ITEM_NOT_FOUND => Ok(None),
            ERR_SEC_SUCCESS => {
                if data.is_null() {
                    return Err(KeychainError::Corrupted);
                }
                let raw = unsafe {
                    let slice = std::slice::from_raw_parts(data.cast::<u8>(), length as usize);
                    let copied = Zeroizing::new(slice.to_vec());
                    std::ptr::write_bytes(data.cast::<u8>(), 0, length as usize);
                    let _ = SecKeychainItemFreeContent(null_mut(), data);
                    copied
                };
                let text = std::str::from_utf8(&raw).map_err(|_| KeychainError::Corrupted)?;
                Ok(Some(Zeroizing::new(text.to_owned())))
            }
            other => Err(map_status(other)),
        }
    }

    #[cfg(test)]
    pub(super) fn add_raw(service: &str, account: &str, bytes: &[u8]) -> Result<(), KeychainError> {
        let status = unsafe {
            SecKeychainAddGenericPassword(
                null_mut(),
                service.len() as u32,
                service.as_ptr(),
                account.len() as u32,
                account.as_ptr(),
                bytes.len() as u32,
                bytes.as_ptr(),
                null_mut(),
            )
        };
        if status == ERR_SEC_SUCCESS {
            Ok(())
        } else {
            Err(map_status(status))
        }
    }

    pub(super) fn save(service: &str, account: &str, secret: &str) -> Result<(), KeychainError> {
        save_with_modifier(service, account, secret, |item, bytes| unsafe {
            SecKeychainItemModifyAttributesAndData(item, null(), bytes.len() as u32, bytes.as_ptr())
        })
    }

    pub(super) fn save_with_modifier(
        service: &str,
        account: &str,
        secret: &str,
        modify: impl FnOnce(SecKeychainItemRef, &[u8]) -> OSStatus,
    ) -> Result<(), KeychainError> {
        let status = unsafe {
            SecKeychainAddGenericPassword(
                null_mut(),
                service.len() as u32,
                service.as_ptr(),
                account.len() as u32,
                account.as_ptr(),
                secret.len() as u32,
                secret.as_ptr(),
                null_mut(),
            )
        };
        match status {
            ERR_SEC_SUCCESS => Ok(()),
            ERR_SEC_DUPLICATE_ITEM => {
                let Some(item) = find_item(service, account)? else {
                    return Err(KeychainError::Unavailable(ERR_SEC_DUPLICATE_ITEM));
                };
                let update_status = modify(item.0, secret.as_bytes());
                if update_status == ERR_SEC_SUCCESS {
                    Ok(())
                } else {
                    Err(map_status(update_status))
                }
            }
            other => Err(map_status(other)),
        }
    }

    pub(super) fn delete(service: &str, account: &str) -> Result<(), KeychainError> {
        for _ in 0..4 {
            let Some(item) = find_item(service, account)? else {
                return Ok(());
            };
            let status = unsafe { SecKeychainItemDelete(item.0) };
            if status != ERR_SEC_SUCCESS && status != ERR_SEC_ITEM_NOT_FOUND {
                return Err(map_status(status));
            }
        }
        Ok(())
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn random_hex() -> String {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes).unwrap();
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    struct ScopedRealKeychain {
        store: Arc<CredentialStore>,
    }

    impl ScopedRealKeychain {
        fn new() -> Self {
            let service = format!("local.knowledgeagent.desktop.test.{}", random_hex());
            Self {
                store: CredentialStore::with_backend(service, Arc::new(PlatformKeychain)),
            }
        }
    }

    impl Drop for ScopedRealKeychain {
        fn drop(&mut self) {
            let _ = self.store.delete(DEEPSEEK_PROVIDER);
        }
    }

    #[test]
    fn real_macos_keychain_crud_rotation_and_corruption_detection() {
        let scoped = ScopedRealKeychain::new();
        let store = &scoped.store;

        assert_eq!(store.load(DEEPSEEK_PROVIDER).unwrap(), None);
        assert_eq!(store.delete(DEEPSEEK_PROVIDER), Ok(()));
        assert_eq!(store.load("other"), Err(KeychainError::Rejected));
        assert_eq!(
            store.save("other", "valid-token"),
            Err(KeychainError::Rejected)
        );
        assert_eq!(
            store.save(DEEPSEEK_PROVIDER, ""),
            Err(KeychainError::Rejected)
        );
        assert_eq!(
            store.save(DEEPSEEK_PROVIDER, "contains whitespace"),
            Err(KeychainError::Rejected)
        );

        let initial = Zeroizing::new(format!("fixture-key-initial-{}", random_hex()));
        store.save(DEEPSEEK_PROVIDER, &initial).unwrap();
        assert_eq!(store.load(DEEPSEEK_PROVIDER).unwrap(), Some(initial));

        let rotated = Zeroizing::new(format!("fixture-key-rotated-{}", random_hex()));
        store.save(DEEPSEEK_PROVIDER, &rotated).unwrap();
        assert_eq!(store.load(DEEPSEEK_PROVIDER).unwrap(), Some(rotated));

        store.delete(DEEPSEEK_PROVIDER).unwrap();
        assert_eq!(store.load(DEEPSEEK_PROVIDER).unwrap(), None);

        // Write invalid UTF-8 bytes directly to the real macOS Keychain to verify corruption detection.
        macos::add_raw(store.service(), DEEPSEEK_PROVIDER, &[0xff, 0xfe, 0x00]).unwrap();
        assert_eq!(store.load(DEEPSEEK_PROVIDER), Err(KeychainError::Corrupted));
        store.delete(DEEPSEEK_PROVIDER).unwrap();
        assert_eq!(store.load(DEEPSEEK_PROVIDER).unwrap(), None);

        // Verify that when modifying an existing Keychain item is denied by macOS ACL/user cancel,
        // save fails closed and never deletes-and-recreates the existing item to bypass ACLs.
        let protected = Zeroizing::new(format!("fixture-key-protected-{}", random_hex()));
        store.save(DEEPSEEK_PROVIDER, &protected).unwrap();
        let replacement = Zeroizing::new(format!("fixture-key-bypass-{}", random_hex()));
        for denied_status in [-25293, -25308, -128] {
            assert_eq!(
                macos::save_with_modifier(
                    store.service(),
                    DEEPSEEK_PROVIDER,
                    &replacement,
                    |_, _| denied_status
                ),
                Err(KeychainError::AccessDenied(denied_status))
            );
            assert_eq!(
                store.load(DEEPSEEK_PROVIDER).unwrap(),
                Some(protected.clone())
            );
        }
        store.delete(DEEPSEEK_PROVIDER).unwrap();
        assert_eq!(store.load(DEEPSEEK_PROVIDER).unwrap(), None);
    }

    struct ScopedRealMcpKeychain {
        store: Arc<McpCredentialStore>,
        server_ids: Vec<String>,
    }

    impl ScopedRealMcpKeychain {
        fn new(server_ids: &[&str]) -> Self {
            let service = format!("local.knowledgeagent.desktop.test.{}.mcp-credentials", random_hex());
            Self {
                store: McpCredentialStore::with_backend(service, Arc::new(PlatformKeychain)),
                server_ids: server_ids.iter().map(|s| (*s).to_owned()).collect(),
            }
        }
    }

    impl Drop for ScopedRealMcpKeychain {
        fn drop(&mut self) {
            for id in &self.server_ids {
                let _ = self.store.delete(id);
            }
        }
    }

    #[test]
    fn real_macos_mcp_keychain_isolation_spaces_length_rotation_and_cleanup() {
        let model_scoped = ScopedRealKeychain::new();
        let mcp_scoped = ScopedRealMcpKeychain::new(&["srv-alpha", "srv-beta"]);
        let mcp = &mcp_scoped.store;
        assert_ne!(model_scoped.store.service(), mcp.service());

        // Initially absent
        assert_eq!(mcp.load("srv-alpha").unwrap(), None);
        assert_eq!(mcp.delete("srv-alpha"), Ok(()));

        // Validation rules: spaces allowed, NUL/CR/LF rejected, >4096 bytes rejected
        let mut bad_newline = BTreeMap::new();
        bad_newline.insert("Authorization".to_owned(), Zeroizing::new("Bearer abc\r\nX-Injected: 1".to_owned()));
        assert_eq!(mcp.save("srv-alpha", &bad_newline), Err(KeychainError::Rejected));

        let mut bad_oversized = BTreeMap::new();
        bad_oversized.insert("API_KEY".to_owned(), Zeroizing::new("x".repeat(MCP_MAX_SECRET_VALUE_BYTES + 1)));
        assert_eq!(mcp.save("srv-alpha", &bad_oversized), Err(KeychainError::Rejected));

        // Valid secret with spaces (Bearer token) + long JWT-like token up to 4096 bytes
        let long_token = format!("Bearer {}", "a".repeat(2048));
        let mut alpha_secrets = BTreeMap::new();
        alpha_secrets.insert("Authorization".to_owned(), Zeroizing::new(long_token.clone()));
        alpha_secrets.insert("X-Tenant-Key".to_owned(), Zeroizing::new(format!("tenant-{}", random_hex())));
        mcp.save("srv-alpha", &alpha_secrets).unwrap();

        // Separate server has independent secrets; model credential store remains completely untouched
        let mut beta_secrets = BTreeMap::new();
        beta_secrets.insert("API_TOKEN".to_owned(), Zeroizing::new(format!("beta-{}", random_hex())));
        mcp.save("srv-beta", &beta_secrets).unwrap();

        assert_eq!(mcp.load("srv-alpha").unwrap(), Some(alpha_secrets.clone()));
        assert_eq!(mcp.load("srv-beta").unwrap(), Some(beta_secrets.clone()));
        assert_eq!(model_scoped.store.load(DEEPSEEK_PROVIDER).unwrap(), None);

        // Rotate secret
        let mut rotated = alpha_secrets.clone();
        rotated.insert("Authorization".to_owned(), Zeroizing::new(format!("Bearer rotated-{}", random_hex())));
        mcp.save("srv-alpha", &rotated).unwrap();
        assert_eq!(mcp.load("srv-alpha").unwrap(), Some(rotated));

        // Saving empty map deletes Keychain entry
        mcp.save("srv-alpha", &BTreeMap::new()).unwrap();
        assert_eq!(mcp.load("srv-alpha").unwrap(), None);

        // Corruption detection on invalid JSON in real Keychain
        let account = mcp_credential_account("srv-alpha").unwrap();
        macos::add_raw(mcp.service(), &account, b"{not-valid-json").unwrap();
        assert_eq!(mcp.load("srv-alpha"), Err(KeychainError::Corrupted));
        mcp.delete("srv-alpha").unwrap();
        assert_eq!(mcp.load("srv-alpha").unwrap(), None);

        // Clean delete
        mcp.delete("srv-beta").unwrap();
        assert_eq!(mcp.load("srv-beta").unwrap(), None);
    }
}
