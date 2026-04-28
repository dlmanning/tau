use async_trait::async_trait;

use crate::Result;

/// Newtype around a secret string. Avoids accidental logging — `Debug`
/// is intentionally not derived; callers must use `expose()` to read.
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretString(***)")
    }
}

/// Pluggable credential storage. Sources don't depend on this — they
/// take `SecretString` at construction. Hosts retrieve the secret from
/// the store and pass it through.
#[async_trait]
pub trait TokenStore: Send + Sync {
    async fn get(&self, key: &str) -> Result<SecretString>;
    async fn put(&self, key: &str, value: SecretString) -> Result<()>;
    async fn list(&self) -> Result<Vec<String>>;
}

/// In-memory store, for tests and ephemeral hosts.
pub struct MemTokenStore {
    // Implementation deferred.
}

#[async_trait]
impl TokenStore for MemTokenStore {
    async fn get(&self, _key: &str) -> Result<SecretString> {
        todo!()
    }
    async fn put(&self, _key: &str, _value: SecretString) -> Result<()> {
        todo!()
    }
    async fn list(&self) -> Result<Vec<String>> {
        todo!()
    }
}

/// macOS Keychain / Linux secret-service backed store.
pub struct KeychainTokenStore {
    // Implementation deferred.
}

#[async_trait]
impl TokenStore for KeychainTokenStore {
    async fn get(&self, _key: &str) -> Result<SecretString> {
        todo!()
    }
    async fn put(&self, _key: &str, _value: SecretString) -> Result<()> {
        todo!()
    }
    async fn list(&self) -> Result<Vec<String>> {
        todo!()
    }
}
