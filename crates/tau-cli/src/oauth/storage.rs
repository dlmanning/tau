//! OAuth credentials storage
//!
//! Stores OAuth tokens in ~/.config/tau/oauth.json with restricted permissions (0o600)

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{collections::HashMap, fs, io, path::PathBuf};

use serde::{Deserialize, Serialize};

/// OAuth credentials for a provider
#[derive(Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    /// Credential type (always "oauth")
    #[serde(rename = "type")]
    pub cred_type: String,
    /// Refresh token
    pub refresh: String,
    /// Access token
    pub access: String,
    /// Expiry timestamp in milliseconds
    pub expires: i64,
}

/// Manual impl so tokens can't leak through `{:?}` in logs, error
/// messages, or panic payloads.
impl std::fmt::Debug for OAuthCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthCredentials")
            .field("cred_type", &self.cred_type)
            .field("refresh", &"<redacted>")
            .field("access", &"<redacted>")
            .field("expires", &self.expires)
            .finish()
    }
}

impl OAuthCredentials {
    pub fn new(refresh: String, access: String, expires_in_secs: i64) -> Self {
        let expires =
            chrono::Utc::now().timestamp_millis() + (expires_in_secs * 1000) - (5 * 60 * 1000);
        Self {
            cred_type: "oauth".to_string(),
            refresh,
            access,
            expires,
        }
    }
}

/// Get the OAuth storage directory
fn oauth_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("tau")
}

/// Get the OAuth storage file path
fn oauth_file() -> PathBuf {
    oauth_dir().join("oauth.json")
}

/// Load all OAuth credentials from storage
fn load_storage() -> HashMap<String, OAuthCredentials> {
    let path = oauth_file();
    if !path.exists() {
        return HashMap::new();
    }

    match fs::read_to_string(&path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => HashMap::new(),
    }
}

/// Save all OAuth credentials to storage
fn save_storage(storage: &HashMap<String, OAuthCredentials>) -> io::Result<()> {
    let dir = oauth_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
        #[cfg(unix)]
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    }

    let path = oauth_file();
    let content = serde_json::to_string_pretty(storage)?;
    fs::write(&path, content)?;

    #[cfg(unix)]
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;

    Ok(())
}

/// Load OAuth credentials for a specific provider
pub fn load_oauth_credentials(provider: &str) -> Option<OAuthCredentials> {
    let storage = load_storage();
    storage.get(provider).cloned()
}

/// Save OAuth credentials for a specific provider
pub fn save_oauth_credentials(provider: &str, credentials: &OAuthCredentials) -> io::Result<()> {
    let mut storage = load_storage();
    storage.insert(provider.to_string(), credentials.clone());
    save_storage(&storage)
}

/// Remove OAuth credentials for a specific provider
pub fn remove_oauth_credentials(provider: &str) -> io::Result<()> {
    let mut storage = load_storage();
    storage.remove(provider);
    save_storage(&storage)
}

/// List all providers with saved OAuth credentials
#[allow(dead_code)]
pub fn list_oauth_providers() -> Vec<String> {
    let storage = load_storage();
    storage.keys().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_expiry_buffer() {
        let creds = OAuthCredentials::new(
            "refresh".to_string(),
            "access".to_string(),
            3600, // 1 hour
        );

        let now = chrono::Utc::now().timestamp_millis();
        // Should expire ~55 minutes from now (1 hour minus 5 min buffer)
        let expected_min = now + (55 * 60 * 1000) - 1000; // Allow 1 sec tolerance
        let expected_max = now + (55 * 60 * 1000) + 1000;

        assert!(creds.expires >= expected_min && creds.expires <= expected_max);
    }

    /// `{:?}` must never expose token material.
    #[test]
    fn debug_redacts_tokens() {
        let creds =
            OAuthCredentials::new("sekrit-refresh".to_string(), "sekrit-access".to_string(), 3600);
        let dbg = format!("{creds:?}");
        assert!(!dbg.contains("sekrit"), "tokens leaked: {dbg}");
        assert!(dbg.contains("<redacted>"));
    }
}
