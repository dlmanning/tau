//! OAuth support for LLM providers

mod anthropic;
mod storage;

pub use anthropic::{login_anthropic, refresh_anthropic_token};
pub use storage::{
    OAuthCredentials, load_oauth_credentials, remove_oauth_credentials, save_oauth_credentials,
};

/// Supported OAuth providers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthProvider {
    Anthropic,
}

impl OAuthProvider {
    pub fn id(&self) -> &'static str {
        match self {
            OAuthProvider::Anthropic => "anthropic",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            OAuthProvider::Anthropic => "Anthropic (Claude Pro/Max)",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "anthropic" => Some(OAuthProvider::Anthropic),
            _ => None,
        }
    }

    /// Get all available OAuth providers
    pub fn available() -> Vec<Self> {
        vec![OAuthProvider::Anthropic]
    }
}

/// Get a valid OAuth token for a provider, refreshing if necessary
pub async fn get_oauth_token(provider: OAuthProvider) -> Option<String> {
    let credentials = load_oauth_credentials(provider.id())?;

    // Check if token is expired (buffer already applied when storing)
    if chrono::Utc::now().timestamp_millis() >= credentials.expires {
        // Token expired - try to refresh
        match refresh_token(provider, &credentials.refresh).await {
            Ok(new_creds) => {
                save_oauth_credentials(provider.id(), &new_creds).ok()?;
                Some(new_creds.access)
            }
            Err(e) => {
                tracing::warn!("Failed to refresh OAuth token for {:?}: {}", provider, e);
                // Remove invalid credentials
                let _ = remove_oauth_credentials(provider.id());
                None
            }
        }
    } else {
        Some(credentials.access)
    }
}

async fn refresh_token(
    provider: OAuthProvider,
    refresh_token: &str,
) -> Result<OAuthCredentials, String> {
    match provider {
        OAuthProvider::Anthropic => refresh_anthropic_token(refresh_token).await,
    }
}

/// Login to an OAuth provider
pub async fn login<F, G, Fut>(
    provider: OAuthProvider,
    on_auth_url: F,
    on_prompt_code: G,
) -> Result<(), String>
where
    F: FnOnce(String),
    G: FnOnce() -> Fut,
    Fut: std::future::Future<Output = String>,
{
    match provider {
        OAuthProvider::Anthropic => {
            let credentials = login_anthropic(on_auth_url, on_prompt_code).await?;
            save_oauth_credentials(provider.id(), &credentials)
                .map_err(|e| format!("Failed to save credentials: {}", e))?;
            Ok(())
        }
    }
}

/// Logout from an OAuth provider
pub fn logout(provider: OAuthProvider) -> Result<(), String> {
    remove_oauth_credentials(provider.id())
        .map_err(|e| format!("Failed to remove credentials: {}", e))
}
