//! Anthropic OAuth implementation using PKCE flow

use super::storage::OAuthCredentials;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use sha2::{Digest, Sha256};

// OAuth constants for Anthropic
const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const SCOPES: &str = "org:create_api_key user:profile user:inference";

/// Generate PKCE verifier and challenge
fn generate_pkce() -> (String, String) {
    // Generate 32 random bytes for verifier
    let mut verifier_bytes = [0u8; 32];
    getrandom::fill(&mut verifier_bytes).expect("Failed to generate random bytes");
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    // Generate SHA-256 challenge from verifier
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge_bytes = hasher.finalize();
    let challenge = URL_SAFE_NO_PAD.encode(challenge_bytes);

    (verifier, challenge)
}

/// Login to Anthropic using OAuth
///
/// # Arguments
/// * `on_auth_url` - Called with the authorization URL to open in browser
/// * `on_prompt_code` - Called to get the authorization code from user (format: code#state)
pub async fn login_anthropic<F, G, Fut>(
    on_auth_url: F,
    on_prompt_code: G,
) -> Result<OAuthCredentials, String>
where
    F: FnOnce(String),
    G: FnOnce() -> Fut,
    Fut: std::future::Future<Output = String>,
{
    let (verifier, challenge) = generate_pkce();

    // Build authorization URL
    let auth_params = [
        ("code", "true"),
        ("client_id", CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", REDIRECT_URI),
        ("scope", SCOPES),
        ("code_challenge", &challenge),
        ("code_challenge_method", "S256"),
        ("state", &verifier),
    ];

    let params_str = auth_params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoding::encode(v)))
        .collect::<Vec<_>>()
        .join("&");

    let auth_url = format!("{}?{}", AUTHORIZE_URL, params_str);

    // Notify caller of auth URL
    on_auth_url(auth_url);

    // Wait for authorization code from user
    let auth_code = on_prompt_code().await;
    let auth_code = auth_code.trim();

    // Parse code#state format
    let (code, state) = auth_code
        .split_once('#')
        .ok_or_else(|| "Invalid authorization code format. Expected: code#state".to_string())?;

    // Exchange code for tokens
    let client = reqwest::Client::new();
    let token_request = serde_json::json!({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    });

    let response = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/json")
        .json(&token_request)
        .send()
        .await
        .map_err(|e| format!("Failed to exchange code: {}", e))?;

    if !response.status().is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Token exchange failed: {}", error_text));
    }

    let token_data: TokenResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse token response: {}", e))?;

    Ok(OAuthCredentials::new(
        token_data.refresh_token.unwrap_or_default(),
        token_data.access_token,
        token_data.expires_in,
    ))
}

/// Refresh an Anthropic OAuth token
pub async fn refresh_anthropic_token(refresh_token: &str) -> Result<OAuthCredentials, String> {
    let client = reqwest::Client::new();
    let token_request = serde_json::json!({
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    });

    let response = client
        .post(TOKEN_URL)
        .header("Content-Type", "application/json")
        .json(&token_request)
        .send()
        .await
        .map_err(|e| format!("Failed to refresh token: {}", e))?;

    if !response.status().is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!("Token refresh failed: {}", error_text));
    }

    let token_data: TokenResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse token response: {}", e))?;

    Ok(OAuthCredentials::new(
        token_data.refresh_token.unwrap_or_default(),
        token_data.access_token,
        token_data.expires_in,
    ))
}

#[derive(serde::Deserialize, Debug)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    expires_in: i64,
}
