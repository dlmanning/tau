//! OAuth CLI handlers

use crate::oauth;

pub(crate) async fn handle_oauth_login(provider_id: &str) -> anyhow::Result<()> {
    let provider = match oauth::OAuthProvider::from_id(provider_id) {
        Some(p) => p,
        None => {
            eprintln!("Unknown OAuth provider: {}", provider_id);
            eprintln!("Available providers: anthropic");
            std::process::exit(1);
        }
    };

    println!("Logging in to {}...", provider.name());
    println!();

    match oauth::login(
        provider,
        |url| {
            println!("Opening browser to authorize...");
            println!();
            println!("If the browser doesn't open, visit this URL:");
            println!("  {}", url);
            println!();

            #[cfg(target_os = "macos")]
            let _ = std::process::Command::new("open").arg(&url).spawn();
            #[cfg(target_os = "linux")]
            let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("cmd")
                .args(["/C", "start", &url])
                .spawn();
        },
        || async {
            println!("After authorizing, paste the code below (format: code#state):");
            print!("> ");
            use std::io::Write;
            std::io::stdout().flush().ok();

            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();
            input.trim().to_string()
        },
    )
    .await
    {
        Ok(()) => {
            println!();
            println!("Successfully logged in to {}!", provider.name());
            println!("Credentials saved to ~/.config/tau/oauth.json");
        }
        Err(e) => {
            eprintln!();
            eprintln!("Login failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

pub(crate) fn handle_oauth_logout(provider_id: &str) -> anyhow::Result<()> {
    let provider = match oauth::OAuthProvider::from_id(provider_id) {
        Some(p) => p,
        None => {
            eprintln!("Unknown OAuth provider: {}", provider_id);
            eprintln!("Available providers: anthropic");
            std::process::exit(1);
        }
    };

    match oauth::logout(provider) {
        Ok(()) => {
            println!("Successfully logged out of {}", provider.name());
        }
        Err(e) => {
            eprintln!("Logout failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

pub(crate) fn show_auth_status() -> anyhow::Result<()> {
    println!("OAuth Authentication Status");
    println!("{}", "-".repeat(40));

    for provider in oauth::OAuthProvider::available() {
        let status = if let Some(creds) = oauth::load_oauth_credentials(provider.id()) {
            let expires = chrono::DateTime::from_timestamp_millis(creds.expires)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                .unwrap_or_else(|| "unknown".to_string());

            if chrono::Utc::now().timestamp_millis() >= creds.expires {
                "Logged in (token expired, will refresh on next use)".to_string()
            } else {
                format!("Logged in (expires: {})", expires)
            }
        } else {
            "Not logged in".to_string()
        };

        println!("{:<25} {}", provider.name(), status);
    }

    println!();
    println!("Login with: tau auth login <provider>");
    println!("Logout with: tau auth logout <provider>");

    Ok(())
}
