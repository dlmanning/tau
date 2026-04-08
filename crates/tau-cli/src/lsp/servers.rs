//! LSP server discovery — detect installed language servers

/// Configuration for a known language server
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub command: String,
    pub args: Vec<String>,
    /// Maps file extension to LSP language ID
    pub extensions: Vec<(String, String)>, // (extension, language_id)
}

impl ServerConfig {
    /// Get the language ID for a given file extension
    pub fn language_id_for(&self, ext: &str) -> Option<&str> {
        self.extensions
            .iter()
            .find(|(e, _)| e == ext)
            .map(|(_, id)| id.as_str())
    }
}

/// Check if a command is available on the system PATH.
/// Uses `spawn_blocking` to avoid blocking the async runtime.
pub async fn is_available(cmd: &str) -> bool {
    let cmd = cmd.to_string();
    tokio::task::spawn_blocking(move || {
        std::process::Command::new("which")
            .arg(&cmd)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false)
}

/// Discover language servers installed on this system.
pub async fn discover_servers() -> Vec<ServerConfig> {
    let mut servers = vec![];

    if is_available("rust-analyzer").await {
        servers.push(ServerConfig {
            command: "rust-analyzer".into(),
            args: vec![],
            extensions: vec![("rs".into(), "rust".into())],
        });
    }

    if is_available("typescript-language-server").await {
        servers.push(ServerConfig {
            command: "typescript-language-server".into(),
            args: vec!["--stdio".into()],
            extensions: vec![
                ("ts".into(), "typescript".into()),
                ("tsx".into(), "typescriptreact".into()),
                ("js".into(), "javascript".into()),
                ("jsx".into(), "javascriptreact".into()),
            ],
        });
    }

    if is_available("pyright-langserver").await {
        servers.push(ServerConfig {
            command: "pyright-langserver".into(),
            args: vec!["--stdio".into()],
            extensions: vec![("py".into(), "python".into())],
        });
    } else if is_available("pylsp").await {
        servers.push(ServerConfig {
            command: "pylsp".into(),
            args: vec![],
            extensions: vec![("py".into(), "python".into())],
        });
    }

    if is_available("gopls").await {
        servers.push(ServerConfig {
            command: "gopls".into(),
            args: vec!["serve".into()],
            extensions: vec![("go".into(), "go".into())],
        });
    }

    if is_available("clangd").await {
        servers.push(ServerConfig {
            command: "clangd".into(),
            args: vec![],
            extensions: vec![
                ("c".into(), "c".into()),
                ("cpp".into(), "cpp".into()),
                ("cc".into(), "cpp".into()),
                ("h".into(), "c".into()),
                ("hpp".into(), "cpp".into()),
            ],
        });
    }

    servers
}

/// Find the server config for a given file extension.
pub fn server_for_extension<'a>(configs: &'a [ServerConfig], ext: &str) -> Option<&'a ServerConfig> {
    configs
        .iter()
        .find(|c| c.extensions.iter().any(|(e, _)| e == ext))
}
