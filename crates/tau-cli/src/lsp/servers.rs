//! LSP server discovery — detect installed language servers

use std::process::Command;

/// Configuration for a known language server
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub command: String,
    pub args: Vec<String>,
    pub extensions: Vec<String>,
    pub language_id: String,
}

/// Check if a command is available on the system PATH.
fn is_available(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Discover language servers installed on this system.
pub fn discover_servers() -> Vec<ServerConfig> {
    let mut servers = vec![];

    if is_available("rust-analyzer") {
        servers.push(ServerConfig {
            command: "rust-analyzer".into(),
            args: vec![],
            extensions: vec!["rs".into()],
            language_id: "rust".into(),
        });
    }

    if is_available("typescript-language-server") {
        servers.push(ServerConfig {
            command: "typescript-language-server".into(),
            args: vec!["--stdio".into()],
            extensions: vec!["ts".into(), "tsx".into(), "js".into(), "jsx".into()],
            language_id: "typescript".into(),
        });
    }

    if is_available("pyright-langserver") {
        servers.push(ServerConfig {
            command: "pyright-langserver".into(),
            args: vec!["--stdio".into()],
            extensions: vec!["py".into()],
            language_id: "python".into(),
        });
    } else if is_available("pylsp") {
        servers.push(ServerConfig {
            command: "pylsp".into(),
            args: vec![],
            extensions: vec!["py".into()],
            language_id: "python".into(),
        });
    }

    if is_available("gopls") {
        servers.push(ServerConfig {
            command: "gopls".into(),
            args: vec!["serve".into()],
            extensions: vec!["go".into()],
            language_id: "go".into(),
        });
    }

    if is_available("clangd") {
        servers.push(ServerConfig {
            command: "clangd".into(),
            args: vec![],
            extensions: vec!["c".into(), "cpp".into(), "cc".into(), "h".into(), "hpp".into()],
            language_id: "cpp".into(),
        });
    }

    servers
}

/// Find the server config for a given file extension.
pub fn server_for_extension<'a>(configs: &'a [ServerConfig], ext: &str) -> Option<&'a ServerConfig> {
    configs.iter().find(|c| c.extensions.iter().any(|e| e == ext))
}
