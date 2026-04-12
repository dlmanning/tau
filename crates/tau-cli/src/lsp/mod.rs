//! LSP client infrastructure — manages language server processes
//!
//! ## Known limitations
//!
//! - No `textDocument/didChange` or `textDocument/didSave` notifications are sent.
//!   After the model edits a file, the server's view becomes stale until the file
//!   is re-opened (which happens on the next tool call for that file if the server
//!   has crashed and restarted, but not otherwise). A future improvement would track
//!   file edits and send change notifications.
//!
//! - The request timeout is fixed at 60 seconds. rust-analyzer's initial workspace
//!   indexing can exceed this on very large codebases. The first request after server
//!   start may time out; a retry will typically succeed once indexing completes.
//!
//! - Server stderr is logged at debug level. Set `RUST_LOG=lsp_stderr=debug` to see
//!   language server diagnostic output.

mod client;
mod servers;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lsp_types::*;
use tokio::sync::Mutex;

use client::LspClient;
pub use servers::discover_servers;
use servers::{ServerConfig, server_for_extension};

/// Maximum file size we'll send to an LSP server via didOpen (10 MB).
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// Convert a file path to a file:// URI with proper percent-encoding.
fn file_uri(path: &Path) -> anyhow::Result<Uri> {
    let encoded: String = path
        .components()
        .map(|c| {
            let s = c.as_os_str().to_string_lossy();
            urlencoding::encode(&s).into_owned()
        })
        .collect::<Vec<_>>()
        .join("/");
    // file:// + / + encoded (components don't include leading /)
    let uri_str = format!("file:///{}", encoded.trim_start_matches('/'));
    uri_str
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid URI '{}': {}", uri_str, e))
}

/// Extract a file path from a file:// URI.
fn uri_to_path(uri: &Uri) -> String {
    let s = uri.as_str();
    let path = s.strip_prefix("file://").unwrap_or(s);
    urlencoding::decode(path)
        .map(|s| s.into_owned())
        .unwrap_or_else(|_| path.to_string())
}

/// Manages LSP server lifecycle and routes requests by file type.
pub struct LspManager {
    /// extension → client (may contain dead clients that need restart)
    servers: Mutex<HashMap<String, Arc<LspClient>>>,
    configs: Vec<ServerConfig>,
    workspace_root: PathBuf,
    /// URIs of files sent didOpen, keyed by server command to reset on restart
    opened_files: Mutex<HashMap<String, HashSet<String>>>,
}

impl LspManager {
    pub async fn new(workspace_root: PathBuf) -> Self {
        let configs = discover_servers().await;
        if configs.is_empty() {
            tracing::debug!("No LSP servers found on PATH");
        } else {
            for cfg in &configs {
                tracing::debug!("Found LSP server: {}", cfg.command);
            }
        }
        Self {
            servers: Mutex::new(HashMap::new()),
            configs,
            workspace_root,
            opened_files: Mutex::new(HashMap::new()),
        }
    }

    /// Whether any language servers are available.
    pub fn is_available(&self) -> bool {
        !self.configs.is_empty()
    }

    /// Get or start a server for the given file path.
    /// If the cached server is dead, removes it and starts a fresh one.
    async fn ensure_server(&self, file_path: &Path) -> anyhow::Result<Arc<LspClient>> {
        let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow::anyhow!("File has no extension"))?;

        let config = server_for_extension(&self.configs, ext)
            .ok_or_else(|| anyhow::anyhow!("No LSP server for .{} files", ext))?
            .clone();

        let mut servers = self.servers.lock().await;

        if let Some(client) = servers.get(ext) {
            if client.is_alive().await {
                return Ok(client.clone());
            }
            tracing::warn!("LSP server {} died, restarting", config.command);
            for (e, _) in &config.extensions {
                servers.remove(e);
            }
            self.opened_files.lock().await.remove(&config.command);
        }

        tracing::info!("Starting LSP server: {} for .{}", config.command, ext);
        let client =
            LspClient::spawn(&config.command, &config.args, &self.workspace_root).await?;

        #[allow(deprecated)] // root_uri still needed by many servers
        let init_params = InitializeParams {
            process_id: Some(std::process::id()),
            root_uri: Some(file_uri(&self.workspace_root)?),
            capabilities: ClientCapabilities {
                text_document: Some(TextDocumentClientCapabilities {
                    definition: Some(GotoCapability {
                        dynamic_registration: Some(false),
                        link_support: Some(false),
                    }),
                    references: Some(DynamicRegistrationClientCapabilities {
                        dynamic_registration: Some(false),
                    }),
                    hover: Some(HoverClientCapabilities {
                        dynamic_registration: Some(false),
                        content_format: Some(vec![MarkupKind::Markdown, MarkupKind::PlainText]),
                    }),
                    document_symbol: Some(DocumentSymbolClientCapabilities {
                        dynamic_registration: Some(false),
                        symbol_kind: None,
                        hierarchical_document_symbol_support: Some(true),
                        tag_support: None,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };

        let _init_result: InitializeResult = client.request("initialize", init_params).await?;
        client.notify("initialized", InitializedParams {}).await?;

        let client = Arc::new(client);
        for (e, _) in &config.extensions {
            servers.insert(e.to_string(), client.clone());
        }

        Ok(client)
    }

    /// Ensure a file is open in the server (textDocument/didOpen).
    /// Tracks per-server so a server restart correctly re-opens files.
    async fn ensure_file_open(
        &self,
        client: &LspClient,
        path: &Path,
        server_command: &str,
        language_id: &str,
    ) -> anyhow::Result<()> {
        let uri = file_uri(path)?;
        let uri_str = uri.as_str().to_string();

        let mut opened = self.opened_files.lock().await;
        let server_files = opened.entry(server_command.to_string()).or_default();
        if server_files.contains(&uri_str) {
            return Ok(());
        }

        let metadata = tokio::fs::metadata(path).await?;
        if metadata.len() > MAX_FILE_SIZE {
            return Err(anyhow::anyhow!(
                "File too large for LSP ({:.1} MB, max {:.1} MB)",
                metadata.len() as f64 / 1_048_576.0,
                MAX_FILE_SIZE as f64 / 1_048_576.0,
            ));
        }

        let text = tokio::fs::read_to_string(path).await?;
        client
            .notify(
                "textDocument/didOpen",
                DidOpenTextDocumentParams {
                    text_document: TextDocumentItem {
                        uri,
                        language_id: language_id.to_string(),
                        version: 1,
                        text,
                    },
                },
            )
            .await?;

        server_files.insert(uri_str);
        Ok(())
    }

    fn config_for_path(&self, path: &Path) -> Option<&ServerConfig> {
        let ext = path.extension().and_then(|e| e.to_str())?;
        server_for_extension(&self.configs, ext)
    }

    /// Prepare a client for the given file: ensure the server is running,
    /// the file is opened, and return the client + file URI.
    async fn prepare(&self, path: &Path) -> anyhow::Result<(Arc<LspClient>, Uri)> {
        let config = self.config_for_path(path).cloned();
        let client = self.ensure_server(path).await?;
        let lang_id = config
            .as_ref()
            .and_then(|c| {
                let ext = path.extension()?.to_str()?;
                c.language_id_for(ext).map(|s| s.to_string())
            })
            .unwrap_or_else(|| "plaintext".into());
        self.ensure_file_open(
            &client,
            path,
            config.as_ref().map(|c| c.command.as_str()).unwrap_or(""),
            &lang_id,
        )
        .await?;
        let uri = file_uri(path)?;
        Ok((client, uri))
    }

    /// Go to definition at the given position.
    pub async fn go_to_definition(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> anyhow::Result<String> {
        let (client, uri) = self.prepare(path).await?;
        let result: Option<GotoDefinitionResponse> = client
            .request(
                "textDocument/definition",
                GotoDefinitionParams {
                    text_document_position_params: TextDocumentPositionParams {
                        text_document: TextDocumentIdentifier { uri },
                        position: Position::new(
                            line.saturating_sub(1),
                            character.saturating_sub(1),
                        ),
                    },
                    work_done_progress_params: Default::default(),
                    partial_result_params: Default::default(),
                },
            )
            .await?;

        Ok(format_definition_result(result))
    }

    /// Find all references at the given position.
    pub async fn find_references(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> anyhow::Result<String> {
        let (client, uri) = self.prepare(path).await?;
        let result: Option<Vec<Location>> = client
            .request(
                "textDocument/references",
                ReferenceParams {
                    text_document_position: TextDocumentPositionParams {
                        text_document: TextDocumentIdentifier { uri },
                        position: Position::new(
                            line.saturating_sub(1),
                            character.saturating_sub(1),
                        ),
                    },
                    context: ReferenceContext {
                        include_declaration: true,
                    },
                    work_done_progress_params: Default::default(),
                    partial_result_params: Default::default(),
                },
            )
            .await?;

        Ok(format_references_result(result))
    }

    /// Get hover information at the given position.
    pub async fn hover(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> anyhow::Result<String> {
        let (client, uri) = self.prepare(path).await?;
        let result: Option<Hover> = client
            .request(
                "textDocument/hover",
                HoverParams {
                    text_document_position_params: TextDocumentPositionParams {
                        text_document: TextDocumentIdentifier { uri },
                        position: Position::new(
                            line.saturating_sub(1),
                            character.saturating_sub(1),
                        ),
                    },
                    work_done_progress_params: Default::default(),
                },
            )
            .await?;

        Ok(format_hover_result(result))
    }

    /// Gracefully shut down all running LSP servers.
    pub async fn shutdown_all(&self) {
        let mut unique: Vec<Arc<LspClient>> = Vec::new();
        {
            let servers = self.servers.lock().await;
            let mut seen = HashSet::new();
            for client in servers.values() {
                if seen.insert(Arc::as_ptr(client) as usize) {
                    unique.push(client.clone());
                }
            }
        }
        for client in &unique {
            if client.is_alive().await {
                if let Err(e) = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    client.shutdown(),
                )
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("shutdown timed out")))
                {
                    tracing::debug!("LSP shutdown error: {}", e);
                }
            }
        }
    }

    /// Get document symbols for a file.
    pub async fn document_symbol(&self, path: &Path) -> anyhow::Result<String> {
        let (client, uri) = self.prepare(path).await?;
        let result: Option<DocumentSymbolResponse> = client
            .request(
                "textDocument/documentSymbol",
                DocumentSymbolParams {
                    text_document: TextDocumentIdentifier { uri },
                    work_done_progress_params: Default::default(),
                    partial_result_params: Default::default(),
                },
            )
            .await?;

        Ok(format_document_symbols(result))
    }
}

fn format_definition_result(result: Option<GotoDefinitionResponse>) -> String {
    match result {
        None => "No definition found".into(),
        Some(GotoDefinitionResponse::Scalar(loc)) => {
            format!(
                "{}:{}:{}",
                uri_to_path(&loc.uri),
                loc.range.start.line + 1,
                loc.range.start.character + 1,
            )
        }
        Some(GotoDefinitionResponse::Array(locs)) => {
            if locs.is_empty() {
                return "No definition found".into();
            }
            locs.iter()
                .map(|loc| {
                    format!(
                        "{}:{}:{}",
                        uri_to_path(&loc.uri),
                        loc.range.start.line + 1,
                        loc.range.start.character + 1,
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        Some(GotoDefinitionResponse::Link(links)) => {
            if links.is_empty() {
                return "No definition found".into();
            }
            links
                .iter()
                .map(|link| {
                    format!(
                        "{}:{}:{}",
                        uri_to_path(&link.target_uri),
                        link.target_range.start.line + 1,
                        link.target_range.start.character + 1,
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

fn format_references_result(result: Option<Vec<Location>>) -> String {
    let locs = match result {
        Some(locs) if !locs.is_empty() => locs,
        _ => return "No references found".into(),
    };

    let mut by_file: HashMap<String, Vec<String>> = HashMap::new();
    for loc in &locs {
        let path = uri_to_path(&loc.uri);
        by_file.entry(path).or_default().push(format!(
            "  line {}:{}",
            loc.range.start.line + 1,
            loc.range.start.character + 1
        ));
    }

    let mut output = format!(
        "Found {} references in {} files:\n",
        locs.len(),
        by_file.len()
    );
    for (file, refs) in &by_file {
        output.push_str(&format!("\n{}:\n", file));
        for r in refs {
            output.push_str(r);
            output.push('\n');
        }
    }
    output
}

fn format_hover_result(result: Option<Hover>) -> String {
    match result {
        None => "No hover information".into(),
        Some(hover) => match hover.contents {
            HoverContents::Markup(markup) => markup.value,
            HoverContents::Scalar(MarkedString::String(s)) => s,
            HoverContents::Scalar(MarkedString::LanguageString(ls)) => {
                format!("```{}\n{}\n```", ls.language, ls.value)
            }
            HoverContents::Array(items) => items
                .into_iter()
                .map(|item| match item {
                    MarkedString::String(s) => s,
                    MarkedString::LanguageString(ls) => {
                        format!("```{}\n{}\n```", ls.language, ls.value)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
        },
    }
}

fn format_document_symbols(result: Option<DocumentSymbolResponse>) -> String {
    match result {
        None => "No symbols found".into(),
        Some(DocumentSymbolResponse::Flat(symbols)) => {
            if symbols.is_empty() {
                return "No symbols found".into();
            }
            symbols
                .iter()
                .map(|s| {
                    format!(
                        "{} {} (line {})",
                        symbol_kind_str(s.kind),
                        s.name,
                        s.location.range.start.line + 1
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        Some(DocumentSymbolResponse::Nested(symbols)) => {
            if symbols.is_empty() {
                return "No symbols found".into();
            }
            let mut output = String::new();
            format_nested_symbols(&symbols, 0, &mut output);
            output
        }
    }
}

fn format_nested_symbols(symbols: &[DocumentSymbol], depth: usize, output: &mut String) {
    let indent = "  ".repeat(depth);
    for sym in symbols {
        output.push_str(&format!(
            "{}{} {} (line {})\n",
            indent,
            symbol_kind_str(sym.kind),
            sym.name,
            sym.range.start.line + 1
        ));
        if let Some(ref children) = sym.children {
            format_nested_symbols(children, depth + 1, output);
        }
    }
}

fn symbol_kind_str(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::FILE => "file",
        SymbolKind::MODULE => "mod",
        SymbolKind::NAMESPACE => "namespace",
        SymbolKind::PACKAGE => "package",
        SymbolKind::CLASS => "class",
        SymbolKind::METHOD => "method",
        SymbolKind::PROPERTY => "property",
        SymbolKind::FIELD => "field",
        SymbolKind::CONSTRUCTOR => "constructor",
        SymbolKind::ENUM => "enum",
        SymbolKind::INTERFACE => "interface",
        SymbolKind::FUNCTION => "fn",
        SymbolKind::VARIABLE => "var",
        SymbolKind::CONSTANT => "const",
        SymbolKind::STRING => "string",
        SymbolKind::NUMBER => "number",
        SymbolKind::BOOLEAN => "bool",
        SymbolKind::ARRAY => "array",
        SymbolKind::OBJECT => "object",
        SymbolKind::KEY => "key",
        SymbolKind::NULL => "null",
        SymbolKind::ENUM_MEMBER => "enum_member",
        SymbolKind::STRUCT => "struct",
        SymbolKind::EVENT => "event",
        SymbolKind::OPERATOR => "operator",
        SymbolKind::TYPE_PARAMETER => "type_param",
        _ => "symbol",
    }
}
