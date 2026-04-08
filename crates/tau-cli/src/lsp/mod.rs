//! LSP client infrastructure — manages language server processes

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

/// Convert a file path to a file:// URI string and parse as lsp_types::Uri
fn file_uri(path: &Path) -> anyhow::Result<Uri> {
    let uri_str = format!("file://{}", path.display());
    uri_str.parse().map_err(|e| anyhow::anyhow!("Invalid URI: {}", e))
}

/// Extract a file path from a file:// URI
fn uri_to_path_string(uri: &Uri) -> String {
    let s = uri.as_str();
    s.strip_prefix("file://").unwrap_or(s).to_string()
}

/// Manages LSP server lifecycle and routes requests by file type.
pub struct LspManager {
    servers: Mutex<HashMap<String, Arc<LspClient>>>,
    configs: Vec<ServerConfig>,
    workspace_root: PathBuf,
    opened_files: Mutex<HashSet<String>>,
}

impl LspManager {
    pub fn new(workspace_root: PathBuf) -> Self {
        let configs = discover_servers();
        if configs.is_empty() {
            tracing::debug!("No LSP servers found on PATH");
        } else {
            for cfg in &configs {
                tracing::debug!("Found LSP server: {} ({})", cfg.command, cfg.language_id);
            }
        }
        Self {
            servers: Mutex::new(HashMap::new()),
            configs,
            workspace_root,
            opened_files: Mutex::new(HashSet::new()),
        }
    }

    /// Whether any language servers are available.
    pub fn is_available(&self) -> bool {
        !self.configs.is_empty()
    }

    /// Get or start a server for the given file path.
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
            return Ok(client.clone());
        }

        tracing::info!("Starting LSP server: {} for .{}", config.command, ext);
        let client = LspClient::spawn(&config.command, &config.args, &self.workspace_root).await?;

        // Initialize
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
        client
            .notify("initialized", InitializedParams {})
            .await?;

        let client = Arc::new(client);
        // Cache for all extensions this server handles
        for e in &config.extensions {
            servers.insert(e.clone(), client.clone());
        }

        Ok(client)
    }

    /// Ensure a file is open in the server (textDocument/didOpen).
    async fn ensure_file_open(
        &self,
        client: &LspClient,
        path: &Path,
        language_id: &str,
    ) -> anyhow::Result<()> {
        let uri = file_uri(path)?;
        let uri_str = uri.to_string();

        let mut opened = self.opened_files.lock().await;
        if opened.contains(&uri_str) {
            return Ok(());
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

        opened.insert(uri_str);
        Ok(())
    }

    fn language_id_for_path(&self, path: &Path) -> String {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        server_for_extension(&self.configs, ext)
            .map(|c| c.language_id.clone())
            .unwrap_or_else(|| "plaintext".into())
    }

    /// Go to definition at the given position.
    pub async fn go_to_definition(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> anyhow::Result<String> {
        let client = self.ensure_server(path).await?;
        let lang_id = self.language_id_for_path(path);
        self.ensure_file_open(&client, path, &lang_id).await?;

        let uri = file_uri(path).unwrap();
        let result: Option<GotoDefinitionResponse> = client
            .request(
                "textDocument/definition",
                GotoDefinitionParams {
                    text_document_position_params: TextDocumentPositionParams {
                        text_document: TextDocumentIdentifier { uri },
                        position: Position::new(line.saturating_sub(1), character.saturating_sub(1)),
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
        let client = self.ensure_server(path).await?;
        let lang_id = self.language_id_for_path(path);
        self.ensure_file_open(&client, path, &lang_id).await?;

        let uri = file_uri(path).unwrap();
        let result: Option<Vec<Location>> = client
            .request(
                "textDocument/references",
                ReferenceParams {
                    text_document_position: TextDocumentPositionParams {
                        text_document: TextDocumentIdentifier { uri },
                        position: Position::new(line.saturating_sub(1), character.saturating_sub(1)),
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
        let client = self.ensure_server(path).await?;
        let lang_id = self.language_id_for_path(path);
        self.ensure_file_open(&client, path, &lang_id).await?;

        let uri = file_uri(path).unwrap();
        let result: Option<Hover> = client
            .request(
                "textDocument/hover",
                HoverParams {
                    text_document_position_params: TextDocumentPositionParams {
                        text_document: TextDocumentIdentifier { uri },
                        position: Position::new(line.saturating_sub(1), character.saturating_sub(1)),
                    },
                    work_done_progress_params: Default::default(),
                },
            )
            .await?;

        Ok(format_hover_result(result))
    }

    /// Get document symbols for a file.
    pub async fn document_symbol(&self, path: &Path) -> anyhow::Result<String> {
        let client = self.ensure_server(path).await?;
        let lang_id = self.language_id_for_path(path);
        self.ensure_file_open(&client, path, &lang_id).await?;

        let uri = file_uri(path).unwrap();
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

// ============================================================================
// Result formatters
// ============================================================================

fn uri_to_path(uri: &Uri) -> String {
    uri_to_path_string(uri)
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
    {
            // Group by file
            let mut by_file: HashMap<String, Vec<String>> = HashMap::new();
            for loc in &locs {
                let path = uri_to_path(&loc.uri);
                by_file
                    .entry(path)
                    .or_default()
                    .push(format!(
                        "  line {}:{}",
                        loc.range.start.line + 1,
                        loc.range.start.character + 1
                    ));
            }

            let mut output = format!("Found {} references in {} files:\n", locs.len(), by_file.len());
            for (file, refs) in &by_file {
                output.push_str(&format!("\n{}:\n", file));
                for r in refs {
                    output.push_str(r);
                    output.push('\n');
                }
            }
            output
    }
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
