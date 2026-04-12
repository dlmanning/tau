//! LSP code intelligence tool

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

use crate::lsp::LspManager;

#[derive(Deserialize, JsonSchema)]
struct LspArgs {
    /// The LSP operation to perform
    #[schemars(extend("enum" = ["goToDefinition", "findReferences", "hover", "documentSymbol"]))]
    operation: String,
    /// Absolute path to the file
    #[serde(rename = "filePath")]
    file_path: String,
    /// Line number (1-based). Required for all operations except documentSymbol.
    line: Option<u32>,
    /// Character offset on the line (1-based). Required for all operations except documentSymbol.
    character: Option<u32>,
}

/// Tool for querying language servers (go-to-definition, find-references, hover, etc.)
pub struct LspTool {
    manager: Arc<LspManager>,
}

impl LspTool {
    pub fn new(manager: Arc<LspManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for LspTool {
    fn name(&self) -> &str {
        "lsp"
    }

    fn activity_description(&self, _arguments: &serde_json::Value) -> String {
        "Querying language server".to_string()
    }

    fn description(&self) -> &str {
        "Query language servers for code intelligence. Supports go-to-definition, \
         find-references, hover (type info), and document symbols. Requires a language \
         server to be installed (e.g. rust-analyzer, typescript-language-server, pyright, gopls)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(LspArgs)
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _ctx: ExecutionContext,
    ) -> ToolResult {
        let args: LspArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let path = PathBuf::from(&args.file_path);
        if !path.is_absolute() {
            return ToolResult::error("filePath must be an absolute path");
        }
        if !path.exists() {
            return ToolResult::error(format!("File not found: {}", args.file_path));
        }

        let result = match args.operation.as_str() {
            "goToDefinition" => {
                let (line, character) = match (args.line, args.character) {
                    (Some(l), Some(c)) => (l, c),
                    _ => return ToolResult::error("goToDefinition requires 'line' and 'character'"),
                };
                self.manager.go_to_definition(&path, line, character).await
            }
            "findReferences" => {
                let (line, character) = match (args.line, args.character) {
                    (Some(l), Some(c)) => (l, c),
                    _ => return ToolResult::error("findReferences requires 'line' and 'character'"),
                };
                self.manager.find_references(&path, line, character).await
            }
            "hover" => {
                let (line, character) = match (args.line, args.character) {
                    (Some(l), Some(c)) => (l, c),
                    _ => return ToolResult::error("hover requires 'line' and 'character'"),
                };
                self.manager.hover(&path, line, character).await
            }
            "documentSymbol" => self.manager.document_symbol(&path).await,
            _ => {
                return ToolResult::error(format!(
                    "Unknown operation '{}'. Valid: goToDefinition, findReferences, hover, documentSymbol",
                    args.operation
                ))
            }
        };

        match result {
            Ok(text) => ToolResult::text(text),
            Err(e) => ToolResult::error(format!("LSP error: {}", e)),
        }
    }
}
