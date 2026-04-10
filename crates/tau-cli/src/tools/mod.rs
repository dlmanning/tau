//! Built-in tools for the coding agent

pub mod agent;
mod ask;
mod bash;
mod edit;
mod glob;
mod grep;
mod list;
pub mod lsp;
mod read;
mod write;

use std::path::{Path, PathBuf};

/// Resolve a path against a CWD.
/// Absolute paths are returned as-is. Relative paths are joined with `cwd`.
pub(crate) fn resolve_path(path_str: &str, cwd: &Path) -> PathBuf {
    let path = PathBuf::from(path_str);
    if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    }
}

pub use agent::AgentTool;
pub use ask::AskTool;
pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use list::ListTool;
pub use lsp::LspTool;
pub use read::ReadTool;
pub use write::WriteTool;
