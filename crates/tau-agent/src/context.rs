//! Hierarchical context file loading
//!
//! Loads context files (AGENTS.md or CLAUDE.md) from:
//! 1. Global: ~/.config/tau/AGENTS.md (or CLAUDE.md)
//! 2. Parent directories: from repo root down to current directory
//! 3. Current directory: highest priority

use std::path::{Path, PathBuf};

/// Names of context files to look for (in order of preference)
const CONTEXT_FILE_NAMES: &[&str] = &["AGENTS.md", "CLAUDE.md"];

/// Load all context files and return their combined content
pub fn load_context() -> Option<String> {
    let mut context_parts = Vec::new();

    if let Some(global) = load_global_context() {
        context_parts.push(global);
    }

    let parent_contexts = load_parent_contexts();
    context_parts.extend(parent_contexts);

    if let Some(current) = load_current_context() {
        context_parts.push(current);
    }

    if context_parts.is_empty() {
        None
    } else {
        Some(context_parts.join("\n\n---\n\n"))
    }
}

/// Load context from global config directory
fn load_global_context() -> Option<String> {
    let config_dir = dirs::config_dir()?.join("tau");
    load_context_from_dir(&config_dir)
}

/// Load context from current directory
fn load_current_context() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    load_context_from_dir(&cwd)
}

/// Load context from all parent directories, from root to current
fn load_parent_contexts() -> Vec<String> {
    let mut contexts = Vec::new();

    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return contexts,
    };

    let repo_root = find_repo_root(&cwd);
    let start_dir = repo_root.unwrap_or_else(|| PathBuf::from("/"));
    let mut dirs_to_check = Vec::new();
    let mut current = cwd.parent();

    while let Some(dir) = current {
        if !dir.starts_with(&start_dir) && dir != start_dir {
            break;
        }
        dirs_to_check.push(dir.to_path_buf());
        current = dir.parent();
    }

    dirs_to_check.reverse();
    for dir in dirs_to_check {
        if let Some(content) = load_context_from_dir(&dir) {
            contexts.push(content);
        }
    }

    contexts
}

/// Find the repository root by looking for .git directory
fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);

    while let Some(dir) = current {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }

    None
}

/// Load context file from a specific directory
fn load_context_from_dir(dir: &Path) -> Option<String> {
    for name in CONTEXT_FILE_NAMES {
        let path = dir.join(name);
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let content = content.trim();
                if !content.is_empty() {
                    return Some(content.to_string());
                }
            }
        }
    }
    None
}

/// Get list of context files that would be loaded (for debugging/info)
#[allow(dead_code)]
pub fn list_context_files() -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Some(config_dir) = dirs::config_dir() {
        let tau_dir = config_dir.join("tau");
        for name in CONTEXT_FILE_NAMES {
            let path = tau_dir.join(name);
            if path.exists() {
                files.push(path);
                break;
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        let repo_root = find_repo_root(&cwd);
        let start_dir = repo_root.unwrap_or_else(|| PathBuf::from("/"));

        let mut dirs_to_check = Vec::new();
        let mut current = cwd.parent();

        while let Some(dir) = current {
            if !dir.starts_with(&start_dir) && dir != start_dir {
                break;
            }
            dirs_to_check.push(dir.to_path_buf());
            current = dir.parent();
        }
        dirs_to_check.reverse();

        for dir in dirs_to_check {
            for name in CONTEXT_FILE_NAMES {
                let path = dir.join(name);
                if path.exists() {
                    files.push(path);
                    break;
                }
            }
        }

        for name in CONTEXT_FILE_NAMES {
            let path = cwd.join(name);
            if path.exists() {
                files.push(path);
                break;
            }
        }
    }

    files
}
