//! File-operation extraction for the summary metadata.
//!
//! Scans the to-be-summarized prefix for read/write tool calls so the
//! summarization prompt can list which files were touched.

use tau_ai::{Content, Message};

const READ_TOOLS: &[&str] = &["read", "glob", "grep", "list"];
const WRITE_TOOLS: &[&str] = &["write", "edit"];

pub(super) fn extract_file_operations(messages: &[Message]) -> (Vec<String>, Vec<String>) {
    let mut read_files = Vec::new();
    let mut modified_files = Vec::new();
    for msg in messages {
        let Message::Assistant { content, .. } = msg else {
            continue;
        };
        for c in content {
            let Content::ToolCall {
                name, arguments, ..
            } = c
            else {
                continue;
            };
            let n = name.as_str();
            if READ_TOOLS.contains(&n) {
                if let Some(p) = arguments.get("path").and_then(|v| v.as_str()) {
                    if !read_files.contains(&p.to_string()) {
                        read_files.push(p.into());
                    }
                }
            } else if WRITE_TOOLS.contains(&n) {
                for key in ["path", "file_path"] {
                    if let Some(p) = arguments.get(key).and_then(|v| v.as_str()) {
                        if !modified_files.contains(&p.to_string()) {
                            modified_files.push(p.into());
                        }
                    }
                }
            }
        }
    }
    (read_files, modified_files)
}
