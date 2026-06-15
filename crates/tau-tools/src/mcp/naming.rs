//! Provider-safe tool naming: `mcp__<server>__<tool>`.
//!
//! Anthropic constrains tool names to `^[a-zA-Z0-9_-]{1,64}$`. Server
//! names are already validated to `^[a-zA-Z0-9_-]{1,32}$` by the host
//! config, so the prefix is always valid; the remote tool part is
//! sanitized here.

use std::collections::HashSet;

use sha2::{Digest, Sha256};

const MAX_NAME_LEN: usize = 64;

/// Compute the exposed name for a remote tool, registering it in
/// `taken`. Returns `None` only on a pathological collision that the
/// deterministic hash suffix cannot resolve (caller should skip the
/// tool with a warning).
pub(crate) fn tool_name(
    server: &str,
    remote: &str,
    taken: &mut HashSet<String>,
) -> Option<String> {
    let prefix = format!("mcp__{server}__");
    let sanitized: String = remote
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();

    let plain = fit(&prefix, &sanitized, remote);
    let candidate = if taken.contains(&plain) {
        // Collision (two remote names sanitizing identically): retry
        // with the hash suffix, which depends on the original remote
        // name and is therefore deterministic across runs.
        let hashed = with_hash(&prefix, &sanitized, remote);
        if taken.contains(&hashed) {
            return None;
        }
        hashed
    } else {
        plain
    };
    taken.insert(candidate.clone());
    Some(candidate)
}

/// `prefix + sanitized`, truncated with a hash suffix when over the
/// length limit.
fn fit(prefix: &str, sanitized: &str, remote: &str) -> String {
    if prefix.len() + sanitized.len() <= MAX_NAME_LEN {
        format!("{prefix}{sanitized}")
    } else {
        with_hash(prefix, sanitized, remote)
    }
}

/// Truncate the tool part and append `_` + 6 hex chars of the original
/// remote name's SHA-256, keeping the result within the limit while
/// staying deterministic and collision-resistant.
fn with_hash(prefix: &str, sanitized: &str, remote: &str) -> String {
    let digest = Sha256::digest(remote.as_bytes());
    let suffix = format!("_{:02x}{:02x}{:02x}", digest[0], digest[1], digest[2]);
    let budget = MAX_NAME_LEN - prefix.len() - suffix.len();
    let truncated: String = sanitized.chars().take(budget).collect();
    format!("{prefix}{truncated}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_name_passes_through() {
        let mut taken = HashSet::new();
        assert_eq!(
            tool_name("linear", "create_issue", &mut taken).unwrap(),
            "mcp__linear__create_issue"
        );
    }

    #[test]
    fn invalid_chars_become_underscores() {
        let mut taken = HashSet::new();
        assert_eq!(
            tool_name("s", "files.read/all", &mut taken).unwrap(),
            "mcp__s__files_read_all"
        );
    }

    #[test]
    fn long_names_truncate_deterministically_within_limit() {
        let remote = "a".repeat(100);
        let mut taken = HashSet::new();
        let first = tool_name("server", &remote, &mut taken).unwrap();
        let mut taken2 = HashSet::new();
        let second = tool_name("server", &remote, &mut taken2).unwrap();
        assert_eq!(first, second, "must be stable across runs");
        assert!(first.len() <= MAX_NAME_LEN);
        assert!(first.starts_with("mcp__server__aaaa"));
    }

    #[test]
    fn sanitization_collisions_get_hash_suffixes() {
        let mut taken = HashSet::new();
        let a = tool_name("s", "read.file", &mut taken).unwrap();
        let b = tool_name("s", "read/file", &mut taken).unwrap();
        assert_ne!(a, b, "second tool must be disambiguated");
        assert_eq!(a, "mcp__s__read_file");
        assert!(b.starts_with("mcp__s__read_file_"));
        // An exact duplicate remote name (server bug) resolves once
        // via the hash suffix; a third identical name has nowhere
        // deterministic left to go -> None (caller skips it).
        let mut taken = HashSet::new();
        let first = tool_name("s", "x", &mut taken).unwrap();
        let second = tool_name("s", "x", &mut taken).unwrap();
        assert_ne!(first, second);
        assert!(tool_name("s", "x", &mut taken).is_none());
    }
}
