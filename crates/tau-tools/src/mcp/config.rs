//! Transport-agnostic server specs, decoupled from the host's config
//! format (tau-cli's TOML types convert into these).

use std::collections::BTreeMap;
use std::time::Duration;

/// How much the user trusts a server's self-reported tool annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum McpTrust {
    /// Tools default to `ToolRisk::Elevated` (approval-gated) unless
    /// annotated read-only.
    #[default]
    Untrusted,
    /// All of the server's tools run as `ToolRisk::Local`
    /// (auto-approved by the default policy).
    Trusted,
}

#[derive(Debug, Clone)]
pub enum McpTransportSpec {
    /// Spawn `command args..` with `env` merged over the inherited
    /// environment; JSON-RPC over the child's stdio.
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
    },
    /// Streamable-HTTP endpoint. `auth_header`, when present, is the
    /// full `Authorization` header value.
    Http {
        url: String,
        auth_header: Option<String>,
    },
}

/// One configured MCP server, ready to connect.
#[derive(Debug, Clone)]
pub struct McpServerSpec {
    /// Validated by the host config: `^[a-zA-Z0-9_-]{1,32}$`, so the
    /// `mcp__<name>__` tool prefix always fits provider constraints.
    pub name: String,
    pub transport: McpTransportSpec,
    /// Per tool call (not connect) timeout.
    pub call_timeout: Duration,
    pub trust: McpTrust,
    /// Remote tool names to expose; `None` = all.
    pub include_tools: Option<Vec<String>>,
    /// Remote tool names to hide; applied after `include_tools`.
    pub exclude_tools: Vec<String>,
}

impl McpServerSpec {
    /// Whether a remote tool passes this spec's include/exclude lists.
    pub(crate) fn allows_tool(&self, remote_name: &str) -> bool {
        if let Some(include) = &self.include_tools
            && !include.iter().any(|t| t == remote_name)
        {
            return false;
        }
        !self.exclude_tools.iter().any(|t| t == remote_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(include: Option<Vec<&str>>, exclude: Vec<&str>) -> McpServerSpec {
        McpServerSpec {
            name: "s".into(),
            transport: McpTransportSpec::Http {
                url: "http://x/mcp".into(),
                auth_header: None,
            },
            call_timeout: Duration::from_secs(60),
            trust: McpTrust::Untrusted,
            include_tools: include.map(|v| v.into_iter().map(String::from).collect()),
            exclude_tools: exclude.into_iter().map(String::from).collect(),
        }
    }

    #[test]
    fn include_exclude_filtering() {
        assert!(spec(None, vec![]).allows_tool("a"));
        assert!(spec(Some(vec!["a"]), vec![]).allows_tool("a"));
        assert!(!spec(Some(vec!["a"]), vec![]).allows_tool("b"));
        assert!(!spec(None, vec!["a"]).allows_tool("a"));
        // Exclude wins over include.
        assert!(!spec(Some(vec!["a"]), vec!["a"]).allows_tool("a"));
    }
}
