//! Risk/category classification for third-party MCP tools.
//!
//! Server-asserted [`ToolAnnotations`] are *hints*, not guarantees: a
//! read-only hint may downgrade a tool to `Local` (auto-approved by
//! the default policy) but never to `Safe`, and nothing the server
//! says can bypass gating for an untrusted server's mutating tools.
//! `McpTrust::Trusted` is the user's explicit opt-out.

use rmcp::model::ToolAnnotations;
use tau_agent::{ToolCategory, ToolRisk};

use super::config::McpTrust;

pub(crate) fn classify(
    trust: McpTrust,
    annotations: Option<&ToolAnnotations>,
) -> (ToolRisk, ToolCategory) {
    let read_only = annotations.and_then(|a| a.read_only_hint) == Some(true);
    let category = if read_only {
        ToolCategory::Read
    } else {
        ToolCategory::Other
    };
    let risk = match trust {
        McpTrust::Trusted => ToolRisk::Local,
        McpTrust::Untrusted if read_only => ToolRisk::Local,
        McpTrust::Untrusted => ToolRisk::Elevated,
    };
    (risk, category)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ann(read_only: Option<bool>, destructive: Option<bool>) -> ToolAnnotations {
        let mut a = ToolAnnotations::new();
        a.read_only_hint = read_only;
        a.destructive_hint = destructive;
        a
    }

    #[test]
    fn untrusted_defaults_to_elevated() {
        assert_eq!(
            classify(McpTrust::Untrusted, None),
            (ToolRisk::Elevated, ToolCategory::Other)
        );
        // Destructive or unannotated mutating tools stay gated.
        assert_eq!(
            classify(McpTrust::Untrusted, Some(&ann(Some(false), Some(true)))).0,
            ToolRisk::Elevated
        );
    }

    #[test]
    fn read_only_hint_downgrades_to_local() {
        assert_eq!(
            classify(McpTrust::Untrusted, Some(&ann(Some(true), None))),
            (ToolRisk::Local, ToolCategory::Read)
        );
    }

    #[test]
    fn trusted_server_is_local_regardless() {
        assert_eq!(
            classify(McpTrust::Trusted, Some(&ann(Some(false), Some(true)))),
            (ToolRisk::Local, ToolCategory::Other)
        );
        assert_eq!(
            classify(McpTrust::Trusted, Some(&ann(Some(true), None))),
            (ToolRisk::Local, ToolCategory::Read)
        );
    }
}
