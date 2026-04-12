//! Shared utilities

/// Truncate a string to `max` characters, appending "..." if truncated.
/// Operates on Unicode char boundaries, not bytes.
pub fn truncate_chars(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let truncated: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

/// Format a token count compactly (e.g. 1234567 → "1.2M", 1234 → "1.2k", 56 → "56")
pub fn format_tokens(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Format a compaction reason for display.
pub fn compaction_reason_str(reason: tau_agent::CompactionReason) -> &'static str {
    match reason {
        tau_agent::CompactionReason::Threshold => "threshold",
        tau_agent::CompactionReason::Overflow => "overflow",
        tau_agent::CompactionReason::Manual => "manual",
    }
}
