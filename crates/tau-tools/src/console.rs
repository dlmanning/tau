//! Console-line classifier — best-effort heuristics that pick a
//! [`ConsoleLevel`] for a single output line.
//!
//! Lives next to `bash.rs` because bash is the primary producer of
//! ambiguous streamed output. Other tools that stream text either know the
//! level themselves (use [`ProgressSender::send_at`]) or stay at
//! `ConsoleLevel::Normal` (use [`ProgressSender::send`]).
//!
//! ANSI escape codes in the input are inspected for color hints but kept in
//! the line content so terminal hosts pass them through.

use tau_agent::events::{ConsoleLevel, ConsoleLine};
use tau_agent::tool::ProgressSender;

/// Classify a single line. Examines ANSI color codes first (they're a
/// strong producer signal), then prefix patterns common to cargo / sh / Rust
/// panic output.
///
/// Returns `Muted` for blank lines so the host can fade them visually.
pub fn classify_line(content: &str) -> ConsoleLevel {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return ConsoleLevel::Muted;
    }

    if let Some(level) = classify_by_ansi(content) {
        return level;
    }

    classify_by_prefix(trimmed)
}

/// Look for SGR color codes anywhere in the line. First match wins;
/// hosts that want richer styling can re-classify themselves.
fn classify_by_ansi(content: &str) -> Option<ConsoleLevel> {
    // ESC[<params>m — we just look for the code+`m` substring.
    // Order matters: check more specific (red/green) before generic (dim).
    if contains_sgr(content, &["31", "91", "1;31", "31;1"]) {
        return Some(ConsoleLevel::Danger);
    }
    if contains_sgr(content, &["33", "93", "1;33", "33;1"]) {
        return Some(ConsoleLevel::Warning);
    }
    if contains_sgr(content, &["32", "92", "1;32", "32;1"]) {
        return Some(ConsoleLevel::Success);
    }
    if contains_sgr(content, &["2", "90"]) {
        return Some(ConsoleLevel::Muted);
    }
    None
}

fn contains_sgr(content: &str, codes: &[&str]) -> bool {
    codes
        .iter()
        .any(|c| content.contains(&format!("\x1b[{c}m")))
}

fn classify_by_prefix(line: &str) -> ConsoleLevel {
    // Strip a leading ANSI sequence so prefix checks match colored output.
    let bare = strip_leading_ansi(line).trim_start();

    // Order: Danger first (most specific failure markers), then
    // Success, Warning, Muted, fall through to Normal.
    if starts_with_ci(bare, "error:")
        || starts_with_ci(bare, "error[")
        || bare.starts_with("panicked at")
        || bare.starts_with("thread '")
        || bare.starts_with("test result: FAILED")
        || bare.starts_with("failures:")
    {
        return ConsoleLevel::Danger;
    }

    if bare.starts_with("test result: ok")
        || bare.ends_with("... ok")
        || bare.starts_with('✓')
        || bare.starts_with("ok ")
        || bare.starts_with("PASS ")
        || bare.starts_with("PASSED")
    {
        return ConsoleLevel::Success;
    }

    if starts_with_ci(bare, "warning:")
        || bare.starts_with("Running ")
        || bare.starts_with("WARN ")
    {
        return ConsoleLevel::Warning;
    }

    if bare.starts_with("$ ")
        || bare.starts_with("# ")
        || bare.starts_with("Compiling ")
        || bare.starts_with("Finished ")
        || bare.starts_with("Checking ")
    {
        return ConsoleLevel::Muted;
    }

    ConsoleLevel::Normal
}

fn starts_with_ci(haystack: &str, needle: &str) -> bool {
    if haystack.len() < needle.len() {
        return false;
    }
    haystack[..needle.len()].eq_ignore_ascii_case(needle)
}

/// Strip a single leading `ESC[ … m` sequence if present.
fn strip_leading_ansi(s: &str) -> &str {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&0x1b) || bytes.get(1) != Some(&b'[') {
        return s;
    }
    if let Some(end) = bytes.iter().position(|b| *b == b'm') {
        // Safe: ANSI prefix is ASCII.
        return &s[end + 1..];
    }
    s
}

/// Classify `content` with [`classify_line`] and emit it as a single line on
/// the progress channel. Sugar for the common bash-style "I'm streaming
/// raw stdout/stderr" case.
pub fn send_classified(progress: &ProgressSender, content: impl Into<String>) {
    let content = content.into();
    let level = classify_line(&content);
    progress.send_lines(vec![ConsoleLine::new(content, level)]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_line_is_muted() {
        assert_eq!(classify_line(""), ConsoleLevel::Muted);
        assert_eq!(classify_line("   "), ConsoleLevel::Muted);
    }

    #[test]
    fn cargo_compile_chain_classifies_correctly() {
        assert_eq!(classify_line("Compiling foo v0.1.0"), ConsoleLevel::Muted);
        assert_eq!(
            classify_line("    Finished `dev` profile in 1.23s"),
            ConsoleLevel::Muted
        );
        assert_eq!(
            classify_line("    Checking tau-agent v0.1.0"),
            ConsoleLevel::Muted
        );
    }

    #[test]
    fn cargo_test_lines_classify() {
        assert_eq!(classify_line("Running 3 tests"), ConsoleLevel::Warning);
        assert_eq!(
            classify_line("test foo::bar ... ok"),
            ConsoleLevel::Success
        );
        assert_eq!(
            classify_line("test result: ok. 3 passed; 0 failed"),
            ConsoleLevel::Success
        );
        assert_eq!(
            classify_line("test result: FAILED. 1 passed; 2 failed"),
            ConsoleLevel::Danger
        );
    }

    #[test]
    fn rust_compiler_diagnostics_classify() {
        assert_eq!(
            classify_line("error: unresolved import"),
            ConsoleLevel::Danger
        );
        assert_eq!(
            classify_line("error[E0061]: this function takes 1 arg"),
            ConsoleLevel::Danger
        );
        assert_eq!(
            classify_line("warning: unused variable: `x`"),
            ConsoleLevel::Warning
        );
    }

    #[test]
    fn panic_lines_classify() {
        assert_eq!(
            classify_line("panicked at 'assertion failed', src/lib.rs:42"),
            ConsoleLevel::Danger
        );
        assert_eq!(
            classify_line("thread 'main' panicked at ..."),
            ConsoleLevel::Danger
        );
    }

    #[test]
    fn shell_prompt_is_muted() {
        assert_eq!(classify_line("$ ls -la"), ConsoleLevel::Muted);
        assert_eq!(classify_line("# pwd"), ConsoleLevel::Muted);
    }

    #[test]
    fn ansi_red_overrides_unknown_text() {
        let line = "\x1b[31msomething bad\x1b[0m";
        assert_eq!(classify_line(line), ConsoleLevel::Danger);
    }

    #[test]
    fn ansi_green_classifies_success() {
        let line = "\x1b[32mPASS\x1b[0m";
        assert_eq!(classify_line(line), ConsoleLevel::Success);
    }

    #[test]
    fn ansi_dim_classifies_muted() {
        let line = "\x1b[2mFinished\x1b[0m";
        assert_eq!(classify_line(line), ConsoleLevel::Muted);
    }

    #[test]
    fn check_mark_is_success() {
        assert_eq!(classify_line("✓ all good"), ConsoleLevel::Success);
    }

    #[test]
    fn benign_failed_substring_is_not_danger() {
        // The literal "FAILED" appearing inside a passing test name must not
        // trigger Danger — the line ends with "... ok" so it's Success.
        assert_eq!(
            classify_line("test handles_FAILED_input ... ok"),
            ConsoleLevel::Success
        );
    }

    #[test]
    fn benign_failures_substring_is_not_danger() {
        // "no failures: 0" mentions the substring but isn't a failure block.
        assert_eq!(
            classify_line("[INFO] no failures: 0 reported"),
            ConsoleLevel::Normal
        );
    }

    #[test]
    fn ok_must_be_at_end_of_line() {
        // "... ok" appearing mid-sentence shouldn't classify Success.
        assert_eq!(
            classify_line("I asked for ... ok response and got it"),
            ConsoleLevel::Normal
        );
    }

    #[test]
    fn unknown_is_normal() {
        assert_eq!(
            classify_line("just some normal output"),
            ConsoleLevel::Normal
        );
    }

    #[test]
    fn strip_leading_ansi_drops_only_first_seq() {
        let s = "\x1b[31mHello\x1b[0m world";
        assert_eq!(strip_leading_ansi(s), "Hello\x1b[0m world");
    }

    #[test]
    fn strip_leading_ansi_passes_through_plain() {
        assert_eq!(strip_leading_ansi("plain"), "plain");
    }
}
