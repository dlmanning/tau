//! Context compaction.
//!
//! When a conversation grows past the model's context window (or hits
//! a `keep_recent_tokens` reserve), the actor pauses to summarize the
//! oldest messages and replaces them with a `<context-summary>`
//! block. Split-turn detection ensures a partially-summarized turn
//! gets its prefix described separately so the kept assistant
//! message reads coherently.
//!
//! Module layout:
//! - [`cut_point`] — pure split-turn / cut-point detection;
//! - [`prompts`] — prompt templates and prompt assembly;
//! - [`file_ops`] — read/write file-operation extraction;
//! - this root — orchestration ([`compact`]), result application,
//!   token estimation, and the [`Summarizer`] seam that decouples the
//!   algorithm from the transport.

mod cut_point;
mod file_ops;
mod prompts;

use std::sync::Arc;

use futures::StreamExt;
use tau_ai::{Content, Message};
use tokio_util::sync::CancellationToken;

use crate::core::config::AgentConfig;
use crate::core::transport::{AgentRunConfig, Transport};
use crate::types::events::AgentEvent;

// `CompactionReason` is the event payload — re-exported for ergonomic
// imports from this module.
pub use crate::types::events::CompactionReason;

/// A token budget that can be expressed as either an absolute count
/// or a fraction of the model's context window.
///
/// **Prefer `Fraction` for defaults that must work across model
/// sizes.** Absolute counts that are sensible on a 200K-context model
/// can swallow half the window on a 32K-context model — a `Tokens(16_384)`
/// reserve is ~8% of Opus's window but ~50% of GPT-3.5's. `Fraction(0.08)`
/// scales correctly and produces the same headroom on Opus.
///
/// `Tokens` is the right choice when you want explicit control that's
/// independent of which model the agent ends up running — e.g.
/// "always keep at least the last 5000 tokens, regardless of window
/// size."
#[derive(Debug, Clone, Copy)]
pub enum CompactionThreshold {
    /// Fraction of `model.context_window`. Clamped to `[0.0, 1.0]`
    /// before use.
    Fraction(f32),
    /// Absolute token count.
    Tokens(u64),
}

impl CompactionThreshold {
    /// Resolve to an absolute token count against a model's
    /// `context_window`. `Fraction` values are clamped to `[0.0, 1.0]`
    /// before scaling, so out-of-range inputs degrade gracefully
    /// (negative → zero, > 1.0 → the full window).
    pub fn resolve(&self, context_window: u64) -> u64 {
        match self {
            Self::Fraction(f) => {
                // NaN propagates through clamp as NaN, so guard it
                // explicitly — degrade to zero rather than producing
                // garbage.
                if f.is_nan() {
                    return 0;
                }
                let clamped = f.clamp(0.0, 1.0) as f64;
                // Round, not truncate: with f32 0.08 the exact product
                // is 15_999.999... on a 200K window, which truncates to
                // 15_999 and would silently disagree with the intuitive
                // "8% of 200K = 16K". Rounding lands on 16_000.
                (context_window as f64 * clamped).round() as u64
            }
            Self::Tokens(n) => *n,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub enabled: bool,
    /// Trigger proactive compaction when `(input + cache_read)` is
    /// within this much of the model's `context_window`. Expressed as
    /// a [`CompactionThreshold`] so the same default scales correctly
    /// across model sizes — see the [`CompactionThreshold`] docs for
    /// why fractions are preferred over absolute counts here.
    pub reserve: CompactionThreshold,
    /// Lower bound on how many tokens of recent messages survive a
    /// compaction pass — the cut-point search walks back until it has
    /// accumulated at least this much, then continues to the next
    /// message boundary.
    pub keep_recent: CompactionThreshold,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // 0.08 of 200K (Opus) = 16_000 tokens — close to the old
            // absolute default of 16_384. On a 32K-context model,
            // 0.08 = 2_560 tokens (vs the old default of 16_384, which
            // would have reserved 50% of the window before any work
            // could start).
            reserve: CompactionThreshold::Fraction(0.08),
            // 0.10 of 200K = 20_000 tokens — matches the old absolute
            // default. Scales sensibly on smaller models.
            keep_recent: CompactionThreshold::Fraction(0.10),
        }
    }
}

pub struct CompactionResult {
    /// Summary text. Wrapped in `<context-summary>` markers and
    /// prepended to the kept messages by [`apply_compaction_result`].
    pub summary: String,
    /// Index of the first message to keep (everything before it gets
    /// summarized).
    pub first_kept_index: usize,
    /// Total estimated tokens before compaction (for the
    /// `CompactionEnd` event).
    pub tokens_before: u64,
}

// ─── Token estimation (char/4 heuristic) ─────────────────────────────

pub fn estimate_tokens(message: &Message) -> u64 {
    let char_count: usize = match message {
        Message::User { content, .. }
        | Message::Assistant { content, .. }
        | Message::ToolResult { content, .. }
        | Message::SystemInjection { content, .. } => content_char_count(content),
    };
    (char_count / 4) as u64
}

pub fn estimate_total_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_tokens).sum()
}

fn content_char_count(content: &[Content]) -> usize {
    content
        .iter()
        .map(|c| match c {
            Content::Text { text } => text.len(),
            Content::Thinking { thinking, .. } => thinking.len(),
            Content::ToolCall {
                name, arguments, ..
            } => name.len() + serde_json::to_string(arguments).unwrap_or_default().len(),
            Content::Image { .. } => 4800,
            Content::RedactedThinking { data } => data.len(),
            Content::ServerToolUse { name, input, .. } => {
                name.len() + serde_json::to_string(input).unwrap_or_default().len()
            }
            Content::ServerToolResult { content, .. } => {
                serde_json::to_string(content).unwrap_or_default().len()
            }
        })
        .sum()
}

// ─── Summarizer seam ─────────────────────────────────────────────────

/// Produces a summary for an assembled summarization prompt.
///
/// This is the seam between the compaction algorithm (cut-point
/// detection, prompt assembly, summary stitching) and the LLM call
/// that actually writes the summary: orchestration depends only on
/// this trait, so it can be tested with a stub — no [`Transport`]
/// required. The production impl is [`TransportSummarizer`].
#[async_trait::async_trait]
trait Summarizer: Send + Sync {
    async fn summarize(&self, prompt: &str, cancel: &CancellationToken)
    -> Result<String, String>;
}

/// Production [`Summarizer`]: a one-shot, tool-less `transport.run()`
/// call against the agent's configured model.
struct TransportSummarizer<'a> {
    agent_config: &'a AgentConfig,
    transport: &'a Arc<dyn Transport>,
}

#[async_trait::async_trait]
impl Summarizer for TransportSummarizer<'_> {
    async fn summarize(
        &self,
        prompt: &str,
        cancel: &CancellationToken,
    ) -> Result<String, String> {
        let run_config = AgentRunConfig {
            system_prompt: Some(prompts::SUMMARIZATION_SYSTEM_PROMPT.into()),
            tools: vec![],
            server_tools: vec![],
            model: self.agent_config.model.clone(),
            reasoning: None,
            thinking_adaptive: false,
            max_tokens: Some(4096),
            temperature: None,
            // Summarization is a one-shot call, not part of any turn loop.
            turn_number: 0,
            cache_scope: None,
            cache_ttl: None,
            system_prompt_boundary: None,
        };

        let user_message = Message::user(prompt);
        let mut stream = self
            .transport
            .run(vec![user_message], &run_config, cancel.clone())
            .await
            .map_err(|e| format!("Compaction LLM call failed: {e}"))?;

        let mut result_text = String::new();
        while let Some(event) = stream.next().await {
            match event {
                AgentEvent::MessageEnd { message } => result_text = message.text(),
                AgentEvent::Error { message } => {
                    return Err(format!("Compaction LLM error: {message}"));
                }
                _ => {}
            }
        }

        if result_text.is_empty() {
            return Err("Compaction LLM returned empty response".into());
        }
        Ok(result_text)
    }
}

// ─── Entry point ─────────────────────────────────────────────────────

/// Run compaction on the given messages.
///
/// `custom_instructions`, when present and non-empty after trimming, is
/// appended as a `## User instructions` section to the main summarization
/// prompt (both the initial and the update variants). The split-turn
/// sub-summary prompt is intentionally left untouched.
///
/// # Failure policy
///
/// This function only *reports* failure (`Err(String)`); what failure
/// *means* is decided at the two actor call sites:
///
/// - **Forced** compaction (overflow / manual, `step_compaction` in the
///   actor) treats an error as fatal: the prompt fails with
///   `Error::Compaction`.
/// - **Proactive** compaction (threshold-based,
///   `run_proactive_compaction` in the actor) is best-effort: the error
///   is logged with `tracing::warn` and the conversation continues
///   uncompacted until the next opportunity.
pub async fn compact(
    messages: &[Message],
    config: &CompactionConfig,
    agent_config: &AgentConfig,
    transport: &Arc<dyn Transport>,
    previous_summary: Option<&str>,
    custom_instructions: Option<&str>,
    cancel: &CancellationToken,
) -> Result<CompactionResult, String> {
    let keep_recent_tokens = config
        .keep_recent
        .resolve(agent_config.model.context_window as u64);
    let summarizer = TransportSummarizer {
        agent_config,
        transport,
    };
    compact_with_summarizer(
        messages,
        keep_recent_tokens,
        previous_summary,
        custom_instructions,
        &summarizer,
        cancel,
    )
    .await
}

/// Transport-free compaction orchestration: find the cut point,
/// assemble the prompt(s), and stitch the summarizer's output into a
/// [`CompactionResult`]. [`compact`] wraps this with the production
/// [`TransportSummarizer`]; tests drive it with a stub.
async fn compact_with_summarizer(
    messages: &[Message],
    keep_recent_tokens: u64,
    previous_summary: Option<&str>,
    custom_instructions: Option<&str>,
    summarizer: &dyn Summarizer,
    cancel: &CancellationToken,
) -> Result<CompactionResult, String> {
    let tokens_before = estimate_total_tokens(messages);

    if cancel.is_cancelled() {
        return Err("Compaction cancelled".into());
    }
    let cut = cut_point::find_cut_point(messages, keep_recent_tokens, cancel).ok_or_else(|| {
        if cancel.is_cancelled() {
            "Compaction cancelled".to_string()
        } else {
            "Not enough messages to compact".to_string()
        }
    })?;

    let messages_to_summarize = &messages[..cut.first_kept_index];
    let (read_files, modified_files) = file_ops::extract_file_operations(messages_to_summarize);
    let conversation_text = prompts::serialize_messages_for_summary(messages_to_summarize);

    let prompt = prompts::build_main_prompt(
        &conversation_text,
        previous_summary,
        &read_files,
        &modified_files,
        custom_instructions,
    );

    let mut full_summary = String::new();

    if cut.is_split_turn {
        if let Some(turn_start) = cut.turn_start_index {
            let turn_prefix = &messages[turn_start..cut.first_kept_index];
            let turn_prefix_text = prompts::serialize_messages_for_summary(turn_prefix);
            let turn_prompt = prompts::build_turn_prefix_prompt(&turn_prefix_text);
            let turn_summary = summarizer.summarize(&turn_prompt, cancel).await?;
            full_summary.push_str("## Split Turn Context\n");
            full_summary.push_str(&turn_summary);
            full_summary.push_str("\n\n");
        }
    }

    if cancel.is_cancelled() {
        return Err("Compaction cancelled".into());
    }
    let main_summary = summarizer.summarize(&prompt, cancel).await?;
    full_summary.push_str(&main_summary);

    Ok(CompactionResult {
        summary: full_summary,
        first_kept_index: cut.first_kept_index,
        tokens_before,
    })
}

/// Apply a successful compaction result to a conversation: splice off
/// the summarized prefix, prepend a `<context-summary>` user message,
/// keep the suffix.
/// The synthetic user message that replaces summarized history.
/// Public so hosts that persist sessions can reconstruct the exact
/// in-memory message on load.
pub fn summary_message(summary: &str) -> Message {
    Message::user(format!(
        "<context-summary>\n{summary}\n</context-summary>\n\nThe conversation was compacted. Continue from where we left off.",
    ))
}

pub fn apply_compaction_result(
    messages: &mut Vec<Message>,
    previous_summary: &mut Option<String>,
    result: CompactionResult,
) {
    *previous_summary = Some(result.summary.clone());
    let kept = messages.split_off(result.first_kept_index);
    *messages = vec![summary_message(&result.summary)];
    messages.extend(kept);
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use tau_ai::{AssistantMetadata, Content, Message};

    fn user(text: &str) -> Message {
        Message::User {
            content: vec![Content::text(text)],
            timestamp: 0,
        }
    }
    fn assistant(text: &str) -> Message {
        Message::Assistant {
            content: vec![Content::text(text)],
            metadata: AssistantMetadata::default(),
        }
    }
    fn assistant_tool(name: &str) -> Message {
        Message::Assistant {
            content: vec![Content::tool_call("id", name, serde_json::json!({}))],
            metadata: AssistantMetadata::default(),
        }
    }
    fn tool_result(tool_call_id: &str) -> Message {
        Message::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "test".into(),
            content: vec![Content::text("result")],
            is_error: false,
            timestamp: 0,
        }
    }

    #[test]
    fn estimate_tokens_char_quarter() {
        // 12 chars / 4 = 3 tokens
        assert_eq!(estimate_tokens(&user("Hello world!")), 3);
    }

    #[test]
    fn threshold_fraction_scales_with_context_window() {
        // Same fraction, different windows → different absolute counts.
        // This is the whole point of using fractions in defaults: the
        // 8% reserve produces 16K on Opus and 2.56K on a 32K model,
        // rather than reserving half a small model's window.
        let t = CompactionThreshold::Fraction(0.08);
        assert_eq!(t.resolve(200_000), 16_000);
        assert_eq!(t.resolve(32_000), 2_560);
        assert_eq!(t.resolve(0), 0);
    }

    #[test]
    fn threshold_tokens_ignores_context_window() {
        // Absolute counts pass through unchanged — for callers that
        // want explicit control regardless of which model runs.
        let t = CompactionThreshold::Tokens(5_000);
        assert_eq!(t.resolve(200_000), 5_000);
        assert_eq!(t.resolve(32_000), 5_000);
    }

    #[test]
    fn threshold_fraction_clamps_out_of_range() {
        // Negative / > 1.0 inputs degrade gracefully rather than
        // producing nonsense token counts.
        assert_eq!(CompactionThreshold::Fraction(-0.5).resolve(100_000), 0);
        assert_eq!(
            CompactionThreshold::Fraction(2.0).resolve(100_000),
            100_000
        );
        assert_eq!(CompactionThreshold::Fraction(f32::NAN).resolve(100_000), 0);
    }

    #[test]
    fn compaction_default_matches_legacy_on_opus_context() {
        // The fraction defaults were chosen to mirror the old absolute
        // defaults on a 200K-context model — within rounding error —
        // so existing Opus deployments don't see a behavior change.
        let cfg = CompactionConfig::default();
        let reserve = cfg.reserve.resolve(200_000);
        let keep = cfg.keep_recent.resolve(200_000);
        // Old absolute defaults were 16_384 and 20_000.
        assert!(
            (15_500..=16_500).contains(&reserve),
            "reserve ≈ old default on Opus: got {reserve}"
        );
        assert_eq!(keep, 20_000);
    }

    #[test]
    fn apply_compaction_replaces_prefix() {
        let mut messages = vec![
            user("old 1"),
            assistant("old 2"),
            user("recent 1"),
            assistant("recent 2"),
        ];
        let mut prev = None;
        apply_compaction_result(
            &mut messages,
            &mut prev,
            CompactionResult {
                summary: "Summary of old conversation".into(),
                first_kept_index: 2,
                tokens_before: 1000,
            },
        );
        // summary + 2 recent
        assert_eq!(messages.len(), 3);
        assert!(messages[0].text().contains("context-summary"));
        assert_eq!(messages[1].text(), "recent 1");
        assert_eq!(messages[2].text(), "recent 2");
        assert_eq!(prev.as_deref(), Some("Summary of old conversation"));
    }

    // ─── Summarizer-seam tests (no Transport) ────────────────────────

    /// Stub that records every prompt it receives and replays canned
    /// replies in order.
    struct StubSummarizer {
        replies: Mutex<Vec<String>>,
        prompts: Mutex<Vec<String>>,
    }

    impl StubSummarizer {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().rev().map(|s| s.to_string()).collect()),
                prompts: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Summarizer for StubSummarizer {
        async fn summarize(
            &self,
            prompt: &str,
            _cancel: &CancellationToken,
        ) -> Result<String, String> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            self.replies
                .lock()
                .unwrap()
                .pop()
                .ok_or_else(|| "stub exhausted".to_string())
        }
    }

    /// The orchestration runs end-to-end against a stub `Summarizer`,
    /// with no `Transport` anywhere: the cut point lands after the old
    /// prefix, the assembled prompt contains only summarized messages
    /// plus the `## User instructions` section, and the stub's reply
    /// comes back as the summary.
    #[tokio::test]
    async fn compact_orchestrates_with_stub_summarizer() {
        let messages = vec![
            user("old question"),
            assistant("old answer"),
            user("recent question"),
            assistant("recent answer"),
        ];
        let stub = StubSummarizer::new(&["STUB SUMMARY"]);
        let cancel = CancellationToken::new();

        // keep_recent = 1 token → the rev-walk overshoots immediately
        // and the fallback keeps the last two messages.
        let result = compact_with_summarizer(
            &messages,
            1,
            None,
            Some("Focus on file paths."),
            &stub,
            &cancel,
        )
        .await
        .expect("compaction succeeds with stub");

        assert_eq!(result.summary, "STUB SUMMARY");
        assert_eq!(result.first_kept_index, 2);
        assert_eq!(result.tokens_before, estimate_total_tokens(&messages));

        let prompts = stub.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 1, "single main summarization call");
        assert!(prompts[0].contains("[User]: old question"));
        assert!(
            !prompts[0].contains("recent question"),
            "kept suffix must not be summarized"
        );
        assert!(prompts[0].contains("## User instructions\n\nFocus on file paths."));
    }

    /// A cut point landing mid-turn produces *two* summarizer calls —
    /// turn-prefix first, then the main prompt — stitched together
    /// under `## Split Turn Context`.
    #[tokio::test]
    async fn compact_split_turn_makes_two_stub_calls() {
        let messages = vec![
            user("first task"),
            assistant("first reply"),
            user("second task"),
            assistant_tool("read"),
            tool_result("id"),
            assistant_tool("write"),
            tool_result("id"),
        ];
        // Choose the threshold so the rev-walk breaks exactly at index 4,
        // putting first_kept on the second tool-calling assistant (5).
        let keep_recent = estimate_tokens(&messages[6])
            + estimate_tokens(&messages[5])
            + estimate_tokens(&messages[4]);
        let stub = StubSummarizer::new(&["TURN PREFIX", "MAIN SUMMARY"]);
        let cancel = CancellationToken::new();

        let result = compact_with_summarizer(&messages, keep_recent, None, None, &stub, &cancel)
            .await
            .expect("split-turn compaction succeeds with stub");

        assert_eq!(result.first_kept_index, 5);
        assert_eq!(
            result.summary,
            "## Split Turn Context\nTURN PREFIX\n\nMAIN SUMMARY"
        );

        let prompts = stub.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 2, "turn-prefix call then main call");
        assert!(prompts[0].contains("<partial-turn>"));
        assert!(prompts[1].contains("[User]: first task"));
    }
}
