//! Sync transitions on agent state.
//!
//! This module replaces the original `tau-agent`'s `logic.rs`. Three
//! function families, each named for the role it plays:
//!
//! - **`decide_*`** — pure decisions. Take `&Frame` and `&Conv` (or
//!   just an outcome), return a typed action enum. Never mutate state.
//! - **`apply_*`** — state transitions. Take `&Frame` and `&mut Conv`
//!   (or just `&mut Conv`), perform the mutation. Never make a new
//!   decision.
//! - **`build_*` and other plain helpers** — pure builders that
//!   neither read mutable state nor mutate it. `build_context`,
//!   `build_run_config`, `build_tool_groups`, `extract_tool_calls`,
//!   `collect_ordered_results`, `drain_queue`. Standalone utilities
//!   the actor calls between `decide_*` and `apply_*`.
//!
//! The actor reads a `decide_*` result, performs any I/O the result
//! demands, then calls one or more `apply_*` to commit the transition.
//! No I/O happens here — the whole module is testable without a tokio
//! runtime.

use std::collections::HashMap;

use tau_ai::{Content, Message, Usage};

use crate::core::config::DequeueMode;
use crate::core::overflow::is_context_overflow;
use crate::core::state::{Conv, Frame, ToolCall};
use crate::core::stream::StreamOutcome;
use crate::core::tool::{BoxedTool, Concurrency, ToolResult, to_api_tool};
use crate::core::transport::AgentRunConfig;
use crate::types::events::CompactionReason;

// ─── Action enums (the language `decide_*` speaks) ───────────────────

/// What to do after an LLM response.
#[derive(Debug)]
pub enum ResponseAction {
    /// Run these tool calls grouped by concurrency.
    RunTools {
        tool_calls: Vec<ToolCall>,
        groups: Vec<Vec<usize>>,
        first_user_message: Option<Message>,
    },
    /// Compact (overflow); optionally retry the in-flight pending
    /// messages after compaction completes.
    Compact {
        reason: CompactionReason,
        resume_pending: Option<(Vec<Message>, Option<Message>)>,
    },
    /// Turn complete — go to the queue-drain phase.
    Done,
    /// Fatal error.
    Error(crate::types::error::Error),
}

#[derive(Debug)]
pub struct ResponseDecision {
    pub action: ResponseAction,
    /// Run proactive (threshold-based) compaction before executing the
    /// action. `false` doesn't preclude the `Compact` action — that's
    /// the overflow path; this flag is the *threshold* path.
    pub needs_proactive_compaction: bool,
}

#[derive(Debug)]
pub enum BatchCompleteAction {
    /// Steering arrived during this batch. The actor should commit
    /// results (treating remaining tools as skipped), then drain the
    /// steering queue and start a new turn. The drain is *not* done
    /// here because it mutates and would force this enum to carry the
    /// drained messages — the actor calls `apply_drain_queues` after
    /// seeing this action.
    Redirect {
        skipped_indices: Vec<(usize, String, String)>,
    },
    /// All groups done — hand off to `apply_tool_results`.
    AllGroupsDone,
    /// One or more groups still pending — caller spawns the next.
    NextGroup(Vec<usize>),
}

// ─── Pure decisions: read state, return action ───────────────────────

/// Build the message list to send to the LLM: history + pending,
/// optionally transformed.
pub fn build_context(frame: &Frame, conv: &Conv, pending: &[Message]) -> Vec<Message> {
    let mut context: Vec<Message> = conv
        .conversation
        .messages
        .iter()
        .cloned()
        .chain(pending.iter().cloned())
        .collect();
    if let Some(ref transform) = frame.transform_context {
        context = transform(context);
    }
    context
}

/// Build the per-call run config from the frame's wiring.
pub fn build_run_config(frame: &Frame, turn_number: u32) -> AgentRunConfig {
    AgentRunConfig {
        system_prompt: frame.config.system_prompt.clone(),
        tools: frame
            .tools
            .iter()
            .map(|t| to_api_tool(t.as_ref()))
            .collect(),
        server_tools: frame.server_tools.clone(),
        model: frame.config.model.clone(),
        reasoning: Some(frame.config.reasoning),
        thinking_adaptive: frame.config.thinking_adaptive,
        max_tokens: frame.config.max_tokens,
        temperature: None,
        turn_number,
        cache_scope: frame.config.cache_scope.clone(),
        cache_ttl: frame.config.cache_ttl.clone(),
        system_prompt_boundary: frame.config.system_prompt_boundary.clone(),
    }
}

/// Whether proactive compaction is needed for this turn's usage.
pub fn decide_proactive_compaction(frame: &Frame, usage: &Usage) -> bool {
    if !frame.config.compaction.enabled {
        return false;
    }
    let used = usage.input + usage.cache_read;
    let cw = frame.config.model.context_window as u64;
    let reserve = frame.config.compaction.reserve.resolve(cw);
    let limit = cw.saturating_sub(reserve);
    used > limit
}

/// Decide what to do after an LLM response. Pure — does not commit
/// the assistant message or accumulate usage. The caller `apply_*`s
/// those if the decision says they should.
pub fn decide_response_action(
    frame: &Frame,
    outcome: &StreamOutcome,
    first_user_message: Option<Message>,
) -> ResponseDecision {
    if let Some(error_msg) = outcome.error.as_deref() {
        let overflow = is_context_overflow(error_msg);
        if overflow && frame.config.compaction.enabled {
            return ResponseDecision {
                action: ResponseAction::Compact {
                    reason: CompactionReason::Overflow,
                    resume_pending: Some((
                        first_user_message.iter().cloned().collect(),
                        first_user_message,
                    )),
                },
                needs_proactive_compaction: false,
            };
        }
        return ResponseDecision {
            action: ResponseAction::Error(crate::types::error::Error::Other(error_msg.to_string())),
            needs_proactive_compaction: false,
        };
    }

    let Some(ref assistant_msg) = outcome.assistant_message else {
        return ResponseDecision {
            action: ResponseAction::Done,
            needs_proactive_compaction: false,
        };
    };

    let needs_proactive = decide_proactive_compaction(frame, &outcome.usage);
    let tool_calls = extract_tool_calls(assistant_msg);
    let action = if tool_calls.is_empty() {
        ResponseAction::Done
    } else {
        let groups = build_tool_groups(&frame.tools, &tool_calls);
        if groups.is_empty() {
            ResponseAction::Done
        } else {
            ResponseAction::RunTools {
                tool_calls,
                groups,
                first_user_message,
            }
        }
    };

    ResponseDecision {
        action,
        needs_proactive_compaction: needs_proactive,
    }
}

/// Which queue `apply_drain_queues` actually pulled from. The actor
/// reads this to decide whether to decrement the bg-pending counter
/// (only when follow-ups were drained, since steering doesn't come
/// through the bg-follow-up path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainedFrom {
    Steering,
    FollowUps,
    Nothing,
}

/// Drained queue payload + identity. Replaces the prior bare
/// `Vec<Message>` return that forced callers to reconstruct which
/// queue they got messages from.
pub struct DrainedQueue {
    pub messages: Vec<Message>,
    pub source: DrainedFrom,
}

/// Drain the steering queue if non-empty, otherwise the follow-up
/// queue. Returns the drained messages alongside which queue they
/// came from so the caller can react accordingly (e.g. decrement the
/// bg-pending counter on `FollowUps` drains).
///
/// The bg-pending counter on `Shared` is *not* touched here — that's
/// the actor's responsibility, because we don't (and shouldn't)
/// thread `Shared` into pure transition functions.
pub fn apply_drain_queues(frame: &Frame, conv: &mut Conv) -> DrainedQueue {
    if !conv.steering_queue.is_empty() {
        return DrainedQueue {
            messages: drain_queue(&mut conv.steering_queue, frame.config.steering_mode),
            source: DrainedFrom::Steering,
        };
    }
    if !conv.follow_up_queue.is_empty() {
        return DrainedQueue {
            messages: drain_queue(&mut conv.follow_up_queue, frame.config.follow_up_mode),
            source: DrainedFrom::FollowUps,
        };
    }
    DrainedQueue {
        messages: vec![],
        source: DrainedFrom::Nothing,
    }
}

/// Decide what to do at a tool-batch boundary. If steering arrived
/// during the batch, commit results and redirect; otherwise advance
/// to the next group or finish.
///
/// Pure — does not commit tool results to `Conv`. The caller
/// `apply_commit_tool_results` for the redirect path or
/// `apply_tool_results` (final commit) for the all-done path.
pub fn decide_batch_complete(
    _frame: &Frame,
    conv: &Conv,
    remaining_groups: &[Vec<usize>],
    tool_calls: &[ToolCall],
) -> BatchCompleteAction {
    if !conv.steering_queue.is_empty() {
        let mut skipped = Vec::new();
        for group in remaining_groups {
            for &idx in group {
                let tc = &tool_calls[idx];
                skipped.push((idx, tc.id.clone(), tc.name.clone()));
            }
        }
        return BatchCompleteAction::Redirect {
            skipped_indices: skipped,
        };
    }

    if remaining_groups.is_empty() {
        BatchCompleteAction::AllGroupsDone
    } else {
        BatchCompleteAction::NextGroup(remaining_groups[0].clone())
    }
}

// ─── State mutations: the `apply_*` family ───────────────────────────

/// Commit `pending` messages into the conversation (e.g. the user's
/// prompt at the start of a turn).
pub fn apply_pending(conv: &mut Conv, pending: &[Message]) {
    conv.conversation.messages.extend(pending.iter().cloned());
}

/// Commit a successful response: pending → conversation, then
/// assistant message → conversation, then accumulate usage.
///
/// If the stream produced no assistant message (`outcome.assistant_message
/// = None`), the pending messages are *not* committed — there's nothing
/// new to react to and the user's prompt should be re-presentable rather
/// than baked into history. Usage is still accumulated since the call
/// itself consumed tokens.
pub fn apply_response(conv: &mut Conv, outcome: StreamOutcome, pending: &[Message]) {
    if let Some(assistant) = outcome.assistant_message {
        apply_pending(conv, pending);
        conv.conversation.messages.push(assistant);
    }
    apply_usage(conv, &outcome.usage);
}

/// Apply the partial-message + pending preservation that the error
/// path of [`decide_response_action`] depends on for context. Called
/// by the actor when the decision is `Compact { reason: Overflow }`
/// or `Error`.
///
/// Pending is committed when *either* `force_pending` is true *or* the
/// stream produced a meaningful partial message. The actor passes
/// `force_pending: true` on the overflow-into-compaction path because
/// the prompt is going to resume after compaction — pending must
/// survive. On a non-overflow error, the prompt is going to terminate;
/// pending stays re-presentable unless preserving it gives a partial
/// some context.
///
/// The partial is committed only when it's meaningful (some text /
/// thinking / a tool call).
pub fn apply_partial_on_error(
    conv: &mut Conv,
    outcome: &StreamOutcome,
    pending: &[Message],
    force_pending: bool,
) {
    let has_meaningful = outcome
        .partial_message
        .as_ref()
        .is_some_and(message_has_content);
    if has_meaningful || force_pending {
        apply_pending(conv, pending);
    }
    if has_meaningful {
        if let Some(ref partial) = outcome.partial_message {
            conv.conversation.messages.push(partial.clone());
        }
    }
}

/// Accumulate per-turn token usage into the running total.
pub fn apply_usage(conv: &mut Conv, usage: &Usage) {
    conv.conversation.total_usage.input += usage.input;
    conv.conversation.total_usage.output += usage.output;
    conv.conversation.total_usage.cache_read += usage.cache_read;
    conv.conversation.total_usage.cache_write += usage.cache_write;
    conv.conversation.total_usage.thinking += usage.thinking;
    conv.conversation.total_usage.cache_creation_1h += usage.cache_creation_1h;
    conv.conversation.total_usage.cache_creation_5m += usage.cache_creation_5m;
}

/// Commit tool results to the conversation in original request order.
pub fn apply_tool_results(
    conv: &mut Conv,
    tool_calls: &[ToolCall],
    results_map: &mut HashMap<usize, (String, String, ToolResult)>,
) {
    let messages = collect_ordered_results(tool_calls, results_map);
    conv.conversation.messages.extend(messages);
}

// ─── Standalone helpers ──────────────────────────────────────────────

pub fn extract_tool_calls(msg: &Message) -> Vec<ToolCall> {
    let Message::Assistant { content, .. } = msg else {
        return vec![];
    };
    content
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall {
                id,
                name,
                arguments,
            } => Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                args: arguments.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// Group tool calls by concurrency: consecutive `Parallel` tools form
/// a group, `Sequential` tools are singletons. An unknown tool is
/// treated as `Sequential` so the unknown-tool error is surfaced in
/// isolation rather than mid-batch.
pub fn build_tool_groups(tools: &[BoxedTool], tool_calls: &[ToolCall]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut current: Vec<usize> = vec![];
    let mut current_parallel = false;

    for (idx, tc) in tool_calls.iter().enumerate() {
        let is_parallel = tools
            .iter()
            .find(|t| t.name() == tc.name)
            .map(|t| t.concurrency() == Concurrency::Parallel)
            .unwrap_or(false);

        if idx == 0 {
            current_parallel = is_parallel;
            current.push(idx);
        } else if is_parallel && current_parallel {
            current.push(idx);
        } else {
            groups.push(std::mem::take(&mut current));
            current.push(idx);
            current_parallel = is_parallel;
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

pub fn collect_ordered_results(
    tool_calls: &[ToolCall],
    results_map: &mut HashMap<usize, (String, String, ToolResult)>,
) -> Vec<Message> {
    let mut out = Vec::new();
    for (idx, tc) in tool_calls.iter().enumerate() {
        let (id, name, result) = results_map.remove(&idx).unwrap_or_else(|| {
            (
                tc.id.clone(),
                tc.name.clone(),
                ToolResult::error("Task failed (panicked or cancelled)"),
            )
        });
        out.push(Message::ToolResult {
            tool_call_id: id,
            tool_name: name,
            content: result.content,
            is_error: result.is_error,
            timestamp: chrono::Utc::now().timestamp_millis(),
        });
    }
    out
}

pub fn drain_queue(queue: &mut Vec<Message>, mode: DequeueMode) -> Vec<Message> {
    match mode {
        DequeueMode::All => std::mem::take(queue),
        DequeueMode::OneAtATime => {
            if queue.is_empty() {
                vec![]
            } else {
                vec![queue.remove(0)]
            }
        }
    }
}

fn message_has_content(msg: &Message) -> bool {
    let content = match msg {
        Message::Assistant { content, .. }
        | Message::User { content, .. }
        | Message::ToolResult { content, .. }
        | Message::SystemInjection { content, .. } => content,
    };
    content.iter().any(|c| match c {
        Content::Text { text } => !text.trim().is_empty(),
        Content::Thinking { thinking, .. } => !thinking.trim().is_empty(),
        Content::ToolCall { name, .. } => !name.is_empty(),
        Content::Image { .. }
        | Content::RedactedThinking { .. }
        | Content::ServerToolUse { .. }
        | Content::ServerToolResult { .. } => true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::conversation::Conversation;
    use tau_ai::{AssistantMetadata, Usage};

    fn empty_conv() -> Conv {
        Conv {
            conversation: Conversation::default(),
            steering_queue: vec![],
            follow_up_queue: vec![],
            cwd: None,
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message::Assistant {
            content: vec![Content::text(text)],
            metadata: AssistantMetadata::default(),
        }
    }

    /// Force-pending must commit the user message even when there's no
    /// partial. Regression for the overflow-without-partial path that
    /// silently dropped pending in an earlier v2 implementation.
    #[test]
    fn apply_partial_on_error_force_pending_commits_without_partial() {
        let mut conv = empty_conv();
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: Usage::default(),
            error: Some("context overflow".into()),
            partial_message: None,
        };
        let pending = vec![Message::user("the user's prompt")];

        apply_partial_on_error(&mut conv, &outcome, &pending, /* force */ true);

        assert_eq!(conv.conversation.messages.len(), 1, "pending committed");
        assert_eq!(conv.conversation.messages[0].role(), "user");
    }

    /// Non-force, no meaningful partial: pending stays uncommitted.
    #[test]
    fn apply_partial_on_error_no_force_no_partial_leaves_pending() {
        let mut conv = empty_conv();
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: Usage::default(),
            error: Some("rate limited".into()),
            partial_message: None,
        };
        let pending = vec![Message::user("the user's prompt")];

        apply_partial_on_error(&mut conv, &outcome, &pending, /* force */ false);

        assert!(
            conv.conversation.messages.is_empty(),
            "pending not committed"
        );
    }

    /// Meaningful partial commits pending and the partial, regardless
    /// of force flag.
    #[test]
    fn apply_partial_on_error_meaningful_partial_commits_both() {
        let mut conv = empty_conv();
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: Usage::default(),
            error: Some("network error".into()),
            partial_message: Some(assistant_text("partial content")),
        };
        let pending = vec![Message::user("prompt")];

        apply_partial_on_error(&mut conv, &outcome, &pending, /* force */ false);

        assert_eq!(conv.conversation.messages.len(), 2);
        assert_eq!(conv.conversation.messages[0].role(), "user");
        assert_eq!(conv.conversation.messages[1].text(), "partial content");
    }
}
