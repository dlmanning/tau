//! Decision logic extracted from the actor loop.
//!
//! All functions in this module are synchronous decision-makers that read or
//! mutate `AgentState` but never do I/O (no channels, no transport calls, no
//! async). This makes them testable without a tokio runtime.

use std::collections::HashMap;

use tau_ai::{Content, Message, Usage};

use crate::compaction::{CompactionReason, CompactionResult};
use crate::config::DequeueMode;
use crate::overflow::is_context_overflow;
use crate::state::{AgentState, ToolCall};
use crate::stream::StreamOutcome;
use crate::tool::{BoxedTool, Concurrency, ToolResult, to_api_tool};
use crate::tool_executor::has_meaningful_content;
use crate::transport::AgentRunConfig;

// ─── Decision types ────────────────────────────────────────────────

/// What to do after receiving an LLM response.
#[derive(Debug)]
pub(crate) enum ResponseAction {
    /// Run these tool calls (grouped by concurrency).
    RunTools {
        tool_calls: Vec<ToolCall>,
        groups: Vec<Vec<usize>>,
        first_user_message: Option<Message>,
    },
    /// Compact due to overflow, then optionally retry.
    Compact {
        reason: CompactionReason,
        resume_pending: Option<(Vec<Message>, Option<Message>)>,
    },
    /// Turn complete, check follow-ups.
    Done,
    /// Fatal error.
    Error(crate::error::Error),
}

/// Result of `process_response`: an action plus an optional proactive compaction flag.
#[derive(Debug)]
pub(crate) struct ResponseDecision {
    pub action: ResponseAction,
    /// If true, run proactive compaction before executing the action.
    pub needs_proactive_compaction: bool,
}

/// What to do after draining follow-up/steering queues.
#[derive(Debug)]
pub(crate) enum FollowUpAction {
    /// Start a new turn with these messages.
    Continue(Vec<Message>),
    /// Wait for background agents to post follow-ups.
    WaitForFollowUps,
    /// All done.
    Done,
}

/// What to do when a tool batch completes.
#[derive(Debug)]
pub(crate) enum BatchCompleteAction {
    /// Steering arrived — commit tool results, start new turn with steering as pending.
    Redirect {
        /// Steering messages to send as pending for the next turn.
        steering: Vec<Message>,
        /// Skipped tool calls (index, id, name) for the caller to emit events.
        skipped_indices: Vec<(usize, String, String)>,
    },
    /// All groups done — hand off to ApplyToolResults.
    AllGroupsDone,
    /// More groups remain — spawn the next one.
    NextGroup(Vec<usize>),
}

// ─── AgentState methods (decision logic) ───────────────────────────

impl AgentState {
    /// Build context: conversation.messages + pending, with optional transform.
    pub(crate) fn build_context(&self, pending: &[Message]) -> Vec<Message> {
        let mut context: Vec<Message> = self
            .conversation
            .messages
            .iter()
            .cloned()
            .chain(pending.iter().cloned())
            .collect();

        if let Some(ref transform) = self.transform_context {
            context = transform(context);
        }
        context
    }

    /// Build the run configuration from current agent state.
    pub(crate) fn build_run_config(&self) -> AgentRunConfig {
        AgentRunConfig {
            system_prompt: self.config.system_prompt.clone(),
            tools: self.tools.iter().map(|t| to_api_tool(t.as_ref())).collect(),
            server_tools: self.server_tools.clone(),
            model: self.config.model.clone(),
            reasoning: Some(self.config.reasoning),
            thinking_adaptive: self.config.thinking_adaptive,
            max_tokens: self.config.max_tokens,
            temperature: None,
            cache_scope: self.config.cache_scope.clone(),
            cache_ttl: self.config.cache_ttl.clone(),
            system_prompt_boundary: self.config.system_prompt_boundary.clone(),
        }
    }

    /// Decide what to do after an LLM response. Delegates to `handle_response_error`
    /// for the error path and `commit_and_decide` for the success path.
    pub(crate) fn process_response(
        &mut self,
        outcome: StreamOutcome,
        pending: Vec<Message>,
        first_user_message: Option<Message>,
    ) -> ResponseDecision {
        if outcome.error.is_some() {
            return self.handle_response_error(&outcome, pending, first_user_message);
        }

        self.commit_and_decide(outcome, pending, first_user_message)
    }

    /// Handle the error path of a response: save partial content if meaningful,
    /// trigger compaction on overflow, or report the error.
    fn handle_response_error(
        &mut self,
        outcome: &StreamOutcome,
        pending: Vec<Message>,
        first_user_message: Option<Message>,
    ) -> ResponseDecision {
        let error_msg = outcome.error.as_ref().unwrap();
        let overflow = is_context_overflow(error_msg);
        let needs_compaction = overflow && self.config.compaction.enabled;
        let has_meaningful_partial = outcome
            .partial_message
            .as_ref()
            .is_some_and(has_meaningful_content);

        // Flush pending once if we need to preserve context
        if has_meaningful_partial || needs_compaction {
            flush_pending(&mut self.conversation.messages, &pending);
        }

        if has_meaningful_partial {
            if let Some(ref partial) = outcome.partial_message {
                self.conversation.messages.push(partial.clone());
            }
        }

        if needs_compaction {
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

        self.conversation.error = Some(error_msg.clone());
        ResponseDecision {
            action: ResponseAction::Error(crate::error::Error::Other(error_msg.clone())),
            needs_proactive_compaction: false,
        }
    }

    /// Commit a successful response to conversation state and decide next action.
    fn commit_and_decide(
        &mut self,
        outcome: StreamOutcome,
        pending: Vec<Message>,
        first_user_message: Option<Message>,
    ) -> ResponseDecision {
        self.accumulate_usage(&outcome.usage);

        let Some(ref assistant_msg) = outcome.assistant_message else {
            return ResponseDecision {
                action: ResponseAction::Done,
                needs_proactive_compaction: false,
            };
        };

        // Commit pending + assistant message
        flush_pending(&mut self.conversation.messages, &pending);
        self.conversation.messages.push(assistant_msg.clone());

        let needs_proactive_compaction = self.should_compact_proactively(&outcome.usage);
        let tool_calls = extract_tool_calls(assistant_msg);

        let action = if tool_calls.is_empty() {
            ResponseAction::Done
        } else {
            let groups = build_tool_groups(&self.tools, &tool_calls);
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
            needs_proactive_compaction,
        }
    }

    /// Check whether proactive compaction is needed based on usage vs context window.
    pub(crate) fn should_compact_proactively(&self, usage: &Usage) -> bool {
        if !self.config.compaction.enabled {
            return false;
        }
        let used = usage.input + usage.cache_read;
        let limit = (self.config.model.context_window as u64)
            .saturating_sub(self.config.compaction.reserve_tokens);
        used > limit
    }

    /// Accumulate token usage from a turn into total_usage.
    pub(crate) fn accumulate_usage(&mut self, usage: &Usage) {
        self.conversation.total_usage.input += usage.input;
        self.conversation.total_usage.output += usage.output;
        self.conversation.total_usage.cache_read += usage.cache_read;
        self.conversation.total_usage.cache_write += usage.cache_write;
        self.conversation.total_usage.thinking += usage.thinking;
        self.conversation.total_usage.cache_creation_1h += usage.cache_creation_1h;
        self.conversation.total_usage.cache_creation_5m += usage.cache_creation_5m;
    }

    /// Decide what to do after a turn completes (no tool calls).
    /// Checks steering and follow-up queues.
    pub(crate) fn drain_follow_ups(&mut self) -> FollowUpAction {
        // Steering first
        let steering = drain_queue(&mut self.steering_queue, self.config.steering_mode);
        if !steering.is_empty() {
            return FollowUpAction::Continue(steering);
        }

        // Follow-ups
        let follow_ups = drain_queue(&mut self.follow_up_queue, self.config.follow_up_mode);
        if !follow_ups.is_empty() {
            let count = follow_ups.len() as u32;
            let _ = self.pending_follow_ups.fetch_update(
                std::sync::atomic::Ordering::Release,
                std::sync::atomic::Ordering::Acquire,
                |n| Some(n.saturating_sub(count)),
            );
            return FollowUpAction::Continue(follow_ups);
        }

        // Wait for background agents?
        if self
            .pending_follow_ups
            .load(std::sync::atomic::Ordering::Acquire)
            > 0
        {
            return FollowUpAction::WaitForFollowUps;
        }

        FollowUpAction::Done
    }

    /// Decide what to do when a tool batch completes.
    ///
    /// Checks if steering arrived during execution, and if so, collects results
    /// (marking remaining groups as skipped) and commits tool results to conversation.
    pub(crate) fn handle_batch_complete(
        &mut self,
        remaining_groups: &[Vec<usize>],
        tool_calls: &[ToolCall],
        results_map: &mut HashMap<usize, (String, String, ToolResult)>,
    ) -> BatchCompleteAction {
        if !self.steering_queue.is_empty() {
            // Drain steering
            let steering = drain_queue(&mut self.steering_queue, self.config.steering_mode);

            // Collect skipped tool info for the caller to emit events
            let mut skipped = Vec::new();
            for group in remaining_groups {
                for &idx in group {
                    let tc = &tool_calls[idx];
                    let skip_result = ToolResult::error("Skipped due to steering message");
                    skipped.push((idx, tc.id.clone(), tc.name.clone()));
                    results_map.insert(idx, (tc.id.clone(), tc.name.clone(), skip_result));
                }
            }

            // Commit tool results to conversation
            let tool_results = collect_ordered_results(tool_calls, results_map);
            self.conversation.messages.extend(tool_results);

            return BatchCompleteAction::Redirect {
                steering,
                skipped_indices: skipped,
            };
        }

        if remaining_groups.is_empty() {
            BatchCompleteAction::AllGroupsDone
        } else {
            BatchCompleteAction::NextGroup(remaining_groups[0].clone())
        }
    }

    /// Apply a compaction result to conversation state.
    pub(crate) fn apply_compaction_result(&mut self, result: CompactionResult) {
        self.conversation.previous_summary = Some(result.summary.clone());
        let kept = self
            .conversation
            .messages
            .split_off(result.first_kept_index);
        self.conversation.messages = vec![Message::user(format!(
            "<context-summary>\n{}\n</context-summary>\n\nThe conversation was compacted. Continue from where we left off.",
            result.summary
        ))];
        self.conversation.messages.extend(kept);
    }
}

// ─── Standalone pure functions ─────────────────────────────────────

/// Extract tool calls from an assistant message.
pub(crate) fn extract_tool_calls(msg: &Message) -> Vec<ToolCall> {
    let content = match msg {
        Message::Assistant { content, .. } => content,
        _ => return vec![],
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

/// Build groups of tool calls: consecutive Parallel tools form a group,
/// Sequential tools are singletons.
pub(crate) fn build_tool_groups(tools: &[BoxedTool], tool_calls: &[ToolCall]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut current_group: Vec<usize> = vec![];
    let mut current_is_parallel = false;

    for (idx, tc) in tool_calls.iter().enumerate() {
        let is_parallel = tools
            .iter()
            .find(|t| t.name() == tc.name.as_str())
            .map(|t| t.concurrency() == Concurrency::Parallel)
            .unwrap_or(false);

        if idx == 0 {
            current_is_parallel = is_parallel;
            current_group.push(idx);
        } else if is_parallel && current_is_parallel {
            current_group.push(idx);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(idx);
            current_is_parallel = is_parallel;
        }
    }
    if !current_group.is_empty() {
        groups.push(current_group);
    }
    groups
}

/// Collect tool results in original request order, producing ToolResult messages.
pub(crate) fn collect_ordered_results(
    tool_calls: &[ToolCall],
    results_map: &mut HashMap<usize, (String, String, ToolResult)>,
) -> Vec<Message> {
    let mut messages = Vec::new();
    for (idx, tc) in tool_calls.iter().enumerate() {
        let (id, name, result) = results_map.remove(&idx).unwrap_or_else(|| {
            (
                tc.id.clone(),
                tc.name.clone(),
                ToolResult::error("Task failed (panicked or cancelled)"),
            )
        });
        messages.push(Message::ToolResult {
            tool_call_id: id,
            tool_name: name,
            content: result.content,
            is_error: result.is_error,
            timestamp: chrono::Utc::now().timestamp_millis(),
        });
    }
    messages
}

/// Commit pending messages into the conversation.
pub(crate) fn flush_pending(messages: &mut Vec<Message>, pending: &[Message]) {
    for m in pending {
        messages.push(m.clone());
    }
}

/// Drain a message queue based on the dequeue mode.
pub(crate) fn drain_queue(queue: &mut Vec<Message>, mode: DequeueMode) -> Vec<Message> {
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

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32};

    use crate::conversation::Conversation;
    use crate::tool::FileAccessTracker;

    use crate::test_utils::*;

    fn make_test_state() -> AgentState {
        let (event_tx, _) = tokio::sync::broadcast::channel(256);
        AgentState {
            config: make_test_config(),
            conversation: Conversation::default(),
            tools: vec![Arc::new(EchoTool) as BoxedTool],
            transport: Arc::new(MockTransport::new()),
            event_tx,
            server_tools: vec![],
            schema_cache: Default::default(),
            cwd: None,
            file_access: Arc::new(parking_lot::Mutex::new(FileAccessTracker::default())),
            interaction_tx: None,
            approval_policy: Arc::new(crate::approval::DefaultApprovalPolicy),
            transform_context: None,
            steering_queue: vec![],
            pending_conversation_ops: vec![],
            follow_up_queue: vec![],
            pending_follow_ups: Arc::new(AtomicU32::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
            cancel: Arc::new(parking_lot::Mutex::new(
                tokio_util::sync::CancellationToken::new(),
            )),
        }
    }

    // ─── extract_tool_calls ────────────────────────────────────────

    #[test]
    fn extract_tool_calls_from_assistant_message() {
        let msg = make_tool_call_message("echo", "call_1", serde_json::json!({"text": "hi"}));
        let calls = extract_tool_calls(&msg);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "echo");
        assert_eq!(calls[0].id, "call_1");
    }

    #[test]
    fn extract_tool_calls_from_text_only_message() {
        let msg = make_assistant_message("just text");
        let calls = extract_tool_calls(&msg);
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_tool_calls_from_user_message() {
        let msg = make_user_message("hello");
        let calls = extract_tool_calls(&msg);
        assert!(calls.is_empty());
    }

    // ─── build_tool_groups ─────────────────────────────────────────

    #[test]
    fn build_tool_groups_all_parallel() {
        let tools: Vec<BoxedTool> = vec![Arc::new(EchoTool)];
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
            ToolCall {
                id: "3".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
        ];
        let groups = build_tool_groups(&tools, &calls);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0, 1, 2]);
    }

    #[test]
    fn build_tool_groups_unknown_tool_is_sequential() {
        let tools: Vec<BoxedTool> = vec![Arc::new(EchoTool)];
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
            ToolCall {
                id: "2".into(),
                name: "unknown".into(),
                args: serde_json::json!({}),
            },
            ToolCall {
                id: "3".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
        ];
        let groups = build_tool_groups(&tools, &calls);
        // echo(parallel), unknown(sequential), echo(parallel)
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![0]);
        assert_eq!(groups[1], vec![1]);
        assert_eq!(groups[2], vec![2]);
    }

    #[test]
    fn build_tool_groups_empty() {
        let tools: Vec<BoxedTool> = vec![];
        let groups = build_tool_groups(&tools, &[]);
        assert!(groups.is_empty());
    }

    // ─── drain_queue ───────────────────────────────────────────────

    #[test]
    fn drain_queue_all_mode() {
        let mut queue = vec![make_user_message("a"), make_user_message("b")];
        let drained = drain_queue(&mut queue, DequeueMode::All);
        assert_eq!(drained.len(), 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn drain_queue_one_at_a_time() {
        let mut queue = vec![make_user_message("a"), make_user_message("b")];
        let drained = drain_queue(&mut queue, DequeueMode::OneAtATime);
        assert_eq!(drained.len(), 1);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn drain_queue_empty() {
        let mut queue: Vec<Message> = vec![];
        let drained = drain_queue(&mut queue, DequeueMode::All);
        assert!(drained.is_empty());
    }

    // ─── process_response ──────────────────────────────────────────

    #[test]
    fn process_response_text_only() {
        let mut state = make_test_state();
        let outcome = StreamOutcome {
            assistant_message: Some(make_assistant_message("hello")),
            usage: tau_ai::Usage {
                input: 100,
                output: 50,
                ..Default::default()
            },
            error: None,
            partial_message: None,
        };

        let decision = state.process_response(outcome, vec![], None);

        assert!(matches!(decision.action, ResponseAction::Done));
        assert!(!decision.needs_proactive_compaction);
        assert_eq!(state.conversation.total_usage.input, 100);
        assert_eq!(state.conversation.total_usage.output, 50);
        // Assistant message should be committed
        assert_eq!(state.conversation.messages.len(), 1);
    }

    #[test]
    fn process_response_with_tool_calls() {
        let mut state = make_test_state();
        let msg = make_tool_call_message("echo", "call_1", serde_json::json!({"text": "x"}));
        let outcome = StreamOutcome {
            assistant_message: Some(msg),
            usage: tau_ai::Usage::default(),
            error: None,
            partial_message: None,
        };

        let decision = state.process_response(outcome, vec![], None);

        match decision.action {
            ResponseAction::RunTools {
                tool_calls, groups, ..
            } => {
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].name, "echo");
                assert_eq!(groups.len(), 1);
            }
            other => panic!("expected RunTools, got {other:?}"),
        }
    }

    #[test]
    fn process_response_error_non_overflow() {
        let mut state = make_test_state();
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: tau_ai::Usage::default(),
            error: Some("rate limit exceeded".into()),
            partial_message: None,
        };

        let decision = state.process_response(outcome, vec![], None);

        assert!(matches!(decision.action, ResponseAction::Error(_)));
    }

    #[test]
    fn process_response_overflow_triggers_compaction() {
        let mut state = make_test_state();
        state.config.compaction.enabled = true;
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: tau_ai::Usage::default(),
            error: Some("context_length_exceeded".into()),
            partial_message: None,
        };

        let decision = state.process_response(outcome, vec![], None);

        match decision.action {
            ResponseAction::Compact { reason, .. } => {
                assert!(matches!(reason, CompactionReason::Overflow));
            }
            other => panic!("expected Compact, got {other:?}"),
        }
    }

    #[test]
    fn process_response_proactive_compaction_flag() {
        let mut state = make_test_state();
        state.config.compaction.enabled = true;
        state.config.model.context_window = 1000;
        state.config.compaction.reserve_tokens = 100;

        let outcome = StreamOutcome {
            assistant_message: Some(make_assistant_message("hi")),
            // input + cache_read = 950 > 1000 - 100 = 900
            usage: tau_ai::Usage {
                input: 950,
                ..Default::default()
            },
            error: None,
            partial_message: None,
        };

        let decision = state.process_response(outcome, vec![], None);

        assert!(decision.needs_proactive_compaction);
        assert!(matches!(decision.action, ResponseAction::Done));
    }

    #[test]
    fn process_response_proactive_compaction_with_tool_calls() {
        let mut state = make_test_state();
        state.config.compaction.enabled = true;
        state.config.model.context_window = 1000;
        state.config.compaction.reserve_tokens = 100;

        let msg = make_tool_call_message("echo", "call_1", serde_json::json!({"text": "x"}));
        let outcome = StreamOutcome {
            assistant_message: Some(msg),
            usage: tau_ai::Usage {
                input: 950,
                ..Default::default()
            },
            error: None,
            partial_message: None,
        };

        let decision = state.process_response(outcome, vec![], None);

        // Both flags should be set: compaction needed AND tools to run
        assert!(decision.needs_proactive_compaction);
        match decision.action {
            ResponseAction::RunTools { tool_calls, .. } => {
                assert_eq!(tool_calls.len(), 1);
            }
            other => panic!("expected RunTools, got {other:?}"),
        }
    }

    #[test]
    fn process_response_commits_pending_messages() {
        let mut state = make_test_state();
        let pending = vec![make_user_message("my prompt")];
        let outcome = StreamOutcome {
            assistant_message: Some(make_assistant_message("response")),
            usage: tau_ai::Usage::default(),
            error: None,
            partial_message: None,
        };

        state.process_response(outcome, pending, None);

        assert_eq!(state.conversation.messages.len(), 2); // user + assistant
        assert_eq!(state.conversation.messages[0].role(), "user");
        assert_eq!(state.conversation.messages[1].role(), "assistant");
    }

    #[test]
    fn process_response_no_assistant_message() {
        let mut state = make_test_state();
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: tau_ai::Usage::default(),
            error: None,
            partial_message: None,
        };

        let decision = state.process_response(outcome, vec![], None);
        assert!(matches!(decision.action, ResponseAction::Done));
    }

    // ─── drain_follow_ups ──────────────────────────────────────────

    #[test]
    fn drain_follow_ups_steering_first() {
        let mut state = make_test_state();
        state.steering_queue.push(make_user_message("steer"));
        state.follow_up_queue.push(make_user_message("follow"));

        match state.drain_follow_ups() {
            FollowUpAction::Continue(msgs) => {
                assert_eq!(msgs.len(), 1);
                assert_eq!(msgs[0].text(), "steer");
            }
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn drain_follow_ups_follow_up_when_no_steering() {
        let mut state = make_test_state();
        state.follow_up_queue.push(make_user_message("follow"));

        match state.drain_follow_ups() {
            FollowUpAction::Continue(msgs) => {
                assert_eq!(msgs[0].text(), "follow");
            }
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    #[test]
    fn drain_follow_ups_waits_for_background() {
        let mut state = make_test_state();
        state
            .pending_follow_ups
            .store(1, std::sync::atomic::Ordering::Release);

        assert!(matches!(
            state.drain_follow_ups(),
            FollowUpAction::WaitForFollowUps
        ));
    }

    #[test]
    fn drain_follow_ups_done_when_empty() {
        let mut state = make_test_state();
        assert!(matches!(state.drain_follow_ups(), FollowUpAction::Done));
    }

    // ─── should_compact_proactively ────────────────────────────────

    #[test]
    fn should_compact_when_over_threshold() {
        let state = make_test_state();
        // context_window=200000, reserve=16384, limit=183616
        let usage = tau_ai::Usage {
            input: 190000,
            ..Default::default()
        };
        assert!(state.should_compact_proactively(&usage));
    }

    #[test]
    fn should_not_compact_when_under_threshold() {
        let state = make_test_state();
        let usage = tau_ai::Usage {
            input: 100,
            ..Default::default()
        };
        assert!(!state.should_compact_proactively(&usage));
    }

    #[test]
    fn should_not_compact_when_disabled() {
        let mut state = make_test_state();
        state.config.compaction.enabled = false;
        let usage = tau_ai::Usage {
            input: 999999,
            ..Default::default()
        };
        assert!(!state.should_compact_proactively(&usage));
    }

    // ─── apply_compaction_result ───────────────────────────────────

    #[test]
    fn apply_compaction_replaces_old_messages() {
        let mut state = make_test_state();
        state.conversation.messages = vec![
            make_user_message("old 1"),
            make_assistant_message("old 2"),
            make_user_message("recent 1"),
            make_assistant_message("recent 2"),
        ];

        let result = CompactionResult {
            summary: "Summary of old conversation".to_string(),
            first_kept_index: 2,
            tokens_before: 1000,
            read_files: vec![],
            modified_files: vec![],
        };

        state.apply_compaction_result(result);

        // Should have: summary message + 2 recent messages
        assert_eq!(state.conversation.messages.len(), 3);
        assert!(
            state.conversation.messages[0]
                .text()
                .contains("context-summary")
        );
        assert_eq!(state.conversation.messages[1].text(), "recent 1");
        assert_eq!(state.conversation.messages[2].text(), "recent 2");
        assert_eq!(
            state.conversation.previous_summary.as_deref(),
            Some("Summary of old conversation")
        );
    }

    // ─── collect_ordered_results ───────────────────────────────────

    #[test]
    fn collect_ordered_results_preserves_order() {
        let calls = vec![
            ToolCall {
                id: "a".into(),
                name: "t1".into(),
                args: serde_json::json!({}),
            },
            ToolCall {
                id: "b".into(),
                name: "t2".into(),
                args: serde_json::json!({}),
            },
        ];
        let mut results = HashMap::new();
        // Insert in reverse order
        results.insert(
            1,
            (
                "b".to_string(),
                "t2".to_string(),
                ToolResult::text("result_b"),
            ),
        );
        results.insert(
            0,
            (
                "a".to_string(),
                "t1".to_string(),
                ToolResult::text("result_a"),
            ),
        );

        let messages = collect_ordered_results(&calls, &mut results);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text(), "result_a");
        assert_eq!(messages[1].text(), "result_b");
    }

    #[test]
    fn collect_ordered_results_handles_missing() {
        let calls = vec![ToolCall {
            id: "a".into(),
            name: "t1".into(),
            args: serde_json::json!({}),
        }];
        let mut results = HashMap::new(); // No results at all

        let messages = collect_ordered_results(&calls, &mut results);
        assert_eq!(messages.len(), 1);
        // Should have a fallback error
        match &messages[0] {
            Message::ToolResult { is_error, .. } => assert!(is_error),
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    // ─── build_context ─────────────────────────────────────────────

    #[test]
    fn build_context_combines_conversation_and_pending() {
        let mut state = make_test_state();
        state.conversation.messages = vec![make_user_message("existing")];
        let pending = vec![make_user_message("new")];

        let context = state.build_context(&pending);
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].text(), "existing");
        assert_eq!(context[1].text(), "new");
    }

    #[test]
    fn build_context_applies_transform() {
        let mut state = make_test_state();
        state.conversation.messages = vec![make_user_message("original")];
        state.transform_context = Some(Arc::new(|mut msgs| {
            msgs.push(make_user_message("injected"));
            msgs
        }));

        let context = state.build_context(&[]);
        assert_eq!(context.len(), 2);
        assert_eq!(context[1].text(), "injected");
    }

    // ─── error with partial message ────────────────────────────────

    #[test]
    fn process_response_error_saves_partial_message() {
        let mut state = make_test_state();
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: tau_ai::Usage::default(),
            error: Some("some error".into()),
            partial_message: Some(make_assistant_message("partial content")),
        };

        let decision = state.process_response(outcome, vec![], None);

        assert!(matches!(decision.action, ResponseAction::Error(_)));
        // Partial message should be saved in conversation
        assert_eq!(state.conversation.messages.len(), 1);
        assert_eq!(state.conversation.messages[0].text(), "partial content");
    }

    #[test]
    fn process_response_overflow_with_partial_saves_then_compacts() {
        let mut state = make_test_state();
        state.config.compaction.enabled = true;
        let pending = vec![make_user_message("my prompt")];
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: tau_ai::Usage::default(),
            error: Some("context_length_exceeded".into()),
            partial_message: Some(make_assistant_message("partial")),
        };

        let decision = state.process_response(outcome, pending, None);

        match decision.action {
            ResponseAction::Compact { reason, .. } => {
                assert!(matches!(reason, CompactionReason::Overflow));
            }
            other => panic!("expected Compact, got {other:?}"),
        }
        // Pending flushed once + partial saved = 2 messages (no double-flush)
        assert_eq!(state.conversation.messages.len(), 2);
        assert_eq!(state.conversation.messages[0].role(), "user");
        assert_eq!(state.conversation.messages[1].role(), "assistant");
    }

    #[test]
    fn drain_follow_ups_decrements_pending_count() {
        let mut state = make_test_state();
        state
            .pending_follow_ups
            .store(2, std::sync::atomic::Ordering::Release);
        state.follow_up_queue.push(make_user_message("a"));
        state.follow_up_queue.push(make_user_message("b"));

        let action = state.drain_follow_ups();
        assert!(matches!(action, FollowUpAction::Continue(_)));
        assert_eq!(
            state
                .pending_follow_ups
                .load(std::sync::atomic::Ordering::Acquire),
            0
        );
    }

    // ─── handle_batch_complete ─────────────────────────────────────

    #[test]
    fn batch_complete_no_steering_all_done() {
        let mut state = make_test_state();
        let calls = vec![ToolCall {
            id: "a".into(),
            name: "echo".into(),
            args: serde_json::json!({}),
        }];
        let mut results = HashMap::new();
        results.insert(
            0,
            ("a".to_string(), "echo".to_string(), ToolResult::text("ok")),
        );

        let action = state.handle_batch_complete(&[], &calls, &mut results);
        assert!(matches!(action, BatchCompleteAction::AllGroupsDone));
    }

    #[test]
    fn batch_complete_more_groups_remaining() {
        let mut state = make_test_state();
        let calls = vec![
            ToolCall {
                id: "a".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
            ToolCall {
                id: "b".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
        ];
        let mut results = HashMap::new();
        results.insert(
            0,
            ("a".to_string(), "echo".to_string(), ToolResult::text("ok")),
        );

        let remaining = vec![vec![1]];
        let action = state.handle_batch_complete(&remaining, &calls, &mut results);
        match action {
            BatchCompleteAction::NextGroup(group) => {
                assert_eq!(group, vec![1]);
            }
            other => panic!("expected NextGroup, got {other:?}"),
        }
    }

    #[test]
    fn batch_complete_steering_redirects() {
        let mut state = make_test_state();
        state.steering_queue.push(make_user_message("redirect"));

        let calls = vec![
            ToolCall {
                id: "a".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
            ToolCall {
                id: "b".into(),
                name: "echo".into(),
                args: serde_json::json!({}),
            },
        ];
        let mut results = HashMap::new();
        results.insert(
            0,
            ("a".to_string(), "echo".to_string(), ToolResult::text("ok")),
        );
        // Group 1 (index 1) not yet executed
        let remaining = vec![vec![1]];

        let action = state.handle_batch_complete(&remaining, &calls, &mut results);
        match action {
            BatchCompleteAction::Redirect {
                steering,
                skipped_indices,
            } => {
                assert_eq!(steering.len(), 1);
                assert_eq!(steering[0].text(), "redirect");
                // Index 1 ("b") should be skipped
                assert_eq!(skipped_indices.len(), 1);
                assert_eq!(skipped_indices[0].1, "b");
            }
            other => panic!("expected Redirect, got {other:?}"),
        }

        // Tool results should be committed to conversation (2: one real, one skipped)
        let tool_results: Vec<_> = state
            .conversation
            .messages
            .iter()
            .filter(|m| matches!(m, Message::ToolResult { .. }))
            .collect();
        assert_eq!(tool_results.len(), 2);

        // Steering queue should be drained
        assert!(state.steering_queue.is_empty());
    }

    // ─── handle_response_error (split from process_response) ──────

    #[test]
    fn handle_response_error_simple_error() {
        let mut state = make_test_state();
        let outcome = StreamOutcome {
            assistant_message: None,
            usage: tau_ai::Usage::default(),
            error: Some("bad request".into()),
            partial_message: None,
        };

        let decision = state.handle_response_error(&outcome, vec![], None);

        assert!(matches!(decision.action, ResponseAction::Error(_)));
        // Pending should NOT be flushed on simple error
        assert!(state.conversation.messages.is_empty());
    }

    #[test]
    fn commit_and_decide_with_tools() {
        let mut state = make_test_state();
        let msg = make_tool_call_message("echo", "c1", serde_json::json!({"text": "x"}));
        let outcome = StreamOutcome {
            assistant_message: Some(msg),
            usage: tau_ai::Usage {
                input: 100,
                output: 50,
                ..Default::default()
            },
            error: None,
            partial_message: None,
        };
        let pending = vec![make_user_message("prompt")];

        let decision = state.commit_and_decide(outcome, pending, None);

        // Usage accumulated
        assert_eq!(state.conversation.total_usage.input, 100);
        // Messages committed: user + assistant
        assert_eq!(state.conversation.messages.len(), 2);
        // Action: run tools
        assert!(matches!(decision.action, ResponseAction::RunTools { .. }));
    }
}
