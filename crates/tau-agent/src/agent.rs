//! Agent state management and execution

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::Ordering,
};
use tau_ai::{Content, Message, Model, ReasoningLevel, Usage};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::{
    compaction::{self, CompactionConfig, CompactionReason},
    events::AgentEvent,
    tool::{BoxedTool, ToolResult, to_api_tool},
    transport::{AgentRunConfig, Transport, is_context_overflow},
};

/// Controls how messages are drained from a queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequeueMode {
    /// Drain all pending messages at once.
    All,
    /// Drain one message at a time.
    OneAtATime,
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// System prompt
    pub system_prompt: Option<String>,
    /// Model to use
    pub model: Model,
    /// Reasoning/thinking level
    pub reasoning: ReasoningLevel,
    /// Maximum tokens per response
    pub max_tokens: Option<u32>,
    /// Context compaction configuration
    pub compaction: CompactionConfig,
    /// How to drain the steering queue
    pub steering_mode: DequeueMode,
    /// How to drain the follow-up queue
    pub follow_up_mode: DequeueMode,
}

// Re-export types that were moved to their own modules so existing
// `use crate::agent::Conversation` etc. paths keep working.
pub use crate::conversation::{AgentState, Conversation};
pub use crate::handle::AgentHandle;

/// The main agent that orchestrates conversations
pub struct Agent {
    config: AgentConfig,
    conversation: Conversation,
    tools: Vec<BoxedTool>,
    transport: Arc<dyn Transport>,
    event_tx: broadcast::Sender<AgentEvent>,
    handle: AgentHandle,

    // --- transformContext hook ---
    /// Optional hook to transform context messages before sending to transport
    transform_context: Option<Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>>,

    // --- Schema validator cache ---
    /// Cached compiled JSON schema validators keyed by tool name
    schema_cache: HashMap<String, Arc<jsonschema::Validator>>,
}

impl Agent {
    /// Create a new agent
    pub fn new(config: AgentConfig, transport: Arc<dyn Transport>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            config,
            conversation: Conversation::default(),
            tools: vec![],
            transport,
            event_tx,
            handle: AgentHandle::new(),
            transform_context: None,
            schema_cache: HashMap::new(),
        }
    }

    /// Subscribe to agent events
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Get the current state
    pub fn state(&self) -> &Conversation {
        &self.conversation
    }

    /// Get the agent config
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Set the system prompt
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.config.system_prompt = Some(prompt.into());
    }

    /// Set the model
    pub fn set_model(&mut self, model: Model) {
        self.config.model = model;
    }

    /// Set the reasoning level
    pub fn set_reasoning(&mut self, level: ReasoningLevel) {
        self.config.reasoning = level;
    }

    /// Set compaction configuration
    pub fn set_compaction_config(&mut self, config: CompactionConfig) {
        self.config.compaction = config;
    }

    /// Add a tool
    pub fn add_tool(&mut self, tool: BoxedTool) {
        self.cache_tool_schema(&tool);
        self.tools.push(tool);
    }

    /// Set tools (replaces existing)
    pub fn set_tools(&mut self, tools: Vec<BoxedTool>) {
        self.schema_cache.clear();
        for tool in &tools {
            self.cache_tool_schema(tool);
        }
        self.tools = tools;
    }

    /// Compile and cache the JSON schema validator for a tool.
    fn cache_tool_schema(&mut self, tool: &BoxedTool) {
        let schema = tool.parameters_schema();
        match jsonschema::validator_for(&schema) {
            Ok(validator) => {
                self.schema_cache
                    .insert(tool.name().to_string(), Arc::new(validator));
            }
            Err(e) => {
                tracing::warn!(
                    "Invalid tool parameter schema for '{}', skipping validation: {}",
                    tool.name(),
                    e
                );
            }
        }
    }

    /// Get tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Clear all messages
    pub fn clear_messages(&mut self) {
        self.conversation.messages.clear();
        self.conversation.total_usage = Usage::default();
        self.conversation.error = None;
        self.conversation.previous_summary = None;
    }

    /// Set messages (for loading from session)
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.conversation.messages = messages;
    }

    /// Set the previous summary (for session resume after compaction)
    pub fn set_previous_summary(&mut self, summary: Option<String>) {
        self.conversation.previous_summary = summary;
    }

    /// Get all messages
    pub fn messages(&self) -> &[Message] {
        &self.conversation.messages
    }

    /// Get a cloneable handle for poking the agent from external code.
    pub fn handle(&self) -> AgentHandle {
        self.handle.clone()
    }

    /// Abort the current operation
    pub fn abort(&self) {
        self.handle.abort();
    }

    /// Get a handle to cancel the current operation from outside
    /// Returns an Arc that can be used to cancel even while a prompt is running
    pub fn cancel_handle(&self) -> Arc<Mutex<CancellationToken>> {
        self.handle.cancel_token()
    }

    /// Enqueue a steering message that interrupts after the current tool completes.
    pub fn steer(&self, message: Message) {
        self.handle.steer(message);
    }

    /// Enqueue a follow-up message consumed after the loop finishes.
    pub fn follow_up(&self, message: Message) {
        self.handle.follow_up(message);
    }

    /// Wait until the agent loop becomes idle (finishes running).
    pub async fn wait_for_idle(&self) {
        self.handle.wait_for_idle().await;
    }

    /// Whether the agent loop is currently running.
    pub fn is_running(&self) -> bool {
        self.handle.is_running()
    }

    /// Set the steering queue dequeue mode.
    pub fn set_steering_mode(&mut self, mode: DequeueMode) {
        self.config.steering_mode = mode;
    }

    /// Set the follow-up queue dequeue mode.
    pub fn set_follow_up_mode(&mut self, mode: DequeueMode) {
        self.config.follow_up_mode = mode;
    }

    /// Set a transform_context hook called before sending messages to transport.
    pub fn set_transform_context(
        &mut self,
        f: impl Fn(Vec<Message>) -> Vec<Message> + Send + Sync + 'static,
    ) {
        self.transform_context = Some(Arc::new(f));
    }

    /// Remove the transform_context hook.
    pub fn clear_transform_context(&mut self) {
        self.transform_context = None;
    }

    /// Send a message and run the agent loop
    pub async fn prompt(&mut self, input: &str) -> crate::error::Result<()> {
        self.prompt_with_content(vec![Content::text(input)]).await
    }

    /// Run compaction on the current conversation
    pub async fn run_compaction(&mut self, reason: CompactionReason) -> crate::error::Result<()> {
        let _ = self.event_tx.send(AgentEvent::CompactionStart { reason });

        let tokens_before = compaction::estimate_total_tokens(&self.conversation.messages);

        let result = compaction::compact(
            &self.conversation.messages,
            &self.config.compaction,
            &self.config,
            &self.transport,
            self.conversation.previous_summary.as_deref(),
        )
        .await
        .map_err(crate::error::Error::Compaction)?;

        // Replace messages: [summary as User message] + [kept messages]
        let summary_msg = Message::user(format!(
            "<context-summary>\n{}\n</context-summary>",
            result.summary
        ));
        let kept = self.conversation.messages[result.first_kept_index..].to_vec();
        self.conversation.messages = vec![summary_msg];
        self.conversation.messages.extend(kept);
        self.conversation.previous_summary = Some(result.summary);

        let tokens_after = compaction::estimate_total_tokens(&self.conversation.messages);
        let _ = self.event_tx.send(AgentEvent::CompactionEnd {
            tokens_before,
            tokens_after,
        });

        Ok(())
    }

    /// Send a message with multiple content blocks
    pub async fn prompt_with_content(&mut self, content: Vec<Content>) -> crate::error::Result<()> {
        // Create user message
        let user_message = Message::User {
            content,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        self.run_with_messages(vec![user_message]).await
    }

    /// Re-enter the agent loop, draining steering then follow-up queues.
    pub async fn continue_loop(&mut self) -> crate::error::Result<()> {
        // Drain steering queue first, then follow-up
        let mut messages = self.drain_queue(&self.handle.steering_queue, self.config.steering_mode);
        if messages.is_empty() {
            messages = self.drain_queue(&self.handle.follow_up_queue, self.config.follow_up_mode);
        }
        if messages.is_empty() {
            return Ok(());
        }
        self.run_with_messages(messages).await
    }

    /// Drain messages from a queue according to the given mode.
    fn drain_queue(&self, queue: &Arc<Mutex<Vec<Message>>>, mode: DequeueMode) -> Vec<Message> {
        let mut q = queue.lock();
        match mode {
            DequeueMode::All => q.drain(..).collect(),
            DequeueMode::OneAtATime => {
                if q.is_empty() {
                    vec![]
                } else {
                    vec![q.remove(0)]
                }
            }
        }
    }

    /// Skip remaining tool calls by emitting start/end events and producing error results.
    fn skip_remaining_tools(
        &self,
        tool_calls: &[(String, String, serde_json::Value)],
        tool_results: &mut Vec<Message>,
    ) {
        for (skip_id, skip_name, _) in tool_calls {
            let _ = self.event_tx.send(AgentEvent::ToolExecutionStart {
                tool_call_id: skip_id.clone(),
                tool_name: skip_name.clone(),
                arguments: serde_json::Value::Null,
            });
            let skip_result = ToolResult::error("Skipped due to steering message");
            let _ = self.event_tx.send(AgentEvent::ToolExecutionEnd {
                tool_call_id: skip_id.clone(),
                tool_name: skip_name.clone(),
                result: skip_result.text_content(),
                is_error: skip_result.is_error,
            });
            tool_results.push(Message::tool_result(
                skip_id,
                skip_name,
                skip_result.content,
                skip_result.is_error,
            ));
        }
    }

    // ---- Private helpers extracted from run_with_messages ----

    /// Build the run config from current agent state.
    fn build_run_config(&self) -> AgentRunConfig {
        AgentRunConfig {
            system_prompt: self.config.system_prompt.clone(),
            tools: self.tools.iter().map(|t| to_api_tool(t.as_ref())).collect(),
            model: self.config.model.clone(),
            reasoning: Some(self.config.reasoning),
            max_tokens: self.config.max_tokens,
            temperature: None,
        }
    }

    /// Assemble context messages from conversation history + pending, applying transformContext hook.
    fn build_context(&self, pending: &[Message]) -> Vec<Message> {
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

    /// Process the event stream, forwarding events to subscribers.
    /// Returns (assistant_message, turn_usage, error_if_any).
    async fn process_stream(
        &mut self,
        event_stream: &mut crate::transport::AgentEventStream,
    ) -> (Option<Message>, Usage, Option<String>) {
        use futures::StreamExt;

        let mut assistant_message: Option<Message> = None;
        let mut turn_usage = Usage::default();
        let mut error: Option<String> = None;

        while let Some(event) = event_stream.next().await {
            let _ = self.event_tx.send(event.clone());

            match event {
                AgentEvent::MessageUpdate { message } => {
                    self.conversation.stream_message = Some(message);
                }
                AgentEvent::MessageEnd { message } => {
                    self.conversation.stream_message = None;
                    assistant_message = Some(message);
                }
                AgentEvent::TurnEnd { usage, .. } => {
                    turn_usage = usage;
                }
                AgentEvent::Error { message } => {
                    error = Some(message);
                }
                _ => {}
            }
        }

        (assistant_message, turn_usage, error)
    }

    /// Add turn usage to cumulative totals.
    fn accumulate_usage(&mut self, turn_usage: &Usage) {
        self.conversation.total_usage.input += turn_usage.input;
        self.conversation.total_usage.output += turn_usage.output;
        self.conversation.total_usage.cache_read += turn_usage.cache_read;
        self.conversation.total_usage.cache_write += turn_usage.cache_write;
        self.conversation.total_usage.thinking += turn_usage.thinking;
    }

    /// Attempt overflow recovery via compaction. Returns `true` if recovery succeeded
    /// and the caller should `continue` the loop.
    async fn try_overflow_recovery(
        &mut self,
        error: &str,
        messages_to_add: &mut Vec<Message>,
        first_user_message: &Option<Message>,
        turn: &mut u32,
    ) -> bool {
        if !self.config.compaction.enabled || !is_context_overflow(error) {
            return false;
        }
        // Flush pending messages into conversation before compacting
        for m in messages_to_add.drain(..) {
            self.conversation.messages.push(m);
        }
        if self.run_compaction(CompactionReason::Overflow).await.is_ok() {
            if let Some(msg) = first_user_message {
                *messages_to_add = vec![msg.clone()];
            }
            *turn = 0;
            return true;
        }
        false
    }

    /// Execute tool calls, checking the steering queue between each.
    /// Returns (tool_result_messages, was_steered).
    async fn execute_tool_calls(
        &self,
        tool_calls: Vec<(String, String, serde_json::Value)>,
    ) -> (Vec<Message>, bool) {
        let mut tool_results = vec![];
        let mut steered = false;

        for idx in 0..tool_calls.len() {
            let (ref id, ref name, ref args) = tool_calls[idx];

            // Check steering queue before executing (except for the first tool)
            if idx > 0 {
                let steering_msgs =
                    self.drain_queue(&self.handle.steering_queue, self.config.steering_mode);
                if !steering_msgs.is_empty() {
                    self.skip_remaining_tools(&tool_calls[idx..], &mut tool_results);
                    tool_results.extend(steering_msgs);
                    steered = true;
                    break;
                }
            }

            let tool = self.tools.iter().find(|t| t.name() == name.as_str());

            let _ = self.event_tx.send(AgentEvent::ToolExecutionStart {
                tool_call_id: id.clone(),
                tool_name: name.clone(),
                arguments: args.clone(),
            });

            let result = if let Some(tool) = tool {
                let validation_error = self
                    .schema_cache
                    .get(name.as_str())
                    .and_then(|validator| validate_with_validator(args, validator));

                if let Some(err) = validation_error {
                    ToolResult::error(err)
                } else {
                    let cancel = self.handle.cancel.lock().clone();
                    let progress = crate::tool::ProgressSender::new(
                        self.event_tx.clone(),
                        id.clone(),
                        name.clone(),
                    );
                    tool.execute_with_progress(id, args.clone(), cancel, progress)
                        .await
                }
            } else {
                ToolResult::error(format!("Tool not found: {}", name))
            };

            let _ = self.event_tx.send(AgentEvent::ToolExecutionEnd {
                tool_call_id: id.clone(),
                tool_name: name.clone(),
                result: result.text_content(),
                is_error: result.is_error,
            });

            tool_results.push(Message::tool_result(id, name, result.content, result.is_error));

            // Check steering queue after each tool
            let steering_msgs =
                self.drain_queue(&self.handle.steering_queue, self.config.steering_mode);
            if !steering_msgs.is_empty() {
                self.skip_remaining_tools(&tool_calls[idx + 1..], &mut tool_results);
                tool_results.extend(steering_msgs);
                steered = true;
                break;
            }
        }

        (tool_results, steered)
    }

    /// If input tokens are approaching the context window, compact proactively.
    async fn check_compaction_threshold(
        &mut self,
        turn_usage: &Usage,
        messages_to_add: &mut Vec<Message>,
    ) {
        if !self.config.compaction.enabled {
            return;
        }
        let used = turn_usage.input + turn_usage.cache_read;
        let limit = self
            .config
            .model
            .context_window
            .saturating_sub(self.config.compaction.reserve_tokens);
        if used > limit {
            for m in messages_to_add.drain(..) {
                self.conversation.messages.push(m);
            }
            let _ = self.run_compaction(CompactionReason::Threshold).await;
        }
    }

    /// Flush pending messages into the conversation.
    fn flush_pending(&mut self, messages_to_add: &mut Vec<Message>) {
        for m in messages_to_add.drain(..) {
            self.conversation.messages.push(m);
        }
    }

    /// Core agent loop, shared between prompt_with_content and continue_loop.
    async fn run_with_messages(&mut self, initial_messages: Vec<Message>) -> crate::error::Result<()> {
        // Reset cancellation token
        *self.handle.cancel.lock() = CancellationToken::new();
        self.handle.is_running.store(true, Ordering::Release);

        let run_config = self.build_run_config();
        self.conversation.is_streaming = true;
        self.conversation.error = None;
        let _ = self.event_tx.send(AgentEvent::AgentStart);

        let mut turn = 0u32;
        let mut messages_to_add: Vec<Message> = initial_messages;
        let first_user_message = messages_to_add.first().cloned();

        let result = loop {
            turn += 1;

            // Build context and user message for this turn
            let context_messages = self.build_context(&messages_to_add);
            let current_user_msg = if turn == 1 {
                messages_to_add
                    .first()
                    .cloned()
                    .unwrap_or_else(|| Message::User {
                        content: vec![],
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    })
            } else {
                Message::User {
                    content: vec![],
                    timestamp: chrono::Utc::now().timestamp_millis(),
                }
            };

            // Run the transport
            let cancel_token = self.handle.cancel.lock().clone();
            let mut event_stream = match self
                .transport
                .run(context_messages, current_user_msg, &run_config, cancel_token)
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    let error_msg = e.to_string();
                    let overflow = e.is_context_overflow() || is_context_overflow(&error_msg);
                    if overflow
                        && self
                            .try_overflow_recovery(
                                &error_msg,
                                &mut messages_to_add,
                                &first_user_message,
                                &mut turn,
                            )
                            .await
                    {
                        continue;
                    }
                    self.conversation.error = Some(error_msg.clone());
                    let _ = self.event_tx.send(AgentEvent::Error {
                        message: error_msg.clone(),
                    });
                    break Err(crate::error::Error::Other(error_msg));
                }
            };

            // Process the event stream
            let (assistant_message, turn_usage, stream_error) =
                self.process_stream(&mut event_stream).await;

            // Handle streaming errors with overflow recovery
            if let Some(error_message) = stream_error {
                if let Some(partial) = self.conversation.stream_message.take() {
                    if has_meaningful_content(&partial) {
                        self.flush_pending(&mut messages_to_add);
                        self.conversation.messages.push(partial);
                    }
                }
                if self
                    .try_overflow_recovery(
                        &error_message,
                        &mut messages_to_add,
                        &first_user_message,
                        &mut turn,
                    )
                    .await
                {
                    continue;
                }
                self.conversation.error = Some(error_message.clone());
                break Err(crate::error::Error::Other(error_message));
            }

            self.accumulate_usage(&turn_usage);
            self.check_compaction_threshold(&turn_usage, &mut messages_to_add).await;

            // Process assistant message
            if let Some(msg) = assistant_message {
                self.flush_pending(&mut messages_to_add);
                self.conversation.messages.push(msg.clone());

                let tool_calls = msg.tool_calls();
                if tool_calls.is_empty() {
                    let follow_ups =
                        self.drain_queue(&self.handle.follow_up_queue, self.config.follow_up_mode);
                    if !follow_ups.is_empty() {
                        messages_to_add = follow_ups;
                        continue;
                    }
                    break Ok(());
                }

                // Convert to owned types and execute
                let tool_calls_vec: Vec<(String, String, serde_json::Value)> = tool_calls
                    .into_iter()
                    .map(|(id, name, args)| (id.to_string(), name.to_string(), args.clone()))
                    .collect();

                let (tool_results, steered) = self.execute_tool_calls(tool_calls_vec).await;
                messages_to_add = tool_results;
                if steered {
                    continue;
                }
            } else {
                break Ok(());
            }
        };

        self.conversation.is_streaming = false;

        // Proactive compaction check
        if self.config.compaction.enabled {
            let last_input = self.conversation.total_usage.input;
            let limit = self
                .config
                .model
                .context_window
                .saturating_sub(self.config.compaction.reserve_tokens);
            if last_input > limit {
                let _ = self.run_compaction(CompactionReason::Threshold).await;
            }
        }

        let _ = self.event_tx.send(AgentEvent::AgentEnd {
            total_turns: turn,
            total_usage: self.conversation.total_usage.clone(),
        });

        self.handle.is_running.store(false, Ordering::Release);
        self.handle.idle_notify.notify_waiters();

        result
    }
}

/// Check if a message has meaningful content worth preserving.
/// Returns true if the message contains non-whitespace text, thinking blocks,
/// or tool calls with a name.
fn has_meaningful_content(message: &Message) -> bool {
    let content = match message {
        Message::Assistant { content, .. } => content,
        Message::User { content, .. } => content,
        Message::ToolResult { content, .. } => content,
    };

    content.iter().any(|c| match c {
        Content::Text { text } => !text.trim().is_empty(),
        Content::Thinking { thinking } => !thinking.trim().is_empty(),
        Content::ToolCall { name, .. } => !name.is_empty(),
        Content::Image { .. } => true,
    })
}

/// Validate tool arguments using a pre-compiled validator.
/// Returns `Some(error_message)` if validation fails, `None` if valid.
fn validate_with_validator(
    args: &serde_json::Value,
    validator: &jsonschema::Validator,
) -> Option<String> {
    let errors: Vec<String> = validator
        .iter_errors(args)
        .map(|e| {
            let path = e.instance_path.to_string();
            if path.is_empty() {
                e.to_string()
            } else {
                format!("{}: {}", path, e)
            }
        })
        .collect();

    if errors.is_empty() {
        None
    } else {
        Some(format!(
            "Tool argument validation failed:\n{}",
            errors.join("\n")
        ))
    }
}

/// Validate tool arguments against a JSON Schema (compiles on each call).
/// Used in tests. Returns `Some(error_message)` if validation fails, `None` if valid.
#[cfg(test)]
fn validate_tool_args(args: &serde_json::Value, schema: &serde_json::Value) -> Option<String> {
    let validator = match jsonschema::validator_for(schema) {
        Ok(v) => v,
        Err(_) => return None,
    };
    validate_with_validator(args, &validator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::AgentEvent;
    use crate::transport::{AgentEventStream, AgentRunConfig, Transport};
    use async_trait::async_trait;
    use std::sync::Arc;
    use tau_ai::{AssistantMetadata, Content, Message};

    // ===== Item 2: validate_tool_args tests =====

    fn simple_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "count": { "type": "integer" }
            },
            "required": ["path"]
        })
    }

    #[test]
    fn test_validate_args_valid() {
        let args = serde_json::json!({"path": "/foo.rs", "count": 10});
        assert!(validate_tool_args(&args, &simple_schema()).is_none());
    }

    #[test]
    fn test_validate_args_valid_optional_missing() {
        let args = serde_json::json!({"path": "/foo.rs"});
        assert!(validate_tool_args(&args, &simple_schema()).is_none());
    }

    #[test]
    fn test_validate_args_missing_required() {
        let args = serde_json::json!({"count": 5});
        let err = validate_tool_args(&args, &simple_schema());
        assert!(err.is_some());
        let msg = err.unwrap();
        assert!(msg.contains("validation failed"), "got: {}", msg);
        assert!(msg.contains("path"), "should mention missing field, got: {}", msg);
    }

    #[test]
    fn test_validate_args_wrong_type() {
        let args = serde_json::json!({"path": 123});
        let err = validate_tool_args(&args, &simple_schema());
        assert!(err.is_some());
        let msg = err.unwrap();
        assert!(msg.contains("validation failed"), "got: {}", msg);
    }

    #[test]
    fn test_validate_args_invalid_schema_returns_none() {
        // A schema that jsonschema can't compile — invalid type value
        let bad_schema = serde_json::json!({"type": "not_a_real_type"});
        let args = serde_json::json!({"anything": true});
        // Should log a warning and return None (skip validation)
        assert!(validate_tool_args(&args, &bad_schema).is_none());
    }

    #[test]
    fn test_validate_args_empty_object_valid() {
        // Schema with no required fields
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "optional": { "type": "string" }
            }
        });
        let args = serde_json::json!({});
        assert!(validate_tool_args(&args, &schema).is_none());
    }

    // ===== Item 3: has_meaningful_content tests =====

    #[test]
    fn test_meaningful_content_text() {
        let msg = Message::Assistant {
            content: vec![Content::text("hello")],
            metadata: AssistantMetadata::default(),
        };
        assert!(has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_whitespace_only() {
        let msg = Message::Assistant {
            content: vec![Content::text("   \n\t  ")],
            metadata: AssistantMetadata::default(),
        };
        assert!(!has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_empty() {
        let msg = Message::Assistant {
            content: vec![],
            metadata: AssistantMetadata::default(),
        };
        assert!(!has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_thinking() {
        let msg = Message::Assistant {
            content: vec![Content::thinking("let me think...")],
            metadata: AssistantMetadata::default(),
        };
        assert!(has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_tool_call() {
        let msg = Message::Assistant {
            content: vec![Content::tool_call("id1", "read", serde_json::json!({}))],
            metadata: AssistantMetadata::default(),
        };
        assert!(has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_tool_call_empty_name() {
        let msg = Message::Assistant {
            content: vec![Content::tool_call("id1", "", serde_json::json!({}))],
            metadata: AssistantMetadata::default(),
        };
        assert!(!has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_mixed_empty_and_real() {
        let msg = Message::Assistant {
            content: vec![Content::text(""), Content::text("real content")],
            metadata: AssistantMetadata::default(),
        };
        assert!(has_meaningful_content(&msg));
    }

    // ===== Item 5: Steering & Follow-up queue tests =====

    /// A mock transport that returns a canned assistant response.
    struct MockTransport {
        /// Messages the assistant will respond with.
        responses: Arc<Mutex<Vec<Message>>>,
    }

    impl MockTransport {
        fn new(responses: Vec<Message>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
            }
        }
    }

    #[async_trait]
    impl Transport for MockTransport {
        async fn run(
            &self,
            _messages: Vec<Message>,
            _user_message: Message,
            _config: &AgentRunConfig,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let msg = {
                let mut responses = self.responses.lock();
                if responses.is_empty() {
                    Message::Assistant {
                        content: vec![Content::text("done")],
                        metadata: AssistantMetadata::default(),
                    }
                } else {
                    responses.remove(0)
                }
            };

            let usage = tau_ai::Usage::default();

            let stream: AgentEventStream = Box::pin(async_stream::stream! {
                yield AgentEvent::TurnStart { turn_number: 1 };
                yield AgentEvent::MessageEnd { message: msg.clone() };
                yield AgentEvent::TurnEnd {
                    turn_number: 1,
                    message: msg,
                    usage,
                };
            });

            Ok(stream)
        }
    }

    fn make_test_agent(responses: Vec<Message>) -> Agent {
        let transport = Arc::new(MockTransport::new(responses));
        let config = AgentConfig {
            system_prompt: Some("test".into()),
            model: tau_ai::Model {
                id: "test".into(),
                name: "test".into(),
                api: tau_ai::Api::AnthropicMessages,
                provider: tau_ai::Provider::Anthropic,
                base_url: "http://localhost".into(),
                reasoning: false,
                input_types: vec![],
                cost: tau_ai::CostInfo::default(),
                context_window: 200000,
                max_tokens: 4096,
                headers: Default::default(),
            },
            reasoning: tau_ai::ReasoningLevel::Off,
            max_tokens: None,
            compaction: CompactionConfig::default(),
            steering_mode: DequeueMode::All,
            follow_up_mode: DequeueMode::All,
        };
        Agent::new(config, transport)
    }

    #[tokio::test]
    async fn test_follow_up_continues_loop() {
        // First response has no tool calls -> normally would end.
        // But we enqueue a follow-up before calling prompt.
        let responses = vec![
            // Response to initial prompt
            Message::Assistant {
                content: vec![Content::text("first response")],
                metadata: AssistantMetadata::default(),
            },
            // Response to follow-up
            Message::Assistant {
                content: vec![Content::text("second response")],
                metadata: AssistantMetadata::default(),
            },
        ];

        let mut agent = make_test_agent(responses);

        // Enqueue a follow-up message before starting
        agent.follow_up(Message::user("follow-up question"));

        agent.prompt("initial prompt").await.unwrap();

        // Both responses should be in messages:
        // [user, assistant("first response"), user("follow-up question"), assistant("second response")]
        let msgs = agent.messages();
        assert!(msgs.len() >= 4, "expected at least 4 messages, got {}", msgs.len());

        let texts: Vec<String> = msgs.iter().map(|m| m.text()).collect();
        assert!(texts.iter().any(|t| t.contains("first response")));
        assert!(texts.iter().any(|t| t.contains("second response")));
        assert!(texts.iter().any(|t| t.contains("follow-up question")));
    }

    #[tokio::test]
    async fn test_is_running_and_idle() {
        let responses = vec![Message::Assistant {
            content: vec![Content::text("done")],
            metadata: AssistantMetadata::default(),
        }];
        let mut agent = make_test_agent(responses);

        assert!(!agent.is_running());
        agent.prompt("hello").await.unwrap();
        // After prompt returns, should be idle
        assert!(!agent.is_running());
    }

    #[tokio::test]
    async fn test_dequeue_mode_one_at_a_time() {
        let responses = vec![
            Message::Assistant {
                content: vec![Content::text("r1")],
                metadata: AssistantMetadata::default(),
            },
            Message::Assistant {
                content: vec![Content::text("r2")],
                metadata: AssistantMetadata::default(),
            },
        ];
        let mut agent = make_test_agent(responses);
        agent.set_follow_up_mode(DequeueMode::OneAtATime);

        // Enqueue two follow-ups
        agent.follow_up(Message::user("fu1"));
        agent.follow_up(Message::user("fu2"));

        agent.prompt("start").await.unwrap();

        // With OneAtATime, only one follow-up should be consumed per loop run.
        // First prompt: initial -> r1, then drains "fu1" -> r2, then drains "fu2"
        // but there's no third response so it just gets "done" (default).
        let msgs = agent.messages();
        let texts: Vec<String> = msgs.iter().map(|m| m.text()).collect();
        assert!(texts.iter().any(|t| t.contains("fu1")));
    }

    // ===== Steering with actual tool calls =====

    /// A simple no-op tool for testing
    struct NoopTool {
        tool_name: String,
        /// Count how many times execute was called
        call_count: Arc<std::sync::atomic::AtomicU32>,
    }

    impl NoopTool {
        fn new(name: &str) -> (Self, Arc<std::sync::atomic::AtomicU32>) {
            let count = Arc::new(std::sync::atomic::AtomicU32::new(0));
            (
                Self {
                    tool_name: name.to_string(),
                    call_count: count.clone(),
                },
                count,
            )
        }
    }

    #[async_trait]
    impl crate::tool::Tool for NoopTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            "A no-op tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            _arguments: serde_json::Value,
            _cancel: CancellationToken,
        ) -> ToolResult {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ToolResult::text("ok")
        }
    }

    #[tokio::test]
    async fn test_steer_skips_remaining_tool_calls() {
        // The assistant responds with two tool calls: tool_a and tool_b.
        // We steer after tool_a executes, so tool_b should be skipped.
        let responses = vec![
            // First response: two tool calls
            Message::Assistant {
                content: vec![
                    Content::tool_call("call_a", "tool_a", serde_json::json!({})),
                    Content::tool_call("call_b", "tool_b", serde_json::json!({})),
                ],
                metadata: AssistantMetadata::default(),
            },
            // Response after steering (to the tool results + steering message)
            Message::Assistant {
                content: vec![Content::text("steered response")],
                metadata: AssistantMetadata::default(),
            },
        ];

        let mut agent = make_test_agent(responses);

        let (tool_a, count_a) = NoopTool::new("tool_a");
        let (tool_b, count_b) = NoopTool::new("tool_b");
        agent.add_tool(Arc::new(tool_a));
        agent.add_tool(Arc::new(tool_b));

        // We need to steer after tool_a runs. Since we can't inject mid-loop
        // from outside in a single-threaded test, we'll pre-fill the steering
        // queue. The check happens after each tool, so steer() before prompt
        // means it will be picked up after tool_a finishes.
        agent.steer(Message::user("stop and do this instead"));

        agent.prompt("run both tools").await.unwrap();

        // tool_a should have been called once
        assert_eq!(
            count_a.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "tool_a should have been executed"
        );
        // tool_b should NOT have been called
        assert_eq!(
            count_b.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "tool_b should have been skipped by steering"
        );

        // Verify the steering message and steered response are in the conversation
        let texts: Vec<String> = agent.messages().iter().map(|m| m.text()).collect();
        assert!(
            texts.iter().any(|t| t.contains("stop and do this instead")),
            "steering message should be in conversation"
        );
        assert!(
            texts.iter().any(|t| t.contains("steered response")),
            "response to steering should be in conversation"
        );
    }

    #[tokio::test]
    async fn test_steer_before_first_tool_skips_all() {
        // The assistant responds with two tool calls.
        // Steering check before idx>0 tools means: if we have 2 tools and steer
        // is queued, tool_a runs (idx=0, no check), then before tool_b (idx=1)
        // steering is found and tool_b is skipped.
        // But what if we want to test the idx>0 pre-check path?
        // With 3 tools: tool_a runs, then before tool_b the queue is checked.
        let responses = vec![
            Message::Assistant {
                content: vec![
                    Content::tool_call("c1", "tool_a", serde_json::json!({})),
                    Content::tool_call("c2", "tool_b", serde_json::json!({})),
                    Content::tool_call("c3", "tool_c", serde_json::json!({})),
                ],
                metadata: AssistantMetadata::default(),
            },
            Message::Assistant {
                content: vec![Content::text("done after steer")],
                metadata: AssistantMetadata::default(),
            },
        ];

        let mut agent = make_test_agent(responses);

        let (tool_a, count_a) = NoopTool::new("tool_a");
        let (tool_b, count_b) = NoopTool::new("tool_b");
        let (tool_c, count_c) = NoopTool::new("tool_c");
        agent.add_tool(Arc::new(tool_a));
        agent.add_tool(Arc::new(tool_b));
        agent.add_tool(Arc::new(tool_c));

        // Pre-fill steering queue
        agent.steer(Message::user("interrupt"));

        agent.prompt("go").await.unwrap();

        // tool_a executes (idx=0), then steering is found after tool_a,
        // so tool_b and tool_c are skipped
        assert_eq!(count_a.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(count_b.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(count_c.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    // ===== Item 6: transformContext tests =====

    #[tokio::test]
    async fn test_transform_context_is_called() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let responses = vec![Message::Assistant {
            content: vec![Content::text("hi")],
            metadata: AssistantMetadata::default(),
        }];
        let mut agent = make_test_agent(responses);

        agent.set_transform_context(move |msgs| {
            called_clone.store(true, Ordering::Release);
            msgs
        });

        agent.prompt("hello").await.unwrap();
        assert!(called.load(Ordering::Acquire), "transform_context hook should have been called");
    }

    #[tokio::test]
    async fn test_transform_context_modifies_messages() {
        // We can verify the hook runs by injecting a system message.
        // The mock transport doesn't inspect messages, but we can verify the hook is wired in.
        let injected = Arc::new(Mutex::new(false));
        let injected_clone = injected.clone();

        let responses = vec![Message::Assistant {
            content: vec![Content::text("ok")],
            metadata: AssistantMetadata::default(),
        }];
        let mut agent = make_test_agent(responses);

        agent.set_transform_context(move |mut msgs| {
            // Inject an extra user message
            msgs.push(Message::user("injected"));
            *injected_clone.lock() = true;
            msgs
        });

        agent.prompt("test").await.unwrap();
        assert!(*injected.lock(), "transform hook should have modified messages");
    }

    #[tokio::test]
    async fn test_clear_transform_context() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let call_count = Arc::new(AtomicU32::new(0));
        let count_clone = call_count.clone();

        let responses = vec![
            Message::Assistant {
                content: vec![Content::text("r1")],
                metadata: AssistantMetadata::default(),
            },
            Message::Assistant {
                content: vec![Content::text("r2")],
                metadata: AssistantMetadata::default(),
            },
        ];
        let mut agent = make_test_agent(responses);

        agent.set_transform_context(move |msgs| {
            count_clone.fetch_add(1, Ordering::Relaxed);
            msgs
        });

        agent.prompt("first").await.unwrap();
        let count_after_first = call_count.load(Ordering::Relaxed);
        assert!(count_after_first > 0);

        // Clear the hook
        agent.clear_transform_context();

        // Second prompt should NOT increment counter
        agent.prompt("second").await.unwrap();
        let count_after_second = call_count.load(Ordering::Relaxed);
        assert_eq!(count_after_first, count_after_second, "hook should not be called after clear");
    }
}
