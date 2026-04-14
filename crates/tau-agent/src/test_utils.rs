//! Test utilities for tau-agent and downstream crates.
//!
//! Available within tau-agent via `#[cfg(test)]`, and to downstream crates
//! via the `test-utils` feature flag:
//!
//! ```toml
//! [dev-dependencies]
//! tau-agent = { workspace = true, features = ["test-utils"] }
//! ```

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use parking_lot::Mutex as ParkingMutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::config::{AgentConfig, DequeueMode};
use crate::events::AgentEvent;
use crate::handle::AgentHandle;
use crate::tool::{
    BoxedTool, Concurrency, ExecutionContext, FileAccessTracker, ProgressSender, Tool, ToolResult,
};
use crate::transport::{AgentEventStream, AgentRunConfig, Transport};

// ---------------------------------------------------------------------------
// Message constructors
// ---------------------------------------------------------------------------

/// Create a user message with text content.
pub fn make_user_message(text: &str) -> tau_ai::Message {
    tau_ai::Message::user(text)
}

/// Create an assistant message with text content.
pub fn make_assistant_message(text: &str) -> tau_ai::Message {
    tau_ai::Message::Assistant {
        content: vec![tau_ai::Content::text(text)],
        metadata: tau_ai::AssistantMetadata::default(),
    }
}

/// Create an assistant message containing a tool call.
pub fn make_tool_call_message(name: &str, id: &str, args: serde_json::Value) -> tau_ai::Message {
    tau_ai::Message::Assistant {
        content: vec![tau_ai::Content::tool_call(id, name, args)],
        metadata: tau_ai::AssistantMetadata::default(),
    }
}

/// Create a tool result message.
pub fn make_tool_result(id: &str, name: &str, text: &str) -> tau_ai::Message {
    tau_ai::Message::tool_result(id, name, vec![tau_ai::Content::text(text)], false)
}

// ---------------------------------------------------------------------------
// Test model & config
// ---------------------------------------------------------------------------

/// Create a minimal Model suitable for tests. Does not hit any real API.
pub fn make_test_model() -> tau_ai::Model {
    tau_ai::Model {
        id: "test-model".to_string(),
        name: "Test Model".to_string(),
        api: tau_ai::Api::AnthropicMessages,
        provider: tau_ai::Provider::Anthropic,
        base_url: "https://test.invalid".to_string(),
        reasoning: false,
        input_types: vec![tau_ai::InputType::Text],
        cost: tau_ai::CostInfo::default(),
        context_window: 200_000,
        max_tokens: 8192,
        headers: Default::default(),
    }
}

/// Create a test AgentConfig with a system prompt set. Used by integration tests.
pub fn test_config() -> AgentConfig {
    let mut cfg = make_test_config();
    cfg.system_prompt = Some("You are a test agent.".into());
    cfg
}

/// Create a minimal AgentConfig suitable for tests.
pub fn make_test_config() -> AgentConfig {
    AgentConfig {
        system_prompt: None,
        model: make_test_model(),
        reasoning: tau_ai::ReasoningLevel::Off,
        thinking_adaptive: false,
        max_tokens: None,
        max_turns: None,
        compaction: crate::compaction::CompactionConfig::default(),
        steering_mode: DequeueMode::All,
        follow_up_mode: DequeueMode::All,
        cache_scope: None,
        cache_ttl: None,
        system_prompt_boundary: None,
    }
}

/// Create an `ExecutionContext` for testing tools in isolation.
pub fn make_execution_context() -> ExecutionContext {
    let (tx, _rx) = tokio::sync::broadcast::channel(256);
    ExecutionContext {
        cwd: PathBuf::from("/tmp"),
        cancel: CancellationToken::new(),
        progress: ProgressSender::new(tx, "test_call", "test_tool"),
        interaction: None,
        file_access: Arc::new(ParkingMutex::new(FileAccessTracker::default())),
    }
}

// ---------------------------------------------------------------------------
// MockTransport
// ---------------------------------------------------------------------------

/// A transport that returns canned responses. Each call to `run()` pops the
/// next queued response and streams it as an `AgentEventStream`.
///
/// Panics if `run()` is called after all queued responses have been consumed.
///
/// # Example
///
/// ```ignore
/// let transport = MockTransport::new()
///     .with_text_response("Hello!")
///     .with_text_response("Goodbye!");
///
/// // First run() returns "Hello!", second returns "Goodbye!"
/// // Third run() would panic.
/// ```
pub struct MockTransport {
    responses: Mutex<VecDeque<Vec<AgentEvent>>>,
    call_count: Mutex<usize>,
    total_queued: Mutex<usize>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            call_count: Mutex::new(0),
            total_queued: Mutex::new(0),
        }
    }

    /// Queue a complete turn that returns a text-only assistant message.
    pub fn with_text_response(self, text: &str) -> Self {
        let msg = make_assistant_message(text);
        let events = vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::MessageStart {
                message: msg.clone(),
            },
            AgentEvent::MessageEnd {
                message: msg.clone(),
            },
            AgentEvent::TurnEnd {
                turn_number: 1,
                message: msg,
                usage: tau_ai::Usage {
                    input: 100,
                    output: 50,
                    ..Default::default()
                },
            },
        ];
        self.with_events(events)
    }

    /// Queue a complete turn where the assistant requests a tool call.
    pub fn with_tool_call_response(
        self,
        tool_name: &str,
        tool_call_id: &str,
        args: serde_json::Value,
    ) -> Self {
        let msg = make_tool_call_message(tool_name, tool_call_id, args);
        let events = vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::MessageStart {
                message: msg.clone(),
            },
            AgentEvent::MessageEnd {
                message: msg.clone(),
            },
            AgentEvent::TurnEnd {
                turn_number: 1,
                message: msg,
                usage: tau_ai::Usage {
                    input: 100,
                    output: 50,
                    ..Default::default()
                },
            },
        ];
        self.with_events(events)
    }

    /// Queue an error response (error arrives as a stream event).
    pub fn with_error(self, msg: &str) -> Self {
        let events = vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::Error {
                message: msg.to_string(),
            },
        ];
        self.with_events(events)
    }

    /// Queue a raw sequence of events as one response.
    pub fn with_events(self, events: Vec<AgentEvent>) -> Self {
        self.responses.lock().unwrap().push_back(events);
        *self.total_queued.lock().unwrap() += 1;
        self
    }
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn run(
        &self,
        _messages: Vec<tau_ai::Message>,
        _config: &AgentRunConfig,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let events = {
            let mut responses = self.responses.lock().unwrap();
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            let total = *self.total_queued.lock().unwrap();
            responses.pop_front().unwrap_or_else(|| {
                panic!(
                    "MockTransport: no more queued responses (queued {total}, call #{count})"
                )
            })
        };

        Ok(Box::pin(stream::iter(events)))
    }
}

// ---------------------------------------------------------------------------
// EventCollector — deterministic async event collection
// ---------------------------------------------------------------------------

/// Collects `AgentEvent`s from a broadcast channel in a background task,
/// providing deterministic wait methods that replace sleep+poll patterns.
///
/// Modeled after pipecat-rs's `FrameCollector`.
///
/// # Example
///
/// ```ignore
/// let handle = builder.spawn();
/// let collector = EventCollector::from_handle(&handle);
///
/// handle.prompt("hello").await.unwrap();
/// collector.wait_for_end().await;
///
/// let events = collector.events();
/// assert!(events.iter().any(|e| matches!(e, AgentEvent::AgentEnd { .. })));
/// ```
pub struct EventCollector {
    inner: Arc<Mutex<Vec<AgentEvent>>>,
    count_rx: watch::Receiver<usize>,
    // Keep alive so the channel doesn't close
    _count_tx: Arc<watch::Sender<usize>>,
    _task: tokio::task::JoinHandle<()>,
}

impl EventCollector {
    /// Subscribe to an `AgentHandle`'s event stream and start collecting.
    pub fn from_handle(handle: &AgentHandle) -> Self {
        let mut event_rx = handle.subscribe();
        let inner = Arc::new(Mutex::new(Vec::<AgentEvent>::new()));
        let (count_tx, count_rx) = watch::channel(0usize);
        let count_tx = Arc::new(count_tx);

        let inner_clone = inner.clone();
        let count_tx_clone = count_tx.clone();
        let task = tokio::spawn(async move {
            loop {
                match event_rx.recv().await {
                    Ok(event) => {
                        let new_count = {
                            let mut events = inner_clone.lock().unwrap();
                            events.push(event);
                            events.len()
                        };
                        let _ = count_tx_clone.send(new_count);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        panic!(
                            "EventCollector: broadcast receiver lagged by {n} events. \
                             Increase broadcast channel capacity or reduce event volume."
                        );
                    }
                }
            }
        });

        Self {
            inner,
            count_rx,
            _count_tx: count_tx,
            _task: task,
        }
    }

    /// Wait until an `AgentEnd` or `Error` event has been collected.
    /// Panics if neither appears within 5 seconds.
    pub async fn wait_for_end(&self) {
        self.wait_for_event_timeout(
            |e| matches!(e, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. }),
            Duration::from_secs(5),
        )
        .await;
    }

    /// Wait until an event matching the predicate has been collected.
    /// Panics if no match appears within 5 seconds.
    pub async fn wait_for_event(&self, pred: impl Fn(&AgentEvent) -> bool) {
        self.wait_for_event_timeout(pred, Duration::from_secs(5))
            .await;
    }

    /// Wait until at least `count` events have been collected.
    /// Panics if the count isn't reached within 5 seconds.
    pub async fn wait_for_count(&self, count: usize) {
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            let mut rx = self.count_rx.clone();
            loop {
                rx.borrow_and_update();
                if self.count() >= count {
                    return;
                }
                if rx.changed().await.is_err() {
                    panic!(
                        "EventCollector: channel closed at count {} while waiting for {count}",
                        self.count()
                    );
                }
            }
        })
        .await;
        if result.is_err() {
            panic!(
                "EventCollector: timed out waiting for count {count}, got {}",
                self.count()
            );
        }
    }

    /// Wait with a custom timeout.
    pub async fn wait_for_event_timeout(
        &self,
        pred: impl Fn(&AgentEvent) -> bool,
        timeout: Duration,
    ) {
        let result =
            tokio::time::timeout(timeout, self.wait_for_event_inner(&pred)).await;
        if result.is_err() {
            let events = self.event_names();
            panic!(
                "EventCollector: timed out after {timeout:?} waiting for event. \
                 Collected so far: {events:?}"
            );
        }
    }

    async fn wait_for_event_inner(&self, pred: &(impl Fn(&AgentEvent) -> bool + ?Sized)) {
        let mut rx = self.count_rx.clone();
        loop {
            // IMPORTANT: lock must be dropped before .await
            rx.borrow_and_update();
            {
                let events = self.inner.lock().unwrap();
                if events.iter().any(pred) {
                    return;
                }
            }
            if rx.changed().await.is_err() {
                let names = self.event_names();
                panic!(
                    "EventCollector: channel closed while waiting. Collected: {names:?}"
                );
            }
        }
    }

    /// Clone all collected events.
    pub fn events(&self) -> Vec<AgentEvent> {
        self.inner.lock().unwrap().clone()
    }

    /// Take all collected events, clearing the buffer. Subsequent waits will
    /// only see events arriving after this call.
    pub fn take_events(&self) -> Vec<AgentEvent> {
        std::mem::take(&mut *self.inner.lock().unwrap())
    }

    /// Extract assistant messages from collected events (from MessageEnd events).
    pub fn assistant_messages(&self) -> Vec<tau_ai::Message> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                AgentEvent::MessageEnd { message } => Some(message.clone()),
                _ => None,
            })
            .collect()
    }

    /// Get human-readable event type names for debugging.
    pub fn event_names(&self) -> Vec<&'static str> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .map(event_type_name)
            .collect()
    }

    /// Number of events collected so far.
    pub fn count(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

/// Return the variant name of an `AgentEvent` for debugging.
fn event_type_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "AgentStart",
        AgentEvent::TurnStart { .. } => "TurnStart",
        AgentEvent::MessageStart { .. } => "MessageStart",
        AgentEvent::MessageUpdate { .. } => "MessageUpdate",
        AgentEvent::MessageEnd { .. } => "MessageEnd",
        AgentEvent::ToolExecutionStart { .. } => "ToolExecutionStart",
        AgentEvent::ToolExecutionUpdate { .. } => "ToolExecutionUpdate",
        AgentEvent::ToolExecutionEnd { .. } => "ToolExecutionEnd",
        AgentEvent::TurnEnd { .. } => "TurnEnd",
        AgentEvent::AgentEnd { .. } => "AgentEnd",
        AgentEvent::CompactionStart { .. } => "CompactionStart",
        AgentEvent::CompactionEnd { .. } => "CompactionEnd",
        AgentEvent::Error { .. } => "Error",
        AgentEvent::Subagent { .. } => "Subagent",
    }
}

// ---------------------------------------------------------------------------
// EchoTool — simple test tool
// ---------------------------------------------------------------------------

/// A tool that returns its `text` argument as output. Useful for testing
/// the full tool execution round-trip without side effects.
pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the text argument back"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo back"
                }
            },
            "required": ["text"]
        })
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Parallel
    }

    async fn execute(&self, arguments: serde_json::Value, _ctx: ExecutionContext) -> ToolResult {
        let text = arguments
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("(empty)");
        ToolResult::text(text)
    }
}

// ---------------------------------------------------------------------------
// FailTool — always errors
// ---------------------------------------------------------------------------

/// A tool that always returns an error. Useful for testing error handling.
pub struct FailTool;

#[async_trait]
impl Tool for FailTool {
    fn name(&self) -> &str {
        "fail"
    }

    fn description(&self) -> &str {
        "Always fails"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _arguments: serde_json::Value, _ctx: ExecutionContext) -> ToolResult {
        ToolResult::error("intentional failure")
    }
}

// ---------------------------------------------------------------------------
// Helper: spawn an agent with MockTransport for testing
// ---------------------------------------------------------------------------

/// Build and spawn an agent with the given MockTransport and tools.
/// Returns `(AgentHandle, EventCollector)` ready for testing.
pub fn spawn_test_agent(
    transport: MockTransport,
    tools: Vec<BoxedTool>,
) -> (AgentHandle, EventCollector) {
    spawn_test_agent_with_config(make_test_config(), transport, tools)
}

/// Build and spawn an agent with a custom config, MockTransport, and tools.
pub fn spawn_test_agent_with_config(
    config: AgentConfig,
    transport: MockTransport,
    tools: Vec<BoxedTool>,
) -> (AgentHandle, EventCollector) {
    let mut builder = crate::builder::AgentBuilder::new(config, Arc::new(transport));
    builder.set_tools(tools);

    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    (handle, collector)
}

// ---------------------------------------------------------------------------
// Integration test transports (moved from tests/harness.rs)
// ---------------------------------------------------------------------------

/// Returns a fixed text response on every call. Useful for tests that
/// prompt multiple times and always expect the same response.
pub struct TextTransport {
    pub text: String,
}

impl TextTransport {
    pub fn create(text: impl Into<String>) -> Arc<dyn Transport> {
        Arc::new(Self { text: text.into() })
    }
}

#[async_trait]
impl Transport for TextTransport {
    async fn run(
        &self,
        _messages: Vec<tau_ai::Message>,
        _config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let msg = make_assistant_message(&self.text);
        let usage = tau_ai::Usage { input: 100, output: 50, ..Default::default() };
        let events = vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::MessageEnd { message: msg.clone() },
            AgentEvent::TurnEnd { turn_number: 1, message: msg, usage },
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

/// Returns tool calls for `n` turns, then a text response.
/// Each turn's tool call has a unique id based on the turn counter.
pub struct ToolCallTransport {
    remaining: std::sync::atomic::AtomicU32,
    pub tool_name: String,
}

impl ToolCallTransport {
    pub fn create(tool_turns: u32, tool_name: impl Into<String>) -> Arc<dyn Transport> {
        Arc::new(Self {
            remaining: std::sync::atomic::AtomicU32::new(tool_turns),
            tool_name: tool_name.into(),
        })
    }
}

#[async_trait]
impl Transport for ToolCallTransport {
    async fn run(
        &self,
        _messages: Vec<tau_ai::Message>,
        _config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        use std::sync::atomic::Ordering;
        let prev = self.remaining.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_sub(1))
        }).unwrap_or(0);

        let msg = if prev > 0 {
            tau_ai::Message::Assistant {
                content: vec![
                    tau_ai::Content::text("Calling tool."),
                    tau_ai::Content::tool_call(
                        format!("call_{}", prev),
                        &self.tool_name,
                        serde_json::json!({"text": "hello"}),
                    ),
                ],
                metadata: tau_ai::AssistantMetadata::default(),
            }
        } else {
            make_assistant_message("Done.")
        };

        let usage = tau_ai::Usage { input: 50, output: 25, ..Default::default() };
        let events = vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::MessageEnd { message: msg.clone() },
            AgentEvent::TurnEnd { turn_number: 1, message: msg, usage },
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

/// Streams events with a delay between start and completion.
/// Responds to cancellation.
pub struct SlowTransport {
    delay_ms: u64,
}

impl SlowTransport {
    pub fn create(delay_ms: u64) -> Arc<dyn Transport> {
        Arc::new(Self { delay_ms })
    }
}

#[async_trait]
impl Transport for SlowTransport {
    async fn run(
        &self,
        _messages: Vec<tau_ai::Message>,
        _config: &AgentRunConfig,
        cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let delay = self.delay_ms;
        let events = async_stream::stream! {
            yield AgentEvent::TurnStart { turn_number: 1 };
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(delay)) => {}
                _ = cancel.cancelled() => {
                    yield AgentEvent::Error { message: "Cancelled".into() };
                    return;
                }
            }
            let msg = make_assistant_message("done");
            yield AgentEvent::MessageEnd { message: msg.clone() };
            yield AgentEvent::TurnEnd {
                turn_number: 1,
                message: msg,
                usage: tau_ai::Usage::default(),
            };
        };
        Ok(Box::pin(events))
    }
}

/// Captured transport call: messages + config snapshot.
#[derive(Clone)]
pub struct CapturedCall {
    pub messages: Vec<tau_ai::Message>,
    pub system_prompt: Option<String>,
    pub tool_names: Vec<String>,
    pub model_id: String,
}

/// Returns a fixed response but captures the messages and config it received.
pub struct CapturingTransport {
    pub text: String,
    pub captured: Mutex<Vec<CapturedCall>>,
}

impl CapturingTransport {
    pub fn create(text: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            captured: Mutex::new(vec![]),
        })
    }

    pub fn calls(&self) -> Vec<CapturedCall> {
        self.captured.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for CapturingTransport {
    async fn run(
        &self,
        messages: Vec<tau_ai::Message>,
        config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        self.captured.lock().unwrap().push(CapturedCall {
            messages,
            system_prompt: config.system_prompt.clone(),
            tool_names: config.tools.iter().map(|t| t.name.clone()).collect(),
            model_id: config.model.id.clone(),
        });
        let msg = make_assistant_message(&self.text);
        let usage = tau_ai::Usage { input: 100, output: 50, ..Default::default() };
        let events = vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::MessageEnd { message: msg.clone() },
            AgentEvent::TurnEnd { turn_number: 1, message: msg, usage },
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

/// Transport that always fails with an error from `run()` (transport-level failure).
pub struct ErrorTransport {
    pub message: String,
}

impl ErrorTransport {
    pub fn create(message: impl Into<String>) -> Arc<dyn Transport> {
        Arc::new(Self { message: message.into() })
    }
}

#[async_trait]
impl Transport for ErrorTransport {
    async fn run(
        &self,
        _messages: Vec<tau_ai::Message>,
        _config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        Err(tau_ai::Error::Api {
            error_type: "test_error".into(),
            message: self.message.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Integration test tools
// ---------------------------------------------------------------------------

/// A sequential tool that sleeps, useful for testing steering between groups.
pub struct SlowTool {
    pub delay_ms: u64,
}

#[async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &str { "slow" }
    fn description(&self) -> &str { "Slow tool" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    fn concurrency(&self) -> Concurrency { Concurrency::Sequential }
    async fn execute(&self, _args: serde_json::Value, _ctx: ExecutionContext) -> ToolResult {
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        ToolResult::text("slow done")
    }
}

// ---------------------------------------------------------------------------
// Event collection utility
// ---------------------------------------------------------------------------

/// Drain all available events from a broadcast receiver.
pub fn collect_events(rx: &mut tokio::sync::broadcast::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = vec![];
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    events
}

// ---------------------------------------------------------------------------
// Tests for the test utilities themselves
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_user_message_creates_user_role() {
        let msg = make_user_message("hello");
        assert_eq!(msg.role(), "user");
        assert_eq!(msg.text(), "hello");
    }

    #[test]
    fn make_assistant_message_creates_assistant_role() {
        let msg = make_assistant_message("hi");
        assert_eq!(msg.role(), "assistant");
        assert_eq!(msg.text(), "hi");
    }

    #[test]
    fn make_tool_call_message_has_tool_call() {
        let msg = make_tool_call_message("echo", "call_1", serde_json::json!({"text": "x"}));
        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "echo");
    }

    #[test]
    fn make_tool_result_creates_tool_result() {
        let msg = make_tool_result("call_1", "echo", "output");
        assert_eq!(msg.role(), "tool_result");
    }

    #[tokio::test]
    async fn mock_transport_returns_queued_responses() {
        let transport = MockTransport::new()
            .with_text_response("first")
            .with_text_response("second");

        let config = AgentRunConfig {
            system_prompt: None,
            tools: vec![],
            server_tools: vec![],
            model: make_test_model(),
            reasoning: None,
            thinking_adaptive: false,
            max_tokens: None,
            temperature: None,
            cache_scope: None,
            cache_ttl: None,
            system_prompt_boundary: None,
        };
        let cancel = tokio_util::sync::CancellationToken::new();

        // First call
        use futures::StreamExt;
        let stream = transport.run(vec![], &config, cancel.clone()).await.unwrap();
        let events: Vec<_> = stream.collect().await;
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::MessageEnd { .. })));

        // Second call
        let stream = transport.run(vec![], &config, cancel).await.unwrap();
        let events: Vec<_> = stream.collect().await;
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::MessageEnd { .. })));
    }

    #[tokio::test]
    #[should_panic(expected = "no more queued responses")]
    async fn mock_transport_panics_when_exhausted() {
        let transport = MockTransport::new().with_text_response("only one");

        let config = AgentRunConfig {
            system_prompt: None,
            tools: vec![],
            server_tools: vec![],
            model: make_test_model(),
            reasoning: None,
            thinking_adaptive: false,
            max_tokens: None,
            temperature: None,
            cache_scope: None,
            cache_ttl: None,
            system_prompt_boundary: None,
        };
        let cancel = tokio_util::sync::CancellationToken::new();

        // First call succeeds
        let _ = transport.run(vec![], &config, cancel.clone()).await.unwrap();
        // Second call panics
        let _ = transport.run(vec![], &config, cancel).await.unwrap();
    }

    #[tokio::test]
    async fn spawn_test_agent_prompt_and_collect() {
        let transport = MockTransport::new().with_text_response("Hello from mock!");
        let (handle, collector) = spawn_test_agent(transport, vec![]);

        handle.prompt("test").await.unwrap();
        collector.wait_for_end().await;

        let events = collector.events();
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentStart)));
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentEnd { .. })));

        let messages = collector.assistant_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), "Hello from mock!");
    }

    #[tokio::test]
    async fn tool_execution_round_trip() {
        let transport = MockTransport::new()
            .with_tool_call_response("echo", "call_1", serde_json::json!({"text": "ping"}))
            .with_text_response("Got: ping");

        let echo_tool: BoxedTool = Arc::new(EchoTool);
        let (handle, collector) = spawn_test_agent(transport, vec![echo_tool]);

        handle.prompt("echo ping").await.unwrap();
        collector.wait_for_end().await;

        let events = collector.events();

        // Should have tool execution events
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionStart { tool_name, .. } if tool_name == "echo")));
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } if tool_name == "echo" && !is_error)));

        // Should have final text response
        let messages = collector.assistant_messages();
        assert!(messages.iter().any(|m| m.text() == "Got: ping"));
    }

    #[tokio::test]
    async fn error_response_propagates() {
        let transport = MockTransport::new().with_error("something went wrong");
        let (handle, collector) = spawn_test_agent(transport, vec![]);

        handle.prompt("test").await.unwrap();
        collector.wait_for_end().await;

        let events = collector.events();
        // The error event from the stream gets forwarded
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::Error { message } if message.contains("something went wrong")
        )));
    }

    #[tokio::test]
    async fn fail_tool_returns_error_result() {
        let transport = MockTransport::new()
            .with_tool_call_response("fail", "call_1", serde_json::json!({}))
            .with_text_response("Acknowledged error");

        let fail_tool: BoxedTool = Arc::new(FailTool);
        let (handle, collector) = spawn_test_agent(transport, vec![fail_tool]);

        handle.prompt("do something").await.unwrap();
        collector.wait_for_end().await;

        let events = collector.events();
        assert!(events.iter().any(
            |e| matches!(e, AgentEvent::ToolExecutionEnd { is_error, .. } if *is_error)
        ));
    }

    #[tokio::test]
    async fn event_names_are_human_readable() {
        let transport = MockTransport::new().with_text_response("hi");
        let (handle, collector) = spawn_test_agent(transport, vec![]);

        handle.prompt("test").await.unwrap();
        collector.wait_for_end().await;

        let names = collector.event_names();
        assert!(names.contains(&"AgentStart"));
        assert!(names.contains(&"AgentEnd"));
        assert!(names.contains(&"MessageEnd"));
    }

    #[tokio::test]
    async fn take_events_clears_buffer() {
        let transport = MockTransport::new().with_text_response("hi");
        let (handle, collector) = spawn_test_agent(transport, vec![]);

        handle.prompt("test").await.unwrap();
        collector.wait_for_end().await;

        let events = collector.take_events();
        assert!(!events.is_empty());
        assert_eq!(collector.count(), 0);
    }

    #[tokio::test]
    async fn make_execution_context_works_for_tool_testing() {
        let ctx = make_execution_context();
        let tool = EchoTool;
        let result = tool
            .execute(serde_json::json!({"text": "hello"}), ctx)
            .await;
        assert!(!result.is_error);
        assert_eq!(result.text_content(), "hello");
    }
}
