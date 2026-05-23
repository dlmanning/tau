//! Test utilities for tau-agent and downstream crates.
//!
//! Available within tau-agent via `#[cfg(test)]` and to downstream
//! crates via the `test-utils` feature flag:
//!
//! ```toml
//! [dev-dependencies]
//! tau-agent = { workspace = true, features = ["test-utils"] }
//! ```
//!
//! Port of v1 `tau-agent`'s test_utils, adapted to v2's type paths.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;
use futures::stream as fstream;
use parking_lot::Mutex as ParkingMutex;
use serde_json::Value;
use tau_ai::{
    Api, AssistantMetadata, Content, CostInfo, InputType, Message, Model, Provider, ReasoningLevel,
    Usage,
};
use tokio::sync::{broadcast, watch};
use tokio::time;
use tokio_util::sync::CancellationToken;

use crate::core::compaction::CompactionConfig;
use crate::core::config::{AgentConfig, DequeueMode};
use crate::core::handle::AgentHandle;
use crate::core::tool::{
    BoxedTool, Concurrency, ExecutionContext, FileAccessTracker, ProgressSender, Tool, ToolResult,
};
use crate::core::transport::{AgentEventStream, AgentRunConfig, Transport};
use crate::types::events::AgentEvent;

// ─── Message constructors ────────────────────────────────────────────

pub fn make_assistant_message(text: &str) -> Message {
    Message::Assistant {
        content: vec![Content::text(text)],
        metadata: AssistantMetadata::default(),
    }
}

pub fn make_tool_call_message(name: &str, id: &str, args: Value) -> Message {
    Message::Assistant {
        content: vec![Content::tool_call(id, name, args)],
        metadata: AssistantMetadata::default(),
    }
}

// ─── Test model & config ─────────────────────────────────────────────

pub fn make_test_model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: Api::AnthropicMessages,
        provider: Provider::Anthropic,
        base_url: "https://test.invalid".into(),
        reasoning: false,
        input_types: vec![InputType::Text],
        cost: CostInfo::default(),
        context_window: 200_000,
        max_tokens: 8192,
        headers: Default::default(),
    }
}

/// Like `make_test_config` but with a `system_prompt` set. Used by
/// integration tests that exercise system-prompt-in-context.
pub fn test_config() -> AgentConfig {
    let mut cfg = make_test_config();
    cfg.system_prompt = Some("You are a test agent.".into());
    cfg
}

pub fn make_test_config() -> AgentConfig {
    AgentConfig {
        system_prompt: None,
        model: make_test_model(),
        reasoning: ReasoningLevel::Off,
        thinking_adaptive: false,
        max_tokens: None,
        max_turns: None,
        compaction: CompactionConfig::default(),
        steering_mode: DequeueMode::All,
        follow_up_mode: DequeueMode::All,
        cache_scope: None,
        cache_ttl: None,
        system_prompt_boundary: None,
    }
}

pub fn make_execution_context() -> ExecutionContext {
    let (tx, _rx) = broadcast::channel(256);
    ExecutionContext {
        cwd: PathBuf::from("/tmp"),
        cancel: CancellationToken::new(),
        progress: ProgressSender::new(tx, "test_call", "test_tool"),
        interaction: None,
        interaction_timeout: None,
        file_access: Arc::new(ParkingMutex::new(FileAccessTracker::default())),
        agent_id: None,
        subagent_depth: 0,
    }
}

// ─── MockTransport: queued canned responses ──────────────────────────

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

    pub fn with_text_response(self, text: &str) -> Self {
        let msg = make_assistant_message(text);
        self.with_events(vec![
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
                usage: Usage {
                    input: 100,
                    output: 50,
                    ..Default::default()
                },
            },
        ])
    }

    pub fn with_tool_call_response(self, tool_name: &str, tool_call_id: &str, args: Value) -> Self {
        let msg = make_tool_call_message(tool_name, tool_call_id, args);
        self.with_events(vec![
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
                usage: Usage {
                    input: 100,
                    output: 50,
                    ..Default::default()
                },
            },
        ])
    }

    pub fn with_error(self, msg: &str) -> Self {
        self.with_events(vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::Error {
                message: msg.into(),
            },
        ])
    }

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
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let events = {
            let mut responses = self.responses.lock().unwrap();
            let mut count = self.call_count.lock().unwrap();
            *count += 1;
            let total = *self.total_queued.lock().unwrap();
            responses.pop_front().unwrap_or_else(|| {
                panic!("MockTransport: no more queued responses (queued {total}, call #{count})")
            })
        };
        Ok(Box::pin(fstream::iter(events)))
    }
}

// ─── EventCollector: deterministic async event collection ────────────

pub struct EventCollector {
    inner: Arc<Mutex<Vec<AgentEvent>>>,
    count_rx: watch::Receiver<usize>,
    _count_tx: Arc<watch::Sender<usize>>,
    _task: tokio::task::JoinHandle<()>,
}

impl EventCollector {
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
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
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

    pub async fn wait_for_end(&self) {
        self.wait_for_event_timeout(
            |e| matches!(e, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. }),
            Duration::from_secs(5),
        )
        .await;
    }

    pub async fn wait_for_event(&self, pred: impl Fn(&AgentEvent) -> bool) {
        self.wait_for_event_timeout(pred, Duration::from_secs(5))
            .await;
    }

    pub async fn wait_for_count(&self, count: usize) {
        let result = time::timeout(Duration::from_secs(5), async {
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

    pub async fn wait_for_event_timeout(
        &self,
        pred: impl Fn(&AgentEvent) -> bool,
        timeout: Duration,
    ) {
        let result = time::timeout(timeout, self.wait_for_event_inner(&pred)).await;
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
            rx.borrow_and_update();
            {
                let events = self.inner.lock().unwrap();
                if events.iter().any(pred) {
                    return;
                }
            }
            if rx.changed().await.is_err() {
                let names = self.event_names();
                panic!("EventCollector: channel closed while waiting. Collected: {names:?}");
            }
        }
    }

    pub fn events(&self) -> Vec<AgentEvent> {
        self.inner.lock().unwrap().clone()
    }

    pub fn take_events(&self) -> Vec<AgentEvent> {
        std::mem::take(&mut *self.inner.lock().unwrap())
    }

    pub fn assistant_messages(&self) -> Vec<Message> {
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

    pub fn event_names(&self) -> Vec<&'static str> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .map(event_type_name)
            .collect()
    }

    pub fn count(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

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
        AgentEvent::ToolApprovalResolved { .. } => "ToolApprovalResolved",
        AgentEvent::FileChanged { .. } => "FileChanged",
        AgentEvent::AgentReport { .. } => "AgentReport",
        AgentEvent::TurnEnd { .. } => "TurnEnd",
        AgentEvent::AgentEnd { .. } => "AgentEnd",
        AgentEvent::CompactionStart { .. } => "CompactionStart",
        AgentEvent::CompactionEnd { .. } => "CompactionEnd",
        AgentEvent::Error { .. } => "Error",
    }
}

// ─── Tools ───────────────────────────────────────────────────────────

pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echoes the text argument back"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string", "description": "Text to echo back" } },
            "required": ["text"]
        })
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Parallel
    }
    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let text = arguments
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("(empty)");
        ToolResult::text(text)
    }
}

pub struct FailTool;

#[async_trait]
impl Tool for FailTool {
    fn name(&self) -> &str {
        "fail"
    }
    fn description(&self) -> &str {
        "Always fails"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        ToolResult::error("intentional failure")
    }
}

pub struct PanicTool;

#[async_trait]
impl Tool for PanicTool {
    fn name(&self) -> &str {
        "panic"
    }
    fn description(&self) -> &str {
        "Panics inside execute"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        panic!("intentional panic in tool");
    }
}

pub struct SlowTool {
    pub delay_ms: u64,
}

#[async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &str {
        "slow"
    }
    fn description(&self) -> &str {
        "Slow tool"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }
    async fn execute(&self, _args: Value, _ctx: ExecutionContext) -> ToolResult {
        time::sleep(Duration::from_millis(self.delay_ms)).await;
        ToolResult::text("slow done")
    }
}

// ─── Transports ──────────────────────────────────────────────────────

pub struct PanicTransport;

impl PanicTransport {
    pub fn create() -> Arc<dyn Transport> {
        Arc::new(Self)
    }
}

#[async_trait]
impl Transport for PanicTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        panic!("intentional panic in transport");
    }
}

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
        _messages: Vec<Message>,
        config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let msg = make_assistant_message(&self.text);
        let turn_number = config.turn_number;
        let usage = Usage {
            input: 100,
            output: 50,
            ..Default::default()
        };
        let events = vec![
            AgentEvent::TurnStart { turn_number },
            AgentEvent::MessageEnd {
                message: msg.clone(),
            },
            AgentEvent::TurnEnd {
                turn_number,
                message: msg,
                usage,
            },
        ];
        Ok(Box::pin(fstream::iter(events)))
    }
}

pub struct ToolCallTransport {
    remaining: AtomicU32,
    pub tool_name: String,
}

impl ToolCallTransport {
    pub fn create(tool_turns: u32, tool_name: impl Into<String>) -> Arc<dyn Transport> {
        Arc::new(Self {
            remaining: AtomicU32::new(tool_turns),
            tool_name: tool_name.into(),
        })
    }
}

#[async_trait]
impl Transport for ToolCallTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let prev = self
            .remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n: u32| {
                Some(n.saturating_sub(1))
            })
            .unwrap_or(0);

        let msg = if prev > 0 {
            Message::Assistant {
                content: vec![
                    Content::text("Calling tool."),
                    Content::tool_call(
                        format!("call_{}", prev),
                        &self.tool_name,
                        serde_json::json!({"text": "hello"}),
                    ),
                ],
                metadata: AssistantMetadata::default(),
            }
        } else {
            make_assistant_message("Done.")
        };

        let turn_number = config.turn_number;
        let usage = Usage {
            input: 50,
            output: 25,
            ..Default::default()
        };
        let events = vec![
            AgentEvent::TurnStart { turn_number },
            AgentEvent::MessageEnd {
                message: msg.clone(),
            },
            AgentEvent::TurnEnd {
                turn_number,
                message: msg,
                usage,
            },
        ];
        Ok(Box::pin(fstream::iter(events)))
    }
}

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
        _messages: Vec<Message>,
        config: &AgentRunConfig,
        cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let delay = self.delay_ms;
        let turn_number = config.turn_number;
        let events = stream! {
            yield AgentEvent::TurnStart { turn_number };
            tokio::select! {
                _ = time::sleep(Duration::from_millis(delay)) => {}
                _ = cancel.cancelled() => {
                    yield AgentEvent::Error { message: "Cancelled".into() };
                    return;
                }
            }
            let msg = make_assistant_message("done");
            yield AgentEvent::MessageEnd { message: msg.clone() };
            yield AgentEvent::TurnEnd { turn_number, message: msg, usage: Usage::default() };
        };
        Ok(Box::pin(events))
    }
}

/// Captured transport call: messages + config snapshot.
#[derive(Clone)]
pub struct CapturedCall {
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub tool_names: Vec<String>,
    pub model_id: String,
}

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
        messages: Vec<Message>,
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
        let turn_number = config.turn_number;
        let usage = Usage {
            input: 100,
            output: 50,
            ..Default::default()
        };
        let events = vec![
            AgentEvent::TurnStart { turn_number },
            AgentEvent::MessageEnd {
                message: msg.clone(),
            },
            AgentEvent::TurnEnd {
                turn_number,
                message: msg,
                usage,
            },
        ];
        Ok(Box::pin(fstream::iter(events)))
    }
}

pub struct ErrorTransport {
    pub message: String,
}

impl ErrorTransport {
    pub fn create(message: impl Into<String>) -> Arc<dyn Transport> {
        Arc::new(Self {
            message: message.into(),
        })
    }
}

#[async_trait]
impl Transport for ErrorTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        _cancel: CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        Err(tau_ai::Error::Api {
            error_type: "test_error".into(),
            message: self.message.clone(),
        })
    }
}

// ─── Test-spawn helpers ──────────────────────────────────────────────

/// Build and spawn an agent with a MockTransport + tools. Async
/// because [`AgentBuilder::spawn`] awaits the actor's readiness
/// signal — see the API reference for the contract.
pub async fn spawn_test_agent(
    transport: MockTransport,
    tools: Vec<BoxedTool>,
) -> (AgentHandle, EventCollector) {
    spawn_test_agent_with_config(make_test_config(), transport, tools).await
}

pub async fn spawn_test_agent_with_config(
    config: AgentConfig,
    transport: MockTransport,
    tools: Vec<BoxedTool>,
) -> (AgentHandle, EventCollector) {
    let mut builder = crate::core::builder::AgentBuilder::new(config, Arc::new(transport));
    builder.set_tools(tools);

    let collector = EventCollector::from_handle(&builder.handle());
    let handle = builder.spawn().await.expect("test agent spawn");

    (handle, collector)
}

