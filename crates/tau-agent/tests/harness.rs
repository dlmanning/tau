//! Shared test harness: mock transports, tools, and config builders.
#![allow(dead_code, clippy::new_ret_no_self)]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use tau_ai::{AssistantMetadata, Content, CostInfo, Message, Usage};
use tau_agent::*;
use tau_agent::transport::{AgentEventStream, AgentRunConfig};

// ─── Config ─────────────────────────────────────────────────────────

pub fn test_config() -> AgentConfig {
    AgentConfig {
        system_prompt: Some("You are a test agent.".into()),
        model: tau_ai::Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api: tau_ai::Api::AnthropicMessages,
            provider: tau_ai::Provider::Anthropic,
            base_url: "http://localhost".into(),
            reasoning: false,
            input_types: vec![],
            cost: CostInfo::default(),
            context_window: 200_000,
            max_tokens: 4096,
            headers: Default::default(),
        },
        reasoning: tau_ai::ReasoningLevel::Off,
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

// ─── Transports ─────────────────────────────────────────────────────

/// Returns a fixed text response on every call.
pub struct TextTransport {
    pub text: String,
}

impl TextTransport {
    pub fn new(text: impl Into<String>) -> Arc<dyn Transport> {
        Arc::new(Self { text: text.into() })
    }
}

#[async_trait]
impl Transport for TextTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let msg = Message::Assistant {
            content: vec![Content::text(&self.text)],
            metadata: AssistantMetadata::default(),
        };
        let usage = Usage { input: 100, output: 50, ..Default::default() };
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
    remaining: AtomicU32,
    pub tool_name: String,
}

impl ToolCallTransport {
    pub fn new(tool_turns: u32, tool_name: impl Into<String>) -> Arc<dyn Transport> {
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
        _config: &AgentRunConfig,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let prev = self.remaining.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_sub(1))
        }).unwrap_or(0);

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
            Message::Assistant {
                content: vec![Content::text("Done.")],
                metadata: AssistantMetadata::default(),
            }
        };

        let usage = Usage { input: 50, output: 25, ..Default::default() };
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
    pub fn new(delay_ms: u64) -> Arc<dyn Transport> {
        Arc::new(Self { delay_ms })
    }
}

#[async_trait]
impl Transport for SlowTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let delay = self.delay_ms;
        let events = async_stream::stream! {
            yield AgentEvent::TurnStart { turn_number: 1 };
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_millis(delay)) => {}
                _ = cancel.cancelled() => {
                    yield AgentEvent::Error { message: "Cancelled".into() };
                    return;
                }
            }
            let msg = Message::Assistant {
                content: vec![Content::text("done")],
                metadata: AssistantMetadata::default(),
            };
            yield AgentEvent::MessageEnd { message: msg.clone() };
            yield AgentEvent::TurnEnd { turn_number: 1, message: msg, usage: Usage::default() };
        };
        Ok(Box::pin(events))
    }
}

/// Captured call: messages + config snapshot.
#[derive(Clone)]
pub struct CapturedCall {
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub tool_names: Vec<String>,
    pub model_id: String,
}

/// Returns a fixed response but captures the messages and config it received.
pub struct CapturingTransport {
    pub text: String,
    pub captured: std::sync::Mutex<Vec<CapturedCall>>,
}

impl CapturingTransport {
    pub fn new(text: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            captured: std::sync::Mutex::new(vec![]),
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
        _cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        self.captured.lock().unwrap().push(CapturedCall {
            messages,
            system_prompt: config.system_prompt.clone(),
            tool_names: config.tools.iter().map(|t| t.name.clone()).collect(),
            model_id: config.model.id.clone(),
        });
        let msg = Message::Assistant {
            content: vec![Content::text(&self.text)],
            metadata: AssistantMetadata::default(),
        };
        let usage = Usage { input: 100, output: 50, ..Default::default() };
        let events = vec![
            AgentEvent::TurnStart { turn_number: 1 },
            AgentEvent::MessageEnd { message: msg.clone() },
            AgentEvent::TurnEnd { turn_number: 1, message: msg, usage },
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

/// Transport that always fails with an error.
pub struct ErrorTransport {
    pub message: String,
}

impl ErrorTransport {
    pub fn new(message: impl Into<String>) -> Arc<dyn Transport> {
        Arc::new(Self { message: message.into() })
    }
}

#[async_trait]
impl Transport for ErrorTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        Err(tau_ai::Error::Api {
            error_type: "test_error".into(),
            message: self.message.clone(),
        })
    }
}

// ─── Tools ──────────────────────────────────────────────────────────

pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echoes input" }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } }
        })
    }
    fn concurrency(&self) -> Concurrency { Concurrency::Parallel }
    async fn execute(&self, args: serde_json::Value, _ctx: ExecutionContext) -> ToolResult {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("(empty)");
        ToolResult::text(text)
    }
}

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
        tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        ToolResult::text("slow done")
    }
}

// ─── Event collection ───────────────────────────────────────────────

/// Drain all available events from a broadcast receiver.
pub fn collect_events(rx: &mut tokio::sync::broadcast::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = vec![];
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    events
}
