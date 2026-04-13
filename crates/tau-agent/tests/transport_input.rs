//! Tests that verify what the actor sends TO the transport — system prompt,
//! tool definitions, model config, message context, and tool results.

mod harness;

use async_trait::async_trait;
use futures::stream;
use harness::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tau_agent::transport::{AgentEventStream, AgentRunConfig};
use tau_agent::*;
use tau_ai::{AssistantMetadata, Content, Message, Usage};

#[tokio::test]
async fn system_prompt_sent_to_transport() {
    let transport = CapturingTransport::new("ok");
    let mut builder = AgentBuilder::new(test_config(), transport.clone());
    builder.set_system_prompt("You are helpful.");
    let handle = builder.spawn();

    handle.prompt_and_wait("hi").await.unwrap();

    let calls = transport.calls();
    assert_eq!(calls[0].system_prompt.as_deref(), Some("You are helpful."));
}

#[tokio::test]
async fn tool_definitions_sent_to_transport() {
    let transport = CapturingTransport::new("ok");
    let mut builder = AgentBuilder::new(test_config(), transport.clone());
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();

    handle.prompt_and_wait("hi").await.unwrap();

    let calls = transport.calls();
    assert!(
        calls[0].tool_names.contains(&"echo".to_string()),
        "tool definitions should include 'echo', got {:?}",
        calls[0].tool_names
    );
}

#[tokio::test]
async fn model_id_sent_to_transport() {
    let transport = CapturingTransport::new("ok");
    let builder = AgentBuilder::new(test_config(), transport.clone());
    let handle = builder.spawn();

    handle.prompt_and_wait("hi").await.unwrap();

    let calls = transport.calls();
    assert_eq!(calls[0].model_id, "test-model");
}

#[tokio::test]
async fn user_message_in_first_call_context() {
    let transport = CapturingTransport::new("ok");
    let builder = AgentBuilder::new(test_config(), transport.clone());
    let handle = builder.spawn();

    handle.prompt_and_wait("what is 2+2").await.unwrap();

    let calls = transport.calls();
    let first_ctx = &calls[0].messages;
    let user_msg = first_ctx.iter().find(|m| matches!(m, Message::User { .. }));
    assert!(user_msg.is_some(), "first call should contain user message");
    assert!(user_msg.unwrap().text().contains("what is 2+2"));
}

#[tokio::test]
async fn tool_results_sent_to_transport_on_next_turn() {
    /// Transport that returns one tool call, then captures the next call.
    struct ToolThenCapture {
        call_count: AtomicU32,
        captured: std::sync::Mutex<Vec<CapturedCall>>,
    }

    #[async_trait]
    impl Transport for ToolThenCapture {
        async fn run(
            &self,
            messages: Vec<Message>,
            config: &AgentRunConfig,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let n = self.call_count.fetch_add(1, Ordering::Relaxed);
            self.captured.lock().unwrap().push(CapturedCall {
                messages: messages.clone(),
                system_prompt: config.system_prompt.clone(),
                tool_names: config.tools.iter().map(|t| t.name.clone()).collect(),
                model_id: config.model.id.clone(),
            });

            let msg = if n == 0 {
                Message::Assistant {
                    content: vec![Content::tool_call(
                        "c1",
                        "echo",
                        serde_json::json!({"text": "test"}),
                    )],
                    metadata: AssistantMetadata::default(),
                }
            } else {
                Message::Assistant {
                    content: vec![Content::text("final")],
                    metadata: AssistantMetadata::default(),
                }
            };
            let events = vec![
                AgentEvent::TurnStart { turn_number: 1 },
                AgentEvent::MessageEnd {
                    message: msg.clone(),
                },
                AgentEvent::TurnEnd {
                    turn_number: 1,
                    message: msg,
                    usage: Usage::default(),
                },
            ];
            Ok(Box::pin(stream::iter(events)))
        }
    }

    let transport = Arc::new(ToolThenCapture {
        call_count: AtomicU32::new(0),
        captured: std::sync::Mutex::new(vec![]),
    });
    let mut builder = AgentBuilder::new(test_config(), transport.clone());
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();

    let calls = transport.captured.lock().unwrap().clone();
    assert!(calls.len() >= 2, "should have at least 2 transport calls");

    // Second call should contain the ToolResult
    let second_ctx = &calls[1].messages;
    let has_tool_result = second_ctx
        .iter()
        .any(|m| matches!(m, Message::ToolResult { .. }));
    assert!(
        has_tool_result,
        "second call should include ToolResult, got: {:?}",
        second_ctx
            .iter()
            .map(|m| match m {
                Message::User { .. } => "User",
                Message::Assistant { .. } => "Assistant",
                Message::ToolResult { .. } => "ToolResult",
                Message::SystemInjection { .. } => "SystemInjection",
            })
            .collect::<Vec<_>>()
    );

    // The tool result should contain the echo output
    let tool_result = second_ctx
        .iter()
        .find(|m| matches!(m, Message::ToolResult { .. }))
        .unwrap();
    assert!(
        tool_result.text().contains("test"),
        "tool result should contain echo output"
    );
}

#[tokio::test]
async fn model_change_reflected_in_subsequent_calls() {
    let transport = CapturingTransport::new("ok");
    let builder = AgentBuilder::new(test_config(), transport.clone());
    let handle = builder.spawn();

    handle.prompt_and_wait("first").await.unwrap();

    let mut new_model = test_config().model;
    new_model.id = "changed-model".into();
    handle.set_model(new_model);

    handle.prompt_and_wait("second").await.unwrap();

    let calls = transport.calls();
    assert_eq!(calls[0].model_id, "test-model");
    assert_eq!(calls[1].model_id, "changed-model");
}

#[tokio::test]
async fn transform_context_applied_before_transport() {
    let transport = CapturingTransport::new("ok");
    let mut builder = AgentBuilder::new(test_config(), transport.clone());
    builder.set_transform_context(Arc::new(|mut msgs| {
        msgs.push(Message::user("[injected by transform]"));
        msgs
    }));
    let handle = builder.spawn();

    handle.prompt_and_wait("hello").await.unwrap();

    let calls = transport.calls();
    let has_injected = calls[0]
        .messages
        .iter()
        .any(|m| m.text().contains("[injected by transform]"));
    assert!(
        has_injected,
        "transform_context should inject a message into the context"
    );
}
