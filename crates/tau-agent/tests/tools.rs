//! Tests for tool execution: single tool, multi-turn, parallel ordering,
//! sequential vs parallel grouping, tool-only assistant messages.

use async_trait::async_trait;
use futures::stream;
use tau_agent::test_utils::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tau_agent::transport::{AgentEventStream, AgentRunConfig};
use tau_agent::*;
use tau_ai::{AssistantMetadata, Content, Message, Usage};

#[tokio::test]
async fn single_tool_call_round_trip() {
    let transport = ToolCallTransport::create(1, "echo");
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();
    let mut rx = handle.subscribe();

    handle.prompt_and_wait("go").await.unwrap();

    let events = collect_events(&mut rx);
    let starts: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
        .collect();
    let ends: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .collect();
    assert_eq!(starts.len(), 1);
    assert_eq!(ends.len(), 1);

    let msgs = handle.messages().await.unwrap();
    // user, assistant(tool_call), tool_result, assistant(text)
    assert!(
        msgs.len() >= 4,
        "expected >= 4 messages, got {}",
        msgs.len()
    );
    assert!(msgs.iter().any(|m| matches!(m, Message::ToolResult { .. })));
}

#[tokio::test]
async fn multi_turn_tool_loop() {
    let transport = ToolCallTransport::create(3, "echo");
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();
    let mut rx = handle.subscribe();

    handle.prompt_and_wait("go").await.unwrap();

    let events = collect_events(&mut rx);
    let tool_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .count();
    assert_eq!(
        tool_count, 3,
        "should have executed 3 tool calls across 3 turns"
    );

    let msgs = handle.messages().await.unwrap();
    let tool_results = msgs
        .iter()
        .filter(|m| matches!(m, Message::ToolResult { .. }))
        .count();
    assert_eq!(tool_results, 3);
}

#[tokio::test]
async fn parallel_tool_results_preserve_request_order() {
    struct ThreeToolTransport(AtomicU32);

    #[async_trait]
    impl Transport for ThreeToolTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            _: &AgentRunConfig,
            _: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let prev = self
                .0
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    Some(n.saturating_sub(1))
                })
                .unwrap_or(0);
            let msg = if prev > 0 {
                Message::Assistant {
                    content: vec![
                        Content::tool_call("call_a", "echo", serde_json::json!({"text": "a"})),
                        Content::tool_call("call_b", "echo", serde_json::json!({"text": "b"})),
                        Content::tool_call("call_c", "echo", serde_json::json!({"text": "c"})),
                    ],
                    metadata: AssistantMetadata::default(),
                }
            } else {
                Message::Assistant {
                    content: vec![Content::text("done")],
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

    let transport: Arc<dyn Transport> = Arc::new(ThreeToolTransport(AtomicU32::new(1)));
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();

    let msgs = handle.messages().await.unwrap();
    let ids: Vec<&str> = msgs
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["call_a", "call_b", "call_c"]);
}

#[tokio::test]
async fn tool_only_assistant_message_is_stored() {
    // An assistant message with ONLY a tool call and no text should still be stored.
    struct ToolOnlyTransport(AtomicU32);

    #[async_trait]
    impl Transport for ToolOnlyTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            _: &AgentRunConfig,
            _: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let prev = self
                .0
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    Some(n.saturating_sub(1))
                })
                .unwrap_or(0);
            let msg = if prev > 0 {
                // No text, only tool call
                Message::Assistant {
                    content: vec![Content::tool_call(
                        "c1",
                        "echo",
                        serde_json::json!({"text": "x"}),
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

    let transport: Arc<dyn Transport> = Arc::new(ToolOnlyTransport(AtomicU32::new(1)));
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();

    let msgs = handle.messages().await.unwrap();
    // The tool-call-only assistant message must be in the conversation
    let assistant_with_tool = msgs.iter().any(|m| match m {
        Message::Assistant { content, .. } => content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. })),
        _ => false,
    });
    assert!(
        assistant_with_tool,
        "assistant message with tool call should be stored"
    );
}

#[tokio::test]
async fn context_includes_full_conversation_history() {
    let transport = CapturingTransport::create("response");
    let builder = AgentBuilder::new(test_config(), transport.clone());
    let handle = builder.spawn();

    handle.prompt_and_wait("first").await.unwrap();
    handle.prompt_and_wait("second").await.unwrap();

    let calls = transport.calls();
    // Second call should include messages from the first conversation
    let second = &calls[1];
    let ctx = &second.messages;
    assert!(
        ctx.len() >= 3,
        "second call context should include prior messages, got {}",
        ctx.len()
    );
    assert!(ctx.iter().any(|m| match m {
        Message::User { content, .. } => content.iter().any(|c| c.as_text() == Some("first")),
        _ => false,
    }));
}
