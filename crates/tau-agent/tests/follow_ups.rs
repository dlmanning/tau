//! Tests for steering, follow-ups, and background agent waiting.

use async_trait::async_trait;
use futures::stream;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tau_agent::test_utils::*;
use tau_agent::transport::{AgentEventStream, AgentRunConfig};
use tau_agent::*;
use tau_ai::{AssistantMetadata, Content, Message, Usage};

#[tokio::test]
async fn follow_up_processed_after_prompt() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("ok")).spawn();

    handle.expect_follow_up();
    assert!(handle.has_pending_follow_ups());

    let rx = handle.prompt("initial").await.unwrap();

    // Post follow-up from "background agent" after a delay
    let h2 = handle.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        h2.follow_up(Message::user("background result"));
        h2.consume_follow_up();
    });

    let result = tokio::time::timeout(std::time::Duration::from_secs(5), rx).await;
    assert!(result.is_ok(), "should complete after follow-up");
    assert!(result.unwrap().unwrap().result.is_ok());
    assert!(!handle.has_pending_follow_ups());
}

#[tokio::test]
async fn no_pending_follow_ups_finishes_immediately() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("ok")).spawn();
    // No expect_follow_up — should finish immediately after LLM responds
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        handle.prompt_and_wait("go"),
    )
    .await;
    assert!(
        result.is_ok(),
        "should finish without waiting for follow-ups"
    );
}

#[tokio::test]
async fn steering_during_tool_execution_skips_remaining() {
    // Transport: always returns 2 sequential tool calls (slow + echo).
    // We steer after the slow tool finishes.
    struct TwoToolTransport(AtomicU32);

    #[async_trait]
    impl Transport for TwoToolTransport {
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
                        Content::tool_call("c1", "slow", serde_json::json!({})),
                        Content::tool_call(
                            "c2",
                            "echo",
                            serde_json::json!({"text": "should be skipped"}),
                        ),
                    ],
                    metadata: AssistantMetadata::default(),
                }
            } else {
                Message::Assistant {
                    content: vec![Content::text("steered response")],
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

    let transport: Arc<dyn Transport> = Arc::new(TwoToolTransport(AtomicU32::new(1)));
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(SlowTool { delay_ms: 100 }));
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();
    let mut rx = handle.subscribe();

    let prompt_rx = handle.prompt("go").await.unwrap();

    // Wait for the slow tool to start, then steer
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    handle.steer(Message::user("new direction"));

    let result = tokio::time::timeout(std::time::Duration::from_secs(5), prompt_rx).await;
    assert!(result.is_ok());

    let events = collect_events(&mut rx);
    // The echo tool (c2) should show up as skipped
    let skipped = events.iter().any(|e| match e {
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            is_error,
            ..
        } => tool_call_id == "c2" && *is_error,
        _ => false,
    });
    assert!(skipped, "second tool should be skipped due to steering");
}

#[tokio::test]
async fn max_turns_stops_infinite_tool_loop() {
    struct AlwaysToolTransport;

    #[async_trait]
    impl Transport for AlwaysToolTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            config: &AgentRunConfig,
            _: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let msg = if config.tools.is_empty() {
                Message::Assistant {
                    content: vec![Content::text("Summary")],
                    metadata: AssistantMetadata::default(),
                }
            } else {
                Message::Assistant {
                    content: vec![
                        Content::text("tool time"),
                        Content::tool_call("c", "echo", serde_json::json!({"text": "x"})),
                    ],
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

    let mut cfg = test_config();
    cfg.max_turns = Some(3);
    let mut builder = AgentBuilder::new(cfg, Arc::new(AlwaysToolTransport) as Arc<dyn Transport>);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();
    let mut rx = handle.subscribe();

    let r = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        handle.prompt_and_wait("go"),
    )
    .await;
    assert!(r.is_ok(), "should not hang");
    assert!(r.unwrap().is_ok());

    let events = collect_events(&mut rx);
    if let Some(AgentEvent::AgentEnd { total_turns, .. }) = events
        .iter()
        .find(|e| matches!(e, AgentEvent::AgentEnd { .. }))
    {
        assert!(
            *total_turns <= 3,
            "should stop at max_turns, got {total_turns}"
        );
    } else {
        panic!("missing AgentEnd event");
    }
}
