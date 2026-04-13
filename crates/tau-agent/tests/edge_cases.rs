//! Edge case tests: empty streams, tool panics, multi-group batching,
//! DequeueMode::OneAtATime, steer while idle, follow-up while busy.

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
async fn empty_stream_completes_without_crash() {
    /// Transport that returns an empty stream (no events at all).
    struct EmptyStreamTransport;

    #[async_trait]
    impl Transport for EmptyStreamTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            _: &AgentRunConfig,
            _: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            Ok(Box::pin(stream::empty()))
        }
    }

    let handle = AgentBuilder::new(
        test_config(),
        Arc::new(EmptyStreamTransport) as Arc<dyn Transport>,
    )
    .spawn();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        handle.prompt_and_wait("go"),
    )
    .await;

    assert!(result.is_ok(), "should not hang on empty stream");
    // Result may be Ok or Err, but must not panic or hang
}

#[tokio::test]
async fn multi_group_sequential_then_parallel() {
    /// Returns [sequential, parallel, parallel] tool calls, then text.
    struct MultiGroupTransport(AtomicU32);

    #[async_trait]
    impl Transport for MultiGroupTransport {
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
                        Content::tool_call("c2", "echo", serde_json::json!({"text": "a"})),
                        Content::tool_call("c3", "echo", serde_json::json!({"text": "b"})),
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

    let transport: Arc<dyn Transport> = Arc::new(MultiGroupTransport(AtomicU32::new(1)));
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(SlowTool { delay_ms: 10 }));
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();

    let msgs = handle.messages().await.unwrap();
    let tool_ids: Vec<&str> = msgs
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect();

    // All 3 tools should have executed in order: slow (sequential), then echo+echo (parallel)
    assert_eq!(
        tool_ids,
        vec!["c1", "c2", "c3"],
        "tool results should be in original order across groups"
    );
}

#[tokio::test]
async fn follow_up_during_tool_execution() {
    // Post a follow-up while tools are running. It should be processed after the prompt completes.
    let transport = ToolCallTransport::new(1, "echo");
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();

    let rx = handle.prompt("go").await.unwrap();

    // Immediately post a follow-up (it will arrive while tools execute or during DrainFollowUps)
    handle.follow_up(Message::user("follow-up message"));

    let result = tokio::time::timeout(std::time::Duration::from_secs(5), rx).await;
    assert!(result.is_ok());
    assert!(result.unwrap().unwrap().result.is_ok());

    // The follow-up should have been processed (the transport will be called with it)
    let msgs = handle.messages().await.unwrap();
    let has_follow_up = msgs.iter().any(|m| m.text().contains("follow-up message"));
    assert!(has_follow_up, "follow-up message should be in conversation");
}

#[tokio::test]
async fn dequeue_mode_one_at_a_time() {
    let mut cfg = test_config();
    cfg.follow_up_mode = DequeueMode::OneAtATime;
    let handle = AgentBuilder::new(cfg, TextTransport::new("ok")).spawn();

    // Queue up 2 follow-ups, then prompt
    handle.follow_up(Message::user("fu1"));
    handle.follow_up(Message::user("fu2"));
    handle.expect_follow_up();
    handle.expect_follow_up();

    let rx = handle.prompt("initial").await.unwrap();

    // After initial response, DrainFollowUps should drain ONE follow-up per cycle.
    // Post a consume after a delay to eventually stop waiting.
    let h2 = handle.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        h2.consume_follow_up();
        h2.consume_follow_up();
    });

    let result = tokio::time::timeout(std::time::Duration::from_secs(5), rx).await;
    assert!(result.is_ok(), "should complete");

    let msgs = handle.messages().await.unwrap();
    let fu_count = msgs.iter().filter(|m| m.text().starts_with("fu")).count();
    assert_eq!(
        fu_count, 2,
        "both follow-ups should be processed, got {fu_count}"
    );
}

#[tokio::test]
async fn steer_while_idle_is_not_lost() {
    let transport = CapturingTransport::new("ok");
    let builder = AgentBuilder::new(test_config(), transport.clone());
    let handle = builder.spawn();

    // Steer while idle — the message should be queued
    handle.steer(Message::user("pre-steer"));
    // Now prompt — DrainFollowUps should pick up the steering message
    handle.prompt_and_wait("hello").await.unwrap();

    // The prompt finishes, then DrainFollowUps checks steering queue.
    // If the steer arrived in the queue, it should trigger another turn.
    let calls = transport.calls();
    // At minimum, the original prompt goes through. If steering worked,
    // there should be a second call.
    // Note: steer goes to handle_busy_command which pushes to steering_queue.
    // But if the agent is idle, steer goes through handle_idle_command -> handle_busy_command
    // -> pushes to steering_queue. The steering queue is only checked in DrainFollowUps
    // which happens after the LLM responds. So the steer should be processed.
    if calls.len() >= 2 {
        let second = &calls[1];
        assert!(
            second
                .messages
                .iter()
                .any(|m| m.text().contains("pre-steer")),
            "steering message should appear in second transport call context"
        );
    }
    // If there's only 1 call, the steer may have been processed but the
    // transport was already called. Either way, the steer message should
    // be in the conversation.
    let msgs = handle.messages().await.unwrap();
    let has_steer = msgs.iter().any(|m| m.text().contains("pre-steer"));
    assert!(has_steer, "steering message should be in conversation");
}
