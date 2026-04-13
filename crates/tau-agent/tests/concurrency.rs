//! Tests for concurrent access: abort, busy rejection, queries during streaming,
//! cancel token lifecycle across multiple prompts.

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
async fn reject_concurrent_prompt() {
    let handle = AgentBuilder::new(test_config(), SlowTransport::new(200)).spawn();

    let rx1 = handle.prompt("first").await.unwrap();
    let _ = handle.config().await; // ensure actor picked it up

    let rx2 = handle.prompt("second").await.unwrap();
    let r2 = rx2.await.unwrap();
    assert!(r2.result.is_err());
    assert!(r2.result.unwrap_err().to_string().contains("busy"));

    let r1 = rx1.await.unwrap();
    assert!(r1.result.is_ok());
}

#[tokio::test]
async fn is_running_tracks_lifecycle() {
    let handle = AgentBuilder::new(test_config(), SlowTransport::new(100)).spawn();
    assert!(!handle.is_running());

    let rx = handle.prompt("go").await.unwrap();
    let _ = handle.config().await;
    assert!(handle.is_running());

    let _ = rx.await;
    let _ = handle.config().await;
    assert!(!handle.is_running());
}

#[tokio::test]
async fn abort_stops_active_prompt() {
    let handle = AgentBuilder::new(test_config(), SlowTransport::new(5000)).spawn();

    let rx = handle.prompt("go").await.unwrap();
    let _ = handle.config().await;
    assert!(handle.is_running());

    handle.abort();

    let result = tokio::time::timeout(std::time::Duration::from_secs(2), rx).await;
    assert!(
        result.is_ok(),
        "abort should cause prompt to finish promptly"
    );
}

#[tokio::test]
async fn abort_then_new_prompt_succeeds() {
    struct SwitchTransport(AtomicU32);

    #[async_trait]
    impl Transport for SwitchTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            _: &AgentRunConfig,
            cancel: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let n = self.0.fetch_add(1, Ordering::Relaxed);
            if n == 0 {
                let events = async_stream::stream! {
                    yield AgentEvent::TurnStart { turn_number: 1 };
                    cancel.cancelled().await;
                    yield AgentEvent::Error { message: "Cancelled".into() };
                };
                Ok(Box::pin(events))
            } else {
                let msg = Message::Assistant {
                    content: vec![Content::text("ok after abort")],
                    metadata: AssistantMetadata::default(),
                };
                Ok(Box::pin(stream::iter(vec![
                    AgentEvent::TurnStart { turn_number: 1 },
                    AgentEvent::MessageEnd {
                        message: msg.clone(),
                    },
                    AgentEvent::TurnEnd {
                        turn_number: 1,
                        message: msg,
                        usage: Usage::default(),
                    },
                ])))
            }
        }
    }

    let handle =
        AgentBuilder::new(test_config(), Arc::new(SwitchTransport(AtomicU32::new(0)))).spawn();

    let rx1 = handle.prompt("first").await.unwrap();
    let _ = handle.config().await;
    handle.abort();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), rx1).await;
    let _ = handle.config().await;

    let r = handle.prompt_and_wait("second").await;
    assert!(r.is_ok(), "second prompt after abort should succeed");

    let msgs = handle.messages().await.unwrap();
    assert!(msgs.iter().any(|m| m.text().contains("ok after abort")));
}

#[tokio::test]
async fn queries_work_during_streaming() {
    let handle = AgentBuilder::new(test_config(), SlowTransport::new(200)).spawn();
    let rx = handle.prompt("go").await.unwrap();

    // These should not hang — the actor select!s on cmd_rx during AwaitingModel
    let cfg = handle.config().await;
    assert!(cfg.is_some());
    let msgs = handle.messages().await;
    assert!(msgs.is_some());
    let state = handle.state().await;
    assert!(state.is_some());

    let _ = rx.await;
}

#[tokio::test]
async fn config_mutation_during_streaming_takes_effect_next_turn() {
    let handle = AgentBuilder::new(test_config(), SlowTransport::new(100)).spawn();
    let rx = handle.prompt("go").await.unwrap();

    handle.set_reasoning(tau_ai::ReasoningLevel::High);
    let _ = rx.await;

    let cfg = handle.config().await.unwrap();
    assert_eq!(cfg.reasoning, tau_ai::ReasoningLevel::High);
}
