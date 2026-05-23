//! Tests for error handling: transport errors, conversation.error, overflow.

use async_trait::async_trait;
use futures::stream;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tau_agent::{AgentEventStream, AgentRunConfig};
use tau_agent::test_utils::*;
use tau_agent::*;
use tau_ai::{AssistantMetadata, Content, Message, Usage};

#[tokio::test]
async fn transport_error_sets_conversation_error() {
    let handle = AgentBuilder::new(test_config(), ErrorTransport::create("something broke"))
        .spawn()
        .await
        .unwrap();

    let result = handle.prompt_and_wait("go").await;
    assert!(result.is_err());

    let state = handle.state().await.unwrap();
    assert!(state.error.is_some(), "conversation.error should be set");
    assert!(state.error.unwrap().contains("something broke"));
}

#[tokio::test]
async fn transport_error_emits_error_event() {
    let handle = AgentBuilder::new(test_config(), ErrorTransport::create("boom"))
        .spawn()
        .await
        .unwrap();
    let mut rx = handle.subscribe();

    let _ = handle.prompt_and_wait("go").await;

    let mut events = Vec::new(); while let Ok(e) = rx.try_recv() { events.push(e); }
    let has_error = events.iter().any(|e| matches!(e, AgentEvent::Error { .. }));
    assert!(has_error, "should emit Error event");
}

#[tokio::test]
async fn error_cleared_on_next_prompt() {
    // First call errors, second succeeds.
    struct FlipTransport(AtomicU32);

    #[async_trait]
    impl Transport for FlipTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            _: &AgentRunConfig,
            _: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let n = self.0.fetch_add(1, Ordering::Relaxed);
            if n == 0 {
                Err(tau_ai::Error::Api {
                    error_type: "test".into(),
                    message: "first fail".into(),
                })
            } else {
                let msg = Message::Assistant {
                    content: vec![Content::text("ok now")],
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

    let handle = AgentBuilder::new(test_config(), Arc::new(FlipTransport(AtomicU32::new(0))))
        .spawn()
        .await
        .unwrap();

    let r1 = handle.prompt_and_wait("go").await;
    assert!(r1.is_err());
    assert!(handle.state().await.unwrap().error.is_some());

    let r2 = handle.prompt_and_wait("retry").await;
    assert!(r2.is_ok());
    assert!(
        handle.state().await.unwrap().error.is_none(),
        "error should be cleared on new prompt"
    );
}

#[tokio::test]
async fn agent_is_idle_after_error() {
    let handle = AgentBuilder::new(test_config(), ErrorTransport::create("fail"))
        .spawn()
        .await
        .unwrap();
    let _ = handle.prompt_and_wait("go").await;

    assert!(
        matches!(handle.health(), AgentHealth::Idle),
        "should be idle after error"
    );
}
