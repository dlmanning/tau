//! Tests for the basic agent lifecycle: spawn, prompt, events, shutdown.

use tau_agent::test_utils::*;
use tau_agent::*;
use tau_ai::Message;

#[tokio::test]
async fn spawn_and_query_config() {
    let mut builder = AgentBuilder::new(test_config(), TextTransport::create("hi"));
    builder.set_system_prompt("custom");
    let handle = builder.spawn();

    let cfg = handle.config().await.unwrap();
    assert_eq!(cfg.system_prompt.as_deref(), Some("custom"));
}

#[tokio::test]
async fn set_model_via_handle() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("hi")).spawn();

    let mut m = test_config().model;
    m.id = "new-model".into();
    handle.set_model(m).await.unwrap();

    let cfg = handle.config().await.unwrap();
    assert_eq!(cfg.model.id, "new-model");
}

#[tokio::test]
async fn prompt_returns_assistant_message() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("Hello!")).spawn();

    handle.prompt_and_wait("hi").await.unwrap();

    let msgs = handle.messages().await.unwrap();
    assert_eq!(msgs.len(), 2); // user + assistant
    assert!(matches!(&msgs[0], Message::User { .. }));
    assert_eq!(msgs[1].text(), "Hello!");
}

#[tokio::test]
async fn prompt_emits_start_and_end_events() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("ok")).spawn();
    let mut rx = handle.subscribe();

    handle.prompt_and_wait("go").await.unwrap();

    let events = collect_events(&mut rx);
    assert!(events.iter().any(|e| matches!(e, AgentEvent::AgentStart)));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentEnd { .. }))
    );
}

#[tokio::test]
async fn usage_is_accumulated() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("ok")).spawn();

    handle.prompt_and_wait("a").await.unwrap();
    let s1 = handle.state().await.unwrap();
    assert_eq!(s1.total_usage.input, 100);

    handle.prompt_and_wait("b").await.unwrap();
    let s2 = handle.state().await.unwrap();
    assert_eq!(s2.total_usage.input, 200);
}

#[tokio::test]
async fn is_streaming_false_after_prompt() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("ok")).spawn();
    handle.prompt_and_wait("go").await.unwrap();

    let state = handle.state().await.unwrap();
    assert!(!state.is_streaming);
    assert!(state.error.is_none());
}

// `clear_messages` and `set_messages` were removed from the handle in the
// runtime refactor — conversation mutation is no longer a runtime
// capability. Tests for them are dropped.

#[tokio::test]
async fn timestamps_are_milliseconds() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("ok")).spawn();
    handle.prompt_and_wait("go").await.unwrap();

    for msg in handle.messages().await.unwrap() {
        let ts = match &msg {
            Message::User { timestamp, .. } | Message::ToolResult { timestamp, .. } => *timestamp,
            _ => continue,
        };
        assert!(
            ts > 1_000_000_000_000,
            "timestamp {ts} looks like seconds not millis"
        );
    }
}

#[tokio::test]
async fn handle_is_clone_send_sync() {
    fn assert_bounds<T: Clone + Send + Sync>() {}
    assert_bounds::<AgentHandle>();
}

#[tokio::test]
async fn dropping_all_handles_makes_queries_return_none() {
    let handle = AgentBuilder::new(test_config(), TextTransport::create("ok")).spawn();
    // Clone to keep one alive, drop the original
    let h2 = handle.clone();
    drop(handle);
    // The actor is still alive because h2 holds a sender.
    assert!(h2.config().await.is_some());
    // Now drop h2 — actor should shut down.
    drop(h2);
    // Nothing to assert — just verifying no panic/hang on drop.
}

/// Test transport whose `run()` parks the actor on a Notify until released.
/// This blocks the actor *before* it enters AwaitingModel, so the actor is
/// not in a select!-loop and commands accumulate in the channel.
struct GatedTransport {
    release: std::sync::Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl tau_agent::transport::Transport for GatedTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &tau_agent::transport::AgentRunConfig,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<tau_agent::transport::AgentEventStream> {
        self.release.notified().await;
        // After release, return an empty stream so the prompt completes.
        let s = async_stream::stream! {
            yield AgentEvent::TurnEnd {
                turn_number: 1,
                message: tau_ai::Message::Assistant {
                    content: vec![tau_ai::Content::text("ok")],
                    metadata: Default::default(),
                },
                usage: tau_ai::Usage::default(),
            };
        };
        Ok(Box::pin(s))
    }
}

// `mid_prompt_clear_is_deferred_until_done` was removed: the deferred-op
// mechanism it tested was deleted along with the conversation-mutator
// commands in the runtime refactor.

#[tokio::test]
async fn try_send_returns_err_when_channel_full() {
    use std::sync::Arc;
    use std::time::Duration;

    let release = Arc::new(tokio::sync::Notify::new());
    let transport = Arc::new(GatedTransport {
        release: release.clone(),
    });

    let builder = AgentBuilder::with_channel_capacities(
        test_config(),
        transport,
        /* urgent */ 2,
        /* normal */ 2,
    );
    let handle = builder.spawn();

    // Kick off a prompt. The actor parks in `transport.run().await`,
    // so it cannot drain the normal channel.
    let _rx = handle.prompt("go").await.unwrap();

    // The Prompt itself consumed one normal slot. Fill the rest using
    // `try_set_compaction_config` (a non-removed normal-channel command).
    let mut full_seen = false;
    for _ in 0..10 {
        match handle.try_set_compaction_config(tau_agent::CompactionConfig::default()) {
            Ok(()) => {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            Err(Error::ChannelFull { channel: "normal" }) => {
                full_seen = true;
                break;
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
    assert!(full_seen, "try_set_compaction_config should eventually see Full");

    // Cleanup: release the actor so the test doesn't leak the task.
    release.notify_one();
    let _ = tokio::time::timeout(Duration::from_secs(1), _rx).await;
}

#[tokio::test]
async fn async_send_blocks_then_succeeds_when_channel_full() {
    use std::sync::Arc;
    use std::time::Duration;

    let release = Arc::new(tokio::sync::Notify::new());
    let transport = Arc::new(GatedTransport {
        release: release.clone(),
    });

    let builder = AgentBuilder::with_channel_capacities(test_config(), transport, 2, 2);
    let handle = builder.spawn();

    let rx = handle.prompt("go").await.unwrap();

    // Saturate the normal channel.
    while handle
        .try_set_compaction_config(tau_agent::CompactionConfig::default())
        .is_ok()
    {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    // The async send should block. Race it against a release of the actor:
    // until the actor unblocks, the send must not complete.
    let h2 = handle.clone();
    let send_future = tokio::spawn(async move {
        h2.set_compaction_config(tau_agent::CompactionConfig::default())
            .await
    });

    // Confirm it's not finishing on its own.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!send_future.is_finished(), "send should still be blocked");

    // Release the actor → channel drains → send completes.
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), send_future)
        .await
        .expect("send should not time out")
        .expect("join")
        .expect("send should succeed");

    let _ = tokio::time::timeout(Duration::from_secs(1), rx).await;
}

#[tokio::test]
async fn actor_panic_surfaces_via_shutdown_reason_and_error_event() {
    // PanicTransport panics inside `run` on the actor task itself — that
    // kills the actor (unlike a tool panic, which JoinSet catches).
    let builder = AgentBuilder::new(test_config(), PanicTransport::create());
    let pre_handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&pre_handle);
    let handle = builder.spawn();

    // Fire the prompt that triggers the panic. The catch_unwind wrapper
    // records `shutdown_reason` and notifies before returning, so we don't
    // need to poll.
    let _ = handle.prompt_and_wait("trigger panic").await;

    // The next handle call's async send awaits `shutdown_signaled`, so it
    // is guaranteed to surface `Error::ActorPanic` (not `Error::Other`)
    // even though the channel-close beats the reason write to the handle.
    let err = handle
        .prompt_and_wait("again")
        .await
        .expect_err("actor is dead — prompt must fail");
    assert!(
        matches!(&err, Error::ActorPanic(msg) if msg.contains("intentional panic in transport")),
        "expected Error::ActorPanic with panic message, got: {err:?}"
    );

    let reason = handle
        .shutdown_reason()
        .expect("supervisor should have recorded panic reason by now");
    assert!(
        reason.contains("intentional panic in transport"),
        "expected panic message in shutdown_reason, got: {reason}"
    );

    // The collector should have observed an AgentEvent::Error carrying the
    // panic message (best-effort broadcast from the supervisor).
    let events = collector.events();
    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::Error { message } if message.contains("intentional panic in transport")
        )),
        "expected an AgentEvent::Error with the panic text"
    );
}
