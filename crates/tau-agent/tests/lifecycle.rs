//! Tests for the basic agent lifecycle: spawn, prompt, events, shutdown.

mod harness;

use harness::*;
use tau_agent::*;
use tau_ai::Message;

#[tokio::test]
async fn spawn_and_query_config() {
    let mut builder = AgentBuilder::new(test_config(), TextTransport::new("hi"));
    builder.set_system_prompt("custom");
    let handle = builder.spawn();

    let cfg = handle.config().await.unwrap();
    assert_eq!(cfg.system_prompt.as_deref(), Some("custom"));
}

#[tokio::test]
async fn set_model_via_handle() {
    let handle = AgentBuilder::new(test_config(), TextTransport::new("hi")).spawn();

    let mut m = test_config().model;
    m.id = "new-model".into();
    handle.set_model(m);

    let cfg = handle.config().await.unwrap();
    assert_eq!(cfg.model.id, "new-model");
}

#[tokio::test]
async fn prompt_returns_assistant_message() {
    let handle = AgentBuilder::new(test_config(), TextTransport::new("Hello!")).spawn();

    handle.prompt_and_wait("hi").await.unwrap();

    let msgs = handle.messages().await.unwrap();
    assert_eq!(msgs.len(), 2); // user + assistant
    assert!(matches!(&msgs[0], Message::User { .. }));
    assert_eq!(msgs[1].text(), "Hello!");
}

#[tokio::test]
async fn prompt_emits_start_and_end_events() {
    let handle = AgentBuilder::new(test_config(), TextTransport::new("ok")).spawn();
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
    let handle = AgentBuilder::new(test_config(), TextTransport::new("ok")).spawn();

    handle.prompt_and_wait("a").await.unwrap();
    let s1 = handle.state().await.unwrap();
    assert_eq!(s1.total_usage.input, 100);

    handle.prompt_and_wait("b").await.unwrap();
    let s2 = handle.state().await.unwrap();
    assert_eq!(s2.total_usage.input, 200);
}

#[tokio::test]
async fn is_streaming_false_after_prompt() {
    let handle = AgentBuilder::new(test_config(), TextTransport::new("ok")).spawn();
    handle.prompt_and_wait("go").await.unwrap();

    let state = handle.state().await.unwrap();
    assert!(!state.is_streaming);
    assert!(state.error.is_none());
}

#[tokio::test]
async fn clear_messages_resets_state() {
    let handle = AgentBuilder::new(test_config(), TextTransport::new("ok")).spawn();
    handle.prompt_and_wait("go").await.unwrap();

    handle.clear_messages();
    let _ = handle.config().await; // sync

    let msgs = handle.messages().await.unwrap();
    assert!(msgs.is_empty());

    let state = handle.state().await.unwrap();
    assert_eq!(state.total_usage.input, 0);
    assert!(state.previous_summary.is_none());
}

#[tokio::test]
async fn set_messages_replaces_conversation() {
    let handle = AgentBuilder::new(test_config(), TextTransport::new("ok")).spawn();
    handle.set_messages(vec![Message::user("a"), Message::user("b")]);
    let _ = handle.config().await;

    let msgs = handle.messages().await.unwrap();
    assert_eq!(msgs.len(), 2);
}

#[tokio::test]
async fn timestamps_are_milliseconds() {
    let handle = AgentBuilder::new(test_config(), TextTransport::new("ok")).spawn();
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
    let handle = AgentBuilder::new(test_config(), TextTransport::new("ok")).spawn();
    // Clone to keep one alive, drop the original
    let h2 = handle.clone();
    drop(handle);
    // The actor is still alive because h2 holds a sender.
    assert!(h2.config().await.is_some());
    // Now drop h2 — actor should shut down.
    drop(h2);
    // Nothing to assert — just verifying no panic/hang on drop.
}
