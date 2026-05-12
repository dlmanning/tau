//! Snapshot query for context-window usage.
//!
//! Drives a prompt through a real conversation, then queries
//! `context_stats()` and asserts the snapshot reflects the messages
//! that landed in `state.conv.conversation.messages`.

use tau_agent::core::builder::AgentBuilder;
use tau_agent::test_utils::*;

#[tokio::test]
async fn context_stats_reports_used_against_model_limit() {
    let transport = TextTransport::create("hello from v2");
    let builder = AgentBuilder::new(test_config(), transport);
    let handle = builder.spawn();
    let collector = EventCollector::from_handle(&handle);

    handle
        .prompt_and_wait("a longer prompt that produces non-trivial token usage")
        .await
        .expect("prompt completes");
    collector.wait_for_end().await;

    let stats = handle
        .context_stats()
        .await
        .expect("context_stats query returns");

    // `make_test_model` advertises a 200_000-token context window;
    // `test_config` doesn't change it.
    assert_eq!(stats.limit, 200_000, "limit reflects model.context_window");
    assert!(stats.used > 0, "non-empty conversation reports non-zero used");
    assert!(stats.used <= stats.limit, "used <= limit");
    assert_eq!(
        stats.remaining,
        stats.limit.saturating_sub(stats.used),
        "remaining = limit - used (saturated)"
    );
}

#[tokio::test]
async fn context_stats_before_any_prompt_is_empty() {
    let transport = TextTransport::create("unused");
    let builder = AgentBuilder::new(test_config(), transport);
    let handle = builder.spawn();

    let stats = handle
        .context_stats()
        .await
        .expect("context_stats query returns");

    assert_eq!(stats.used, 0, "no messages yet");
    assert_eq!(stats.limit, 200_000);
    assert_eq!(stats.remaining, 200_000);
}
