//! Tests for `compact()`'s `custom_instructions` parameter.
//!
//! The summarization LLM call is captured via `CapturingTransport`. The
//! prompt arrives as the (single) user-message content of the call.

use std::sync::Arc;

use tau_agent::test_utils::*;
use tau_agent::{AgentBuilder, CompactionConfig, CompactionReason, Transport};
use tau_ai::{AssistantMetadata, Content, Message};

fn user_msg(text: &str) -> Message {
    Message::User {
        content: vec![Content::text(text)],
        timestamp: 0,
    }
}

fn assistant_msg(text: &str) -> Message {
    Message::Assistant {
        content: vec![Content::text(text)],
        metadata: AssistantMetadata::default(),
    }
}

/// Seed a conversation long enough to satisfy `find_cut_point` under a
/// small `keep_recent_tokens` setting.
fn seed_messages() -> Vec<Message> {
    let mut msgs = Vec::new();
    for i in 0..20 {
        msgs.push(user_msg(&format!("user message number {i:02} blah blah")));
        msgs.push(assistant_msg(&format!(
            "assistant reply number {i:02} blah blah"
        )));
    }
    msgs
}

fn small_compaction_config() -> CompactionConfig {
    CompactionConfig {
        enabled: true,
        reserve: tau_agent::CompactionThreshold::Tokens(100),
        keep_recent: tau_agent::CompactionThreshold::Tokens(50),
    }
}

/// Extract the summarization prompt from the (single) user content text
/// of the latest captured transport call.
fn extract_prompt(transport: &Arc<CapturingTransport>) -> String {
    let calls = transport.calls();
    assert!(
        !calls.is_empty(),
        "expected at least one transport call for summarization"
    );
    calls
        .last()
        .unwrap()
        .messages
        .iter()
        .find_map(|m| match m {
            Message::User { content, .. } => Some(
                content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .expect("summarization call should include a user message")
}

async fn build_handle(transport: Arc<CapturingTransport>) -> tau_agent::AgentHandle {
    let cfg = test_config()
        .into_builder()
        .compaction(small_compaction_config())
        .build();
    let mut builder = AgentBuilder::new(cfg, transport as Arc<dyn Transport>);
    builder.seed(tau_agent::AgentSeed::Messages {
        messages: seed_messages(),
        previous_summary: None,
    });
    builder.spawn().await.unwrap()
}

#[tokio::test]
async fn with_instructions() {
    let transport = CapturingTransport::create("SUMMARY-OK");
    let handle = build_handle(transport.clone()).await;

    let rx = handle
        .compact(CompactionReason::Manual, Some("be terse".into()))
        .await
        .expect("compact send");
    let result = rx.await.expect("compact reply");
    assert!(
        result.result.is_ok(),
        "compaction failed: {:?}",
        result.result.err()
    );

    let prompt = extract_prompt(&transport);
    assert!(
        prompt.contains("## User instructions"),
        "expected '## User instructions' header; got:\n{prompt}"
    );
    assert!(
        prompt.contains("be terse"),
        "expected custom instructions text; got:\n{prompt}"
    );
}

#[tokio::test]
async fn without_instructions() {
    let transport = CapturingTransport::create("SUMMARY-OK");
    let handle = build_handle(transport.clone()).await;

    let rx = handle
        .compact(CompactionReason::Manual, None)
        .await
        .expect("compact send");
    let result = rx.await.expect("compact reply");
    assert!(
        result.result.is_ok(),
        "compaction failed: {:?}",
        result.result.err()
    );

    let prompt = extract_prompt(&transport);
    assert!(
        !prompt.contains("## User instructions"),
        "did not expect '## User instructions' header; got:\n{prompt}"
    );
}

#[tokio::test]
async fn whitespace_only() {
    let transport = CapturingTransport::create("SUMMARY-OK");
    let handle = build_handle(transport.clone()).await;

    let rx = handle
        .compact(CompactionReason::Manual, Some("   \n  ".into()))
        .await
        .expect("compact send");
    let result = rx.await.expect("compact reply");
    assert!(
        result.result.is_ok(),
        "compaction failed: {:?}",
        result.result.err()
    );

    let prompt = extract_prompt(&transport);
    assert!(
        !prompt.contains("## User instructions"),
        "whitespace-only instructions should be treated as absent; got:\n{prompt}"
    );
}
