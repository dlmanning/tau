//! End-to-end smoke test: build an agent, prompt it, verify the
//! response and the event stream.

use tau_agent::core::builder::AgentBuilder;
use tau_agent::test_utils::*;

#[tokio::test]
async fn text_response_round_trip() {
    let transport = TextTransport::create("hello from v2");
    let builder = AgentBuilder::new(test_config(), transport);
    let handle = builder.spawn();
    let collector = EventCollector::from_handle(&handle);

    handle
        .prompt_and_wait("hi")
        .await
        .expect("prompt completes");
    collector.wait_for_end().await;

    let messages = collector.assistant_messages();
    assert_eq!(messages.len(), 1, "one assistant message");
    assert_eq!(messages[0].text(), "hello from v2");

    let history = handle.messages().await.expect("messages query");
    assert_eq!(history.len(), 2, "user + assistant in conversation");
    assert_eq!(history[0].role(), "user");
    assert_eq!(history[1].role(), "assistant");
}

#[tokio::test]
async fn tool_call_round_trip() {
    use std::sync::Arc;

    let transport = ToolCallTransport::create(1, "echo");
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn();
    let collector = EventCollector::from_handle(&handle);

    handle
        .prompt_and_wait("call echo")
        .await
        .expect("prompt completes");
    collector.wait_for_end().await;

    let history = handle.messages().await.expect("messages query");
    // user → assistant (tool call) → tool result → assistant ("done")
    assert_eq!(history.len(), 4, "user + 2 assistant + 1 tool result");
    assert_eq!(history[0].role(), "user");
    assert_eq!(history[1].role(), "assistant");
    assert_eq!(history[2].role(), "tool_result");
    assert_eq!(history[3].role(), "assistant");
    assert_eq!(history[3].text(), "Done.");
}
