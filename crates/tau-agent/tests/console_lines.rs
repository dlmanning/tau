//! End-to-end test: bash streams classified lines via the new
//! `ToolExecutionUpdate { lines }` shape.

use std::sync::Arc;

use tau_agent::test_utils::*;
use tau_agent::{
    AgentBuilder, AgentEvent, AutoAcceptAllPolicy, BoxedTool, ConsoleLevel, ConsoleLine,
};
use tau_tools::BashTool;

#[test]
fn tool_execution_update_serde_roundtrip_lines_shape() {
    // Pin the JSON wire-format. Hosts that persist events depend on this
    // exact shape; gap #5 changed `content: String` to `lines: Vec<…>`,
    // so anything that drifts from this shape is a breaking change.
    let event = AgentEvent::ToolExecutionUpdate {
        tool_call_id: "c1".into(),
        tool_name: "bash".into(),
        lines: vec![
            ConsoleLine::new("$ ls", ConsoleLevel::Muted),
            ConsoleLine::new("error: bad", ConsoleLevel::Danger),
        ],
    };
    let json = serde_json::to_value(&event).expect("serialize");
    let expected = serde_json::json!({
        "type": "tool_execution_update",
        "tool_call_id": "c1",
        "tool_name": "bash",
        "lines": [
            { "content": "$ ls",      "level": "muted" },
            { "content": "error: bad", "level": "danger" },
        ]
    });
    assert_eq!(json, expected, "wire format drift");

    // Round-trip back to a typed event.
    let back: AgentEvent = serde_json::from_value(json).expect("deserialize");
    match back {
        AgentEvent::ToolExecutionUpdate { lines, .. } => {
            assert_eq!(lines.len(), 2);
            assert_eq!(lines[0].level, ConsoleLevel::Muted);
            assert_eq!(lines[1].level, ConsoleLevel::Danger);
        }
        other => panic!("expected ToolExecutionUpdate, got {other:?}"),
    }
}

#[tokio::test]
#[cfg(unix)]
async fn bash_emits_command_header_at_muted() {
    let transport = MockTransport::new()
        .with_tool_call_response(
            "bash",
            "c1",
            serde_json::json!({"command": "echo hello"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(BashTool::new()) as BoxedTool);
    builder.set_approval_policy(Arc::new(AutoAcceptAllPolicy));
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let updates: Vec<_> = collector
        .events()
        .into_iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionUpdate { lines, .. } => Some(lines),
            _ => None,
        })
        .collect();

    // First update should be the muted "$ ..." command header.
    let first = updates.first().expect("at least one update");
    let header = first.first().expect("non-empty lines");
    assert_eq!(header.level, ConsoleLevel::Muted);
    assert!(
        header.content.starts_with("$ "),
        "command header content: {:?}",
        header.content
    );
    assert!(header.content.contains("echo hello"));
}

#[tokio::test]
#[cfg(unix)]
async fn bash_streams_stdout_classified() {
    // Simulate a minimal cargo-test-ish output via printf.
    let cmd = r#"printf 'Compiling foo\nRunning 1 test\ntest a ... ok\nerror: bad\n'"#;
    let transport = MockTransport::new()
        .with_tool_call_response("bash", "c1", serde_json::json!({"command": cmd}))
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(BashTool::new()) as BoxedTool);
    builder.set_approval_policy(Arc::new(AutoAcceptAllPolicy));
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let lines: Vec<(String, ConsoleLevel)> = collector
        .events()
        .into_iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionUpdate { lines, .. } => Some(lines),
            _ => None,
        })
        .flatten()
        .map(|l| (l.content, l.level))
        .collect();

    assert!(
        lines.iter().any(|(c, l)| c == "Compiling foo" && *l == ConsoleLevel::Muted),
        "expected muted Compiling, got {lines:?}"
    );
    assert!(
        lines.iter().any(|(c, l)| c == "Running 1 test" && *l == ConsoleLevel::Warning),
        "expected warning Running, got {lines:?}"
    );
    assert!(
        lines.iter().any(|(c, l)| c == "test a ... ok" && *l == ConsoleLevel::Success),
        "expected success test ok, got {lines:?}"
    );
    assert!(
        lines.iter().any(|(c, l)| c == "error: bad" && *l == ConsoleLevel::Danger),
        "expected danger error, got {lines:?}"
    );
}
