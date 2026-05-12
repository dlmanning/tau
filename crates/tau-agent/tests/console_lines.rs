//! Wire-format test for `ToolExecutionUpdate { lines: Vec<ConsoleLine> }`.
//!
//! The v1 version of this file also tested `tau_tools::BashTool`'s
//! line classification (muted command headers, warning/danger/success
//! stream lines). Those tests are bash-specific host code, not v2
//! runtime concerns, so they're omitted here. The wire-format pin
//! survives — hosts that persist events depend on this exact JSON
//! shape.

use tau_agent::{AgentEvent, ConsoleLevel, ConsoleLine};

#[test]
fn tool_execution_update_serde_roundtrip_lines_shape() {
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
            { "content": "$ ls",       "level": "muted" },
            { "content": "error: bad", "level": "danger" },
        ]
    });
    assert_eq!(json, expected, "wire format drift");

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
