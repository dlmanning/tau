//! Tests for `handle.interrupt()` — graceful stop distinct from `abort()`.
//!
//! `interrupt()` requests "finish the in-flight tool/turn, then exit
//! the loop instead of starting another." The flag is checked at the
//! top of each new turn, AFTER the prior tool batch completes and
//! BEFORE the next LLM call.

use std::sync::Arc;
use std::time::Duration;

use tau_agent::test_utils::*;
use tau_agent::*;

/// After the first `TurnEnd`, request a graceful interrupt. The actor
/// should finish applying the pending tool result, then stop before
/// calling the LLM again. The terminal `AgentEnd` should have
/// `interrupted: true`, and no further `TurnStart` should fire.
#[tokio::test]
async fn interrupt_stops_after_current_turn() {
    // 3 tool-call turns followed by a "Done" turn. Without interrupt,
    // the agent would emit 4 TurnStarts. With interrupt after the
    // first TurnEnd, we expect the agent to stop early.
    let transport = ToolCallTransport::create(3, "slow");
    let mut builder = AgentBuilder::new(test_config(), transport);
    // Slow tool: gives the test time to observe TurnStart and call
    // interrupt() before the actor races into the next turn.
    builder.add_tool(Arc::new(SlowTool { delay_ms: 200 }));
    let handle = builder.spawn().await.unwrap();
    let collector = EventCollector::from_handle(&handle);

    let rx = handle.prompt("go").await.unwrap();

    // Request a graceful stop as soon as the first turn STARTS. The
    // semantic under test: the actor finishes the in-flight turn
    // (stream + tool), then exits before the next LLM call. This is
    // race-free: regardless of how fast the in-flight turn completes,
    // the flag is set before the next `step_prepare` runs.
    collector
        .wait_for_event(|e| matches!(e, AgentEvent::TurnStart { turn_number: 1 }))
        .await;
    handle.interrupt();

    // Wait for the prompt to complete (the receiver fires when the
    // actor calls reply.send, which happens during emit_end_and_idle).
    let _ = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .expect("prompt did not complete after interrupt")
        .expect("prompt sender dropped");

    // Drain a beat to ensure any straggler events are collected.
    collector.wait_for_end().await;

    let events = collector.events();

    // The terminal event must be `AgentEnd { interrupted: true }`.
    let end = events
        .iter()
        .find(|e| matches!(e, AgentEvent::AgentEnd { .. }))
        .expect("missing AgentEnd");
    match end {
        AgentEvent::AgentEnd { interrupted, .. } => {
            assert!(
                *interrupted,
                "AgentEnd should have interrupted=true; events: {:?}",
                collector.event_names()
            );
        }
        _ => unreachable!(),
    }

    // Only one TurnStart should appear: the interrupt was requested
    // right after the first TurnEnd, so the second turn must not have
    // started.
    let turn_starts = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnStart { .. }))
        .count();
    assert_eq!(
        turn_starts,
        1,
        "expected exactly 1 TurnStart, got {turn_starts}; events: {:?}",
        collector.event_names()
    );

    // The pending tool from the first turn must have been executed
    // before the interrupt was honored (post-tool, pre-next-LLM).
    let tool_ends = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
        .count();
    assert_eq!(
        tool_ends, 1,
        "first turn's tool should have completed before interrupt"
    );
}

/// Regression: without `interrupt()`, normal multi-turn flow proceeds
/// to completion and the terminal event has `interrupted: false`.
#[tokio::test]
async fn normal_flow_not_interrupted() {
    let transport = ToolCallTransport::create(2, "echo");
    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(Arc::new(EchoTool));
    let handle = builder.spawn().await.unwrap();
    let collector = EventCollector::from_handle(&handle);

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let events = collector.events();
    let end = events
        .iter()
        .find(|e| matches!(e, AgentEvent::AgentEnd { .. }))
        .expect("missing AgentEnd");
    match end {
        AgentEvent::AgentEnd { interrupted, .. } => {
            assert!(
                !*interrupted,
                "AgentEnd.interrupted should be false in normal flow"
            );
        }
        _ => unreachable!(),
    }

    // All 3 turns (2 tool-call + 1 final) should have started.
    let turn_starts = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::TurnStart { .. }))
        .count();
    assert_eq!(turn_starts, 3, "expected all 3 turns to run");
}

/// The interrupt flag must not latch across prompts: a second prompt
/// after an interrupted one should run normally even though the flag
/// was set during the first.
#[tokio::test]
async fn interrupt_does_not_latch_across_prompts() {
    let transport = ToolCallTransport::create(3, "slow");
    let mut builder = AgentBuilder::new(test_config(), transport);
    // Slow tool: gives the test time to observe TurnStart and call
    // interrupt() before the actor races into the next turn.
    builder.add_tool(Arc::new(SlowTool { delay_ms: 200 }));
    let handle = builder.spawn().await.unwrap();

    // First prompt: interrupt as soon as the first turn starts.
    let collector1 = EventCollector::from_handle(&handle);
    let rx = handle.prompt("first").await.unwrap();
    collector1
        .wait_for_event(|e| matches!(e, AgentEvent::TurnStart { turn_number: 1 }))
        .await;
    handle.interrupt();
    let _ = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .unwrap();
    collector1.wait_for_end().await;

    let end1 = collector1
        .events()
        .into_iter()
        .find(|e| matches!(e, AgentEvent::AgentEnd { .. }))
        .unwrap();
    assert!(matches!(
        end1,
        AgentEvent::AgentEnd {
            interrupted: true,
            ..
        }
    ));

    // Second prompt: no interrupt; should run to completion.
    let collector2 = EventCollector::from_handle(&handle);
    handle.prompt_and_wait("second").await.unwrap();
    collector2.wait_for_end().await;

    let end2 = collector2
        .events()
        .into_iter()
        .find(|e| matches!(e, AgentEvent::AgentEnd { .. }))
        .unwrap();
    assert!(
        matches!(
            end2,
            AgentEvent::AgentEnd {
                interrupted: false,
                ..
            }
        ),
        "second prompt should not be marked interrupted"
    );
}
