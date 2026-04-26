//! Tests for the plan-step boundary tools (gap #3).
//!
//! The tools are dumb event emitters — call → event on the agent's
//! channel. These tests just verify the wiring.

use std::sync::Arc;

use tau_agent::test_utils::*;
use tau_agent::{AgentBuilder, AgentEvent, BoxedTool};
use tau_tools::{PlanCompleteTool, StepCompletedTool, StepStartedTool};

#[tokio::test]
async fn step_started_emits_plan_step_started_event() {
    let transport = MockTransport::new()
        .with_tool_call_response(
            "step_started",
            "c1",
            serde_json::json!({"step_id": "s1", "activity": "running tests"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(StepStartedTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let matched = collector.events().into_iter().any(|e| matches!(
        e,
        AgentEvent::PlanStepStarted { step_id, activity, .. }
            if step_id == "s1" && activity.as_deref() == Some("running tests")
    ));
    assert!(matched, "PlanStepStarted should fire with the args");
}

#[tokio::test]
async fn step_completed_emits_plan_step_completed_event() {
    let transport = MockTransport::new()
        .with_tool_call_response(
            "step_completed",
            "c1",
            serde_json::json!({"step_id": "s1", "summary": "tests pass"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(StepCompletedTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let matched = collector.events().into_iter().any(|e| matches!(
        e,
        AgentEvent::PlanStepCompleted { step_id, summary, .. }
            if step_id == "s1" && summary.as_deref() == Some("tests pass")
    ));
    assert!(matched);
}

#[tokio::test]
async fn plan_complete_emits_plan_completed_event() {
    let transport = MockTransport::new()
        .with_tool_call_response(
            "plan_complete",
            "c1",
            serde_json::json!({"summary": "all 3 steps shipped"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(PlanCompleteTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let matched = collector.events().into_iter().any(|e| matches!(
        e,
        AgentEvent::PlanCompleted { summary, .. } if summary == "all 3 steps shipped"
    ));
    assert!(matched);
}

#[tokio::test]
async fn step_pair_duration_is_measurable() {
    // Event ordering: a started → completed pair should yield a positive
    // duration the host can compute from the two timestamps.
    let transport = MockTransport::new()
        .with_tool_call_response(
            "step_started",
            "c1",
            serde_json::json!({"step_id": "s1"}),
        )
        .with_tool_call_response(
            "step_completed",
            "c2",
            serde_json::json!({"step_id": "s1"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(StepStartedTool::new()) as BoxedTool);
    builder.add_tool(Arc::new(StepCompletedTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let events = collector.events();
    let started_at = events.iter().find_map(|e| match e {
        AgentEvent::PlanStepStarted {
            step_id,
            started_at,
            ..
        } if step_id == "s1" => Some(*started_at),
        _ => None,
    });
    let completed_at = events.iter().find_map(|e| match e {
        AgentEvent::PlanStepCompleted {
            step_id,
            completed_at,
            ..
        } if step_id == "s1" => Some(*completed_at),
        _ => None,
    });
    let s = started_at.expect("started event");
    let c = completed_at.expect("completed event");
    assert!(c >= s, "completed_at >= started_at");
}

#[tokio::test]
async fn step_event_from_subagent_arrives_wrapped_in_subagent_variant() {
    use tau_agent::manager::{AgentManager, AgentType, SpawnRequest};
    use tau_agent::transport::Transport;

    // Subagent's transport: one turn calling step_started, then a final
    // text turn so the loop terminates.
    let sub_transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            .with_tool_call_response(
                "step_started",
                "c1",
                serde_json::json!({"step_id": "s1", "activity": "doing"}),
            )
            .with_text_response("done"),
    );

    let tools: Vec<BoxedTool> = vec![Arc::new(StepStartedTool::new())];
    let (parent_event_tx, mut parent_event_rx) =
        tokio::sync::broadcast::channel::<AgentEvent>(64);
    let manager = Arc::new(AgentManager::new(
        parent_event_tx,
        tools,
        test_config(),
        sub_transport,
        4,
    ));

    let cancel = tokio_util::sync::CancellationToken::new();
    let req = SpawnRequest {
        agent_type: AgentType::GeneralPurpose,
        prompt: "go".into(),
        description: "executor".into(),
        model: None,
        cwd: None,
        isolation: None,
        depth: 0,
        inherit_history_from: None,
    };
    let result = manager.spawn(req, cancel).await.expect("spawn");
    let expected_id = result.agent_id;

    // Drain parent's events and look for a wrapped PlanStepStarted.
    let mut wrapped = false;
    while let Ok(ev) = parent_event_rx.try_recv() {
        if let AgentEvent::Subagent {
            agent_id, event, ..
        } = ev
        {
            if agent_id == expected_id {
                if let AgentEvent::PlanStepStarted {
                    step_id, activity, ..
                } = *event
                {
                    if step_id == "s1" && activity.as_deref() == Some("doing") {
                        wrapped = true;
                        break;
                    }
                }
            }
        }
    }
    assert!(
        wrapped,
        "subagent's PlanStepStarted reaches parent wrapped in Subagent variant"
    );
}

#[tokio::test]
async fn invalid_step_args_returns_error_without_emitting_event() {
    // Schema-violating args should make the actor emit a validation error
    // without our tool ever being invoked, so no PlanStepStarted should fire.
    let transport = MockTransport::new()
        .with_tool_call_response(
            "step_started",
            "c1",
            serde_json::json!({"wrong_field": "x"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(StepStartedTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let emitted = collector
        .events()
        .into_iter()
        .any(|e| matches!(e, AgentEvent::PlanStepStarted { .. }));
    assert!(!emitted, "validation error should prevent emission");
}
