//! Integration tests for `AgentManager::snapshot()` and the
//! registry-side bookkeeping that feeds it.

use std::sync::Arc;

use tau_agent::*;

fn make_manager(transport: Arc<dyn Transport>) -> Arc<AgentManager> {
    let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(256);
    Arc::new(AgentManager::new(
        tx,
        test_utils::test_config(),
        transport,
        4,
    ))
}

fn spec_with_tools(tools: Vec<BoxedTool>) -> AgentSpec {
    AgentSpec {
        system_prompt: String::new(),
        tools,
        max_turns: 5,
        allows_worktree: false,
        allowed_subagent_specs: None,
    }
}

fn opts(description: &str) -> SpawnOpts {
    SpawnOpts {
        description: description.into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn snapshot_empty_manager() {
    let mgr = make_manager(test_utils::TextTransport::create("hi"));
    let snap = mgr.snapshot();
    assert!(snap.agents.is_empty());
}

#[tokio::test]
async fn snapshot_reports_completed_agent_with_usage_and_timestamps() {
    let mgr = make_manager(test_utils::TextTransport::create("done"));
    let result = mgr
        .spawn(
            spec_with_tools(vec![]),
            "first".into(),
            opts("alpha"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    let snap = mgr.snapshot();
    assert_eq!(snap.agents.len(), 1, "one agent tracked");
    let entry = snap
        .agents
        .iter()
        .find(|a| a.agent_id == result.agent_id)
        .expect("agent in snapshot");
    assert_eq!(entry.description, "alpha");
    assert_eq!(entry.status, AgentStatus::Idle, "finished → idle");
    assert!(
        entry.usage.input > 0,
        "TurnEnd usage accumulated (got {})",
        entry.usage.input
    );
    assert!(
        entry.usage.output > 0,
        "TurnEnd usage accumulated (got {})",
        entry.usage.output
    );
    assert!(entry.started_at.is_some(), "started_at stamped");
    assert!(entry.completed_at.is_some(), "completed_at stamped");
    assert!(
        entry.started_at.unwrap() <= entry.completed_at.unwrap(),
        "started_at <= completed_at"
    );
}

#[tokio::test]
async fn snapshot_counts_tool_invocations() {
    // 2 turns of tool calls, then a terminal turn.
    let transport = test_utils::ToolCallTransport::create(2, "echo");
    let mgr = make_manager(transport);
    let tools: Vec<BoxedTool> = vec![Arc::new(test_utils::EchoTool)];

    let result = mgr
        .spawn(
            spec_with_tools(tools),
            "go".into(),
            opts("tool-user"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    let snap = mgr.snapshot();
    let entry = snap
        .agents
        .iter()
        .find(|a| a.agent_id == result.agent_id)
        .expect("agent in snapshot");

    assert_eq!(
        entry.tool_use_count, 2,
        "two tool calls observed via ToolExecutionEnd"
    );
}

#[tokio::test]
async fn snapshot_reports_two_agents() {
    let mgr = make_manager(test_utils::TextTransport::create("ok"));
    let a = mgr
        .spawn(
            spec_with_tools(vec![]),
            "first".into(),
            opts("alpha"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn a");
    let b = mgr
        .spawn(
            spec_with_tools(vec![]),
            "second".into(),
            opts("beta"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn b");

    let snap = mgr.snapshot();
    assert_eq!(snap.agents.len(), 2, "both agents tracked");

    let ids: Vec<&str> = snap.agents.iter().map(|s| s.agent_id.as_str()).collect();
    assert!(ids.contains(&a.agent_id.as_str()));
    assert!(ids.contains(&b.agent_id.as_str()));

    for s in &snap.agents {
        assert_eq!(s.status, AgentStatus::Idle);
        assert!(s.usage.input > 0, "usage tracked for {}", s.agent_id);
        assert!(s.completed_at.is_some());
    }
}

#[tokio::test]
async fn snapshot_after_resume_keeps_started_at_refreshes_completed_at() {
    let mgr = make_manager(test_utils::TextTransport::create("ok"));
    let result = mgr
        .spawn(
            spec_with_tools(vec![]),
            "first".into(),
            opts("resumable"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    let first_snap = mgr.snapshot();
    let first_entry = first_snap
        .agents
        .iter()
        .find(|a| a.agent_id == result.agent_id)
        .expect("agent in first snapshot");
    let first_started = first_entry.started_at.expect("started_at set");
    let first_completed = first_entry.completed_at.expect("completed_at set");
    let first_usage_input = first_entry.usage.input;

    // Small delay so timestamps can move forward.
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    let _resumed = mgr
        .send(
            &result.agent_id,
            "follow up",
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("send");

    let second_snap = mgr.snapshot();
    let second_entry = second_snap
        .agents
        .iter()
        .find(|a| a.agent_id == result.agent_id)
        .expect("agent in second snapshot");

    assert_eq!(
        second_entry.started_at.unwrap(),
        first_started,
        "started_at preserved across resume"
    );
    assert!(
        second_entry.completed_at.unwrap() >= first_completed,
        "completed_at refreshed (or equal at clock granularity)"
    );
    assert!(
        second_entry.usage.input > first_usage_input,
        "usage continues accumulating across resume (had {}, now {})",
        first_usage_input,
        second_entry.usage.input
    );
}
