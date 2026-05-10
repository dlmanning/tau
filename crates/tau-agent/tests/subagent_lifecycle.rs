//! Tests for subagent lifecycle events, the subagent_report tool, and
//! approval-policy inheritance (gap #6).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use tau_agent::approval::{ApprovalDecision, ApprovalPolicy, AutoAcceptAllPolicy, ToolRisk};
use tau_agent::manager::{AgentManager, AgentSpec, SpawnOpts};
use tau_agent::test_utils::*;
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};
use tau_agent::transport::Transport;
use tau_agent::{AgentEvent, BoxedTool, SubagentOutcome};
use tau_tools::SubagentReportTool;

fn make_manager(transport: Arc<dyn Transport>) -> Arc<AgentManager> {
    let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(256);
    Arc::new(AgentManager::new(tx, test_config(), transport, 4))
}

fn make_manager_with_rx(
    transport: Arc<dyn Transport>,
) -> (Arc<AgentManager>, tokio::sync::broadcast::Receiver<AgentEvent>) {
    let (tx, rx) = tokio::sync::broadcast::channel::<AgentEvent>(256);
    (
        Arc::new(AgentManager::new(tx, test_config(), transport, 4)),
        rx,
    )
}

fn echo_spec() -> AgentSpec {
    AgentSpec {
        system_prompt: String::new(),
        tools: vec![],
        max_turns: 200,
        allows_worktree: false,
        allowed_subagent_specs: None,
    }
}

fn echo_spec_with_tools(tools: Vec<BoxedTool>) -> AgentSpec {
    AgentSpec {
        system_prompt: String::new(),
        tools,
        max_turns: 200,
        allows_worktree: false,
        allowed_subagent_specs: None,
    }
}

fn spawn_opts(description: &str) -> SpawnOpts {
    SpawnOpts {
        description: description.to_string(),
        ..Default::default()
    }
}

#[tokio::test]
async fn spawn_emits_started_then_completed() {
    let transport: Arc<dyn Transport> = TextTransport::create("done");
    let (manager, mut rx) = make_manager_with_rx(transport);

    manager
        .spawn(echo_spec(), ("hello").to_string(), spawn_opts("test"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    let mut started_id: Option<String> = None;
    let mut completed_id: Option<String> = None;
    let mut completed_outcome: Option<SubagentOutcome> = None;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            AgentEvent::SubagentStarted { agent_id, .. } => started_id = Some(agent_id),
            AgentEvent::SubagentCompleted {
                agent_id, outcome, ..
            } => {
                completed_id = Some(agent_id);
                completed_outcome = Some(outcome);
            }
            _ => {}
        }
    }
    assert!(started_id.is_some(), "SubagentStarted emitted");
    assert_eq!(started_id, completed_id, "ids match across bracket");
    assert!(matches!(
        completed_outcome,
        Some(SubagentOutcome::Completed)
    ));
}

#[tokio::test]
async fn aborted_spawn_emits_completed_with_aborted_outcome() {
    // SlowTransport responds to cancel; we abort before it finishes.
    let transport: Arc<dyn Transport> = SlowTransport::create(2_000);
    let (manager, mut rx) = make_manager_with_rx(transport);

    let cancel = tokio_util::sync::CancellationToken::new();
    let manager_clone = manager.clone();
    let cancel_clone = cancel.clone();
    let spawn_handle = tokio::spawn(async move {
        let _ = manager_clone
            .spawn(echo_spec(), ("hi").to_string(), spawn_opts("slow"), cancel_clone)
            .await;
    });

    // Wait for SubagentStarted to confirm it's in flight.
    let started = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Ok(AgentEvent::SubagentStarted { .. }) = rx.recv().await {
                return;
            }
        }
    })
    .await;
    assert!(started.is_ok(), "subagent should start");

    cancel.cancel();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), spawn_handle).await;

    let mut found_aborted = false;
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::SubagentCompleted {
            outcome: SubagentOutcome::Aborted { .. },
            ..
        } = ev
        {
            found_aborted = true;
            break;
        }
    }
    assert!(found_aborted, "abort should yield Aborted outcome");
}

#[tokio::test]
async fn subagent_report_tool_emits_wrapped_event() {
    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            .with_tool_call_response(
                "subagent_report",
                "c1",
                serde_json::json!({"tag": "passed", "summary": "all green"}),
            )
            .with_text_response("done"),
    );
    let tools: Vec<BoxedTool> = vec![Arc::new(SubagentReportTool::new())];
    let (manager, mut rx) = make_manager_with_rx(transport);

    let result = manager
        .spawn(
            echo_spec_with_tools(tools),
            "report it".to_string(),
            spawn_opts("report agent"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    let mut found = false;
    while let Ok(ev) = rx.try_recv() {
        if let AgentEvent::Subagent {
            agent_id, event, ..
        } = ev
        {
            if agent_id == result.agent_id {
                if let AgentEvent::SubagentReport { tag, summary } = *event {
                    assert_eq!(tag.as_deref(), Some("passed"));
                    assert_eq!(summary, "all green");
                    found = true;
                    break;
                }
            }
        }
    }
    assert!(found, "SubagentReport reaches parent wrapped");
}

// ─── Approval policy inheritance ─────────────────────────────────────

/// A risky tool that records every successful invocation.
struct CountingDangerTool {
    invocations: Arc<AtomicU32>,
}

#[async_trait]
impl Tool for CountingDangerTool {
    fn name(&self) -> &str {
        "danger"
    }
    fn description(&self) -> &str {
        "elevated test tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }
    fn risk(&self, _arguments: &serde_json::Value) -> ToolRisk {
        ToolRisk::Elevated
    }
    async fn execute(&self, _args: serde_json::Value, _ctx: ExecutionContext) -> ToolResult {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        ToolResult::text("ran")
    }
}

#[tokio::test]
async fn subagent_inherits_parent_approval_policy() {
    // Without inheritance the elevated `danger` call would be rejected
    // (no interaction channel). With AutoAccept inherited from the parent
    // it should run.
    let invocations = Arc::new(AtomicU32::new(0));
    let tool: BoxedTool = Arc::new(CountingDangerTool {
        invocations: invocations.clone(),
    });
    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            .with_tool_call_response("danger", "c1", serde_json::json!({}))
            .with_text_response("done"),
    );

    let manager = make_manager(transport);
    let manager = Arc::try_unwrap(manager)
        .ok()
        .expect("unique")
        .with_parent_approval_policy(Arc::new(AutoAcceptAllPolicy));
    let manager = Arc::new(manager);

    manager
        .spawn(
            echo_spec_with_tools(vec![tool]),
            "go".to_string(),
            spawn_opts("subagent"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    assert_eq!(
        invocations.load(Ordering::SeqCst),
        1,
        "subagent inherited AutoAccept from parent and ran the elevated tool"
    );
}

/// A policy that always rejects with a known reason — lets us tell which
/// policy was actually applied to a subagent's call.
struct AlwaysRejectPolicy(&'static str);
impl ApprovalPolicy for AlwaysRejectPolicy {
    fn classify(
        &self,
        _tool: &str,
        _arguments: &serde_json::Value,
        _risk: ToolRisk,
    ) -> ApprovalDecision {
        ApprovalDecision::Reject(self.0.to_string())
    }
}

#[tokio::test]
async fn resume_emits_subagent_resumed_then_completed() {
    // First spawn (just so we have a stored agent), then resume it; expect
    // SubagentResumed before SubagentCompleted on the resume turn.
    use tau_agent::test_utils::TextTransport;
    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let (manager, mut rx) = make_manager_with_rx(transport);

    let cancel = tokio_util::sync::CancellationToken::new();
    let first = manager
        .spawn(echo_spec(), ("first").to_string(), spawn_opts("agent under test"), cancel.clone())
        .await
        .expect("spawn");

    // Drain initial spawn events so we only inspect the resume bracket.
    while rx.try_recv().is_ok() {}

    manager
        .send(&first.agent_id, "resume", cancel)
        .await
        .expect("resume");

    let mut saw_resumed_id: Option<String> = None;
    let mut saw_completed_after_resume = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            AgentEvent::SubagentResumed { agent_id, .. } => {
                saw_resumed_id = Some(agent_id);
            }
            AgentEvent::SubagentCompleted { agent_id, .. } => {
                if saw_resumed_id.as_deref() == Some(agent_id.as_str()) {
                    saw_completed_after_resume = true;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(saw_resumed_id.is_some(), "SubagentResumed emitted on send()");
    assert!(
        saw_completed_after_resume,
        "SubagentCompleted bracketed the resume"
    );
}

#[tokio::test]
async fn nested_subagents_inherit_top_level_override() {
    // Verify the spec: per-spawn approval_policy "applies at that level
    // and below unless overridden again deeper". The host owns recursive
    // spawning via a `SpecResolver` on the parent's `AgentTool`; the
    // resolver builds child specs whose `tools` already include another
    // `AgentTool` configured with the inherited policy.
    use tau_agent::AgentSpec;
    use tau_tools::{AgentTool, SpecResolver};

    let invocations = Arc::new(AtomicU32::new(0));
    let danger: BoxedTool = Arc::new(CountingDangerTool {
        invocations: invocations.clone(),
    });

    // Depth-1 transport: model spawns a depth-2 with no explicit policy.
    // Depth-2 transport: model calls the danger tool.
    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            // Depth-1, turn 1: spawn a sub via agent tool
            .with_tool_call_response(
                "agent",
                "c1",
                serde_json::json!({
                    "subagent_type": "deep",
                    "description": "deeper",
                    "prompt": "do dangerous thing"
                }),
            )
            // Depth-2, turn 1: call danger
            .with_tool_call_response("danger", "c2", serde_json::json!({}))
            // Depth-2, turn 2: final
            .with_text_response("d2 done")
            // Depth-1, turn 2: final
            .with_text_response("d1 done"),
    );

    let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(256);
    let manager = Arc::new(
        AgentManager::new(tx, test_config(), transport, 4)
            .with_parent_approval_policy(Arc::new(AutoAcceptAllPolicy)),
    );

    // depth-2 ("deep") spec contains the danger tool.
    let danger_for_resolver = danger.clone();
    let resolver: SpecResolver = Arc::new(move |name: &str, _depth: u32| match name {
        "deep" => Some(AgentSpec {
            system_prompt: String::new(),
            tools: vec![danger_for_resolver.clone()],
            max_turns: 200,
            allows_worktree: false,
            allowed_subagent_specs: None,
        }),
        _ => None,
    });

    // Depth-1's spawn runs under a permissive policy (so the `agent` tool
    // call itself succeeds). The AgentTool the depth-1 agent uses to
    // spawn depth-2 carries `with_inherited_policy(AlwaysReject)` — that
    // is what gets propagated into the depth-2 SpawnOpts via the tool's
    // execute path. So depth-2 runs under AlwaysReject and its danger
    // call gets rejected.
    let depth1_agent_tool = AgentTool::new(manager.clone(), 0)
        .with_spec_resolver(resolver)
        .with_inherited_policy(Arc::new(AlwaysRejectPolicy("inherited at depth-2")));
    let depth1_spec = AgentSpec {
        system_prompt: String::new(),
        tools: vec![Arc::new(depth1_agent_tool)],
        max_turns: 200,
        allows_worktree: false,
        allowed_subagent_specs: Some(vec!["deep".into()]),
    };

    // Depth-1 itself runs permissive — only its descendants get rejected.
    let opts = SpawnOpts {
        description: "depth1".into(),
        approval_policy: Some(Arc::new(AutoAcceptAllPolicy)),
        ..Default::default()
    };

    manager
        .spawn(
            depth1_spec,
            "nest".to_string(),
            opts,
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "depth-2 inherited depth-1's tool-attached reject policy; danger did NOT run"
    );
}

#[tokio::test]
async fn spawn_request_approval_policy_overrides_parent() {
    // Parent policy = AutoAccept (would let danger through). Spawn opts
    // override with AlwaysReject. Expect 0 invocations.
    let invocations = Arc::new(AtomicU32::new(0));
    let tool: BoxedTool = Arc::new(CountingDangerTool {
        invocations: invocations.clone(),
    });
    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            .with_tool_call_response("danger", "c1", serde_json::json!({}))
            .with_text_response("noted"),
    );

    let manager = make_manager(transport);
    let manager = Arc::try_unwrap(manager)
        .ok()
        .expect("unique")
        .with_parent_approval_policy(Arc::new(AutoAcceptAllPolicy));
    let manager = Arc::new(manager);

    let opts = SpawnOpts {
        description: "rejected exec".into(),
        approval_policy: Some(Arc::new(AlwaysRejectPolicy("override applied"))),
        ..Default::default()
    };

    manager
        .spawn(
            echo_spec_with_tools(vec![tool]),
            "go".to_string(),
            opts,
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "spawn-request override won; tool did not run"
    );
}

// ─── Tool::bind_to_agent post-construction hook ─────────────────────

/// A tool that records every call to `bind_to_agent` along with the
/// agent_id stamped on the handle. Used to verify the manager calls the
/// hook exactly once per tool per spawn, with the right handle.
struct BindRecorder {
    bound_ids: Arc<parking_lot::Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for BindRecorder {
    fn name(&self) -> &str {
        "bind_recorder"
    }
    fn description(&self) -> &str {
        ""
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    fn bind_to_agent(&self, handle: &tau_agent::handle::AgentHandle) {
        let id = handle.agent_id().map(str::to_string).unwrap_or_default();
        self.bound_ids.lock().push(id);
    }
    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: ExecutionContext,
    ) -> ToolResult {
        ToolResult::text("ok")
    }
}

#[tokio::test]
async fn bind_to_agent_called_with_owning_agent_handle() {
    // The runtime must call bind_to_agent on every tool in the spec
    // with the owning agent's handle. Verify by spawning two separate
    // agents and checking each tool instance saw the right id.
    let bound_ids = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
    let recorder: BoxedTool = Arc::new(BindRecorder {
        bound_ids: bound_ids.clone(),
    });

    let transport: Arc<dyn Transport> = TextTransport::create("done");
    let manager = make_manager(transport);

    let spec = AgentSpec {
        system_prompt: String::new(),
        tools: vec![recorder.clone()],
        max_turns: 200,
        allows_worktree: false,
        allowed_subagent_specs: None,
    };

    let r1 = manager
        .spawn(
            spec.clone(),
            "first".to_string(),
            spawn_opts("first"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("first spawn");

    let r2 = manager
        .spawn(
            spec,
            "second".to_string(),
            spawn_opts("second"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("second spawn");

    let ids = bound_ids.lock().clone();
    assert_eq!(
        ids.len(),
        2,
        "bind_to_agent fired once per spawn (got {ids:?})"
    );
    assert!(
        ids.contains(&r1.agent_id),
        "first agent's id was bound: {ids:?}"
    );
    assert!(
        ids.contains(&r2.agent_id),
        "second agent's id was bound: {ids:?}"
    );
}
