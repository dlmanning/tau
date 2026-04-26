//! Tests for subagent lifecycle events, the subagent_report tool, and
//! approval-policy inheritance (gap #6).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use tau_agent::approval::{ApprovalDecision, ApprovalPolicy, AutoAcceptAllPolicy, ToolRisk};
use tau_agent::manager::{AgentManager, AgentType, SpawnRequest};
use tau_agent::test_utils::*;
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};
use tau_agent::transport::Transport;
use tau_agent::{AgentEvent, BoxedTool, SubagentOutcome};
use tau_tools::SubagentReportTool;

fn make_manager(transport: Arc<dyn Transport>, tools: Vec<BoxedTool>) -> Arc<AgentManager> {
    let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(256);
    Arc::new(AgentManager::new(tx, tools, test_config(), transport, 4))
}

fn make_manager_with_rx(
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
) -> (Arc<AgentManager>, tokio::sync::broadcast::Receiver<AgentEvent>) {
    let (tx, rx) = tokio::sync::broadcast::channel::<AgentEvent>(256);
    (
        Arc::new(AgentManager::new(tx, tools, test_config(), transport, 4)),
        rx,
    )
}

fn fresh_request(prompt: &str, description: &str) -> SpawnRequest {
    SpawnRequest {
        agent_type: AgentType::GeneralPurpose,
        prompt: prompt.into(),
        description: description.into(),
        model: None,
        cwd: None,
        isolation: None,
        depth: 0,
        inherit_history_from: None,
        approval_policy: None,
    }
}

#[tokio::test]
async fn spawn_emits_started_then_completed() {
    let transport: Arc<dyn Transport> = TextTransport::create("done");
    let (manager, mut rx) = make_manager_with_rx(transport, vec![]);

    manager
        .spawn(
            fresh_request("hello", "test"),
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
    let (manager, mut rx) = make_manager_with_rx(transport, vec![]);

    let cancel = tokio_util::sync::CancellationToken::new();
    let manager_clone = manager.clone();
    let cancel_clone = cancel.clone();
    let spawn_handle = tokio::spawn(async move {
        let _ = manager_clone
            .spawn(fresh_request("hi", "slow"), cancel_clone)
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
    let (manager, mut rx) = make_manager_with_rx(transport, tools);

    let result = manager
        .spawn(
            fresh_request("report it", "report agent"),
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

    let manager = make_manager(transport, vec![tool]);
    let manager = Arc::try_unwrap(manager)
        .ok()
        .expect("unique")
        .with_parent_approval_policy(Arc::new(AutoAcceptAllPolicy));
    let manager = Arc::new(manager);

    manager
        .spawn(
            fresh_request("go", "subagent"),
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
    let (manager, mut rx) = make_manager_with_rx(transport, vec![]);

    let cancel = tokio_util::sync::CancellationToken::new();
    let first = manager
        .spawn(fresh_request("first", "agent under test"), cancel.clone())
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
    // and below unless overridden again deeper". Set up a manager with
    // an agent_tool_factory so a depth-1 subagent can spawn a depth-2,
    // and pass an AlwaysReject override at depth-1. The depth-2 should
    // inherit it (not the manager's default AutoAccept), so its
    // elevated tool gets rejected and never runs.
    use tau_agent::handle::AgentHandle;
    use tau_tools::AgentTool;

    let invocations = Arc::new(AtomicU32::new(0));
    let danger: BoxedTool = Arc::new(CountingDangerTool {
        invocations: invocations.clone(),
    });

    // Depth-1 transport: model spawns a depth-2 with no explicit policy.
    // Depth-2 transport: model calls the danger tool.
    // We need a single transport that responds differently at different
    // depths — simulate by giving depth-1 an `agent` tool call first,
    // then a final text turn after the subagent returns.
    //
    // The depth-2 prompt comes through the agent tool, which calls
    // manager.spawn — same transport handles it. We queue both turns
    // for depth-1 and depth-2 in order.
    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            // Depth-1, turn 1: spawn a sub via agent tool
            .with_tool_call_response(
                "agent",
                "c1",
                serde_json::json!({
                    "subagent_type": "general-purpose",
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

    let tools: Vec<BoxedTool> = vec![danger];
    let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(256);
    let manager = Arc::new(
        AgentManager::new(tx, tools, test_config(), transport, 4)
            .with_parent_approval_policy(Arc::new(AutoAcceptAllPolicy)),
    );

    let mgr_for_factory = manager.clone();
    manager.set_agent_tool_factory(Arc::new(
        move |depth, handle: AgentHandle, _parent_type, parent_policy| {
            let tool = AgentTool::new(mgr_for_factory.clone(), depth)
                .with_handle(handle)
                .with_inherited_policy(parent_policy);
            Arc::new(tool)
        },
    ));

    // Depth-1 gets the override; depth-2 should inherit it.
    let mut req = fresh_request("nest", "depth1");
    req.approval_policy = Some(Arc::new(AlwaysRejectPolicy("inherited at depth-2")));

    manager
        .spawn(req, tokio_util::sync::CancellationToken::new())
        .await
        .expect("spawn");

    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "depth-2 inherited depth-1's reject policy; danger tool did NOT run"
    );
}

#[tokio::test]
async fn spawn_request_approval_policy_overrides_parent() {
    // Parent policy = AutoAccept (would let danger through). Spawn request
    // overrides with AlwaysReject. Expect 0 invocations.
    let invocations = Arc::new(AtomicU32::new(0));
    let tool: BoxedTool = Arc::new(CountingDangerTool {
        invocations: invocations.clone(),
    });
    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            .with_tool_call_response("danger", "c1", serde_json::json!({}))
            .with_text_response("noted"),
    );

    let manager = make_manager(transport, vec![tool]);
    let manager = Arc::try_unwrap(manager)
        .ok()
        .expect("unique")
        .with_parent_approval_policy(Arc::new(AutoAcceptAllPolicy));
    let manager = Arc::new(manager);

    let mut req = fresh_request("go", "rejected exec");
    req.approval_policy = Some(Arc::new(AlwaysRejectPolicy("override applied")));

    manager
        .spawn(req, tokio_util::sync::CancellationToken::new())
        .await
        .expect("spawn");

    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "spawn-request override won; tool did not run"
    );
}
