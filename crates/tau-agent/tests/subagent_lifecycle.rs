//! Tests for subagent lifecycle events, the subagent_report tool, and
//! approval-policy inheritance (gap #6).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use tau_agent::{ApprovalDecision, ApprovalPolicy, AutoAcceptAll, ToolRisk};
use tau_agent::{Concurrency, ExecutionContext, Tool, ToolResult};
use tau_agent::Transport;
use tau_agent::{AgentManager, AgentSpec, SpawnOpts};
use tau_agent::test_utils::*;
use tau_agent::{AgentEvent, BoxedTool, FleetEvent, SubagentOutcome};
// SubagentReportTool replicated inline below (v1 had it in tau-tools;
// v2 has no host-tools dep, so we fixture it locally).
struct SubagentReportTool;

impl SubagentReportTool {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for SubagentReportTool {
    fn name(&self) -> &str {
        "subagent_report"
    }
    fn description(&self) -> &str {
        "Self-label this subagent's outcome."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "tag": { "type": "string" },
                "summary": { "type": "string" }
            },
            "required": ["summary"]
        })
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }
    async fn execute(&self, args: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let tag = args.get("tag").and_then(|v| v.as_str()).map(String::from);
        let summary = args
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        ctx.progress
            .emit(AgentEvent::AgentReport { tag, summary });
        ToolResult::text("reported")
    }
}

fn make_manager(transport: Arc<dyn Transport>) -> Arc<AgentManager> {
    Arc::new(AgentManager::new(test_config(), transport, 4))
}

fn make_manager_with_rx(
    transport: Arc<dyn Transport>,
) -> (
    Arc<AgentManager>,
    tokio::sync::broadcast::Receiver<FleetEvent>,
) {
    let manager = Arc::new(AgentManager::new(test_config(), transport, 4));
    let rx = manager.subscribe();
    (manager, rx)
}

fn echo_spec() -> AgentSpec {
    AgentSpec {
        system_prompt: String::new(),
        tools: vec![],
        max_turns: 200,
    }
}

fn echo_spec_with_tools(tools: Vec<BoxedTool>) -> AgentSpec {
    AgentSpec {
        system_prompt: String::new(),
        tools,
        max_turns: 200,
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
        .spawn(
            echo_spec(),
            ("hello").to_string(),
            spawn_opts("test"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    let mut started_id: Option<String> = None;
    let mut completed_id: Option<String> = None;
    let mut completed_outcome: Option<SubagentOutcome> = None;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            FleetEvent::AgentStarted { agent_id, .. } => started_id = Some(agent_id),
            FleetEvent::AgentCompleted {
                agent_id, outcome, ..
            } => {
                completed_id = Some(agent_id);
                completed_outcome = Some(outcome);
            }
            _ => {}
        }
    }
    assert!(started_id.is_some(), "AgentStarted emitted");
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
            .spawn(
                echo_spec(),
                ("hi").to_string(),
                spawn_opts("slow"),
                cancel_clone,
            )
            .await;
    });

    // Wait for AgentStarted to confirm it's in flight.
    let started = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Ok(FleetEvent::AgentStarted { .. }) = rx.recv().await {
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
        if let FleetEvent::AgentCompleted {
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
        if let FleetEvent::AgentReport {
            agent_id,
            tag,
            summary,
            ..
        } = ev
        {
            if agent_id == result.agent_id {
                assert_eq!(tag.as_deref(), Some("passed"));
                assert_eq!(summary, "all green");
                found = true;
                break;
            }
        }
    }
    assert!(found, "AgentReport reaches the fleet channel");
}

/// Regression: when a subagent's prompt fails *after* it was committed
/// to the registry's `running` set, the error path must drop it from
/// `running` (via `drop_running`), not merely remove its spec (via
/// `abandon`). Otherwise the dead agent leaks as a perpetually-`Running`
/// entry in every `snapshot()` and the "spec ⇔ bucket" invariant breaks.
#[tokio::test]
async fn failed_spawn_does_not_leak_running_entry() {
    // ErrorTransport fails the prompt *after* commit_running has run.
    let transport: Arc<dyn Transport> = ErrorTransport::create("boom");
    let manager = make_manager(transport);

    let result = manager
        .spawn(
            echo_spec(),
            "go".to_string(),
            spawn_opts("doomed"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await;

    assert!(result.is_err(), "spawn should fail when the prompt errors");

    let snapshot = manager.snapshot();
    assert!(
        snapshot.agents.is_empty(),
        "failed spawn must leave no registry entry; got {} agent(s): {:?}",
        snapshot.agents.len(),
        snapshot
            .agents
            .iter()
            .map(|a| (&a.agent_id, &a.status))
            .collect::<Vec<_>>()
    );
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
        .with_default_approval_policy(Arc::new(AutoAcceptAll));
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
        .spawn(
            echo_spec(),
            ("first").to_string(),
            spawn_opts("agent under test"),
            cancel.clone(),
        )
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
            FleetEvent::AgentResumed { agent_id, .. } => {
                saw_resumed_id = Some(agent_id);
            }
            FleetEvent::AgentCompleted { agent_id, .. } => {
                if saw_resumed_id.as_deref() == Some(agent_id.as_str()) {
                    saw_completed_after_resume = true;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(
        saw_resumed_id.is_some(),
        "AgentResumed emitted on send()"
    );
    assert!(
        saw_completed_after_resume,
        "AgentCompleted bracketed the resume"
    );
}

// Skipped in v2: depends on `tau_tools::{AgentTool, SpecResolver}` —
// host-side recursive-spawn machinery v2 doesn't ship with. The
// underlying behavior (per-spawn approval_policy propagating into a
// descendant via `SpawnOpts::approval_policy`) is exercised one level
// deep by `spawn_request_approval_policy_overrides_parent` below.

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
        .with_default_approval_policy(Arc::new(AutoAcceptAll));
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

// ─── ExecutionContext::agent_id ─────────────────────────────────────

/// A tool that records `ctx.agent_id` on each execute call. Verifies
/// the runtime threads the owning agent's id through to tools at
/// invocation time.
struct IdRecorder {
    seen_ids: Arc<parking_lot::Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for IdRecorder {
    fn name(&self) -> &str {
        "id_recorder"
    }
    fn description(&self) -> &str {
        ""
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    async fn execute(&self, _args: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let id = ctx.agent_id.clone().unwrap_or_default();
        self.seen_ids.lock().push(id);
        ToolResult::text("ok")
    }
}

/// `ExecutionContext::agent_id` must be the spawning agent's id, even
/// when the same tool instance is shared across multiple spawns. Tools
/// carry no per-agent state; the runtime supplies identity per call.
#[tokio::test]
async fn execution_context_carries_owning_agent_id() {
    let seen_ids = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
    let recorder: BoxedTool = Arc::new(IdRecorder {
        seen_ids: seen_ids.clone(),
    });

    // Each manager gets its own ToolCallTransport so its quota isn't
    // drained by the other spawn. The recorder is shared — that's the
    // point: one stateless tool, two spawns, identity arrives via
    // ExecutionContext rather than tool state.
    let spec_for = || AgentSpec {
        system_prompt: String::new(),
        tools: vec![recorder.clone()],
        max_turns: 200,
    };

    let mgr1 = make_manager(ToolCallTransport::create(1, "id_recorder"));
    let r1 = mgr1
        .spawn(
            spec_for(),
            "first".to_string(),
            spawn_opts("first"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("first spawn");

    let mgr2 = make_manager(ToolCallTransport::create(1, "id_recorder"));
    let r2 = mgr2
        .spawn(
            spec_for(),
            "second".to_string(),
            spawn_opts("second"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("second spawn");

    let ids = seen_ids.lock().clone();
    assert_eq!(ids.len(), 2, "tool invoked once per spawn (got {ids:?})");
    assert!(
        ids.contains(&r1.agent_id),
        "first agent's id reached the tool: {ids:?}"
    );
    assert!(
        ids.contains(&r2.agent_id),
        "second agent's id reached the tool: {ids:?}"
    );
}

/// `respec` is a transition: on success the old id no longer resolves
/// (`spec_for(old) -> None`, `find_agent(old) -> None`) and the new
/// handle is usable.
#[tokio::test]
async fn respec_transitions_old_id_to_new() {
    let transport: Arc<dyn Transport> = TextTransport::create("done");
    let manager = make_manager(transport);

    let r = manager
        .spawn(
            echo_spec(),
            "first".to_string(),
            spawn_opts("orig"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");
    let old_id = r.agent_id.clone();
    assert!(manager.spec_for(&old_id).is_some());

    let new_handle = manager
        .respec(
            &old_id,
            AgentSpec {
                system_prompt: "new prompt".to_string(),
                tools: vec![],
                max_turns: 200,
            },
        )
        .await
        .expect("respec succeeds");

    assert!(manager.spec_for(&old_id).is_none(), "old spec dropped");
    let new_id = new_handle.agent_id().expect("stamped").to_string();
    assert_ne!(new_id, old_id, "respec assigns a new id");
    assert!(manager.spec_for(&new_id).is_some(), "new spec recorded");
    // v2: spawn_interactive leaves the agent in the registry's running
    // map; the test simply ends and the manager is dropped.
    let _ = new_id;
}

/// Respec while the agent is currently running (resumed via `send`)
/// must fail and leave the registry intact. Regression for the race
/// where `respec` checked `running_handles`, released the lock, then
/// dropped the spec while a concurrent `send` was driving the agent.
///
/// We exercise this by running `send` (which mid-flight tracks the
/// agent in `running_handles` and removes it from `agents`) on one
/// task while another tries to respec the same id. Either:
///   - respec sees `running_handles` and rejects, OR
///   - respec sees agents-empty (mid-resume) and rejects with
///     "no idle agent."
/// In neither case may the spec be dropped while the agent is alive.
#[tokio::test]
async fn respec_concurrent_with_send_does_not_drop_live_spec() {
    let transport: Arc<dyn Transport> = SlowTransport::create(100);
    let manager = make_manager(transport);

    let r = manager
        .spawn(
            echo_spec(),
            "first".to_string(),
            spawn_opts("racey"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");
    let id = r.agent_id.clone();

    let mgr_send = manager.clone();
    let id_send = id.clone();
    let send_task = tokio::spawn(async move {
        mgr_send
            .send(
                &id_send,
                "follow up",
                tokio_util::sync::CancellationToken::new(),
            )
            .await
    });

    // Give send() a moment to grab the lock and start the resume.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    let respec_result = manager
        .respec(
            &id,
            AgentSpec {
                system_prompt: "new".into(),
                tools: vec![],
                max_turns: 200,
            },
        )
        .await;

    let send_result = send_task.await.expect("send task joins");

    // respec must not have succeeded while the agent was being resumed.
    assert!(
        respec_result.is_err(),
        "respec must reject while agent is resuming"
    );
    // send must have completed normally; the spec must still be alive
    // for the agent that just finished its resume.
    assert!(send_result.is_ok(), "send completes despite race");
    assert!(
        manager.spec_for(&id).is_some(),
        "spec preserved through race"
    );
}

/// `respec` against an id that isn't in the registry surfaces a
/// structured `AgentNotFound` — callers can branch on it without
/// string-matching the Display impl.
#[tokio::test]
async fn respec_missing_id_returns_agent_not_found() {
    let transport: Arc<dyn Transport> = TextTransport::create("done");
    let manager = make_manager(transport);

    let err = match manager
        .respec(
            "definitely-not-an-agent",
            AgentSpec {
                system_prompt: "x".into(),
                tools: vec![],
                max_turns: 1,
            },
        )
        .await
    {
        Ok(_) => panic!("respec on missing id must fail"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, tau_agent::Error::AgentNotFound { id } if id == "definitely-not-an-agent"),
        "expected Error::AgentNotFound, got: {err:?}"
    );
}
