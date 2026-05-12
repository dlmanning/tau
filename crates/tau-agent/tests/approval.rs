//! Tests for the tool-call approval gate.

use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use tau_agent::core::approval::{
    ApprovalDecision, ApprovalPolicy, AutoAcceptAll, DefaultPolicy, ToolApprovalOutcome, ToolRisk,
};
use tau_agent::core::interaction::{InteractionKind, InteractionRequest, InteractionResponse};
use tau_agent::core::tool::{Concurrency, ExecutionContext, Tool, ToolResult};
use tau_agent::test_utils::*;
use tau_agent::*;
use tau_ai::{AssistantMetadata, Content, Message, Usage};

/// A tool that always reports `Elevated` risk and records each invocation.
struct ElevatedTool {
    name: &'static str,
    invocations: Arc<AtomicU32>,
}

#[async_trait]
impl Tool for ElevatedTool {
    fn name(&self) -> &str {
        self.name
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

/// Spawn a one-shot interaction handler that resolves every `tool.confirm`
/// request with the given response.
fn handle_confirm_with(
    mut rx: tokio::sync::mpsc::Receiver<InteractionRequest>,
    response: fn() -> InteractionResponse,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            if let InteractionKind::Typed { schema_id, .. } = &req.kind {
                if schema_id == "tool.confirm" {
                    let _ = req.response_tx.send(response());
                }
            }
        }
    })
}

#[tokio::test]
async fn auto_accept_policy_dispatches_elevated_without_gating() {
    let invocations = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(ElevatedTool {
        name: "danger",
        invocations: invocations.clone(),
    });

    let transport = MockTransport::new()
        .with_tool_call_response("danger", "c1", serde_json::json!({}))
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(tool);
    builder.set_approval_policy(Arc::new(AutoAcceptAll));
    let handle = builder.handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    assert_eq!(invocations.load(Ordering::SeqCst), 1, "tool should run");

    let approved = collector.events().into_iter().any(|e| {
        matches!(
            e,
            AgentEvent::ToolApprovalResolved {
                outcome: ToolApprovalOutcome::AutoApproved,
                ..
            }
        )
    });
    assert!(approved, "AutoAcceptAll should emit AutoApproved");
}

#[tokio::test]
async fn default_policy_local_risk_auto_approves() {
    let transport = MockTransport::new()
        .with_tool_call_response("echo", "c1", serde_json::json!({"text": "hi"}))
        .with_text_response("done");

    let echo: BoxedTool = Arc::new(EchoTool);
    let (handle, collector) = spawn_test_agent(transport, vec![echo]);

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let auto = collector.events().into_iter().any(|e| {
        matches!(
            e,
            AgentEvent::ToolApprovalResolved {
                outcome: ToolApprovalOutcome::AutoApproved,
                ..
            }
        )
    });
    assert!(auto, "Local risk should be AutoApproved by DefaultPolicy");

    let ran = collector.events().into_iter().any(|e| {
        matches!(
            e,
            AgentEvent::ToolExecutionEnd {
                tool_name,
                is_error: false,
                ..
            } if tool_name == "echo"
        )
    });
    assert!(ran, "echo tool should have actually executed");
}

#[tokio::test]
async fn default_policy_elevated_without_interaction_channel_rejects() {
    let invocations = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(ElevatedTool {
        name: "danger",
        invocations: invocations.clone(),
    });

    let transport = MockTransport::new()
        .with_tool_call_response("danger", "c1", serde_json::json!({}))
        .with_text_response("acknowledged");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(tool);
    let handle = builder.handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    assert_eq!(invocations.load(Ordering::SeqCst), 0, "tool must not run");

    let rejected_with_reason = collector.events().into_iter().any(|e| {
        matches!(
            e,
            AgentEvent::ToolApprovalResolved {
                outcome: ToolApprovalOutcome::Rejected { reason },
                ..
            } if reason.contains("no interaction channel")
        )
    });
    assert!(rejected_with_reason);

    let msgs = handle.messages().await.unwrap();
    let synth_error = msgs.iter().any(|m| {
        matches!(
            m,
            Message::ToolResult { is_error: true, content, .. }
                if content.iter().any(|c| c.as_text().is_some_and(|t| t.contains("rejected")))
        )
    });
    assert!(synth_error, "model should see a synth error tool result");
}

#[tokio::test]
async fn elevated_with_user_approval_dispatches() {
    let invocations = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(ElevatedTool {
        name: "danger",
        invocations: invocations.clone(),
    });

    let transport = MockTransport::new()
        .with_tool_call_response("danger", "c1", serde_json::json!({}))
        .with_text_response("done");

    let (interaction_tx, interaction_rx) = tokio::sync::mpsc::channel(8);
    let _handler = handle_confirm_with(interaction_rx, || InteractionResponse::Approved {
        payload: None,
    });

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(tool);
    builder.set_interaction_sender(interaction_tx);
    let handle = builder.handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    assert_eq!(invocations.load(Ordering::SeqCst), 1);

    let approved = collector.events().into_iter().any(|e| {
        matches!(
            e,
            AgentEvent::ToolApprovalResolved {
                outcome: ToolApprovalOutcome::Approved,
                ..
            }
        )
    });
    assert!(approved, "Should emit Approved (user approved)");
}

#[tokio::test]
async fn elevated_with_user_rejection_synth_errors() {
    let invocations = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(ElevatedTool {
        name: "danger",
        invocations: invocations.clone(),
    });

    let transport = MockTransport::new()
        .with_tool_call_response("danger", "c1", serde_json::json!({}))
        .with_text_response("noted");

    let (interaction_tx, interaction_rx) = tokio::sync::mpsc::channel(8);
    let _handler = handle_confirm_with(interaction_rx, || InteractionResponse::Rejected {
        reason: "user said no".into(),
    });

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(tool);
    builder.set_interaction_sender(interaction_tx);
    let handle = builder.handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    assert_eq!(invocations.load(Ordering::SeqCst), 0, "must not run");

    let rejected = collector.events().into_iter().any(|e| {
        matches!(
            e,
            AgentEvent::ToolApprovalResolved {
                outcome: ToolApprovalOutcome::Rejected { reason },
                ..
            } if reason == "user said no"
        )
    });
    assert!(rejected);

    let msgs = handle.messages().await.unwrap();
    let synth_error = msgs.iter().any(|m| {
        matches!(
            m,
            Message::ToolResult { is_error: true, content, .. }
                if content.iter().any(|c| c.as_text().is_some_and(|t| t.contains("user said no")))
        )
    });
    assert!(synth_error);
}

#[tokio::test]
async fn mixed_approval_preserves_tool_call_order() {
    // Two elevated tools in one assistant message: approve first, reject second.
    // Tool results in conversation must be in original order regardless of decision.
    struct PairTransport(AtomicU32);

    #[async_trait]
    impl Transport for PairTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            _: &tau_agent::core::transport::AgentRunConfig,
            _: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<tau_agent::core::transport::AgentEventStream> {
            let prev = self
                .0
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    Some(n.saturating_sub(1))
                })
                .unwrap_or(0);
            let msg = if prev > 0 {
                Message::Assistant {
                    content: vec![
                        Content::tool_call("call_a", "danger", serde_json::json!({})),
                        Content::tool_call("call_b", "danger", serde_json::json!({})),
                    ],
                    metadata: AssistantMetadata::default(),
                }
            } else {
                Message::Assistant {
                    content: vec![Content::text("done")],
                    metadata: AssistantMetadata::default(),
                }
            };
            let events = vec![
                AgentEvent::TurnStart { turn_number: 1 },
                AgentEvent::MessageEnd {
                    message: msg.clone(),
                },
                AgentEvent::TurnEnd {
                    turn_number: 1,
                    message: msg,
                    usage: Usage::default(),
                },
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    let transport: Arc<dyn Transport> = Arc::new(PairTransport(AtomicU32::new(1)));

    let invocations = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(ElevatedTool {
        name: "danger",
        invocations: invocations.clone(),
    });

    // Per-call response: Approve "call_a", Reject "call_b".
    let (interaction_tx, mut interaction_rx) = tokio::sync::mpsc::channel::<InteractionRequest>(8);
    let _handler = tokio::spawn(async move {
        while let Some(req) = interaction_rx.recv().await {
            let id = match &req.kind {
                InteractionKind::Typed { schema_id, payload } if schema_id == "tool.confirm" => {
                    payload
                        .get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                }
                _ => continue,
            };
            let resp = if id == "call_a" {
                InteractionResponse::Approved { payload: None }
            } else {
                InteractionResponse::Rejected {
                    reason: "denied".into(),
                }
            };
            let _ = req.response_tx.send(resp);
        }
    });

    let mut builder = AgentBuilder::new(test_config(), transport);
    builder.add_tool(tool);
    builder.set_interaction_sender(interaction_tx);
    let handle = builder.handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    assert_eq!(invocations.load(Ordering::SeqCst), 1, "only call_a runs");

    let msgs = handle.messages().await.unwrap();
    let ids: Vec<&str> = msgs
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult { tool_call_id, .. } => Some(tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["call_a", "call_b"], "results in original order");

    // call_b should be an error result.
    let b_is_error = msgs.iter().any(|m| {
        matches!(
            m,
            Message::ToolResult { tool_call_id, is_error: true, .. } if tool_call_id == "call_b"
        )
    });
    assert!(b_is_error);
}

#[tokio::test]
async fn set_approval_policy_takes_effect_for_subsequent_prompt() {
    let invocations = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(ElevatedTool {
        name: "danger",
        invocations: invocations.clone(),
    });

    // Two prompts, both call the elevated tool. First runs with default
    // (rejects in headless), second runs after we install AutoAcceptAll.
    let transport = MockTransport::new()
        .with_tool_call_response("danger", "c1", serde_json::json!({}))
        .with_text_response("first")
        .with_tool_call_response("danger", "c2", serde_json::json!({}))
        .with_text_response("second");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(tool);
    let handle = builder.handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("first").await.unwrap();
    assert_eq!(invocations.load(Ordering::SeqCst), 0, "rejected on first");

    handle
        .set_approval_policy(Arc::new(AutoAcceptAll))
        .await
        .unwrap();
    // Sync: send a noop query to ensure the SetApprovalPolicy command was processed.
    let _ = handle.config().await;

    handle.prompt_and_wait("second").await.unwrap();
    collector.wait_for_end().await;

    assert_eq!(invocations.load(Ordering::SeqCst), 1, "ran on second");
}

#[tokio::test]
async fn abort_during_pending_gate_terminates_cleanly() {
    let invocations = Arc::new(AtomicU32::new(0));
    let tool = Arc::new(ElevatedTool {
        name: "danger",
        invocations: invocations.clone(),
    });

    let transport = MockTransport::new()
        .with_tool_call_response("danger", "c1", serde_json::json!({}))
        .with_text_response("never reached");

    // Receiver that captures the request but never responds, so the gate
    // stays pending until we abort.
    let (interaction_tx, mut interaction_rx) = tokio::sync::mpsc::channel::<InteractionRequest>(8);
    let captured: Arc<tokio::sync::Mutex<Option<InteractionRequest>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let captured_clone = captured.clone();
    let _capturer = tokio::spawn(async move {
        if let Some(req) = interaction_rx.recv().await {
            *captured_clone.lock().await = Some(req);
        }
    });

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(tool);
    builder.set_interaction_sender(interaction_tx);
    let handle = builder.handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    let prompt_rx = handle.prompt("go").await.unwrap();

    // Wait until the actor has emitted the assistant tool-call message and
    // the gate request has reached our capturer — that's the window where
    // the actor is parked in AwaitingApproval.
    collector
        .wait_for_event(|e| matches!(e, AgentEvent::MessageEnd { .. }))
        .await;
    for _ in 0..50 {
        if captured.lock().await.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        captured.lock().await.is_some(),
        "tool.confirm interaction never reached the host"
    );

    handle.abort();

    // Prompt resolves cleanly (treated as success on cancel).
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), prompt_rx)
        .await
        .expect("prompt did not return after abort")
        .expect("oneshot dropped");
    assert!(result.result.is_ok(), "abort should yield Ok");

    collector
        .wait_for_event(|e| matches!(e, AgentEvent::AgentEnd { .. }))
        .await;

    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "tool must not run after abort"
    );

    // Drop the dangling oneshot so the request is cleaned up.
    drop(captured.lock().await.take());
}

#[tokio::test]
async fn default_policy_passes_through() {
    // Sanity: confirm DefaultPolicy gates Elevated and lets Local through.
    let p = DefaultPolicy;
    assert!(matches!(
        p.classify("x", &serde_json::Value::Null, ToolRisk::Elevated),
        ApprovalDecision::Gate
    ));
    assert!(matches!(
        p.classify("x", &serde_json::Value::Null, ToolRisk::Local),
        ApprovalDecision::Auto
    ));
}
