//! Tests for the structured plan submission flow.

use std::sync::Arc;

use tau_agent::interaction::{InteractionKind, InteractionRequest, InteractionResponse};
use tau_agent::test_utils::*;
use tau_agent::{AgentBuilder, AgentEvent, BoxedTool, Plan, PlanFile, PlanFileOp, PlanStep};
use tau_ai::Message;
use tau_tools::SubmitPlanTool;

fn sample_plan() -> Plan {
    Plan {
        items: vec![PlanStep {
            id: "s1".into(),
            title: "Add module".into(),
            description: "Create src/foo.rs".into(),
            touches: vec!["src/foo.rs".into()],
        }],
        files: vec![PlanFile {
            op: PlanFileOp::Add,
            path: "src/foo.rs".into(),
            adds: 10,
            dels: 0,
        }],
        flags: vec![],
    }
}

fn plan_with_extra_step() -> Plan {
    let mut p = sample_plan();
    p.items.push(PlanStep {
        id: "s2".into(),
        title: "Wire it up".into(),
        description: "Edited by user".into(),
        touches: vec!["src/lib.rs".into()],
    });
    p
}

#[tokio::test]
async fn submit_plan_round_trips_with_user_edit() {
    // Transport: model calls submit_plan with sample_plan, then emits "done".
    let plan = sample_plan();
    let plan_args = serde_json::to_value(&plan).unwrap();
    let transport = MockTransport::new()
        .with_tool_call_response("submit_plan", "c1", plan_args)
        .with_text_response("plan acknowledged");

    // Host receives SubmitPlan, returns PlanApproved with an *edited* plan.
    let (interaction_tx, mut interaction_rx) =
        tokio::sync::mpsc::channel::<InteractionRequest>(8);
    let captured: Arc<tokio::sync::Mutex<Option<Plan>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let captured_clone = captured.clone();
    let _handler = tokio::spawn(async move {
        if let Some(req) = interaction_rx.recv().await {
            if let InteractionKind::SubmitPlan { plan } = req.kind {
                *captured_clone.lock().await = Some(plan);
                let _ = req.response_tx.send(InteractionResponse::PlanApproved {
                    plan: plan_with_extra_step(),
                });
            }
        }
    });

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(SubmitPlanTool::new()) as BoxedTool);
    builder.set_interaction_sender(interaction_tx);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("plan it").await.unwrap();
    collector.wait_for_end().await;

    // Host saw the original plan.
    let received = captured.lock().await.clone().expect("host got plan");
    assert_eq!(received.items.len(), 1);
    assert_eq!(received.items[0].id, "s1");

    // Tool result fed the *edited* plan back to the model.
    let msgs = handle.messages().await.unwrap();
    let result_text = msgs
        .iter()
        .find_map(|m| match m {
            Message::ToolResult {
                tool_name, content, ..
            } if tool_name == "submit_plan" => Some(content),
            _ => None,
        })
        .expect("tool result present");
    let text = result_text
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<String>();
    assert!(text.contains("Plan approved"));
    assert!(text.contains("\"id\": \"s2\""), "edited plan reaches model");
}

#[tokio::test]
async fn submit_plan_rejection_returns_error_with_reason() {
    let plan_args = serde_json::to_value(sample_plan()).unwrap();
    let transport = MockTransport::new()
        .with_tool_call_response("submit_plan", "c1", plan_args)
        .with_text_response("understood, will revise");

    let (interaction_tx, mut interaction_rx) =
        tokio::sync::mpsc::channel::<InteractionRequest>(8);
    let _handler = tokio::spawn(async move {
        if let Some(req) = interaction_rx.recv().await {
            let _ = req.response_tx.send(InteractionResponse::Rejected {
                reason: "missing migration step".into(),
            });
        }
    });

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(SubmitPlanTool::new()) as BoxedTool);
    builder.set_interaction_sender(interaction_tx);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("plan it").await.unwrap();
    collector.wait_for_end().await;

    let msgs = handle.messages().await.unwrap();
    let result = msgs
        .iter()
        .find_map(|m| match m {
            Message::ToolResult {
                tool_name,
                content,
                is_error,
                ..
            } if tool_name == "submit_plan" => Some((*is_error, content)),
            _ => None,
        })
        .expect("tool result present");
    assert!(result.0, "rejection should be is_error");
    let text: String = result.1.iter().filter_map(|c| c.as_text()).collect();
    assert!(text.contains("missing migration step"), "got: {text}");
}

#[tokio::test]
async fn submit_plan_rejection_then_revision_succeeds() {
    // Model: submit, get rejected, submit again with edits, get approved, stop.
    let initial = sample_plan();
    let revised = plan_with_extra_step();
    let initial_args = serde_json::to_value(&initial).unwrap();
    let revised_args = serde_json::to_value(&revised).unwrap();

    let transport = MockTransport::new()
        .with_tool_call_response("submit_plan", "c1", initial_args)
        .with_tool_call_response("submit_plan", "c2", revised_args)
        .with_text_response("approved, stopping");

    // Host: reject the first call, approve the second.
    let (interaction_tx, mut interaction_rx) =
        tokio::sync::mpsc::channel::<InteractionRequest>(8);
    let _handler = tokio::spawn(async move {
        let mut call = 0u32;
        while let Some(req) = interaction_rx.recv().await {
            call += 1;
            if let InteractionKind::SubmitPlan { plan } = req.kind {
                let resp = if call == 1 {
                    InteractionResponse::Rejected {
                        reason: "needs migration step".into(),
                    }
                } else {
                    InteractionResponse::PlanApproved { plan }
                };
                let _ = req.response_tx.send(resp);
            }
        }
    });

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(SubmitPlanTool::new()) as BoxedTool);
    builder.set_interaction_sender(interaction_tx);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("plan it").await.unwrap();
    collector.wait_for_end().await;

    let msgs = handle.messages().await.unwrap();
    let submit_results: Vec<_> = msgs
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult {
                tool_name,
                content,
                is_error,
                ..
            } if tool_name == "submit_plan" => {
                let text: String = content.iter().filter_map(|c| c.as_text()).collect();
                Some((*is_error, text))
            }
            _ => None,
        })
        .collect();

    assert_eq!(submit_results.len(), 2, "two submit_plan calls");
    assert!(submit_results[0].0, "first call returned error (rejected)");
    assert!(
        submit_results[0].1.contains("needs migration step"),
        "first error includes reason: {}",
        submit_results[0].1
    );
    assert!(!submit_results[1].0, "second call returned success (approved)");
    assert!(
        submit_results[1].1.contains("Plan approved"),
        "second result is the approved plan: {}",
        submit_results[1].1
    );
}

#[tokio::test]
async fn submit_plan_without_interaction_channel_errors() {
    let plan_args = serde_json::to_value(sample_plan()).unwrap();
    let transport = MockTransport::new()
        .with_tool_call_response("submit_plan", "c1", plan_args)
        .with_text_response("noted");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(SubmitPlanTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("plan it").await.unwrap();
    collector.wait_for_end().await;

    let msgs = handle.messages().await.unwrap();
    let is_error = msgs.iter().any(|m| matches!(
        m,
        Message::ToolResult { tool_name, is_error: true, content, .. }
            if tool_name == "submit_plan"
                && content.iter().any(|c| c.as_text().is_some_and(|t| t.contains("No interactive")))
    ));
    assert!(is_error);
}

#[tokio::test]
async fn record_transcript_returns_path_and_writes_jsonl() {
    use tau_agent::transcript::record_transcript;
    use tau_ai::AssistantMetadata;

    let agent_id = format!("test-record-{}", uuid::Uuid::new_v4());
    let messages = vec![
        Message::user("hello"),
        Message::Assistant {
            content: vec![tau_ai::Content::text("hi back")],
            metadata: AssistantMetadata::default(),
        },
    ];

    let path = record_transcript(&agent_id, &messages)
        .await
        .expect("transcript path returned on success");

    let body = tokio::fs::read_to_string(&path).await.expect("file exists");
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 2, "one JSON object per message");
    assert!(lines[0].contains("\"hello\""));
    assert!(lines[1].contains("\"hi back\""));

    // Cleanup so re-runs don't accumulate.
    let _ = tokio::fs::remove_file(&path).await;
}

#[tokio::test]
async fn subagent_interaction_is_stamped_with_agent_id() {
    // Wire an interaction channel to the manager so subagents inherit it.
    // Build a fake "subagent" by directly calling submit_plan from a tool
    // execution context that has the wrapped sender — easiest end-to-end
    // check is just to verify the wrapper stamping logic via the manager.
    use tau_agent::manager::{AgentManager, AgentType, SpawnRequest};
    use tau_agent::transport::Transport;

    let (event_tx, _event_rx) = tokio::sync::broadcast::channel::<AgentEvent>(64);
    let (interaction_tx, mut interaction_rx) =
        tokio::sync::mpsc::channel::<InteractionRequest>(8);

    // A transport that has the subagent immediately call submit_plan and
    // then a text turn (after the tool returns).
    let plan = sample_plan();
    let plan_args = serde_json::to_value(&plan).unwrap();
    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            .with_tool_call_response("submit_plan", "sub_c1", plan_args)
            .with_text_response("plan handled"),
    );

    let tools: Vec<BoxedTool> = vec![Arc::new(SubmitPlanTool::new())];
    let manager = Arc::new(
        AgentManager::new(event_tx, tools, test_config(), transport, 4)
            .with_parent_interaction_sender(interaction_tx),
    );

    // Host responds approving with the same plan.
    let captured_id: Arc<tokio::sync::Mutex<Option<String>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let captured = captured_id.clone();
    let _handler = tokio::spawn(async move {
        if let Some(req) = interaction_rx.recv().await {
            *captured.lock().await = req.agent_id.clone();
            if let InteractionKind::SubmitPlan { plan } = req.kind {
                let _ = req.response_tx.send(InteractionResponse::PlanApproved { plan });
            }
        }
    });

    let cancel = tokio_util::sync::CancellationToken::new();
    let request = SpawnRequest {
        agent_type: AgentType::Plan,
        prompt: "plan it".into(),
        description: "plan run".into(),
        model: None,
        cwd: None,
        isolation: None,
        depth: 0,
        inherit_history_from: None,
        approval_policy: None,
    };
    let result = manager.spawn(request, cancel).await.expect("spawn");

    let stamped = captured_id
        .lock()
        .await
        .clone()
        .expect("agent_id stamped on interaction request");
    assert_eq!(
        stamped, result.agent_id,
        "stamped id matches subagent id"
    );

    // The transcript path should also be populated and on disk.
    let path = result
        .transcript_path
        .as_ref()
        .expect("transcript_path populated");
    assert!(
        std::path::Path::new(path).exists(),
        "transcript file written: {path}"
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn inherit_history_from_seeds_executor_with_planner_messages() {
    use tau_agent::manager::{AgentManager, AgentType, SpawnRequest};
    use tau_agent::transport::{AgentEventStream, AgentRunConfig, Transport};

    // Capturing transport: each call records the messages it received.
    struct CapturingTransport {
        text: String,
        captured: std::sync::Mutex<Vec<Vec<Message>>>,
    }

    #[async_trait::async_trait]
    impl Transport for CapturingTransport {
        async fn run(
            &self,
            messages: Vec<Message>,
            _config: &AgentRunConfig,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            self.captured.lock().unwrap().push(messages);
            let msg = tau_ai::Message::Assistant {
                content: vec![tau_ai::Content::text(self.text.clone())],
                metadata: tau_ai::AssistantMetadata::default(),
            };
            let usage = tau_ai::Usage::default();
            let events = vec![
                AgentEvent::TurnStart { turn_number: 1 },
                AgentEvent::MessageEnd {
                    message: msg.clone(),
                },
                AgentEvent::TurnEnd {
                    turn_number: 1,
                    message: msg,
                    usage,
                },
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    let transport = Arc::new(CapturingTransport {
        text: "ack".into(),
        captured: std::sync::Mutex::new(vec![]),
    });
    let transport_dyn: Arc<dyn Transport> = transport.clone();

    let tools: Vec<BoxedTool> = vec![];
    let (event_tx, _event_rx) = tokio::sync::broadcast::channel::<AgentEvent>(64);
    let manager = Arc::new(AgentManager::new(
        event_tx,
        tools,
        test_config(),
        transport_dyn,
        4,
    ));

    let cancel = tokio_util::sync::CancellationToken::new();

    // Step 1: spawn a "planner" subagent. It runs one turn and terminates.
    let plan_req = SpawnRequest {
        agent_type: AgentType::Plan,
        prompt: "design module X".into(),
        description: "planner".into(),
        model: None,
        cwd: None,
        isolation: None,
        depth: 0,
        inherit_history_from: None,
        approval_policy: None,
    };
    let plan_result = manager
        .spawn(plan_req, cancel.clone())
        .await
        .expect("planner spawn");
    let planner_id = plan_result.agent_id.clone();

    // Sanity: planner is now stored.
    let calls_after_planner = transport.captured.lock().unwrap().len();
    assert!(calls_after_planner >= 1, "planner ran at least one turn");

    // Step 2: spawn an "executor" with inherit_history_from = planner_id.
    let exec_req = SpawnRequest {
        agent_type: AgentType::GeneralPurpose,
        prompt: "execute the approved plan".into(),
        description: "executor".into(),
        model: None,
        cwd: None,
        isolation: None,
        depth: 0,
        inherit_history_from: Some(planner_id.clone()),
        approval_policy: None,
    };
    manager.spawn(exec_req, cancel).await.expect("executor spawn");

    // Inspect the executor's first transport call: it should include the
    // planner's user prompt (proving the seed) AND the executor's new prompt.
    let calls = transport.captured.lock().unwrap();
    let executor_call = &calls[calls_after_planner];
    let texts: Vec<String> = executor_call
        .iter()
        .filter_map(|m| match m {
            Message::User { content, .. } => Some(
                content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("design module X")),
        "executor inherited the planner's user prompt: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("execute the approved plan")),
        "executor's own prompt is appended: {texts:?}"
    );
}

#[tokio::test]
async fn executor_prompt_appended_only_when_inheriting() {
    // CapturingTransport records the system_prompt the agent runs with.
    // Spawn one fresh GeneralPurpose subagent and one inheriting the
    // first's history; only the second should see the executor fragment.
    use tau_agent::manager::{AgentManager, AgentType, SpawnRequest};
    use tau_agent::transport::Transport;

    let transport = CapturingTransport::create("ack");
    let transport_dyn: Arc<dyn Transport> = transport.clone();
    let (event_tx, _event_rx) = tokio::sync::broadcast::channel::<AgentEvent>(64);
    let manager = Arc::new(AgentManager::new(
        event_tx,
        vec![],
        test_config(),
        transport_dyn,
        4,
    ));

    let cancel = tokio_util::sync::CancellationToken::new();

    let first = manager
        .spawn(
            SpawnRequest {
                agent_type: AgentType::GeneralPurpose,
                prompt: "do thing".into(),
                description: "first".into(),
                model: None,
                cwd: None,
                isolation: None,
                depth: 0,
                inherit_history_from: None,
                approval_policy: None,
            },
            cancel.clone(),
        )
        .await
        .expect("first spawn");

    manager
        .spawn(
            SpawnRequest {
                agent_type: AgentType::GeneralPurpose,
                prompt: "now execute".into(),
                description: "executor".into(),
                model: None,
                cwd: None,
                isolation: None,
                depth: 0,
                inherit_history_from: Some(first.agent_id),
                approval_policy: None,
            },
            cancel,
        )
        .await
        .expect("executor spawn");

    let calls = transport.calls();
    let prompts: Vec<String> = calls
        .iter()
        .filter_map(|c| c.system_prompt.clone())
        .collect();
    let with_marker = prompts
        .iter()
        .filter(|p| p.contains("Plan Executor Mode"))
        .count();
    let without_marker = prompts
        .iter()
        .filter(|p| !p.contains("Plan Executor Mode"))
        .count();
    assert!(
        with_marker >= 1,
        "executor's prompt contains the executor fragment"
    );
    assert!(
        without_marker >= 1,
        "first agent's prompt does NOT contain the executor fragment"
    );
}

#[tokio::test]
async fn inherit_history_from_unknown_id_errors() {
    use tau_agent::manager::{AgentManager, AgentType, SpawnRequest};
    use tau_agent::transport::Transport;

    let transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new().with_text_response("never reached"),
    );
    let (event_tx, _event_rx) = tokio::sync::broadcast::channel::<AgentEvent>(16);
    let manager = Arc::new(AgentManager::new(event_tx, vec![], test_config(), transport, 4));

    let req = SpawnRequest {
        agent_type: AgentType::GeneralPurpose,
        prompt: "go".into(),
        description: "exec".into(),
        model: None,
        cwd: None,
        isolation: None,
        depth: 0,
        inherit_history_from: Some("nonexistent-id".into()),
        approval_policy: None,
    };

    let err = manager
        .spawn(req, tokio_util::sync::CancellationToken::new())
        .await
        .expect_err("should fail when source id missing");
    assert!(
        err.to_string().contains("nonexistent-id"),
        "error mentions the missing id: {err}"
    );
}
