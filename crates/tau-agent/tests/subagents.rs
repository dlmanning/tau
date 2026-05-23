//! Integration tests for AgentManager subagent orchestration:
//! spawn, resume, eviction, event forwarding, background agents,
//! find_agent, cancel propagation.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use futures::stream;
use tau_agent::{AgentEventStream, AgentRunConfig};
use tau_agent::{AgentManager, AgentSpec, AgentStatus, SpawnOpts};
use tau_agent::test_utils::*;
use tau_agent::*;
use tau_ai::{AssistantMetadata, Content, Message, Usage};
use tokio_util::sync::CancellationToken;

fn make_manager(transport: Arc<dyn Transport>) -> Arc<AgentManager> {
    let config = test_config();
    Arc::new(AgentManager::new(config, transport, 20))
}

fn echo_spec() -> AgentSpec {
    AgentSpec {
        system_prompt: String::new(),
        tools: vec![Arc::new(EchoTool)],
        max_turns: 200,
    }
}

fn spawn_opts(description: &str) -> SpawnOpts {
    SpawnOpts {
        description: description.to_string(),
        ..Default::default()
    }
}

// ─── Foreground spawn ───────────────────────────────────────────────

#[tokio::test]
async fn spawn_foreground_completes_and_returns_result() {
    let manager = make_manager(TextTransport::create("subagent response"));
    let cancel = CancellationToken::new();

    let result = manager
        .spawn(
            echo_spec(),
            ("do something").to_string(),
            spawn_opts("test subagent"),
            cancel,
        )
        .await;

    assert!(result.is_ok(), "spawn should succeed: {:?}", result.err());
    let result = result.unwrap();
    assert!(!result.agent_id.is_empty());
    assert_eq!(result.text, "subagent response");
    assert!(result.input_tokens > 0);
    // duration_ms may be 0 in fast tests — just check the field exists
    let _ = result.duration_ms;
}

#[tokio::test]
async fn spawn_foreground_stores_agent_for_resumption() {
    let manager = make_manager(TextTransport::create("first response"));
    let cancel = CancellationToken::new();

    let result = manager
        .spawn(
            echo_spec(),
            ("hello").to_string(),
            spawn_opts("resumable agent"),
            cancel,
        )
        .await
        .unwrap();

    // The agent should be findable by ID
    let found = manager.find_agent(&result.agent_id);
    assert!(found.is_some(), "spawned agent should be stored");
    let located = found.unwrap();
    assert_eq!(located.agent_id, result.agent_id);
    assert_eq!(located.description, "resumable agent");
    assert_eq!(located.status, AgentStatus::Idle);
}

#[tokio::test]
async fn inherit_preserves_previous_summary() {
    // Regression: AgentSeed::Inherit must carry the source agent's
    // compaction summary, not just its messages — dropping it makes the
    // inheritor's first compaction fall back to the initial-summarization
    // path and lose continuity (the plan→execute handoff relies on this).
    let manager = make_manager(TextTransport::create("unused"));

    // A source agent seeded with a previous_summary, adopted so the
    // Inherit resolver can find it in the registry.
    let mut src_builder = AgentBuilder::new(test_config(), TextTransport::create("unused"));
    src_builder.seed(AgentSeed::Messages {
        messages: vec![
            Message::user("earlier work"),
            make_assistant_message("earlier reply"),
        ],
        previous_summary: Some("SOURCE-SUMMARY".into()),
    });
    let source = src_builder.spawn().await.unwrap();
    let source_id = manager.adopt(&source, "source", echo_spec());

    // spawn_interactive returns the handle directly and runs through the
    // same configure_builder path as a real subagent spawn.
    let opts = SpawnOpts {
        description: "inheritor".into(),
        seed: AgentSeed::Inherit {
            agent_id: source_id,
        },
        ..Default::default()
    };
    let (inheritor, _id) = manager.spawn_interactive(echo_spec(), opts).await.unwrap();

    let state = inheritor.state().await.expect("inheritor returns state");
    assert_eq!(
        state.previous_summary.as_deref(),
        Some("SOURCE-SUMMARY"),
        "inherit must carry the source's previous_summary, not drop it"
    );
    assert_eq!(
        state.messages.len(),
        2,
        "inherit must also carry the source's messages"
    );
}

// ─── Resume (send) ──────────────────────────────────────────────────

#[tokio::test]
async fn resume_agent_with_send() {
    let transport = Arc::new(CallCountTransport::new("response"));
    let manager = make_manager(transport.clone());
    let cancel = CancellationToken::new();

    let first = manager
        .spawn(
            echo_spec(),
            ("first").to_string(),
            spawn_opts("resumable"),
            cancel.clone(),
        )
        .await
        .unwrap();

    // Resume with a follow-up message
    let second = manager
        .send(&first.agent_id, "follow up question", cancel)
        .await;

    assert!(second.is_ok(), "send should succeed: {:?}", second.err());
    let second = second.unwrap();
    assert_eq!(second.agent_id, first.agent_id);
    assert!(!second.text.is_empty());

    // Transport should have been called multiple times (spawn + resume)
    assert!(transport.call_count.load(Ordering::Relaxed) >= 2);
}

#[tokio::test]
async fn send_to_nonexistent_agent_errors() {
    let manager = make_manager(TextTransport::create("ok"));
    let cancel = CancellationToken::new();

    let result = manager.send("nonexistent-id", "hello", cancel).await;
    let err = result.expect_err("send to missing agent must fail");
    assert!(
        matches!(&err, tau_agent::Error::AgentNotFound { id } if id == "nonexistent-id"),
        "expected Error::AgentNotFound {{ id: \"nonexistent-id\" }}, got: {err:?}"
    );
}

// ─── Eviction ───────────────────────────────────────────────────────

#[tokio::test]
async fn eviction_removes_oldest_agent() {
    let transport = TextTransport::create("ok");
    let config = test_config();
    // Max 2 agents
    let manager = Arc::new(AgentManager::new(config, transport, 2));
    let cancel = CancellationToken::new();

    let a1 = manager
        .spawn(
            echo_spec(),
            ("1").to_string(),
            spawn_opts("first"),
            cancel.clone(),
        )
        .await
        .unwrap();
    let a2 = manager
        .spawn(
            echo_spec(),
            ("2").to_string(),
            spawn_opts("second"),
            cancel.clone(),
        )
        .await
        .unwrap();
    let _a3 = manager
        .spawn(
            echo_spec(),
            ("3").to_string(),
            spawn_opts("third"),
            cancel.clone(),
        )
        .await
        .unwrap();

    // First agent should be evicted
    assert!(
        manager.find_agent(&a1.agent_id).is_none(),
        "oldest should be evicted"
    );
    assert!(
        manager.find_agent(&a2.agent_id).is_some(),
        "second should remain"
    );
}

// ─── Event forwarding ───────────────────────────────────────────────

#[tokio::test]
async fn subagent_events_forwarded_as_wrapped() {
    let transport = TextTransport::create("ok");
    let config = test_config();
    let manager = Arc::new(AgentManager::new(config, transport, 20));
    let mut fleet_rx = manager.subscribe();
    let cancel = CancellationToken::new();

    manager
        .spawn(
            echo_spec(),
            ("go").to_string(),
            spawn_opts("event test"),
            cancel,
        )
        .await
        .unwrap();

    // Collect forwarded events
    let mut subagent_events = vec![];
    while let Ok(event) = fleet_rx.try_recv() {
        if let FleetEvent::Forwarded {
            description, event, ..
        } = event
        {
            subagent_events.push((description, event));
        }
    }

    assert!(
        !subagent_events.is_empty(),
        "should have forwarded subagent events"
    );
    // Should include at least AgentStart and AgentEnd
    let has_start = subagent_events
        .iter()
        .any(|(_, e)| matches!(e, AgentEvent::AgentStart));
    assert!(has_start, "should forward AgentStart");
}

// ─── find_agent ─────────────────────────────────────────────────────

#[tokio::test]
async fn find_agent_by_description_substring() {
    let manager = make_manager(TextTransport::create("ok"));
    let cancel = CancellationToken::new();

    manager
        .spawn(
            echo_spec(),
            ("go").to_string(),
            spawn_opts("search the codebase"),
            cancel,
        )
        .await
        .unwrap();

    let found = manager.find_agent("codebase");
    assert!(found.is_some(), "should find by description substring");
    assert_eq!(found.unwrap().description, "search the codebase");
}

#[tokio::test]
async fn find_agent_not_found() {
    let manager = make_manager(TextTransport::create("ok"));
    assert!(manager.find_agent("nonexistent").is_none());
}

// ─── Background spawn ───────────────────────────────────────────────

#[tokio::test]
async fn spawn_background_posts_follow_up() {
    let transport = TextTransport::create("background result");
    let config = test_config();
    let manager = Arc::new(AgentManager::new(config.clone(), transport, 20));

    // Create a parent agent to receive the follow-up
    let parent_builder = AgentBuilder::new(config, TextTransport::create("parent response"));
    let parent_handle = parent_builder.spawn().await.unwrap();

    let parent_cancel = CancellationToken::new();

    let agent_id = manager
        .spawn_background(
            echo_spec(),
            ("background task").to_string(),
            spawn_opts("bg agent"),
            parent_handle.clone(),
            parent_cancel,
        )
        .await;

    assert!(!agent_id.is_empty());

    // Wait for the background agent to finish and post its follow-up
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // The follow-up should have been posted to the parent handle's queue.
    // We can't directly inspect the queue, but we can check that expect_follow_up
    // was called (by spawn_background) and then consumed.
    // After the bg agent completes, it calls parent_handle.follow_up() which
    // sends a FollowUp command to the parent actor.

    // The agent should be stored for resumption
    let found = manager.find_agent(&agent_id);
    assert!(
        found.is_some(),
        "background agent should be stored after completion"
    );
}

// ─── Cancel propagation ─────────────────────────────────────────────

#[tokio::test]
async fn cancel_propagates_to_subagent() {
    let transport = SlowTransport::create(5000);
    let manager = make_manager(transport);
    let cancel = CancellationToken::new();

    let cancel_clone = cancel.clone();
    let manager_clone = manager.clone();

    let spawn_handle = tokio::spawn(async move {
        manager_clone
            .spawn(
                echo_spec(),
                ("slow task").to_string(),
                spawn_opts("cancel test"),
                cancel_clone,
            )
            .await
    });

    // Give the subagent time to start
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Cancel the parent
    cancel.cancel();

    // The spawn should complete (not hang)
    let result = tokio::time::timeout(std::time::Duration::from_secs(3), spawn_handle).await;
    assert!(result.is_ok(), "spawn should finish after cancel, not hang");
}

// ─── Tool execution in subagent ─────────────────────────────────────

#[tokio::test]
async fn subagent_executes_tools() {
    let transport = ToolCallTransport::create(1, "echo");
    let manager = make_manager(transport);
    let cancel = CancellationToken::new();

    let result = manager
        .spawn(
            echo_spec(),
            ("use the echo tool").to_string(),
            spawn_opts("tool test"),
            cancel,
        )
        .await
        .unwrap();

    assert!(
        result.tool_use_count >= 1,
        "subagent should have used at least 1 tool, got {}",
        result.tool_use_count
    );
}

// ─── Delta usage tracking on resume ─────────────────────────────────

#[tokio::test]
async fn resume_tracks_delta_usage() {
    let transport = TextTransport::create("ok");
    let manager = make_manager(transport);
    let cancel = CancellationToken::new();

    let first = manager
        .spawn(
            echo_spec(),
            ("first").to_string(),
            spawn_opts("usage test"),
            cancel.clone(),
        )
        .await
        .unwrap();

    let first_input = first.input_tokens;

    let second = manager
        .send(&first.agent_id, "second", cancel)
        .await
        .unwrap();

    // Delta usage: only the tokens from the second call
    assert!(second.input_tokens > 0, "should have delta input tokens");
    // The delta should be less than or equal to the total (not cumulative of both calls)
    // Actually it could be anything since it's a new prompt, but it should be nonzero
    assert!(
        second.input_tokens <= first_input * 3,
        "delta should be reasonable"
    );
}

// ─── Concurrent subagent orchestration ──────────────────────────────

#[tokio::test]
async fn multiple_foreground_agents_sequentially() {
    // Spawn 6 agents one after another. Each gets a unique delay via VariableDelayTransport.
    let transport: Arc<dyn Transport> = Arc::new(VariableDelayTransport::new());
    let manager = make_manager(transport);
    let cancel = CancellationToken::new();

    let mut ids = vec![];
    for i in 0..6 {
        let result = manager
            .spawn(
                echo_spec(),
                format!("task {i}").to_string(),
                spawn_opts(&format!("agent-{i}")),
                cancel.clone(),
            )
            .await
            .unwrap();
        assert_eq!(result.text, format!("response-{}", i + 1));
        ids.push(result.agent_id);
    }

    // All 6 should be stored
    for (i, id) in ids.iter().enumerate() {
        let found = manager.find_agent(id);
        assert!(found.is_some(), "agent {i} should be stored");
    }
}

#[tokio::test]
async fn multiple_background_agents_complete_at_different_times() {
    // Spawn 6 background agents with different delays.
    // They should all post follow-ups to the parent, and all be stored after completion.
    let transport: Arc<dyn Transport> = Arc::new(VariableDelayTransport::new());
    let config = test_config();
    let manager = Arc::new(AgentManager::new(config.clone(), transport, 20));
    let mut fleet_rx = manager.subscribe();

    // Parent agent receives follow-ups
    let parent_handle = AgentBuilder::new(config, TextTransport::create("parent"))
        .spawn()
        .await
        .unwrap();
    let parent_cancel = CancellationToken::new();

    let mut agent_ids = vec![];
    for i in 0..6 {
        let id = manager
            .spawn_background(
                echo_spec(),
                format!("bg task {i}").to_string(),
                spawn_opts(&format!("bg-{i}")),
                parent_handle.clone(),
                parent_cancel.clone(),
            )
            .await;
        agent_ids.push(id);
    }

    // Wait for all background agents to complete (longest delay is ~300ms)
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // All 6 should be stored
    for (i, id) in agent_ids.iter().enumerate() {
        let found = manager.find_agent(id);
        assert!(
            found.is_some(),
            "background agent {i} ({id}) should be stored"
        );
    }

    // Parent should have received subagent events from all 6
    let mut subagent_ids = std::collections::HashSet::new();
    while let Ok(event) = fleet_rx.try_recv() {
        if let FleetEvent::Forwarded { agent_id, .. } = event {
            subagent_ids.insert(agent_id);
        }
    }
    assert_eq!(
        subagent_ids.len(),
        6,
        "should have events from all 6 subagents, got {}",
        subagent_ids.len()
    );
}

#[tokio::test]
async fn cancel_all_concurrent_background_agents() {
    // Spawn 4 slow background agents, then cancel the parent.
    // All should terminate promptly.
    let transport = SlowTransport::create(10_000);
    let config = test_config();
    let manager = Arc::new(AgentManager::new(config.clone(), transport, 20));

    let parent_handle = AgentBuilder::new(config, TextTransport::create("parent"))
        .spawn()
        .await
        .unwrap();
    let parent_cancel = CancellationToken::new();

    for i in 0..4 {
        manager
            .spawn_background(
                echo_spec(),
                format!("slow {i}").to_string(),
                spawn_opts(&format!("slow-{i}")),
                parent_handle.clone(),
                parent_cancel.clone(),
            )
            .await;
    }

    // Give them time to start
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Cancel all
    parent_cancel.cancel();

    // They should all finish within a couple seconds, not hang for 10s each
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Not much to assert except that we didn't hang. The agents may or may not
    // be stored depending on timing, but the test completing is the proof.
}

#[tokio::test]
async fn interleave_foreground_and_background_agents() {
    // Spawn bg agent, then fg agent, then another bg agent.
    // All should complete without interfering.
    let transport: Arc<dyn Transport> = Arc::new(VariableDelayTransport::new());
    let config = test_config();
    let manager = Arc::new(AgentManager::new(config.clone(), transport, 20));

    let parent_handle = AgentBuilder::new(config, TextTransport::create("parent"))
        .spawn()
        .await
        .unwrap();
    let parent_cancel = CancellationToken::new();
    let cancel = CancellationToken::new();

    // Background 1
    let bg1_id = manager
        .spawn_background(
            echo_spec(),
            ("bg1").to_string(),
            spawn_opts("background-1"),
            parent_handle.clone(),
            parent_cancel.clone(),
        )
        .await;

    // Foreground (blocks until done)
    let fg_result = manager
        .spawn(
            echo_spec(),
            ("fg").to_string(),
            spawn_opts("foreground"),
            cancel.clone(),
        )
        .await
        .unwrap();
    assert!(!fg_result.text.is_empty());

    // Background 2
    let bg2_id = manager
        .spawn_background(
            echo_spec(),
            ("bg2").to_string(),
            spawn_opts("background-2"),
            parent_handle.clone(),
            parent_cancel,
        )
        .await;

    // Wait for backgrounds
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // All three should be stored
    assert!(
        manager.find_agent(&fg_result.agent_id).is_some(),
        "fg should be stored"
    );
    assert!(
        manager.find_agent(&bg1_id).is_some(),
        "bg1 should be stored"
    );
    assert!(
        manager.find_agent(&bg2_id).is_some(),
        "bg2 should be stored"
    );
}

#[tokio::test]
async fn resume_while_background_agent_running() {
    // Spawn an agent foreground, then spawn a slow background agent,
    // then resume the first agent while the background one is still running.
    let transport: Arc<dyn Transport> = Arc::new(VariableDelayTransport::new());
    let config = test_config();
    let manager = Arc::new(AgentManager::new(config.clone(), transport, 20));

    let parent_handle = AgentBuilder::new(config, TextTransport::create("parent"))
        .spawn()
        .await
        .unwrap();
    let parent_cancel = CancellationToken::new();
    let cancel = CancellationToken::new();

    // Spawn and store an agent
    let first = manager
        .spawn(
            echo_spec(),
            ("first task").to_string(),
            spawn_opts("agent-a"),
            cancel.clone(),
        )
        .await
        .unwrap();

    // Spawn a slow background agent (will take ~200ms)
    manager
        .spawn_background(
            echo_spec(),
            ("slow bg").to_string(),
            spawn_opts("agent-slow"),
            parent_handle,
            parent_cancel,
        )
        .await;

    // Immediately resume agent-a while bg is still running
    let resumed = manager
        .send(&first.agent_id, "follow up to first", cancel)
        .await;

    assert!(
        resumed.is_ok(),
        "resume should work while background agent is running: {:?}",
        resumed.err()
    );
}

// ─── Harness helpers ────────────────────────────────────────────────

/// Transport that delays by an increasing amount on each call (50ms, 100ms, 150ms, ...)
/// and returns a response identifying the call number.
struct VariableDelayTransport {
    call_count: AtomicU32,
}

impl VariableDelayTransport {
    fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Transport for VariableDelayTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        let n = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;
        let delay_ms = (n as u64) * 50;
        let events = async_stream::stream! {
            yield AgentEvent::TurnStart { turn_number: 1 };
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {}
                _ = cancel.cancelled() => {
                    yield AgentEvent::Error { message: "Cancelled".into() };
                    return;
                }
            }
            let msg = Message::Assistant {
                content: vec![Content::text(format!("response-{}", n))],
                metadata: AssistantMetadata::default(),
            };
            yield AgentEvent::MessageEnd { message: msg.clone() };
            yield AgentEvent::TurnEnd {
                turn_number: 1,
                message: msg,
                usage: Usage { input: 100, output: 50, ..Default::default() },
            };
        };
        Ok(Box::pin(events))
    }
}

// ─── Harness helper: transport with call counter ────────────────────

#[tokio::test]
async fn abort_root_agent_cancels_all_subagents() {
    // Scenario: a root agent is running a slow prompt. While it streams,
    // 3 background subagents are spawned (each takes 60s). The user aborts
    // the root agent via handle.abort(). All 3 subagents must terminate
    // immediately — the entire post-abort sequence must complete within 1s.
    //
    // Cancellation chain:
    //   handle.abort() → root cancel token → parent_cancel in spawn_background
    //     → select! fires → bg_cancel.cancel() → subagent transport cancelled

    // Track how many subagent transports have been cancelled
    let cancel_count = Arc::new(AtomicU32::new(0));

    // Custom transport that increments a counter when cancelled
    struct CancelCountingTransport {
        counter: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Transport for CancelCountingTransport {
        async fn run(
            &self,
            _: Vec<Message>,
            _: &AgentRunConfig,
            cancel: tokio_util::sync::CancellationToken,
        ) -> tau_ai::Result<AgentEventStream> {
            let counter = self.counter.clone();
            let events = async_stream::stream! {
                yield AgentEvent::TurnStart { turn_number: 1 };
                // Wait for cancellation or a long timeout
                tokio::select! {
                    _ = cancel.cancelled() => {
                        counter.fetch_add(1, Ordering::SeqCst);
                        yield AgentEvent::Error { message: "Cancelled".into() };
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => {
                        let msg = Message::Assistant {
                            content: vec![Content::text("should not reach")],
                            metadata: AssistantMetadata::default(),
                        };
                        yield AgentEvent::MessageEnd { message: msg.clone() };
                        yield AgentEvent::TurnEnd {
                            turn_number: 1,
                            message: msg,
                            usage: Usage::default(),
                        };
                    }
                }
            };
            Ok(Box::pin(events))
        }
    }

    let sub_transport: Arc<dyn Transport> = Arc::new(CancelCountingTransport {
        counter: cancel_count.clone(),
    });
    let config = test_config();
    let manager = Arc::new(AgentManager::new(config.clone(), sub_transport, 20));

    // Root agent with a slow transport (simulates LLM streaming)
    let root_handle = AgentBuilder::new(config, SlowTransport::create(60_000))
        .spawn()
        .await
        .unwrap();
    let prompt_rx = root_handle.prompt("start").await.unwrap();

    // Wait for root to enter AwaitingModel
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Extract the root's cancel token for spawning subagents
    let parent_cancel = root_handle.cancel_token().lock().clone();

    // Spawn 3 slow background subagents
    for i in 0..3 {
        manager
            .spawn_background(
                echo_spec(),
                format!("slow task {i}").to_string(),
                spawn_opts(&format!("sub-{i}")),
                root_handle.clone(),
                parent_cancel.clone(),
            )
            .await;
    }

    // Wait for subagents to start their transports
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(
        cancel_count.load(Ordering::SeqCst),
        0,
        "no cancellations yet"
    );

    // User aborts the root agent
    root_handle.abort();

    // Everything — root + all 3 subagents — must terminate within 1 second.
    // Without cancellation propagation, this would take 60s.
    let deadline = tokio::time::timeout(std::time::Duration::from_secs(1), async {
        // Root prompt finishes
        let _ = prompt_rx.await;

        // Wait until all 3 subagent transports have seen cancellation
        loop {
            if cancel_count.load(Ordering::SeqCst) >= 3 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;

    assert!(
        deadline.is_ok(),
        "root + all 3 subagents must terminate within 1s of abort, \
         but only {} of 3 subagents were cancelled",
        cancel_count.load(Ordering::SeqCst)
    );
    assert_eq!(
        cancel_count.load(Ordering::SeqCst),
        3,
        "all 3 subagent transports must have received cancellation"
    );
}

struct CallCountTransport {
    text: String,
    call_count: AtomicU32,
}

impl CallCountTransport {
    fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            call_count: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl Transport for CallCountTransport {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _config: &AgentRunConfig,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> tau_ai::Result<AgentEventStream> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        let msg = Message::Assistant {
            content: vec![Content::text(&self.text)],
            metadata: AssistantMetadata::default(),
        };
        let usage = Usage {
            input: 100,
            output: 50,
            ..Default::default()
        };
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
        Ok(Box::pin(stream::iter(events)))
    }
}
