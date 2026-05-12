//! End-to-end fleet smoke tests: spawn a subagent through the
//! manager, resume it, respec it, verify the registry invariant
//! holds.

use std::sync::Arc;

use tau_agent::*;

fn make_manager(transport: Arc<dyn Transport>) -> Arc<AgentManager> {
    let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(64);
    Arc::new(AgentManager::new(
        tx,
        test_utils::test_config(),
        transport,
        4,
    ))
}

fn empty_spec() -> AgentSpec {
    AgentSpec {
        system_prompt: String::new(),
        tools: vec![],
        max_turns: 5,
        allows_worktree: false,
        allowed_subagent_specs: None,
    }
}

fn spawn_opts(description: &str) -> SpawnOpts {
    SpawnOpts {
        description: description.into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn spawn_foreground_returns_result() {
    let mgr = make_manager(test_utils::TextTransport::create("subagent says hi"));
    let result = mgr
        .spawn(
            empty_spec(),
            "hello".into(),
            spawn_opts("test"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");
    assert_eq!(result.text, "subagent says hi");
    // Spec should still be in registry (agent is idle now).
    assert!(mgr.spec_for(&result.agent_id).is_some());
}

#[tokio::test]
async fn send_resumes_stored_agent() {
    let mgr = make_manager(test_utils::TextTransport::create("response"));
    let result = mgr
        .spawn(
            empty_spec(),
            "first".into(),
            spawn_opts("resumable"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");

    let resumed = mgr
        .send(
            &result.agent_id,
            "follow up",
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("send");
    assert_eq!(resumed.agent_id, result.agent_id, "same id");
    assert_eq!(resumed.text, "response");
    // Delta tokens, not cumulative: TextTransport reports 100/50 per
    // turn, so the resume's delta equals exactly one turn's usage.
    assert_eq!(resumed.input_tokens, 100, "delta input tokens");
    assert_eq!(resumed.output_tokens, 50, "delta output tokens");
}

#[tokio::test]
async fn respec_produces_new_id_drops_old_spec() {
    let mgr = make_manager(test_utils::TextTransport::create("ok"));
    let original = mgr
        .spawn(
            empty_spec(),
            "first".into(),
            spawn_opts("orig"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");
    let old_id = original.agent_id.clone();

    let new_handle = mgr
        .respec(
            &old_id,
            AgentSpec {
                system_prompt: "new prompt".into(),
                ..empty_spec()
            },
        )
        .await
        .expect("respec");

    assert!(mgr.spec_for(&old_id).is_none(), "old spec dropped");
    let new_id = new_handle.agent_id().expect("stamped");
    assert_ne!(new_id, old_id, "respec assigns a new id");
    assert!(mgr.spec_for(new_id).is_some(), "new spec recorded");

    // Description preservation: the new agent should carry the
    // original description, not a debug breadcrumb like `respec(...)`.
    let located = mgr.find_agent(new_id).expect("locatable");
    assert_eq!(
        located.description, "orig",
        "description preserved across respec"
    );
}

/// Concurrent `respec` arriving while a `send` is in flight must
/// observe the agent as Running, not as "missing". Regression for the
/// earlier non-atomic `take_idle` + `commit_running` split.
#[tokio::test]
async fn respec_concurrent_with_send_sees_running_not_missing() {
    let mgr = make_manager(test_utils::SlowTransport::create(100));

    let r = mgr
        .spawn(
            empty_spec(),
            "first".into(),
            spawn_opts("racey"),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("spawn");
    let id = r.agent_id.clone();

    let mgr_send = mgr.clone();
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

    // Give send() a moment to take the idle entry and place the
    // running placeholder.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    let respec_result = mgr
        .respec(
            &id,
            AgentSpec {
                system_prompt: "new".into(),
                ..empty_spec()
            },
        )
        .await;
    let send_result = send_task.await.expect("send task joins");

    // The respec must have failed (agent was running mid-resume) but
    // the error must say so explicitly, not "missing".
    let err = match respec_result {
        Err(e) => e,
        Ok(_) => panic!("respec must reject while agent is resuming"),
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("currently running"),
        "respec error mentions running, got: {msg}"
    );
    assert!(send_result.is_ok(), "send completes despite race");
    assert!(mgr.spec_for(&id).is_some(), "spec preserved through race");
}

#[tokio::test]
async fn adopt_records_spec_and_allows_respec() {
    let mgr = make_manager(test_utils::TextTransport::create("ok"));

    // Externally-built handle (the typical "host's root agent" pattern).
    let builder = AgentBuilder::new(
        test_utils::test_config(),
        test_utils::TextTransport::create("ok"),
    );
    let handle = builder.spawn();
    let adopted_id = mgr.adopt(&handle, "root", empty_spec());

    assert!(mgr.spec_for(&adopted_id).is_some(), "spec recorded");
    let located = mgr.find_agent(&adopted_id).expect("findable");
    assert!(
        matches!(located.status, AgentStatus::Adopted),
        "shows as adopted"
    );

    // respec on the adopted handle should produce a new agent and drop
    // the old spec. The old `adopt` bug returned Err("missing") here.
    let new_handle = mgr
        .respec(
            &adopted_id,
            AgentSpec {
                system_prompt: "new".into(),
                ..empty_spec()
            },
        )
        .await
        .expect("respec on adopted handle");
    assert!(mgr.spec_for(&adopted_id).is_none(), "old spec dropped");
    assert_ne!(new_handle.agent_id().unwrap(), adopted_id);
    assert!(
        mgr.spec_for(new_handle.agent_id().unwrap()).is_some(),
        "new spec recorded"
    );
}

#[tokio::test]
async fn eviction_drops_oldest_when_at_capacity() {
    // make_manager configures max_agents=4
    let mgr = make_manager(test_utils::TextTransport::create("ok"));
    let mut ids = Vec::new();
    for i in 0..6 {
        let r = mgr
            .spawn(
                empty_spec(),
                format!("p{i}"),
                spawn_opts(&format!("a{i}")),
                tokio_util::sync::CancellationToken::new(),
            )
            .await
            .expect("spawn");
        ids.push(r.agent_id);
    }
    // First two evicted.
    assert!(mgr.spec_for(&ids[0]).is_none());
    assert!(mgr.spec_for(&ids[1]).is_none());
    // Last four retained.
    for id in &ids[2..] {
        assert!(mgr.spec_for(id).is_some(), "agent {id} retained");
    }
}
