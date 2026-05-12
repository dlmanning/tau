//! End-to-end lifecycle tests for tau-session.
//!
//! Each test uses a tempdir-backed FsStorage so artefacts survive the
//! manager but disappear when the test ends.

use std::sync::Arc;

use tau_agent::BoxedTool;
use tau_agent::Transport;
use tau_agent::test_utils::*;
use tau_ai::Message;
use tau_session::{FsStorage, NewSessionRequest, SessionManager, SessionStatus};

fn setup() -> (Arc<SessionManager>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(FsStorage::new(dir.path()));
    let manager = Arc::new(SessionManager::new(storage));
    (manager, dir)
}

fn fresh_request(transport: Arc<dyn Transport>, prompt: Option<&str>) -> NewSessionRequest {
    NewSessionRequest {
        title: None,
        project_path: std::env::current_dir().unwrap_or_else(|_| ".".into()),
        config: test_config(),
        tools: Vec::new(),
        transport,
        seed_messages: Vec::new(),
        previous_summary: None,
        initial_prompt: prompt.map(String::from),
        customize: None,
    }
}

#[tokio::test]
async fn create_then_list_returns_one_session() {
    let (manager, _dir) = setup();
    let transport: Arc<dyn Transport> = TextTransport::create("hi");
    let active = manager
        .create(fresh_request(transport, Some("hello world")))
        .await
        .expect("create");
    assert_eq!(active.snapshot.info.original_request, "hello world");

    let list = manager.list().await.expect("list");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, active.id);
    assert_eq!(list[0].title, "hello world");
}

#[tokio::test]
async fn first_prompt_persists_messages() {
    let (manager, _dir) = setup();
    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let active = manager
        .create(fresh_request(transport, None))
        .await
        .expect("create");

    // Drive a prompt and wait for completion.
    active.handle.prompt_and_wait("ping").await.expect("prompt");

    // Give the persister a moment to drain the event queue.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Storage should now contain the messages.
    let info = manager
        .list()
        .await
        .expect("list")
        .into_iter()
        .next()
        .unwrap();
    assert!(
        info.message_count >= 2,
        "user + assistant: got {}",
        info.message_count
    );
    assert!(info.total_usage.input > 0, "TurnEnd usage rolled into info");
}

#[tokio::test]
async fn hibernate_then_activate_restores_messages() {
    let (manager, _dir) = setup();
    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let active = manager
        .create(fresh_request(transport.clone(), None))
        .await
        .expect("create");
    let id = active.id.clone();

    active
        .handle
        .prompt_and_wait("hello")
        .await
        .expect("prompt");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    manager.hibernate(&id).await.expect("hibernate");

    // Confirm status flipped.
    let info = manager
        .list()
        .await
        .expect("list")
        .into_iter()
        .find(|s| s.id == id)
        .expect("present");
    assert_eq!(info.status, SessionStatus::Hibernated);

    // Activate with the same transport + config.
    let activated = manager
        .activate(&id, test_config(), Vec::<BoxedTool>::new(), transport)
        .await
        .expect("activate");

    // The activated handle should have the prior messages restored.
    let msgs = activated.handle.messages().await.expect("messages");
    assert!(
        msgs.iter().any(|m| matches!(m, Message::User { .. })),
        "user msg restored: {msgs:?}"
    );
    assert!(
        msgs.iter().any(|m| matches!(m, Message::Assistant { .. })),
        "assistant msg restored"
    );
}

#[tokio::test]
async fn ui_state_round_trips() {
    let (manager, _dir) = setup();
    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let active = manager
        .create(fresh_request(transport.clone(), None))
        .await
        .expect("create");
    let id = active.id.clone();

    let ui = serde_json::json!({"composer": "draft text", "scroll": 42});
    manager
        .save_ui_state(&id, ui.clone())
        .await
        .expect("save ui");

    manager.hibernate(&id).await.expect("hibernate");
    let activated = manager
        .activate(&id, test_config(), Vec::<BoxedTool>::new(), transport)
        .await
        .expect("activate");
    assert_eq!(activated.snapshot.ui_state, Some(ui));
}

#[tokio::test]
async fn list_sorts_by_last_activity_desc() {
    let (manager, _dir) = setup();
    let transport: Arc<dyn Transport> = TextTransport::create("ack");

    // Three creates back-to-back. last_activity should put them most-recent first.
    let a = manager
        .create(fresh_request(transport.clone(), Some("first")))
        .await
        .expect("create a");
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let b = manager
        .create(fresh_request(transport.clone(), Some("second")))
        .await
        .expect("create b");
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let c = manager
        .create(fresh_request(transport, Some("third")))
        .await
        .expect("create c");

    let list = manager.list().await.expect("list");
    let ids: Vec<&str> = list.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, vec![c.id.as_str(), b.id.as_str(), a.id.as_str()]);
}

#[tokio::test]
async fn close_then_delete() {
    let (manager, _dir) = setup();
    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let active = manager
        .create(fresh_request(transport, None))
        .await
        .expect("create");
    let id = active.id.clone();

    manager.close(&id).await.expect("close");
    let info = manager
        .list()
        .await
        .expect("list")
        .into_iter()
        .find(|s| s.id == id)
        .expect("still listed after close");
    assert_eq!(info.status, SessionStatus::Closed);

    manager.delete(&id).await.expect("delete");
    let list = manager.list().await.expect("list");
    assert!(list.iter().all(|s| s.id != id), "delete removed it");
}

#[tokio::test]
async fn manager_emits_lifecycle_events() {
    use tau_session::SessionManagerEvent;
    let (manager, _dir) = setup();
    let mut rx = manager.subscribe();

    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let active = manager
        .create(fresh_request(transport, Some("hi")))
        .await
        .expect("create");
    manager.hibernate(&active.id).await.expect("hibernate");
    manager.close(&active.id).await.expect("close");

    let mut saw_created = false;
    let mut saw_hibernated = false;
    let mut saw_closed = false;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            SessionManagerEvent::Created { id, .. } if id == active.id => saw_created = true,
            SessionManagerEvent::Hibernated { id } if id == active.id => saw_hibernated = true,
            SessionManagerEvent::Closed { id } if id == active.id => saw_closed = true,
            _ => {}
        }
    }
    assert!(saw_created && saw_hibernated && saw_closed);
}

#[tokio::test]
async fn crash_recovery_restores_jsonl_messages_without_hibernate() {
    // Simulates a crash mid-session: session prompts and persists to JSONL,
    // then we drop the manager (no hibernate, no fresh snapshot.json) and
    // build a NEW manager from the same storage. Activate must restore
    // the messages from the JSONL log, NOT from a stale snapshot.
    let dir = tempfile::tempdir().unwrap();
    let storage_path = dir.path().to_path_buf();
    let id;

    {
        let storage = Arc::new(FsStorage::new(&storage_path));
        let manager = Arc::new(SessionManager::new(storage));
        let transport: Arc<dyn Transport> = TextTransport::create("ack");
        let active = manager
            .create(fresh_request(transport, None))
            .await
            .expect("create");
        id = active.id.clone();

        active.handle.prompt_and_wait("survive me").await.unwrap();
        // Wait for persister to flush.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Drop the manager WITHOUT hibernating — simulates a crash.
        // (snapshot.json is therefore never written.)
        std::mem::drop(active);
        std::mem::drop(manager);
    }

    // Confirm: snapshot.json doesn't exist, but messages.jsonl does.
    assert!(
        !storage_path.join(&id).join("snapshot.json").exists(),
        "no hibernate fired, so no snapshot file"
    );
    assert!(
        storage_path.join(&id).join("messages.jsonl").exists(),
        "JSONL log was persisted incrementally"
    );

    // Fresh manager — simulating restart.
    let storage = Arc::new(FsStorage::new(&storage_path));
    let manager = Arc::new(SessionManager::new(storage));
    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let activated = manager
        .activate(&id, test_config(), Vec::<BoxedTool>::new(), transport)
        .await
        .expect("activate after crash");

    let msgs = activated.handle.messages().await.expect("messages");
    assert!(
        msgs.iter().any(|m| matches!(m, Message::User { .. })),
        "user msg recovered from JSONL: {msgs:?}"
    );
    assert!(
        msgs.iter().any(|m| matches!(m, Message::Assistant { .. })),
        "assistant msg recovered from JSONL"
    );
}

#[tokio::test]
async fn hibernate_while_running_aborts_cleanly() {
    use std::time::Duration;
    let (manager, _dir) = setup();
    // 2-second slow transport so we can hibernate mid-prompt.
    let transport: Arc<dyn Transport> = SlowTransport::create(2_000);
    let active = manager
        .create(fresh_request(transport.clone(), None))
        .await
        .expect("create");
    let id = active.id.clone();
    let handle = active.handle.clone();

    // Fire a prompt and don't await it.
    let prompt_rx = handle.prompt("slow request").await.expect("prompt sent");

    // Give it a beat to enter the slow turn.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Hibernate while running. Must complete cleanly within a tight window.
    tokio::time::timeout(Duration::from_secs(2), manager.hibernate(&id))
        .await
        .expect("hibernate completes promptly")
        .expect("hibernate ok");

    // The aborted prompt resolves either way — the point is that hibernate
    // unblocks it. (SlowTransport emits an Error event on cancel which
    // propagates as Err; other transports just return Ok. Either is fine.)
    let _ = tokio::time::timeout(Duration::from_secs(2), prompt_rx)
        .await
        .expect("prompt resolves after abort");

    // Status flipped to Hibernated.
    let info = manager
        .list()
        .await
        .expect("list")
        .into_iter()
        .find(|s| s.id == id)
        .expect("present");
    assert_eq!(info.status, SessionStatus::Hibernated);
}

#[tokio::test]
async fn customize_runs_against_builder_before_spawn() {
    // Verify the escape hatch: customize closure can call builder methods
    // the request struct doesn't expose. We use set_system_prompt as a
    // concrete check (it's the simplest builder method to observe via the
    // capturing transport).
    use tau_agent::AgentBuilder;
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(FsStorage::new(dir.path()));
    let manager = Arc::new(SessionManager::new(storage));
    let transport = CapturingTransport::create("ack");
    let transport_dyn: Arc<dyn Transport> = transport.clone();

    let req = NewSessionRequest {
        title: None,
        project_path: std::env::current_dir().unwrap_or_else(|_| ".".into()),
        config: test_config(),
        tools: Vec::new(),
        transport: transport_dyn,
        seed_messages: Vec::new(),
        previous_summary: None,
        initial_prompt: None,
        customize: Some(Box::new(|b: &mut AgentBuilder| {
            b.set_system_prompt("CUSTOM_PROMPT_MARKER");
        })),
    };

    let active = manager.create(req).await.expect("create");
    active.handle.prompt_and_wait("hi").await.unwrap();

    let calls = transport.calls();
    assert!(
        calls
            .iter()
            .any(|c| c.system_prompt.as_deref() == Some("CUSTOM_PROMPT_MARKER")),
        "customize closure's system_prompt reached the transport"
    );
}

#[tokio::test]
async fn cannot_delete_active_session() {
    use tau_session::Error;
    let (manager, _dir) = setup();
    let transport: Arc<dyn Transport> = TextTransport::create("ack");
    let active = manager
        .create(fresh_request(transport, None))
        .await
        .expect("create");
    let err = manager.delete(&active.id).await.expect_err("should fail");
    assert!(matches!(err, Error::Running(_)));
}
