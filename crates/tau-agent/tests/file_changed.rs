//! Integration tests for `FileChanged` event emission and the
//! `SessionDiffOverlay` aggregation helper (gap #4).

use std::path::PathBuf;
use std::sync::Arc;

use tau_agent::test_utils::*;
use tau_agent::{AgentBuilder, AgentEvent, BoxedTool};
use tau_tools::diff::{FileOp, SessionDiffOverlay};
use tau_tools::{EditTool, ReadTool, WriteTool};

fn tmpdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "tau-file-changed-{}-{}",
        name,
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn write_emits_file_changed_for_new_file() {
    let dir = tmpdir("write-new");
    let target = dir.join("hello.txt");

    let transport = MockTransport::new()
        .with_tool_call_response(
            "write",
            "c1",
            serde_json::json!({"path": target.to_string_lossy(), "content": "hello\n"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(WriteTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let events = collector.events();
    let mut overlay = SessionDiffOverlay::new();
    let mut last_diff = None;
    for ev in &events {
        if let Some(d) = overlay.observe(ev) {
            last_diff = Some(d);
        }
    }
    let diff = last_diff.expect("FileChanged event observed");
    assert_eq!(diff.op, FileOp::Add);
    assert_eq!(diff.adds, 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn write_then_edit_accumulates_in_overlay() {
    let dir = tmpdir("write-then-edit");
    let target = dir.join("file.txt");
    let target_str = target.to_string_lossy().to_string();

    // Model: write "a\nb\nc\n", then read (so the read-before-write guard
    // passes), then edit b -> B.
    let transport = MockTransport::new()
        .with_tool_call_response(
            "write",
            "c1",
            serde_json::json!({"path": target_str.clone(), "content": "a\nb\nc\n"}),
        )
        .with_tool_call_response(
            "read",
            "c2",
            serde_json::json!({"path": target_str.clone()}),
        )
        .with_tool_call_response(
            "edit",
            "c3",
            serde_json::json!({"path": target_str.clone(), "old_text": "b", "new_text": "B"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(WriteTool::new()) as BoxedTool);
    builder.add_tool(Arc::new(ReadTool::new()) as BoxedTool);
    builder.add_tool(Arc::new(EditTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let mut overlay = SessionDiffOverlay::new();
    for ev in collector.events() {
        overlay.observe(&ev);
    }

    // We should have exactly one tracked path.
    assert_eq!(overlay.tracked_count(), 1);
    let snap = overlay.snapshot();
    assert_eq!(snap.len(), 1);
    let diff = &snap[0];
    // Net result: file went from non-existent to "a\nB\nc\n". Op = Add.
    assert_eq!(diff.op, FileOp::Add);
    assert!(
        diff.hunks.iter().any(|h| h
            .lines
            .iter()
            .any(|l| l.content.contains('B'))),
        "edited content reaches the cumulative diff"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn subagent_file_change_reaches_parent_overlay() {
    use tau_agent::manager::{AgentManager, AgentType, SpawnRequest};
    use tau_agent::transport::Transport;

    let dir = tmpdir("subagent-edit");
    let target = dir.join("sub.txt");
    let target_str = target.to_string_lossy().to_string();

    // Subagent transport: write to the file then text turn.
    let sub_transport: Arc<dyn Transport> = Arc::new(
        MockTransport::new()
            .with_tool_call_response(
                "write",
                "c1",
                serde_json::json!({"path": target_str.clone(), "content": "from subagent\n"}),
            )
            .with_text_response("done"),
    );

    let tools: Vec<BoxedTool> = vec![Arc::new(WriteTool::new())];
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
        description: "subagent writer".into(),
        model: None,
        cwd: None,
        isolation: None,
        depth: 0,
        inherit_history_from: None,
    };
    let _result = manager.spawn(req, cancel).await.expect("spawn");

    // Drain parent's events into the overlay.
    let mut overlay = SessionDiffOverlay::new();
    while let Ok(ev) = parent_event_rx.try_recv() {
        overlay.observe(&ev);
    }

    let snap = overlay.snapshot();
    assert_eq!(snap.len(), 1, "subagent's file change appears in overlay");
    assert_eq!(snap[0].op, FileOp::Add);
    assert_eq!(snap[0].path, target);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn edit_emits_modify_op() {
    let dir = tmpdir("edit-modify");
    let target = dir.join("file.txt");
    std::fs::write(&target, "hello\nworld\n").unwrap();
    let target_str = target.to_string_lossy().to_string();

    let transport = MockTransport::new()
        .with_tool_call_response(
            "read",
            "c1",
            serde_json::json!({"path": target_str.clone()}),
        )
        .with_tool_call_response(
            "edit",
            "c2",
            serde_json::json!({"path": target_str.clone(), "old_text": "world", "new_text": "WORLD"}),
        )
        .with_text_response("done");

    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(ReadTool::new()) as BoxedTool);
    builder.add_tool(Arc::new(EditTool::new()) as BoxedTool);
    let handle = builder.pre_handle();
    let collector = EventCollector::from_handle(&handle);
    builder.spawn();

    handle.prompt_and_wait("go").await.unwrap();
    collector.wait_for_end().await;

    let fc = collector.events().into_iter().find_map(|e| match e {
        AgentEvent::FileChanged {
            before,
            after,
            tool_call_id,
            path,
        } => Some((before, after, tool_call_id, path)),
        _ => None,
    });
    let (before, after, tool_call_id, path) = fc.expect("FileChanged emitted");
    assert_eq!(path, target);
    assert_eq!(before.as_deref(), Some("hello\nworld\n"));
    assert_eq!(after.as_deref(), Some("hello\nWORLD\n"));
    assert_eq!(tool_call_id, "c2");

    let _ = std::fs::remove_dir_all(&dir);
}
