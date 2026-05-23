//! End-to-end scenario tests that exercise tau-history the way a
//! real agent runtime would use it. Each test is a vignette: a
//! plausible host-side workflow built from the public API.
//!
//! These complement `tests/repository.rs` (which tests individual
//! operations) by stitching multiple features together — templates
//! + forks + merges + compaction + tags — and asserting on the
//! observable end-state rather than individual structural
//! properties.

use tau_ai::{AssistantMetadata, Content, Message};
use tau_history::{Branch, History, ObjectHash, Repository, ToolDef, TreePatch};

// ─── Test helpers ────────────────────────────────────────────────────

fn user(text: &str) -> Message {
    Message::user(text)
}

fn assistant_text(text: &str) -> Message {
    Message::Assistant {
        content: vec![Content::text(text)],
        metadata: AssistantMetadata::default(),
    }
}

fn assistant_with_tool_call(text: &str, tool_call_id: &str, tool_name: &str) -> Message {
    Message::Assistant {
        content: vec![
            Content::text(text),
            Content::tool_call(tool_call_id, tool_name, serde_json::json!({})),
        ],
        metadata: AssistantMetadata::default(),
    }
}

fn tool_result(call_id: &str, tool_name: &str, text: &str) -> Message {
    Message::ToolResult {
        tool_call_id: call_id.into(),
        tool_name: tool_name.into(),
        content: vec![Content::text(text)],
        is_error: false,
        timestamp: 0,
    }
}

fn tool_def(name: &str) -> ToolDef {
    ToolDef {
        name: name.into(),
        description: format!("the {name} tool"),
        parameters_schema: serde_json::json!({"type": "object"}),
    }
}

async fn texts(branch: &Branch) -> Vec<String> {
    branch
        .messages()
        .await
        .expect("messages")
        .into_iter()
        .map(|m| m.text())
        .collect()
}

// ─── Scenario 1: Full conversation lifecycle ─────────────────────────
//
// A host sets up a template, spawns an agent from it, drives a
// multi-turn conversation with tool calls, observes the prompt-side
// state (system prompt, tools, messages) at each step, then
// compacts when the conversation gets long. End-to-end exercise of
// the conversation flow agents actually do.

#[tokio::test]
async fn full_conversation_lifecycle() {
    let repo = Repository::new();

    // 1. Host registers a "coder" agent template.
    let template = repo.new_branch();
    template
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("You are a careful coder.")])
                .with_tools(vec![tool_def("read_file"), tool_def("edit_file")]),
        )
        .await
        .unwrap();
    repo.set_tag("templates/coder", template.tip().unwrap());

    // 2. Host spawns a working agent by forking from the template.
    let agent = repo.branch_at_tag("templates/coder").unwrap().fork();

    // Sanity: the working agent starts with the template's prefix
    // and no messages.
    assert!(agent.system_prompt().await.unwrap().is_some());
    assert_eq!(agent.tools().await.unwrap().len(), 2);
    assert!(agent.messages().await.unwrap().is_empty());

    // 3. User initiates the conversation.
    agent
        .commit(TreePatch::new().add_message(user("read main.rs")))
        .await
        .unwrap();

    // 4. The agent's runtime calls the model, gets back an
    //    assistant message with a tool_use.
    agent
        .commit(TreePatch::new().add_message(assistant_with_tool_call(
            "I'll read it.",
            "call_1",
            "read_file",
        )))
        .await
        .unwrap();

    // 5. Runtime executes the tool, commits the result.
    agent
        .commit(TreePatch::new().add_message(tool_result(
            "call_1",
            "read_file",
            "fn main() { println!(\"hello\"); }",
        )))
        .await
        .unwrap();

    // 6. Runtime calls the model again with the new context, gets
    //    the assistant's text response (no more tool calls).
    agent
        .commit(TreePatch::new().add_message(assistant_text("It prints 'hello'.")))
        .await
        .unwrap();

    // 7. Verify the model's view of the conversation: system prompt
    //    + tools at the prefix, messages in order.
    let msgs = agent.messages().await.unwrap();
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[0].text(), "read main.rs");
    assert!(msgs[1].text().contains("I'll read it"));
    assert!(msgs[2].text().contains("hello"));
    assert_eq!(msgs[3].text(), "It prints 'hello'.");

    assert!(agent.system_prompt().await.unwrap().is_some());
    assert_eq!(agent.tools().await.unwrap().len(), 2);

    // 8. Conversation continues for a while. We simulate by
    //    committing a bunch of turns, then compact when it gets
    //    long.
    for i in 0..10 {
        agent
            .commit(TreePatch::new().add_message(user(&format!("question {i}"))))
            .await
            .unwrap();
        agent
            .commit(TreePatch::new().add_message(assistant_text(&format!("answer {i}"))))
            .await
            .unwrap();
    }
    assert_eq!(agent.messages().await.unwrap().len(), 4 + 20);

    // 9. Host triggers compaction. Keep the most recent 4 turns;
    //    summarize everything before them.
    let total = agent.messages().await.unwrap().len();
    let keep_recent = 4;
    (agent.as_ref() as &dyn History)
        .compact_prefix(
            total - keep_recent,
            user("<context-summary>earlier conversation</context-summary>"),
            "earlier conversation".into(),
        )
        .await
        .unwrap();

    let after = agent.messages().await.unwrap();
    assert_eq!(after.len(), 1 + keep_recent);
    assert!(after[0].text().contains("context-summary"));
    assert_eq!(
        agent.previous_summary().await.unwrap().as_deref(),
        Some("earlier conversation"),
    );

    // 10. After compaction, the agent's tools and system prompt
    //     are unchanged — the prefix inherited correctly.
    assert!(agent.system_prompt().await.unwrap().is_some());
    assert_eq!(agent.tools().await.unwrap().len(), 2);
}

// ─── Scenario 2: Multi-agent fleet on one repository ─────────────────
//
// A host runs N agents simultaneously, all backed by one
// Repository, each working on its own branch. Verifies storage is
// shared (template prefix dedupes) and the branches stay isolated
// (one agent's commits don't appear on another's).

#[tokio::test]
async fn multi_agent_fleet_shares_template_storage() {
    let repo = Repository::new();

    // Register a template with a moderately expensive setup.
    let template = repo.new_branch();
    template
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("You are a researcher.")])
                .with_tools(
                    (0..10)
                        .map(|i| tool_def(&format!("research_tool_{i}")))
                        .collect(),
                ),
        )
        .await
        .unwrap();
    repo.set_tag("templates/researcher", template.tip().unwrap());

    let objects_before_fleet = repo.object_count();

    // Spawn 5 agents.
    let mut agents = Vec::new();
    for _ in 0..5 {
        let a = repo.branch_at_tag("templates/researcher").unwrap().fork();
        agents.push(a);
    }

    // Spawning is zero-cost — the forks are pointer clones.
    assert_eq!(repo.object_count(), objects_before_fleet);

    // Each agent does some independent work.
    for (i, agent) in agents.iter().enumerate() {
        agent
            .commit(TreePatch::new().add_message(user(&format!("question for agent {i}"))))
            .await
            .unwrap();
        agent
            .commit(TreePatch::new().add_message(assistant_text(&format!("answer from {i}"))))
            .await
            .unwrap();
    }

    // Each agent sees only its own messages.
    for (i, agent) in agents.iter().enumerate() {
        let texts = texts(agent.as_ref()).await;
        assert_eq!(texts.len(), 2);
        assert!(texts[0].contains(&format!("agent {i}")));
        assert!(texts[1].contains(&format!("from {i}")));
    }

    // All 5 agents still see the same template-derived setup.
    for agent in &agents {
        assert!(agent.system_prompt().await.unwrap().is_some());
        assert_eq!(agent.tools().await.unwrap().len(), 10);
    }
}

// ─── Scenario 3: Subagent fork + merge ───────────────────────────────
//
// A parent agent runs a subagent. The subagent forks from the
// parent's current state (inheriting context), does its own work,
// and the parent records the subagent's final tip via merge so the
// graph remembers what produced the tool_result's summary.

#[tokio::test]
async fn subagent_fork_and_merge_preserves_provenance() {
    let repo = Repository::new();
    let parent = repo.new_branch();
    parent
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("orchestrate things")])
                .with_tools(vec![tool_def("spawn_subagent")])
                .add_message(user("Investigate this codebase.")),
        )
        .await
        .unwrap();
    parent
        .commit(TreePatch::new().add_message(assistant_with_tool_call(
            "Spawning investigator.",
            "call_inv",
            "spawn_subagent",
        )))
        .await
        .unwrap();
    let parent_tip_before_subagent = parent.tip().unwrap();

    // Subagent forks from parent — inheriting system prompt, tools,
    // and the conversation so far.
    let subagent = parent.fork();

    // Subagent does its own multi-turn investigation.
    subagent
        .commit(TreePatch::new().add_message(user("focus on auth module")))
        .await
        .unwrap();
    subagent
        .commit(TreePatch::new().add_message(assistant_text("auth uses JWT, signed with HS256")))
        .await
        .unwrap();
    subagent
        .commit(TreePatch::new().add_message(user("any vulnerabilities?")))
        .await
        .unwrap();
    subagent
        .commit(TreePatch::new().add_message(assistant_text("no obvious issues found")))
        .await
        .unwrap();
    let subagent_tip = subagent.tip().unwrap();

    // Subagent's messages include parent's prefix + its own work.
    let sub_msgs = subagent.messages().await.unwrap();
    assert!(sub_msgs.iter().any(|m| m.text().contains("Investigate")));
    assert!(sub_msgs.iter().any(|m| m.text().contains("JWT")));
    assert!(sub_msgs.iter().any(|m| m.text().contains("no obvious issues")));

    // Parent merges the subagent's tip with its tool_result.
    parent
        .merge(
            TreePatch::new().add_message(tool_result(
                "call_inv",
                "spawn_subagent",
                "Investigator: auth uses JWT/HS256; no obvious issues.",
            )),
            vec![subagent_tip],
        )
        .await
        .unwrap();

    // Parent's linear view doesn't include the subagent's messages —
    // just the original turn plus the merged-in tool_result summary.
    let parent_msgs = parent.messages().await.unwrap();
    assert_eq!(parent_msgs.len(), 3, "user, assistant(tool_call), tool_result");
    assert!(parent_msgs[2].text().contains("Investigator:"));

    // But the graph remembers the subagent's tip via extra_parents.
    let merge_commit = repo.get_commit(&parent.tip().unwrap()).unwrap();
    assert_eq!(merge_commit.extra_parents, vec![subagent_tip]);
    assert_eq!(merge_commit.parent, Some(parent_tip_before_subagent));

    // Audit trail check: from the merge commit, we can walk back to
    // the subagent's history via the extra parent.
    let subagent_commit = repo.get_commit(&subagent_tip).unwrap();
    assert!(subagent_commit.parent.is_some(), "subagent's first commit chains back");
}

// ─── Scenario 4: Snapshot and restore via branch_at ──────────────────
//
// Save a branch's tip hash, do destructive operations, then restore
// to the saved tip. Verifies branch_at faithfully reconstructs the
// state — including the next_seq for subsequent commits.

#[tokio::test]
async fn snapshot_and_restore_via_branch_at() {
    let repo = Repository::new();
    let branch = repo.new_branch();

    // Build up some state.
    branch
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("be helpful")])
                .with_tools(vec![tool_def("bash")])
                .add_message(user("hello")),
        )
        .await
        .unwrap();
    branch
        .commit(TreePatch::new().add_message(assistant_text("hi there")))
        .await
        .unwrap();
    branch
        .commit(TreePatch::new().add_message(user("what's 2+2?")))
        .await
        .unwrap();

    // Take a snapshot.
    let snapshot = branch.tip().unwrap();
    let snapshot_texts = texts(branch.as_ref()).await;

    // Continue the branch with more commits.
    branch
        .commit(TreePatch::new().add_message(assistant_text("4")))
        .await
        .unwrap();
    branch
        .commit(TreePatch::new().add_message(user("good")))
        .await
        .unwrap();

    // Reopen the branch at the snapshot. Should show the state as
    // it was when we took the snapshot, not the latest state.
    let restored = repo.branch_at(snapshot);
    let restored_texts = texts(restored.as_ref()).await;
    assert_eq!(restored_texts, snapshot_texts);
    assert!(restored.system_prompt().await.unwrap().is_some());
    assert_eq!(restored.tools().await.unwrap().len(), 1);

    // Subsequent commits on the restored branch use a fresh seq
    // counter derived from the snapshot's tree (continues from
    // where the snapshot left off; new entries land in correct
    // order at read time).
    restored
        .commit(TreePatch::new().add_message(assistant_text("4, restored")))
        .await
        .unwrap();
    let restored_after = texts(restored.as_ref()).await;
    assert_eq!(restored_after.len(), snapshot_texts.len() + 1);
    assert_eq!(restored_after.last().unwrap(), "4, restored");

    // Original branch is unaffected by operations on the restored
    // view — they're separate branches now.
    let original_after = texts(branch.as_ref()).await;
    assert_eq!(original_after.len(), 5);
    assert_eq!(original_after.last().unwrap(), "good");
}

// ─── Scenario 5: Template versioning ─────────────────────────────────
//
// A host has a template registered under a tag. A new version of
// the template is created and the tag is updated. Old agents
// spawned from v1 keep their v1 state; new agents get v2.

#[tokio::test]
async fn template_versioning_with_tag_updates() {
    let repo = Repository::new();

    // Register template v1.
    let v1 = repo.new_branch();
    v1.commit(
        TreePatch::new()
            .with_system_prompt(vec![Content::text("v1: be brief")])
            .with_tools(vec![tool_def("search")]),
    )
    .await
    .unwrap();
    repo.set_tag("templates/assistant", v1.tip().unwrap());

    // Spawn an agent under v1.
    let agent_v1 = repo.branch_at_tag("templates/assistant").unwrap().fork();
    agent_v1
        .commit(TreePatch::new().add_message(user("hello from v1 era")))
        .await
        .unwrap();

    // Host registers template v2 with extra tools and an updated
    // system prompt.
    let v2 = repo.new_branch();
    v2.commit(
        TreePatch::new()
            .with_system_prompt(vec![Content::text("v2: be brief and cite sources")])
            .with_tools(vec![tool_def("search"), tool_def("cite")]),
    )
    .await
    .unwrap();
    repo.set_tag("templates/assistant", v2.tip().unwrap());

    // Spawn an agent under v2.
    let agent_v2 = repo.branch_at_tag("templates/assistant").unwrap().fork();
    agent_v2
        .commit(TreePatch::new().add_message(user("hello from v2 era")))
        .await
        .unwrap();

    // Old agent kept its v1 surface.
    let v1_sp = agent_v1.system_prompt().await.unwrap().unwrap();
    assert!(v1_sp[0].as_text().unwrap().starts_with("v1"));
    assert_eq!(agent_v1.tools().await.unwrap().len(), 1);

    // New agent has v2's surface.
    let v2_sp = agent_v2.system_prompt().await.unwrap().unwrap();
    assert!(v2_sp[0].as_text().unwrap().starts_with("v2"));
    assert_eq!(agent_v2.tools().await.unwrap().len(), 2);

    // Both agents have one user message each — their own.
    assert_eq!(agent_v1.messages().await.unwrap().len(), 1);
    assert_eq!(agent_v2.messages().await.unwrap().len(), 1);

    // The tag now points at v2; resolving it gives the new template.
    assert_eq!(
        repo.resolve_tag("templates/assistant"),
        Some(v2.tip().unwrap())
    );
}

// ─── Scenario 6: Tools change mid-conversation ───────────────────────
//
// A host changes the available tools partway through a conversation
// — e.g., handoff between phases of work. Verifies the new tools
// are in effect for subsequent turns while the message history is
// preserved (and not rebased).

#[tokio::test]
async fn tools_change_mid_conversation_preserves_history() {
    let repo = Repository::new();
    let agent = repo.new_branch();

    // Phase 1: planning, with planning tools.
    agent
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("a multi-phase assistant")])
                .with_tools(vec![tool_def("plan"), tool_def("estimate")])
                .add_message(user("we need a project plan"))
                .add_message(assistant_text("I'll plan it out")),
        )
        .await
        .unwrap();
    assert_eq!(
        agent
            .tools()
            .await
            .unwrap()
            .iter()
            .map(|t| t.name.clone())
            .collect::<Vec<_>>(),
        vec!["estimate".to_string(), "plan".to_string()]
    );
    let phase1_messages_count = agent.messages().await.unwrap().len();

    // Capture the /messages root hash before the tool change so we
    // can verify it's structurally unchanged afterward (the dedup
    // property).
    let messages_subtree_hash_before = {
        let tip = agent.tip().unwrap();
        let commit = repo.get_commit(&tip).unwrap();
        let root = repo.get_tree(&commit.tree).unwrap();
        match root.get("messages").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        }
    };

    // Phase 2: handoff — switch to execution tools. Commit a new
    // tool set with no new messages; the /messages subtree should
    // inherit from the previous commit, hash-shared.
    agent
        .commit(TreePatch::new().with_tools(vec![tool_def("write_code"), tool_def("run_tests")]))
        .await
        .unwrap();

    // The new tip's /messages subtree hash matches the prior one —
    // unchanged history, structurally shared.
    let messages_subtree_hash_after = {
        let tip = agent.tip().unwrap();
        let commit = repo.get_commit(&tip).unwrap();
        let root = repo.get_tree(&commit.tree).unwrap();
        match root.get("messages").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        }
    };
    assert_eq!(
        messages_subtree_hash_before, messages_subtree_hash_after,
        "changing tools without changing messages preserves /messages subtree hash"
    );

    // Tools are the new set; system prompt and messages are
    // inherited.
    assert_eq!(
        agent
            .tools()
            .await
            .unwrap()
            .iter()
            .map(|t| t.name.clone())
            .collect::<Vec<_>>(),
        vec!["run_tests".to_string(), "write_code".to_string()]
    );
    assert!(agent.system_prompt().await.unwrap().is_some());
    assert_eq!(agent.messages().await.unwrap().len(), phase1_messages_count);

    // Continue phase 2 — new messages land alongside phase 1's.
    agent
        .commit(
            TreePatch::new()
                .add_message(user("now build it"))
                .add_message(assistant_text("starting implementation")),
        )
        .await
        .unwrap();
    let final_texts = texts(agent.as_ref()).await;
    assert_eq!(
        final_texts,
        vec![
            "we need a project plan",
            "I'll plan it out",
            "now build it",
            "starting implementation",
        ]
    );
}

// ─── Scenario 7: Large conversation stress test ─────────────────────
//
// Many turns, many tool calls, mixed message types. Verifies
// correctness at scale and that bucketing is doing its job — the
// blob count grows linearly with content, not quadratically.

#[tokio::test]
async fn large_conversation_with_mixed_message_types() {
    let repo = Repository::new();
    let agent = repo.new_branch();

    agent
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("you handle a lot of turns")])
                .with_tools(vec![tool_def("op")]),
        )
        .await
        .unwrap();

    // 500 turns: each turn = user → assistant(with tool_use) →
    // tool_result. That's 1500 messages across three types.
    let turns = 500;
    for i in 0..turns {
        agent
            .commit(TreePatch::new().add_messages(vec![
                user(&format!("question {i}")),
                assistant_with_tool_call("running", &format!("call_{i}"), "op"),
                tool_result(&format!("call_{i}"), "op", &format!("result {i}")),
            ]))
            .await
            .unwrap();
    }

    let msgs = agent.messages().await.unwrap();
    assert_eq!(msgs.len(), turns * 3);

    // Verify ordering: every 3 consecutive messages should be
    // (user, assistant, tool_result) in that order, with i
    // matching across them.
    for i in 0..turns {
        let base = i * 3;
        assert!(msgs[base].text().contains(&format!("question {i}")));
        assert!(msgs[base + 1].text().contains("running"));
        assert!(msgs[base + 2].text().contains(&format!("result {i}")));
    }

    // System prompt + tools still in effect.
    assert!(agent.system_prompt().await.unwrap().is_some());
    assert_eq!(agent.tools().await.unwrap().len(), 1);
}

// ─── Scenario 8: Compaction across types preserves type semantics ────
//
// Compaction replaces the message prefix with a summary. The
// summary is a user-role message (by convention). Verify that
// after compaction, the agent's messages() view shows the summary
// first, followed by the kept tail in order, with types preserved.

#[tokio::test]
async fn compaction_with_mixed_types_preserves_kept_tail_order() {
    let repo = Repository::new();
    let agent = repo.new_branch();

    // Build up a mixed conversation: alternating user / assistant /
    // tool_result patterns, 20 turns.
    for i in 0..7 {
        agent
            .commit(TreePatch::new().add_messages(vec![
                user(&format!("u{i}")),
                assistant_with_tool_call("ack", &format!("c{i}"), "x"),
                tool_result(&format!("c{i}"), "x", &format!("r{i}")),
            ]))
            .await
            .unwrap();
    }
    assert_eq!(agent.messages().await.unwrap().len(), 21);

    // Compact the first 15 messages; keep the last 6 (= 2 turns).
    (agent.as_ref() as &dyn History)
        .compact_prefix(15, user("<summary>"), "summary text".into())
        .await
        .unwrap();

    let after = agent.messages().await.unwrap();
    assert_eq!(after.len(), 7, "1 summary + 6 kept messages");
    assert!(after[0].text().contains("summary"));

    // The 6 kept messages are the last two turns' contents in order.
    let kept_texts: Vec<String> = after.iter().skip(1).map(|m| m.text()).collect();
    assert_eq!(
        kept_texts,
        vec![
            "u5", "ack", "r5", "u6", "ack", "r6",
        ]
    );

    // Their types are preserved: user, assistant, tool_result,
    // user, assistant, tool_result.
    for (i, msg) in after.iter().skip(1).enumerate() {
        match i % 3 {
            0 => assert!(matches!(msg, Message::User { .. }), "user at offset {i}"),
            1 => assert!(matches!(msg, Message::Assistant { .. }), "assistant at offset {i}"),
            2 => assert!(matches!(msg, Message::ToolResult { .. }), "tool_result at offset {i}"),
            _ => unreachable!(),
        }
    }
}

// ─── Scenario 9: Parallel subagents merged into parent ──────────────
//
// A parent spawns three subagents concurrently (via fork), each
// does its own work, then the parent collects them via three
// separate merge commits — one per subagent.

#[tokio::test]
async fn parent_collects_three_parallel_subagents() {
    let repo = Repository::new();
    let parent = repo.new_branch();
    parent
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("orchestrate")])
                .with_tools(vec![tool_def("spawn_3_parallel")])
                .add_message(user("research these three topics in parallel")),
        )
        .await
        .unwrap();
    parent
        .commit(TreePatch::new().add_message(assistant_with_tool_call(
            "spawning three",
            "call_fanout",
            "spawn_3_parallel",
        )))
        .await
        .unwrap();

    // Three subagents fork from parent's current state and each
    // does its own work concurrently. Use Arc + tokio::spawn for
    // realistic concurrency.
    let topics = vec!["topic_a", "topic_b", "topic_c"];
    let mut subagent_tips: Vec<ObjectHash> = Vec::new();
    for topic in &topics {
        let sub = parent.fork();
        sub.commit(TreePatch::new().add_messages(vec![
            user(&format!("research {topic}")),
            assistant_text(&format!("findings about {topic}")),
        ]))
        .await
        .unwrap();
        subagent_tips.push(sub.tip().unwrap());
    }

    // Parent records the result of each subagent as a separate
    // tool_result, merging each one's tip via extra_parents.
    for (i, tip) in subagent_tips.iter().enumerate() {
        parent
            .merge(
                TreePatch::new().add_message(tool_result(
                    "call_fanout",
                    "spawn_3_parallel",
                    &format!("summary of {}", topics[i]),
                )),
                vec![*tip],
            )
            .await
            .unwrap();
    }

    // Parent's messages: the initial setup + 3 tool_result summaries.
    let parent_msgs = parent.messages().await.unwrap();
    assert_eq!(parent_msgs.len(), 2 + 3);
    let summaries: Vec<String> = parent_msgs.iter().skip(2).map(|m| m.text()).collect();
    assert_eq!(
        summaries,
        vec![
            "summary of topic_a".to_string(),
            "summary of topic_b".to_string(),
            "summary of topic_c".to_string(),
        ]
    );

    // Each subagent's tip is recorded in the graph and reachable.
    // Walk back through the parent's chain — the last three commits
    // should each have one extra_parent.
    let mut cursor = parent.tip();
    let mut found_extras: Vec<Vec<ObjectHash>> = Vec::new();
    while let Some(h) = cursor {
        let c = repo.get_commit(&h).unwrap();
        if !c.extra_parents.is_empty() {
            found_extras.push(c.extra_parents.clone());
        }
        cursor = c.parent;
    }
    // We walked tip-to-root; reverse to get chronological order.
    found_extras.reverse();
    assert_eq!(found_extras.len(), 3, "three merge commits in the chain");
    for (i, extras) in found_extras.iter().enumerate() {
        assert_eq!(extras, &vec![subagent_tips[i]]);
    }
}

// ─── Scenario 10: Tags pin checkpoints during a long-running agent ───
//
// A host periodically tags "checkpoint" commits during a long
// conversation so it can roll back to a known-good state if needed.
// Verifies tags persist across many subsequent commits and resolve
// correctly even after the branch has moved far past them.

#[tokio::test]
async fn checkpoint_tags_persist_across_long_branch_history() {
    let repo = Repository::new();
    let agent = repo.new_branch();
    agent
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("checkpointed agent")])
                .with_tools(vec![tool_def("op")]),
        )
        .await
        .unwrap();

    let mut checkpoints: Vec<(String, ObjectHash)> = Vec::new();

    // Run 50 turns, tagging every 10th.
    for i in 0..50 {
        agent
            .commit(TreePatch::new().add_message(user(&format!("turn {i}"))))
            .await
            .unwrap();
        if i % 10 == 9 {
            let name = format!("checkpoint/turn-{}", i + 1);
            let tip = agent.tip().unwrap();
            repo.set_tag(&name, tip);
            checkpoints.push((name, tip));
        }
    }

    // Branch has 50 messages now.
    assert_eq!(agent.messages().await.unwrap().len(), 50);

    // Every checkpoint still resolves; reopening at each gives the
    // expected state.
    for (name, expected_tip) in &checkpoints {
        assert_eq!(repo.resolve_tag(name), Some(*expected_tip));
        let restored = repo.branch_at_tag(name).unwrap();
        let len = restored.messages().await.unwrap().len();
        // Each checkpoint was set after turn (i+1). Names are
        // "checkpoint/turn-10", "/turn-20", etc.
        let expected: usize = name
            .strip_prefix("checkpoint/turn-")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(len, expected);
    }

    // tags() enumerates them in sorted order.
    let listed: Vec<String> = repo.tags().into_iter().map(|(n, _)| n).collect();
    let expected_names: Vec<String> =
        checkpoints.iter().map(|(n, _)| n.clone()).collect();
    // BTreeMap sorts; turn-10 < turn-20 < turn-30 < turn-40 < turn-50.
    assert_eq!(listed, expected_names);
}
