//! Integration tests for the git-data-model repository.
//!
//! Exercises Branch's git-flavored API (`commit`, `merge`, `fork`,
//! `tip`) and the conversation-flavored History trait (`messages`,
//! `system_prompt`, `tools`, `previous_summary`, `append`,
//! `compact_prefix`) on the same `Arc<Branch>`. Tests also verify
//! the structural properties that motivated this design: cheap
//! forks, prefix dedup, tools-change-without-rebase, cache-aligned
//! prefix sharing.

use std::sync::Arc;

use tau_ai::{Content, Message};
use tau_history::{Branch, History, ObjectHash, Repository, ToolDef, TreePatch};

fn user(text: &str) -> Message {
    Message::user(text)
}

fn tool(name: &str) -> ToolDef {
    ToolDef {
        name: name.into(),
        description: format!("{name} tool"),
        parameters_schema: serde_json::json!({"type": "object"}),
    }
}

async fn assert_texts(history: &dyn History, expected: &[&str]) {
    let msgs = history.messages().await.expect("messages read");
    let got: Vec<String> = msgs.iter().map(|m| m.text()).collect();
    let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(got, expected, "history messages did not match");
}

// ─── Empty branch ────────────────────────────────────────────────────

#[tokio::test]
async fn empty_branch_is_empty_everywhere() {
    let repo = Repository::new();
    let branch = repo.new_branch();

    assert!(branch.tip().is_none());
    assert!(branch.messages().await.unwrap().is_empty());
    assert!(branch.system_prompt().await.unwrap().is_none());
    assert!(branch.tools().await.unwrap().is_empty());
    assert!(branch.previous_summary().await.unwrap().is_none());
}

#[tokio::test]
async fn empty_patch_fails() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    let err = branch
        .commit(TreePatch::new())
        .await
        .expect_err("empty patch must error");
    assert!(err.to_string().contains("empty patch"), "got: {err}");
}

// ─── Single commits ──────────────────────────────────────────────────

#[tokio::test]
async fn commit_single_message_round_trips() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(TreePatch::new().add_message(user("hello")))
        .await
        .unwrap();

    assert_texts(branch.as_ref(), &["hello"]).await;
    assert!(branch.tip().is_some());
}

#[tokio::test]
async fn commit_with_system_prompt_and_tools() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("you are helpful")])
                .with_tools(vec![tool("bash"), tool("read_file")])
                .add_message(user("hi")),
        )
        .await
        .unwrap();

    let sp = branch.system_prompt().await.unwrap();
    assert_eq!(sp.as_deref().map(|c| c.len()), Some(1));
    let tools = branch.tools().await.unwrap();
    let names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
    // Tools come back sorted by name (BTreeMap iteration).
    assert_eq!(names, vec!["bash".to_string(), "read_file".to_string()]);
    assert_texts(branch.as_ref(), &["hi"]).await;
}

#[tokio::test]
async fn multiple_commits_preserve_message_order() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(TreePatch::new().add_message(user("a")))
        .await
        .unwrap();
    branch
        .commit(TreePatch::new().add_messages([user("b"), user("c")]))
        .await
        .unwrap();
    branch
        .commit(TreePatch::new().add_message(user("d")))
        .await
        .unwrap();
    assert_texts(branch.as_ref(), &["a", "b", "c", "d"]).await;
}

#[tokio::test]
async fn append_via_history_trait_is_equivalent_to_commit_add_message() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    (branch.as_ref() as &dyn History)
        .append(vec![user("via append")])
        .await
        .unwrap();
    assert_texts(branch.as_ref(), &["via append"]).await;
}

// ─── Inheritance: tools and system_prompt persist across commits ─────

#[tokio::test]
async fn unchanged_tools_inherit_across_commits() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(
            TreePatch::new()
                .with_tools(vec![tool("bash")])
                .add_message(user("turn 1")),
        )
        .await
        .unwrap();
    // Subsequent commit doesn't touch tools — they should still be there.
    branch
        .commit(TreePatch::new().add_message(user("turn 2")))
        .await
        .unwrap();

    let tools = branch.tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "bash");
}

#[tokio::test]
async fn unchanged_system_prompt_inherits_across_commits() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("be brief")])
                .add_message(user("hi")),
        )
        .await
        .unwrap();
    branch
        .commit(TreePatch::new().add_message(user("again")))
        .await
        .unwrap();
    assert!(branch.system_prompt().await.unwrap().is_some());
}

// ─── Tools-change-without-rebase: the key property ───────────────────

#[tokio::test]
async fn changing_tools_does_not_alter_existing_message_hashes() {
    // The whole point of putting tools in the tree (not the chain):
    // changing tools doesn't force a rebase. Existing messages stay
    // at their original blob hashes; only the new commit's /tools
    // subtree differs.
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(
            TreePatch::new()
                .with_tools(vec![tool("bash")])
                .add_message(user("m1"))
                .add_message(user("m2")),
        )
        .await
        .unwrap();

    let blob_count_before = repo.blob_count();

    // Commit again, this time changing tools. No new messages.
    branch
        .commit(TreePatch::new().with_tools(vec![tool("bash"), tool("read")]))
        .await
        .unwrap();

    // The new tools subtree references one new blob (read's def).
    // The existing message blobs and bash's def blob are unchanged
    // and not re-stored. So the blob count should grow by exactly
    // 1 (read's blob).
    assert_eq!(
        repo.blob_count(),
        blob_count_before + 1,
        "tool change should add only the new tool blob, not duplicate messages"
    );

    // And the messages are still visible — they didn't move.
    assert_texts(branch.as_ref(), &["m1", "m2"]).await;
    let final_tools = branch.tools().await.unwrap();
    let names: Vec<String> = final_tools.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["bash".to_string(), "read".to_string()]);
}

#[tokio::test]
async fn tools_only_commit_changes_root_tree_but_not_messages_subtree() {
    // Verify the structural property at the tree level: changing
    // /tools updates the root tree hash but the /messages subtree
    // hash is unchanged (it's the same subtree, hash-shared).
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(
            TreePatch::new()
                .with_tools(vec![tool("bash")])
                .add_message(user("m1")),
        )
        .await
        .unwrap();

    // Read the root tree and its /messages subtree hash.
    let tip = branch.tip().unwrap();
    let c1 = repo.get_commit(&tip).unwrap();
    let t1 = repo.get_tree(&c1.tree).unwrap();
    let messages_hash_1 = t1.get("messages").cloned();

    // Tools-only commit.
    branch
        .commit(TreePatch::new().with_tools(vec![tool("read")]))
        .await
        .unwrap();

    let tip2 = branch.tip().unwrap();
    let c2 = repo.get_commit(&tip2).unwrap();
    let t2 = repo.get_tree(&c2.tree).unwrap();
    let messages_hash_2 = t2.get("messages").cloned();

    assert_ne!(c1.tree, c2.tree, "root tree differs after tools change");
    assert_eq!(
        messages_hash_1, messages_hash_2,
        "/messages subtree should share hash across the tool change"
    );
}

// ─── Fork sharing ────────────────────────────────────────────────────

#[tokio::test]
async fn fork_inherits_state_at_time_of_fork() {
    let repo = Repository::new();
    let a = repo.new_branch();
    a.commit(
        TreePatch::new()
            .with_system_prompt(vec![Content::text("shared sp")])
            .with_tools(vec![tool("bash")])
            .add_messages([user("m1"), user("m2")]),
    )
    .await
    .unwrap();

    let b = a.fork();
    assert_eq!(b.tip(), a.tip(), "fork starts at same tip");
    assert_texts(b.as_ref(), &["m1", "m2"]).await;
    assert!(b.system_prompt().await.unwrap().is_some());
    assert_eq!(b.tools().await.unwrap().len(), 1);
}

#[tokio::test]
async fn fork_diverges_independently() {
    let repo = Repository::new();
    let a = repo.new_branch();
    a.commit(TreePatch::new().add_message(user("shared")))
        .await
        .unwrap();
    let b = a.fork();

    a.commit(TreePatch::new().add_message(user("only on a")))
        .await
        .unwrap();
    b.commit(TreePatch::new().add_message(user("only on b")))
        .await
        .unwrap();

    assert_texts(a.as_ref(), &["shared", "only on a"]).await;
    assert_texts(b.as_ref(), &["shared", "only on b"]).await;
}

#[tokio::test]
async fn fork_shares_storage_for_setup_prefix() {
    // Two forks both inheriting the same expensive setup (system
    // prompt + 10 tools) should not duplicate any of it in storage.
    let repo = Repository::new();
    let a = repo.new_branch();
    a.commit(
        TreePatch::new()
            .with_system_prompt(vec![Content::text("shared")])
            .with_tools((0..10).map(|i| tool(&format!("t{i}"))).collect())
            .add_message(user("base")),
    )
    .await
    .unwrap();

    let blobs_before_fork = repo.blob_count();
    let trees_before_fork = repo.tree_count();

    let _b = a.fork();
    // Fork is just a tip-pointer clone; no new objects.
    assert_eq!(repo.blob_count(), blobs_before_fork);
    assert_eq!(repo.tree_count(), trees_before_fork);
}

// ─── Compaction ──────────────────────────────────────────────────────

#[tokio::test]
async fn compact_replaces_prefix_with_summary_and_keeps_tail() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(
            TreePatch::new().add_messages([
                user("old 1"),
                user("old 2"),
                user("old 3"),
                user("recent 1"),
                user("recent 2"),
            ]),
        )
        .await
        .unwrap();

    let summary_msg = user("<context-summary>Old stuff.</context-summary>");
    (branch.as_ref() as &dyn History)
        .compact_prefix(3, summary_msg, "Old stuff.".into())
        .await
        .unwrap();

    assert_texts(
        branch.as_ref(),
        &[
            "<context-summary>Old stuff.</context-summary>",
            "recent 1",
            "recent 2",
        ],
    )
    .await;
    assert_eq!(
        branch.previous_summary().await.unwrap().as_deref(),
        Some("Old stuff.")
    );
}

#[tokio::test]
async fn compact_out_of_bounds_errors() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(TreePatch::new().add_message(user("only")))
        .await
        .unwrap();
    let err = (branch.as_ref() as &dyn History)
        .compact_prefix(99, user("summary"), "s".into())
        .await
        .expect_err("end > len must error");
    let msg = err.to_string();
    assert!(
        msg.contains("end=99") && msg.contains("length 1"),
        "expected CutOutOfBounds, got: {msg}"
    );
}

#[tokio::test]
async fn compact_on_one_fork_does_not_affect_sibling() {
    let repo = Repository::new();
    let original = repo.new_branch();
    original
        .commit(TreePatch::new().add_messages([user("p1"), user("p2"), user("p3")]))
        .await
        .unwrap();
    let sibling = original.fork();

    (original.as_ref() as &dyn History)
        .compact_prefix(2, user("compacted"), "compacted".into())
        .await
        .unwrap();

    assert_texts(original.as_ref(), &["compacted", "p3"]).await;
    assert_texts(sibling.as_ref(), &["p1", "p2", "p3"]).await;
}

// ─── Merge (multi-parent commits) ────────────────────────────────────

#[tokio::test]
async fn merge_records_extra_parents_on_commit() {
    // A parent collects a subagent's result. The merge commit's
    // tree carries the tool_result message; its extra_parents
    // record the subagent's tip.
    let repo = Repository::new();
    let parent = repo.new_branch();
    parent
        .commit(TreePatch::new().add_message(user("parent's turn")))
        .await
        .unwrap();

    let subagent = repo.new_branch();
    subagent
        .commit(TreePatch::new().add_message(user("subagent's work")))
        .await
        .unwrap();
    let s_tip = subagent.tip().unwrap();

    parent
        .merge(
            TreePatch::new().add_message(user("[tool result: <summary>]")),
            vec![s_tip],
        )
        .await
        .unwrap();

    let merge_tip = parent.tip().unwrap();
    let commit = repo.get_commit(&merge_tip).unwrap();
    assert_eq!(commit.extra_parents, vec![s_tip]);
}

#[tokio::test]
async fn merge_does_not_pull_subagent_messages_into_parent_view() {
    // Parent's messages() is the linear walk of the parent branch.
    // The subagent's messages are reachable via the merge commit's
    // extra_parents (host-side traversal), but they're not in the
    // parent's API-visible messages.
    let repo = Repository::new();
    let parent = repo.new_branch();
    parent
        .commit(TreePatch::new().add_messages([user("p1"), user("p2")]))
        .await
        .unwrap();

    let subagent = repo.new_branch();
    subagent
        .commit(TreePatch::new().add_messages([user("s1"), user("s2"), user("s3")]))
        .await
        .unwrap();
    let s_tip = subagent.tip().unwrap();

    parent
        .merge(TreePatch::new().add_message(user("summary")), vec![s_tip])
        .await
        .unwrap();

    assert_texts(parent.as_ref(), &["p1", "p2", "summary"]).await;
    assert_texts(subagent.as_ref(), &["s1", "s2", "s3"]).await;
}

// ─── branch_at ───────────────────────────────────────────────────────

#[tokio::test]
async fn branch_at_known_tip_replays_state() {
    let repo = Repository::new();
    let original = repo.new_branch();
    original
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("sp")])
                .with_tools(vec![tool("bash")])
                .add_messages([user("a"), user("b")]),
        )
        .await
        .unwrap();
    let tip = original.tip().unwrap();

    let resumed = repo.branch_at(tip);
    assert_eq!(resumed.tip(), Some(tip));
    assert_texts(resumed.as_ref(), &["a", "b"]).await;
    assert!(resumed.system_prompt().await.unwrap().is_some());
    assert_eq!(resumed.tools().await.unwrap().len(), 1);
}

#[tokio::test]
async fn branch_at_unknown_tip_errors_on_first_read() {
    let repo = Repository::new();
    let bogus = ObjectHash::from_bytes([0xab; 32]);
    let branch = repo.branch_at(bogus);
    let err = branch
        .messages()
        .await
        .expect_err("unknown tip must error on read");
    assert!(
        err.to_string().contains("not found"),
        "expected NotFound, got: {err}"
    );
}

// ─── Dedup at the blob layer ─────────────────────────────────────────

#[tokio::test]
async fn identical_messages_committed_to_different_branches_share_blobs() {
    let repo = Repository::new();
    let a = repo.new_branch();
    a.commit(TreePatch::new().add_message(user("setup")))
        .await
        .unwrap();
    let b = a.fork();

    let identical = user("same message");
    a.commit(TreePatch::new().add_message(identical.clone()))
        .await
        .unwrap();
    b.commit(TreePatch::new().add_message(identical))
        .await
        .unwrap();

    // The identical message blob should be stored once. The
    // /messages subtrees of a and b differ (different bucket-tree
    // hashes) because they each contain one occurrence of the
    // message at index 1 — wait, both have ["setup", "same"]
    // sequences which means their /messages trees are
    // byte-identical, so they share the tree too.
    // Total blobs: setup blob + same-message blob = 2.
    assert_eq!(repo.blob_count(), 2);
}

// ─── Tags ────────────────────────────────────────────────────────────
//
// Tags are named pointers to commits. They survive any branch's
// lifecycle and serve as bookmarks: agent type templates, session
// snapshots, audit markers. Hosts choose the naming convention.

#[tokio::test]
async fn tag_round_trips() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(TreePatch::new().add_message(user("hi")))
        .await
        .unwrap();
    let tip = branch.tip().unwrap();

    repo.set_tag("snapshot", tip);
    assert_eq!(repo.resolve_tag("snapshot"), Some(tip));
}

#[tokio::test]
async fn resolving_missing_tag_returns_none() {
    let repo = Repository::new();
    assert!(repo.resolve_tag("does-not-exist").is_none());
}

#[tokio::test]
async fn set_tag_overwrites_existing() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch.commit(TreePatch::new().add_message(user("first"))).await.unwrap();
    let h1 = branch.tip().unwrap();
    branch.commit(TreePatch::new().add_message(user("second"))).await.unwrap();
    let h2 = branch.tip().unwrap();

    repo.set_tag("head", h1);
    assert_eq!(repo.resolve_tag("head"), Some(h1));
    // Overwrite — no error, no force flag needed.
    repo.set_tag("head", h2);
    assert_eq!(repo.resolve_tag("head"), Some(h2));
}

#[tokio::test]
async fn remove_tag_returns_previous_value() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch.commit(TreePatch::new().add_message(user("hi"))).await.unwrap();
    let tip = branch.tip().unwrap();

    repo.set_tag("foo", tip);
    assert_eq!(repo.remove_tag("foo"), Some(tip));
    assert!(repo.resolve_tag("foo").is_none());
    assert!(repo.remove_tag("foo").is_none(), "second remove is None");
}

#[tokio::test]
async fn tags_enumerates_in_sorted_order() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch.commit(TreePatch::new().add_message(user("hi"))).await.unwrap();
    let tip = branch.tip().unwrap();

    repo.set_tag("zebra", tip);
    repo.set_tag("alpha", tip);
    repo.set_tag("mike", tip);

    let tags = repo.tags();
    let names: Vec<&str> = tags.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["alpha", "mike", "zebra"]);
}

#[tokio::test]
async fn tag_survives_branch_destruction() {
    // The tag is on the Repository, not on any particular Branch.
    // Dropping a branch shouldn't lose its tagged commit (as long
    // as the commit's nodes are still reachable through some other
    // reference — which they are, via the tag itself when we
    // implement GC).
    let repo = Repository::new();
    let tip = {
        let branch = repo.new_branch();
        branch.commit(TreePatch::new().add_message(user("ephemeral"))).await.unwrap();
        let tip = branch.tip().unwrap();
        repo.set_tag("kept", tip);
        tip
        // branch dropped here
    };

    // Tag still resolves; commit still readable.
    assert_eq!(repo.resolve_tag("kept"), Some(tip));
    let reopened = repo.branch_at_tag("kept").expect("tag resolves");
    assert_texts(reopened.as_ref(), &["ephemeral"]).await;
}

#[tokio::test]
async fn branch_at_tag_returns_none_for_missing() {
    let repo = Repository::new();
    assert!(repo.branch_at_tag("nothing").is_none());
}

#[tokio::test]
async fn tagging_unknown_hash_is_lazy() {
    // Tags don't validate the hash — matches branch_at's behavior.
    // Errors surface when the resulting branch is read from.
    let repo = Repository::new();
    let bogus = ObjectHash::from_bytes([0xcc; 32]);
    repo.set_tag("dangling", bogus);
    let branch = repo.branch_at_tag("dangling").unwrap();
    let err = branch
        .messages()
        .await
        .expect_err("dangling tag must error on first read");
    assert!(err.to_string().contains("not found"));
}

#[tokio::test]
async fn template_then_fork_spawn_pattern() {
    // The motivating use case for tags: define a template by tag,
    // then fork from it to start a new agent.
    let repo = Repository::new();

    // Construct the template: a commit with system prompt + tools.
    let template = repo.new_branch();
    template
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("You are a researcher.")])
                .with_tools(vec![tool("web_search"), tool("read_file")]),
        )
        .await
        .unwrap();
    repo.set_tag("templates/research-agent", template.tip().unwrap());

    // Spawn a new working agent from the template.
    let working = repo
        .branch_at_tag("templates/research-agent")
        .expect("template tag resolves")
        .fork();

    // The working branch inherits the prefix: same system prompt,
    // same tools, no messages yet.
    let sp = working.system_prompt().await.unwrap();
    assert_eq!(sp.as_deref().map(|c| c.len()), Some(1));
    let tools = working.tools().await.unwrap();
    let names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["read_file", "web_search"]);
    assert!(working.messages().await.unwrap().is_empty());

    // Working branch diverges; template stays put.
    working.commit(TreePatch::new().add_message(user("hi"))).await.unwrap();
    assert_eq!(template.messages().await.unwrap().len(), 0);
    assert_eq!(working.messages().await.unwrap().len(), 1);
}

#[tokio::test]
async fn template_storage_is_shared_across_forks() {
    // Two agents forked from the same template should share all
    // the template's storage. Construct a heavy template and verify
    // forks don't duplicate it.
    let repo = Repository::new();
    let template = repo.new_branch();
    template
        .commit(
            TreePatch::new()
                .with_system_prompt(vec![Content::text("be helpful")])
                .with_tools(
                    (0..20)
                        .map(|i| tool(&format!("tool_{i}")))
                        .collect::<Vec<_>>(),
                ),
        )
        .await
        .unwrap();
    repo.set_tag("templates/heavy", template.tip().unwrap());

    let objects_before_fork = repo.object_count();

    // Fork 5 working agents from the template.
    let _agents: Vec<_> = (0..5)
        .map(|_| repo.branch_at_tag("templates/heavy").unwrap().fork())
        .collect();

    // Forking is just tip-pointer cloning; no new objects.
    assert_eq!(repo.object_count(), objects_before_fork);
}

// ─── Bucketing ───────────────────────────────────────────────────────
//
// /messages is stored as bucketed subtrees (100 messages per bucket).
// These tests exercise the bucket boundary: ordering is preserved,
// counts match, dedup still works, the underlying tree shape is what
// the implementation claims.

const BUCKET_SIZE: usize = 100;

#[tokio::test]
async fn first_bucket_holds_up_to_bucket_size_messages() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    let msgs: Vec<Message> = (0..BUCKET_SIZE).map(|i| user(&format!("m{i}"))).collect();
    branch
        .commit(TreePatch::new().add_messages(msgs))
        .await
        .unwrap();

    // Inspect: /messages should have exactly one bucket subtree.
    let tip = branch.tip().unwrap();
    let commit = repo.get_commit(&tip).unwrap();
    let root = repo.get_tree(&commit.tree).unwrap();
    let messages_entry = root.get("messages").expect("/messages exists");
    let messages_hash = match messages_entry {
        tau_history::TreeEntry::Tree(h) => *h,
        _ => panic!("/messages should be a tree"),
    };
    let messages_tree = repo.get_tree(&messages_hash).unwrap();
    assert_eq!(messages_tree.entries.len(), 1, "exactly one bucket");

    // Verify read-back ordering.
    let read = branch.messages().await.unwrap();
    assert_eq!(read.len(), BUCKET_SIZE);
    for (i, msg) in read.iter().enumerate() {
        assert_eq!(msg.text(), format!("m{i}"));
    }
}

#[tokio::test]
async fn crossing_bucket_boundary_creates_new_bucket() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    let total = BUCKET_SIZE + 1;
    let msgs: Vec<Message> = (0..total).map(|i| user(&format!("m{i}"))).collect();
    branch
        .commit(TreePatch::new().add_messages(msgs))
        .await
        .unwrap();

    // All `total` messages are user messages; under the typed
    // layout they go into /messages/user/. Bucket spill happens
    // inside that type subtree.
    let tip = branch.tip().unwrap();
    let commit = repo.get_commit(&tip).unwrap();
    let root = repo.get_tree(&commit.tree).unwrap();
    let messages_hash = match root.get("messages").unwrap() {
        tau_history::TreeEntry::Tree(h) => *h,
        _ => panic!(),
    };
    let messages_tree = repo.get_tree(&messages_hash).unwrap();
    let user_hash = match messages_tree.entries.get("user").unwrap() {
        tau_history::TreeEntry::Tree(h) => *h,
        _ => panic!(),
    };
    let user_tree = repo.get_tree(&user_hash).unwrap();
    assert_eq!(user_tree.entries.len(), 2, "two buckets after spill");

    let read = branch.messages().await.unwrap();
    assert_eq!(read.len(), total);
    for (i, msg) in read.iter().enumerate() {
        assert_eq!(msg.text(), format!("m{i}"));
    }
}

#[tokio::test]
async fn appending_across_multiple_buckets_in_one_batch() {
    // A single commit that adds enough messages to span 3 buckets:
    // the bucket-grouping logic in append must write all three
    // affected buckets, not just the first.
    let repo = Repository::new();
    let branch = repo.new_branch();
    let total = BUCKET_SIZE * 2 + 25;
    let msgs: Vec<Message> = (0..total).map(|i| user(&format!("m{i}"))).collect();
    branch
        .commit(TreePatch::new().add_messages(msgs))
        .await
        .unwrap();

    let read = branch.messages().await.unwrap();
    assert_eq!(read.len(), total);
    for (i, msg) in read.iter().enumerate() {
        assert_eq!(msg.text(), format!("m{i}"));
    }
}

#[tokio::test]
async fn separate_commits_across_bucket_boundary_preserve_order() {
    // Commit one message at a time across a bucket boundary. Each
    // commit either appends to the current bucket or spills into
    // the next one. Order must be preserved across that transition.
    let repo = Repository::new();
    let branch = repo.new_branch();
    let total = BUCKET_SIZE + 10;
    for i in 0..total {
        branch
            .commit(TreePatch::new().add_message(user(&format!("m{i}"))))
            .await
            .unwrap();
    }
    let read = branch.messages().await.unwrap();
    assert_eq!(read.len(), total);
    for (i, msg) in read.iter().enumerate() {
        assert_eq!(msg.text(), format!("m{i}"));
    }
}

#[tokio::test]
async fn full_first_bucket_is_stable_when_appending_to_second() {
    // Once bucket 0 is full and we start adding to bucket 1,
    // bucket 0's hash should not change across subsequent commits
    // — that's where the storage savings come from. Verify by
    // grabbing bucket 0's hash before vs after.
    let repo = Repository::new();
    let branch = repo.new_branch();

    let msgs: Vec<Message> = (0..BUCKET_SIZE).map(|i| user(&format!("m{i}"))).collect();
    branch
        .commit(TreePatch::new().add_messages(msgs))
        .await
        .unwrap();

    // Drill into /messages/user/0000000 — the bucket holding the
    // first 100 user messages.
    let read_user_bucket0 = |branch: &Arc<Branch>| -> ObjectHash {
        let tip = branch.tip().unwrap();
        let commit = repo.get_commit(&tip).unwrap();
        let root = repo.get_tree(&commit.tree).unwrap();
        let messages_hash = match root.get("messages").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        };
        let messages_tree = repo.get_tree(&messages_hash).unwrap();
        let user_hash = match messages_tree.entries.get("user").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        };
        let user_tree = repo.get_tree(&user_hash).unwrap();
        match user_tree.entries.get("0000000").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        }
    };
    let bucket0_hash_before = read_user_bucket0(&branch);

    // Add 5 more messages — these land in bucket 1.
    branch
        .commit(TreePatch::new().add_messages(
            (0..5).map(|i| user(&format!("extra{i}"))).collect::<Vec<_>>(),
        ))
        .await
        .unwrap();

    let bucket0_hash_after = read_user_bucket0(&branch);

    assert_eq!(
        bucket0_hash_before, bucket0_hash_after,
        "full bucket 0's hash must be stable when bucket 1 grows"
    );
}

#[tokio::test]
async fn compaction_across_bucket_boundary() {
    // Compaction replaces /messages entirely; the rebuild logic must
    // handle bucket layout correctly. Test: have 150 messages, keep
    // the last 50 — the result spans bucket 0 (51 entries, including
    // the summary) and nothing in bucket 1, OR — depending on
    // bucketing — one full bucket and a partial. Either way the
    // counts and order should be right.
    let repo = Repository::new();
    let branch = repo.new_branch();
    let msgs: Vec<Message> = (0..150).map(|i| user(&format!("orig{i}"))).collect();
    branch
        .commit(TreePatch::new().add_messages(msgs))
        .await
        .unwrap();

    (branch.as_ref() as &dyn History)
        .compact_prefix(100, user("summary"), "summary".into())
        .await
        .unwrap();

    let read = branch.messages().await.unwrap();
    assert_eq!(read.len(), 1 + 50, "summary + kept tail");
    assert_eq!(read[0].text(), "summary");
    for (i, msg) in read.iter().skip(1).enumerate() {
        assert_eq!(msg.text(), format!("orig{}", 100 + i));
    }
}

// ─── Concurrency / Send + Sync ───────────────────────────────────────

#[tokio::test]
async fn branches_can_be_committed_to_concurrently() {
    let repo = Repository::new();
    let a = repo.new_branch();
    a.commit(TreePatch::new().add_message(user("base")))
        .await
        .unwrap();
    let b = a.fork();

    let a2: Arc<Branch> = Arc::clone(&a);
    let b2: Arc<Branch> = Arc::clone(&b);
    let t1 = tokio::spawn(async move {
        for i in 0..50 {
            a2.commit(TreePatch::new().add_message(user(&format!("a-{i}"))))
                .await
                .unwrap();
        }
    });
    let t2 = tokio::spawn(async move {
        for i in 0..50 {
            b2.commit(TreePatch::new().add_message(user(&format!("b-{i}"))))
                .await
                .unwrap();
        }
    });
    t1.await.unwrap();
    t2.await.unwrap();

    let a_msgs = a.messages().await.unwrap();
    let b_msgs = b.messages().await.unwrap();
    assert_eq!(a_msgs.len(), 51);
    assert_eq!(b_msgs.len(), 51);
    assert_eq!(a_msgs.last().unwrap().text(), "a-49");
    assert_eq!(b_msgs.last().unwrap().text(), "b-49");
}

// ─── Type-split layout ───────────────────────────────────────────────
//
// Verify the structural properties of the typed /messages tree:
// different message types land in different subtrees; cross-type
// ordering at read time is by commit-sequence; tool result batches
// preserve request order via batch_pos.

fn assistant(text: &str) -> Message {
    use tau_ai::{AssistantMetadata, Content};
    Message::Assistant {
        content: vec![Content::text(text)],
        metadata: AssistantMetadata::default(),
    }
}

fn tool_result(call_id: &str, text: &str) -> Message {
    use tau_ai::Content;
    Message::ToolResult {
        tool_call_id: call_id.into(),
        tool_name: "x".into(),
        content: vec![Content::text(text)],
        is_error: false,
        timestamp: 0,
    }
}

#[tokio::test]
async fn each_message_type_lands_in_its_own_subtree() {
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch
        .commit(TreePatch::new().add_messages(vec![
            user("u1"),
            assistant("a1"),
            tool_result("c1", "r1"),
        ]))
        .await
        .unwrap();

    let tip = branch.tip().unwrap();
    let commit = repo.get_commit(&tip).unwrap();
    let root = repo.get_tree(&commit.tree).unwrap();
    let messages_hash = match root.get("messages").unwrap() {
        tau_history::TreeEntry::Tree(h) => *h,
        _ => panic!(),
    };
    let messages_tree = repo.get_tree(&messages_hash).unwrap();

    // Three distinct type subtrees populated. system_injection
    // wasn't touched, so it's absent.
    let names: Vec<&str> = messages_tree.entries.keys().map(|s| s.as_str()).collect();
    assert!(names.contains(&"user"), "user subtree present");
    assert!(names.contains(&"assistant"), "assistant subtree present");
    assert!(names.contains(&"tool_results"), "tool_results subtree present");
    assert!(
        !names.contains(&"system_injection"),
        "system_injection subtree not created when no SystemInjection messages"
    );
}

#[tokio::test]
async fn cross_type_ordering_at_read_time_is_by_commit_sequence() {
    // Multi-type commits in the order:
    //   commit 1: user
    //   commit 2: assistant
    //   commit 3: tool_results (3 of them)
    //   commit 4: assistant
    // Read should yield messages in the order they were committed,
    // not grouped by type.
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch.commit(TreePatch::new().add_message(user("u1"))).await.unwrap();
    branch.commit(TreePatch::new().add_message(assistant("a1"))).await.unwrap();
    branch
        .commit(TreePatch::new().add_messages(vec![
            tool_result("c1", "r1"),
            tool_result("c2", "r2"),
            tool_result("c3", "r3"),
        ]))
        .await
        .unwrap();
    branch.commit(TreePatch::new().add_message(assistant("a2"))).await.unwrap();

    let msgs = branch.messages().await.unwrap();
    let texts: Vec<String> = msgs.iter().map(|m| m.text()).collect();
    assert_eq!(texts, vec!["u1", "a1", "r1", "r2", "r3", "a2"]);
}

#[tokio::test]
async fn parallel_tool_results_preserve_batch_pos_order() {
    // The batch_pos suffix preserves the request order of tool
    // results within a single commit, which the API requires.
    let repo = Repository::new();
    let branch = repo.new_branch();
    branch.commit(TreePatch::new().add_message(assistant("calling tools"))).await.unwrap();

    // Submit 5 tool_results in a deliberate order — the messages
    // method must return them in this exact order regardless of
    // any internal grouping the implementation does.
    let batch = vec![
        tool_result("c_a", "result_for_a"),
        tool_result("c_b", "result_for_b"),
        tool_result("c_c", "result_for_c"),
        tool_result("c_d", "result_for_d"),
        tool_result("c_e", "result_for_e"),
    ];
    branch.commit(TreePatch::new().add_messages(batch)).await.unwrap();

    let msgs = branch.messages().await.unwrap();
    let texts: Vec<String> = msgs.iter().map(|m| m.text()).collect();
    assert_eq!(
        texts,
        vec![
            "calling tools",
            "result_for_a",
            "result_for_b",
            "result_for_c",
            "result_for_d",
            "result_for_e",
        ]
    );
}

#[tokio::test]
async fn branch_at_recovers_next_seq_from_tree() {
    // If we reopen a branch via branch_at, the next commit's seq
    // must continue from where the previous one left off — not
    // restart at 1 (which would cause name collisions and out-of-
    // order reads).
    let repo = Repository::new();
    let branch = repo.new_branch();
    for i in 0..10 {
        branch
            .commit(TreePatch::new().add_message(user(&format!("m{i}"))))
            .await
            .unwrap();
    }
    let tip = branch.tip().unwrap();
    drop(branch);

    let resumed = repo.branch_at(tip);
    // Append a new message; it should appear AFTER all previous ones.
    resumed
        .commit(TreePatch::new().add_message(user("m10")))
        .await
        .unwrap();

    let msgs = resumed.messages().await.unwrap();
    let texts: Vec<String> = msgs.iter().map(|m| m.text()).collect();
    assert_eq!(
        texts,
        (0..=10).map(|i| format!("m{i}")).collect::<Vec<_>>(),
        "resumed branch's new commit lands after the existing tail"
    );
}

#[tokio::test]
async fn type_split_dedup_at_subtree_level() {
    // Two branches that have identical user-message subtrees (same
    // sequence of user messages) but different assistant subtrees
    // should share the user subtree's hash via content addressing.
    let repo = Repository::new();
    let a = repo.new_branch();
    a.commit(TreePatch::new().add_messages(vec![user("u1"), user("u2")])).await.unwrap();
    let b = a.fork();

    // Both branches add a different assistant message.
    a.commit(TreePatch::new().add_message(assistant("a-only"))).await.unwrap();
    b.commit(TreePatch::new().add_message(assistant("b-only"))).await.unwrap();

    // Both branches have the same /messages/user subtree.
    let a_user_hash = {
        let tip = a.tip().unwrap();
        let commit = repo.get_commit(&tip).unwrap();
        let root = repo.get_tree(&commit.tree).unwrap();
        let messages_hash = match root.get("messages").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        };
        let messages_tree = repo.get_tree(&messages_hash).unwrap();
        match messages_tree.entries.get("user").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        }
    };
    let b_user_hash = {
        let tip = b.tip().unwrap();
        let commit = repo.get_commit(&tip).unwrap();
        let root = repo.get_tree(&commit.tree).unwrap();
        let messages_hash = match root.get("messages").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        };
        let messages_tree = repo.get_tree(&messages_hash).unwrap();
        match messages_tree.entries.get("user").unwrap() {
            tau_history::TreeEntry::Tree(h) => *h,
            _ => panic!(),
        }
    };
    assert_eq!(
        a_user_hash, b_user_hash,
        "two branches with identical user-message subtrees share the same hash"
    );
}
