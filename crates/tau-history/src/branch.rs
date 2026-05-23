//! [`Branch`] — a named ref into a [`Repository`], plus [`TreePatch`]
//! for constructing commits.
//!
//! `Branch` exposes the git-flavored API (`commit`, `merge`, `fork`,
//! `tip`) and implements [`History`] for the agent runtime's
//! conversation-flavored view (`messages`, `system_prompt`, `tools`,
//! `previous_summary`, `append`, `compact_prefix`).
//!
//! # Tree layout
//!
//! Every commit's root tree has this canonical structure. Entries
//! are inherited from the parent's tree when unchanged — their
//! hashes match, so the tree object dedupes structurally. Adding a
//! message changes `messages/` only; the other entries' hashes are
//! unchanged. Changing tools changes `tools/` only.
//!
//! `/messages` is **type-split + bucketed**: one subtree per Message
//! variant, each subtree bucketed for storage efficiency.
//!
//! ```text
//! /messages/
//!   user/                       ← user-role messages
//!     0000000/                  ← bucket (100 entries per bucket)
//!       0000000001-0000         ← Blob: <seq>-<batch_pos>
//!       0000000003-0000
//!       ...
//!   assistant/                  ← assistant turns
//!     0000000/
//!       0000000002-0000
//!       ...
//!   tool_results/               ← tool result messages
//!     0000000/
//!       0000000004-0000         ← parallel batch, request-order 0
//!       0000000004-0001         ← parallel batch, request-order 1
//!       0000000004-0002
//!       ...
//!   system_injection/           ← SystemInjection messages (compaction summaries etc.)
//!     0000000/
//!       0000000005-0000
//! ```
//!
//! ## Naming scheme
//!
//! Entry names are `<seq>-<batch_pos>` where:
//!
//! - `seq` (10-digit zero-padded) is the commit's sequence number,
//!   shared across every message in that commit. Each commit
//!   increments the branch's [`BranchState::next_seq`] counter by 1.
//!   This is the primary ordering key.
//!
//! - `batch_pos` (4-digit zero-padded) is the message's position
//!   within the commit batch. For commits adding a single message,
//!   it's `0`. For parallel tool-result batches it ranges over
//!   `0..N-1` in **request order** (i.e., the order the
//!   corresponding tool_uses appeared in the assistant message,
//!   which is what the API requires).
//!
//! Lexicographic ordering of names = numeric ordering of
//! `(seq, batch_pos)` pairs = conversation order, regardless of
//! which type subtree the entry lives in.
//!
//! ## Why type-split
//!
//! Each message variant lives in its own subtree, which:
//!
//! - Makes the type of a commit's change visible in the tree-diff
//!   (a tool-results commit touches `/messages/tool_results/...`;
//!   an assistant commit touches `/messages/assistant/...`).
//! - Enables direct type-scoped queries ("show me every tool
//!   invocation on this branch" = walk `/messages/tool_results/`).
//! - Lets cache and content-addressing dedup share at the type
//!   level — two branches whose tool-result subtrees are
//!   byte-identical share storage even if their assistant turns
//!   diverged.
//!
//! Reading the conversation in order is a 4-way merge across the
//! type subtrees, sorted by entry name. That's O(N) total (same
//! asymptotic as the prior flat scheme), with a modest
//! constant-factor cost from the sort.
//!
//! ## Why bucketing within types
//!
//! Same reason as before: bounds per-commit cost. Each type's
//! subtree organizes entries into buckets of [`BUCKET_SIZE`] (=
//! 100) entries. Bucket names are zero-padded 7 digits (supports up
//! to 10⁹ entries per type). When a bucket fills, new entries land
//! in the next bucket — the full buckets' hashes stay stable, so
//! the storage cost amortizes.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tau_ai::{Content, Message};

use crate::history::{History, HistoryError};
use crate::objects::{
    Blob, Commit, ObjectHash, ObjectKind, StoreError, ToolDef, Tree, TreeEntry,
};
use crate::repository::Repository;

/// Maximum entries per bucket subtree.
pub const BUCKET_SIZE: usize = 100;
/// Width of zero-padded bucket names.
pub const BUCKET_NAME_WIDTH: usize = 7;
/// Width of zero-padded commit-sequence numbers in entry names.
pub const SEQ_WIDTH: usize = 10;
/// Width of zero-padded batch-position numbers in entry names.
pub const BATCH_POS_WIDTH: usize = 4;

// ─── Type subtree names ──────────────────────────────────────────────

const TYPE_USER: &str = "user";
const TYPE_ASSISTANT: &str = "assistant";
const TYPE_TOOL_RESULTS: &str = "tool_results";
const TYPE_SYSTEM_INJECTION: &str = "system_injection";

/// All four message-type subtree names, in a fixed iteration order
/// for full-walk scans.
const ALL_TYPES: [&str; 4] = [
    TYPE_USER,
    TYPE_ASSISTANT,
    TYPE_TOOL_RESULTS,
    TYPE_SYSTEM_INJECTION,
];

fn type_subtree_name(msg: &Message) -> &'static str {
    match msg {
        Message::User { .. } => TYPE_USER,
        Message::Assistant { .. } => TYPE_ASSISTANT,
        Message::ToolResult { .. } => TYPE_TOOL_RESULTS,
        Message::SystemInjection { .. } => TYPE_SYSTEM_INJECTION,
    }
}

fn bucket_name(idx: usize) -> String {
    format!("{:0width$}", idx, width = BUCKET_NAME_WIDTH)
}

fn entry_name(seq: u64, batch_pos: u64) -> String {
    format!(
        "{:0sw$}-{:0bw$}",
        seq,
        batch_pos,
        sw = SEQ_WIDTH,
        bw = BATCH_POS_WIDTH
    )
}

fn parse_entry_key(name: &str) -> Result<(u64, u64), HistoryError> {
    let (seq_s, pos_s) = name
        .split_once('-')
        .ok_or_else(|| HistoryError::msg(format!("malformed message entry name: {name}")))?;
    let seq: u64 = seq_s
        .parse()
        .map_err(|_| HistoryError::msg(format!("malformed seq in entry name: {name}")))?;
    let batch_pos: u64 = pos_s
        .parse()
        .map_err(|_| HistoryError::msg(format!("malformed batch_pos in entry name: {name}")))?;
    Ok((seq, batch_pos))
}

// ─── MessagesOp ──────────────────────────────────────────────────────

/// How a [`TreePatch`] modifies the message log on commit.
///
/// `Unchanged` means inherit the parent's `/messages` subtree
/// untouched; `Append` adds messages at the end of the existing
/// log; `Replace` discards the existing log and stores the given
/// messages as the new log (used by compaction).
#[derive(Debug, Clone, Default)]
pub enum MessagesOp {
    /// Don't touch the message log.
    #[default]
    Unchanged,
    /// Append these messages to the existing log.
    Append(Vec<Message>),
    /// Replace the existing log entirely with these messages.
    Replace(Vec<Message>),
}

// ─── TreePatch ───────────────────────────────────────────────────────

/// A builder describing how the next commit's tree differs from the
/// parent's. Each field is optional — unset fields inherit from
/// the parent.
///
/// Pass to [`Branch::commit`] or [`Branch::merge`].
#[derive(Debug, Clone, Default)]
pub struct TreePatch {
    pub(crate) system_prompt: Option<Vec<Content>>,
    pub(crate) tools: Option<Vec<ToolDef>>,
    pub(crate) messages: MessagesOp,
    pub(crate) previous_summary: Option<Option<String>>,
}

impl TreePatch {
    /// Create an empty patch — every field unset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (or replace) the system prompt at this commit.
    pub fn with_system_prompt(mut self, content: Vec<Content>) -> Self {
        self.system_prompt = Some(content);
        self
    }

    /// Set (or replace) the tool surface at this commit.
    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Append a single message to the log.
    pub fn add_message(mut self, message: Message) -> Self {
        match &mut self.messages {
            MessagesOp::Unchanged => {
                self.messages = MessagesOp::Append(vec![message]);
            }
            MessagesOp::Append(v) => v.push(message),
            MessagesOp::Replace(v) => v.push(message),
        }
        self
    }

    /// Append multiple messages to the log.
    pub fn add_messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        let iter = messages.into_iter();
        match &mut self.messages {
            MessagesOp::Unchanged => {
                self.messages = MessagesOp::Append(iter.collect());
            }
            MessagesOp::Append(v) => v.extend(iter),
            MessagesOp::Replace(v) => v.extend(iter),
        }
        self
    }

    /// Replace the existing message log with these messages.
    /// Subsequent `add_message`/`add_messages` calls append to the
    /// replacement list.
    pub fn replace_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = MessagesOp::Replace(messages);
        self
    }

    /// Set the `previous_summary` slot. `Some(text)` writes the
    /// summary; `None` clears it. Distinct from "unset" (the
    /// default, which inherits the parent's value).
    pub fn with_previous_summary(mut self, summary: Option<String>) -> Self {
        self.previous_summary = Some(summary);
        self
    }

    /// True when this patch makes no changes.
    pub fn is_empty(&self) -> bool {
        self.system_prompt.is_none()
            && self.tools.is_none()
            && matches!(&self.messages, MessagesOp::Unchanged)
            && self.previous_summary.is_none()
    }
}

// ─── BranchState ─────────────────────────────────────────────────────

#[derive(Debug)]
struct BranchState {
    tip: Option<ObjectHash>,
    next_seq: u64,
}

// ─── Branch ──────────────────────────────────────────────────────────

/// A named ref into a [`Repository`] — git's branch analog.
///
/// Each branch carries a mutable tip pointer and an in-memory
/// sequence counter for naming new message entries. Commits update
/// both atomically under one mutex.
pub struct Branch {
    repo: Arc<Repository>,
    state: Mutex<BranchState>,
}

impl Branch {
    /// `git checkout --orphan` — a brand-new branch with no commits.
    pub(crate) fn empty(repo: Arc<Repository>) -> Arc<Self> {
        Arc::new(Self {
            repo,
            state: Mutex::new(BranchState {
                tip: None,
                next_seq: 1,
            }),
        })
    }

    /// `git checkout <commit>` — open a branch positioned at the
    /// given tip. Derives `next_seq` from the tree by inspecting
    /// the last entry across all type subtrees.
    pub(crate) fn at(repo: Arc<Repository>, tip: ObjectHash) -> Arc<Self> {
        let next_seq = derive_next_seq(&repo, &tip).unwrap_or(1);
        Arc::new(Self {
            repo,
            state: Mutex::new(BranchState {
                tip: Some(tip),
                next_seq,
            }),
        })
    }

    /// `git rev-parse HEAD` — the current tip commit, or `None` if
    /// the branch is empty.
    pub fn tip(&self) -> Option<ObjectHash> {
        self.state.lock().tip
    }

    /// `git commit` — apply `patch` and record a new commit on top
    /// of the current tip.
    pub async fn commit(&self, patch: TreePatch) -> Result<(), HistoryError> {
        self.commit_inner(patch, Vec::new())
    }

    /// `git merge --no-ff` — like `commit`, but also record one or
    /// more `extra_parents` on the new commit. Used to graft a
    /// subagent's tip into the parent branch's lineage when
    /// collecting its result.
    pub async fn merge(
        &self,
        patch: TreePatch,
        extra_parents: Vec<ObjectHash>,
    ) -> Result<(), HistoryError> {
        self.commit_inner(patch, extra_parents)
    }

    /// `git checkout -b <new-branch>` — open a new branch sharing
    /// the same tip. Cheap: the new branch is a tip-pointer clone.
    pub fn fork(self: &Arc<Self>) -> Arc<Self> {
        let s = self.state.lock();
        Arc::new(Self {
            repo: Arc::clone(&self.repo),
            state: Mutex::new(BranchState {
                tip: s.tip,
                next_seq: s.next_seq,
            }),
        })
    }

    fn commit_inner(
        &self,
        patch: TreePatch,
        extra_parents: Vec<ObjectHash>,
    ) -> Result<(), HistoryError> {
        if patch.is_empty() && extra_parents.is_empty() {
            return Err(HistoryError::msg(
                "empty patch with no extra_parents — commit must change something",
            ));
        }

        let mut state = self.state.lock();
        let parent_tip = state.tip;
        let commit_seq = state.next_seq;

        // Read parent root tree (if any).
        let parent_root: Option<Arc<Tree>> = match parent_tip {
            Some(h) => Some(get_commit(&self.repo, &h)?.tree).and_then(|tree_hash| {
                Some(get_tree(&self.repo, &tree_hash))
            }).transpose()?,
            None => None,
        };

        let new_root_hash = apply_patch(&self.repo, parent_root.as_deref(), &patch, commit_seq)?;

        let commit = Commit {
            parent: parent_tip,
            extra_parents,
            tree: new_root_hash,
        };
        let commit_hash = self.repo.put_commit(commit);

        state.tip = Some(commit_hash);
        state.next_seq = commit_seq + 1;
        Ok(())
    }
}

// ─── apply_patch: build new root tree from parent tree + patch ───────

fn apply_patch(
    repo: &Repository,
    parent_root: Option<&Tree>,
    patch: &TreePatch,
    commit_seq: u64,
) -> Result<ObjectHash, HistoryError> {
    // Start from parent's root entries (so unchanged subtrees
    // inherit by hash).
    let mut root = match parent_root {
        Some(t) => t.clone(),
        None => Tree::new(),
    };

    // system_prompt
    if let Some(sp) = &patch.system_prompt {
        let bytes = serde_json::to_vec(sp).map_err(HistoryError::from)?;
        let hash = repo.put_blob(Blob::new(bytes));
        root.insert("system_prompt", TreeEntry::Blob(hash));
    }

    // tools — a tree keyed by tool name, each entry a Blob of the
    // serialized ToolDef.
    if let Some(tools) = &patch.tools {
        let mut tools_tree = Tree::new();
        for t in tools {
            let bytes = serde_json::to_vec(t).map_err(HistoryError::from)?;
            let blob_hash = repo.put_blob(Blob::new(bytes));
            tools_tree.insert(t.name.clone(), TreeEntry::Blob(blob_hash));
        }
        let tools_hash = repo.put_tree(tools_tree);
        root.insert("tools", TreeEntry::Tree(tools_hash));
    }

    // previous_summary
    if let Some(opt) = &patch.previous_summary {
        match opt {
            Some(text) => {
                let blob_hash = repo.put_blob(Blob::new(text.as_bytes().to_vec()));
                root.insert("previous_summary", TreeEntry::Blob(blob_hash));
            }
            None => {
                root.remove("previous_summary");
            }
        }
    }

    // messages
    match &patch.messages {
        MessagesOp::Unchanged => {
            // Inherit (already in root via clone).
        }
        MessagesOp::Append(new_msgs) => {
            if !new_msgs.is_empty() {
                let existing = parent_messages_tree(repo, parent_root)?;
                let new_messages_hash =
                    write_messages_tree(repo, existing.as_deref(), new_msgs, commit_seq)?;
                root.insert("messages", TreeEntry::Tree(new_messages_hash));
            }
        }
        MessagesOp::Replace(new_msgs) => {
            // Build from scratch (no existing).
            let new_messages_hash = write_messages_tree(repo, None, new_msgs, commit_seq)?;
            root.insert("messages", TreeEntry::Tree(new_messages_hash));
        }
    }

    Ok(repo.put_tree(root))
}

/// Fetch the parent root's `/messages` subtree if it exists.
fn parent_messages_tree(
    repo: &Repository,
    parent_root: Option<&Tree>,
) -> Result<Option<Arc<Tree>>, HistoryError> {
    let Some(root) = parent_root else {
        return Ok(None);
    };
    let Some(entry) = root.get("messages") else {
        return Ok(None);
    };
    match entry {
        TreeEntry::Tree(h) => Ok(Some(get_tree(repo, h)?)),
        TreeEntry::Blob(_) => Err(HistoryError::from(StoreError::TypeMismatch {
            path: "messages".into(),
            expected: ObjectKind::Tree,
            actual: ObjectKind::Blob,
        })),
    }
}

/// Write a `/messages` tree by either appending to an existing
/// per-type tree (when `existing` is `Some`) or building from
/// scratch.
fn write_messages_tree(
    repo: &Repository,
    existing: Option<&Tree>,
    new_msgs: &[Message],
    commit_seq: u64,
) -> Result<ObjectHash, HistoryError> {
    // Group messages by their type subtree, preserving (batch_pos,
    // Message) pairs.
    let mut grouped: BTreeMap<&'static str, Vec<(usize, &Message)>> = BTreeMap::new();
    for (i, m) in new_msgs.iter().enumerate() {
        grouped.entry(type_subtree_name(m)).or_default().push((i, m));
    }

    // Start from the existing /messages root tree (so untouched
    // type subtrees inherit) or a fresh one.
    let mut messages_tree = match existing {
        Some(t) => t.clone(),
        None => Tree::new(),
    };

    for (type_name, typed_msgs) in &grouped {
        // Existing per-type subtree, if any.
        let existing_type_tree: Option<Arc<Tree>> = match messages_tree.get(*type_name) {
            Some(TreeEntry::Tree(h)) => Some(get_tree(repo, h)?),
            Some(TreeEntry::Blob(_)) => {
                return Err(HistoryError::from(StoreError::TypeMismatch {
                    path: format!("messages/{type_name}"),
                    expected: ObjectKind::Tree,
                    actual: ObjectKind::Blob,
                }))
            }
            None => None,
        };

        let new_type_tree_hash =
            write_typed_messages(repo, existing_type_tree.as_deref(), typed_msgs, commit_seq)?;
        messages_tree.insert(type_name.to_string(), TreeEntry::Tree(new_type_tree_hash));
    }

    Ok(repo.put_tree(messages_tree))
}

/// Write a per-type tree by appending messages into the appropriate
/// buckets. Each `(batch_pos, message)` becomes an entry named
/// `entry_name(commit_seq, batch_pos)`.
fn write_typed_messages(
    repo: &Repository,
    existing: Option<&Tree>,
    messages: &[(usize, &Message)],
    commit_seq: u64,
) -> Result<ObjectHash, HistoryError> {
    // How many entries already live in this type subtree?
    let start_count = match existing {
        Some(t) => count_in_type_tree(repo, t)?,
        None => 0,
    };

    // Group new entries by their target bucket index.
    let mut by_bucket: BTreeMap<usize, Vec<(u64, &Message)>> = BTreeMap::new();
    for (offset, (batch_pos, msg)) in messages.iter().enumerate() {
        let absolute = start_count + offset;
        let bucket_idx = absolute / BUCKET_SIZE;
        by_bucket
            .entry(bucket_idx)
            .or_default()
            .push((*batch_pos as u64, *msg));
    }

    // Start from the existing per-type tree (inherits full buckets
    // by hash) or empty.
    let mut type_tree = match existing {
        Some(t) => t.clone(),
        None => Tree::new(),
    };

    for (bucket_idx, entries) in &by_bucket {
        let name = bucket_name(*bucket_idx);

        // Existing bucket subtree (when filling an in-progress bucket).
        let mut bucket = match type_tree.get(&name) {
            Some(TreeEntry::Tree(h)) => (*get_tree(repo, h)?).clone(),
            Some(TreeEntry::Blob(_)) => {
                return Err(HistoryError::from(StoreError::TypeMismatch {
                    path: format!("messages/<type>/{name}"),
                    expected: ObjectKind::Tree,
                    actual: ObjectKind::Blob,
                }))
            }
            None => Tree::new(),
        };

        for (batch_pos, msg) in entries {
            let bytes = serde_json::to_vec(msg).map_err(HistoryError::from)?;
            let blob_hash = repo.put_blob(Blob::new(bytes));
            let entry = entry_name(commit_seq, *batch_pos);
            bucket.insert(entry, TreeEntry::Blob(blob_hash));
        }

        let bucket_hash = repo.put_tree(bucket);
        type_tree.insert(name, TreeEntry::Tree(bucket_hash));
    }

    Ok(repo.put_tree(type_tree))
}

/// O(1) count of entries in a type subtree. Every bucket except
/// the last has exactly `BUCKET_SIZE` entries, so we only need to
/// load the last bucket to learn the total.
fn count_in_type_tree(repo: &Repository, type_tree: &Tree) -> Result<usize, HistoryError> {
    let n_buckets = type_tree.entries.len();
    if n_buckets == 0 {
        return Ok(0);
    }
    // Last bucket (BTreeMap iteration is ordered, so `.last_key_value()`
    // gives the largest bucket name).
    let (_last_name, last_entry) = type_tree.entries.iter().next_back().unwrap();
    let last_bucket_hash = match last_entry {
        TreeEntry::Tree(h) => h,
        TreeEntry::Blob(_) => {
            return Err(HistoryError::from(StoreError::TypeMismatch {
                path: "messages/<type>/<bucket>".into(),
                expected: ObjectKind::Tree,
                actual: ObjectKind::Blob,
            }))
        }
    };
    let last_bucket = get_tree(repo, last_bucket_hash)?;
    Ok((n_buckets - 1) * BUCKET_SIZE + last_bucket.entries.len())
}

/// Walk the (potentially) last bucket of each type subtree to find
/// the maximum `seq` ever stored, returning that + 1. Used by
/// [`Branch::at`] to resume a branch's seq counter.
fn derive_next_seq(repo: &Repository, tip: &ObjectHash) -> Result<u64, HistoryError> {
    let commit = get_commit(repo, tip)?;
    let root = get_tree(repo, &commit.tree)?;
    let messages_tree = match root.get("messages") {
        Some(TreeEntry::Tree(h)) => get_tree(repo, h)?,
        _ => return Ok(1),
    };

    let mut max_seq: u64 = 0;
    for type_name in ALL_TYPES.iter() {
        let type_tree = match messages_tree.get(*type_name) {
            Some(TreeEntry::Tree(h)) => get_tree(repo, h)?,
            _ => continue,
        };
        // Last bucket.
        let Some((_last_name, last_entry)) = type_tree.entries.iter().next_back() else {
            continue;
        };
        let last_bucket_hash = match last_entry {
            TreeEntry::Tree(h) => h,
            _ => continue,
        };
        let last_bucket = get_tree(repo, last_bucket_hash)?;
        // Last entry name.
        let Some((entry_name_str, _)) = last_bucket.entries.iter().next_back() else {
            continue;
        };
        let (seq, _pos) = parse_entry_key(entry_name_str)?;
        if seq > max_seq {
            max_seq = seq;
        }
    }

    Ok(max_seq + 1)
}

// ─── Read helpers ────────────────────────────────────────────────────

fn get_commit(repo: &Repository, hash: &ObjectHash) -> Result<Arc<Commit>, HistoryError> {
    repo.get_commit(hash).ok_or_else(|| {
        HistoryError::from(StoreError::NotFound {
            hash: *hash,
            kind: ObjectKind::Commit,
        })
    })
}

fn get_tree(repo: &Repository, hash: &ObjectHash) -> Result<Arc<Tree>, HistoryError> {
    repo.get_tree(hash).ok_or_else(|| {
        HistoryError::from(StoreError::NotFound {
            hash: *hash,
            kind: ObjectKind::Tree,
        })
    })
}

fn get_blob(repo: &Repository, hash: &ObjectHash) -> Result<Arc<Blob>, HistoryError> {
    repo.get_blob(hash).ok_or_else(|| {
        HistoryError::from(StoreError::NotFound {
            hash: *hash,
            kind: ObjectKind::Blob,
        })
    })
}

/// Read the tip's root tree. Returns `None` if the branch is empty.
fn read_root_tree(branch: &Branch) -> Result<Option<Arc<Tree>>, HistoryError> {
    let tip = branch.state.lock().tip;
    match tip {
        Some(h) => {
            let commit = get_commit(&branch.repo, &h)?;
            let tree = get_tree(&branch.repo, &commit.tree)?;
            Ok(Some(tree))
        }
        None => Ok(None),
    }
}

// ─── History impl ────────────────────────────────────────────────────

#[async_trait]
impl History for Branch {
    async fn messages(&self) -> Result<Vec<Message>, HistoryError> {
        let Some(root) = read_root_tree(self)? else {
            return Ok(Vec::new());
        };
        let messages_tree = match root.get("messages") {
            Some(TreeEntry::Tree(h)) => get_tree(&self.repo, h)?,
            _ => return Ok(Vec::new()),
        };

        let mut collected: Vec<((u64, u64), Message)> = Vec::new();
        for type_name in ALL_TYPES.iter() {
            let type_tree = match messages_tree.get(*type_name) {
                Some(TreeEntry::Tree(h)) => get_tree(&self.repo, h)?,
                _ => continue,
            };
            for (_bucket_name, bucket_entry) in &type_tree.entries {
                let bucket_hash = match bucket_entry {
                    TreeEntry::Tree(h) => h,
                    _ => {
                        return Err(HistoryError::from(StoreError::TypeMismatch {
                            path: format!("messages/{type_name}/<bucket>"),
                            expected: ObjectKind::Tree,
                            actual: ObjectKind::Blob,
                        }))
                    }
                };
                let bucket = get_tree(&self.repo, bucket_hash)?;
                for (entry_name_str, blob_entry) in &bucket.entries {
                    let blob_hash = match blob_entry {
                        TreeEntry::Blob(h) => h,
                        _ => {
                            return Err(HistoryError::from(StoreError::TypeMismatch {
                                path: format!("messages/{type_name}/<bucket>/{entry_name_str}"),
                                expected: ObjectKind::Blob,
                                actual: ObjectKind::Tree,
                            }))
                        }
                    };
                    let blob = get_blob(&self.repo, blob_hash)?;
                    let msg: Message =
                        serde_json::from_slice(&blob.bytes).map_err(HistoryError::from)?;
                    let key = parse_entry_key(entry_name_str)?;
                    collected.push((key, msg));
                }
            }
        }

        collected.sort_unstable_by_key(|(k, _)| *k);
        Ok(collected.into_iter().map(|(_, m)| m).collect())
    }

    async fn system_prompt(&self) -> Result<Option<Vec<Content>>, HistoryError> {
        let Some(root) = read_root_tree(self)? else {
            return Ok(None);
        };
        let entry = match root.get("system_prompt") {
            Some(TreeEntry::Blob(h)) => h,
            Some(TreeEntry::Tree(_)) => {
                return Err(HistoryError::from(StoreError::TypeMismatch {
                    path: "system_prompt".into(),
                    expected: ObjectKind::Blob,
                    actual: ObjectKind::Tree,
                }))
            }
            None => return Ok(None),
        };
        let blob = get_blob(&self.repo, entry)?;
        let content: Vec<Content> =
            serde_json::from_slice(&blob.bytes).map_err(HistoryError::from)?;
        Ok(Some(content))
    }

    async fn tools(&self) -> Result<Vec<ToolDef>, HistoryError> {
        let Some(root) = read_root_tree(self)? else {
            return Ok(Vec::new());
        };
        let tools_tree = match root.get("tools") {
            Some(TreeEntry::Tree(h)) => get_tree(&self.repo, h)?,
            Some(TreeEntry::Blob(_)) => {
                return Err(HistoryError::from(StoreError::TypeMismatch {
                    path: "tools".into(),
                    expected: ObjectKind::Tree,
                    actual: ObjectKind::Blob,
                }))
            }
            None => return Ok(Vec::new()),
        };
        let mut tools: Vec<ToolDef> = Vec::with_capacity(tools_tree.entries.len());
        for (_name, entry) in &tools_tree.entries {
            let blob_hash = match entry {
                TreeEntry::Blob(h) => h,
                _ => {
                    return Err(HistoryError::from(StoreError::TypeMismatch {
                        path: "tools/<name>".into(),
                        expected: ObjectKind::Blob,
                        actual: ObjectKind::Tree,
                    }))
                }
            };
            let blob = get_blob(&self.repo, blob_hash)?;
            let t: ToolDef = serde_json::from_slice(&blob.bytes).map_err(HistoryError::from)?;
            tools.push(t);
        }
        Ok(tools)
    }

    async fn previous_summary(&self) -> Result<Option<String>, HistoryError> {
        let Some(root) = read_root_tree(self)? else {
            return Ok(None);
        };
        let blob_hash = match root.get("previous_summary") {
            Some(TreeEntry::Blob(h)) => h,
            Some(TreeEntry::Tree(_)) => {
                return Err(HistoryError::from(StoreError::TypeMismatch {
                    path: "previous_summary".into(),
                    expected: ObjectKind::Blob,
                    actual: ObjectKind::Tree,
                }))
            }
            None => return Ok(None),
        };
        let blob = get_blob(&self.repo, blob_hash)?;
        let text = String::from_utf8(blob.bytes.clone()).map_err(HistoryError::from)?;
        Ok(Some(text))
    }

    async fn append(&self, messages: Vec<Message>) -> Result<(), HistoryError> {
        self.commit(TreePatch::new().add_messages(messages)).await
    }

    async fn compact_prefix(
        &self,
        end: usize,
        summary_message: Message,
        summary_text: String,
    ) -> Result<(), HistoryError> {
        let existing = self.messages().await?;
        if end > existing.len() {
            return Err(HistoryError::from(StoreError::CutOutOfBounds {
                end,
                len: existing.len(),
            }));
        }
        let mut new_messages = Vec::with_capacity(existing.len() - end + 1);
        new_messages.push(summary_message);
        new_messages.extend(existing.into_iter().skip(end));
        self.commit(
            TreePatch::new()
                .replace_messages(new_messages)
                .with_previous_summary(Some(summary_text)),
        )
        .await
    }
}
