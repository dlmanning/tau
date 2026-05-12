//! Session-wide diff aggregation for `FileChanged` events.
//!
//! Hosts feed [`SessionDiffOverlay::observe`] each [`AgentEvent`] from the
//! agent stream; the overlay tracks (baseline, current) per path and computes
//! a [`FileDiff`] on demand. The diff pane in the host renders from
//! [`SessionDiffOverlay::snapshot`]. `Subagent`-wrapped events are unwrapped
//! transitively, so file changes from any depth in the subagent tree
//! aggregate into the same overlay.
//!
//! **Live-session only.** `FileChanged` events live on the broadcast
//! channel and are not stored in `conversation.messages`. A subscriber that
//! attaches mid-session has no source of truth to backfill from, so its
//! overlay starts empty.
//!
//! Session resume across a restart is partially supported by `tau-session`
//! (gap #7) for messages, but `FileChanged` events are *not* persisted
//! there yet — overlay state is lost across hibernate/activate. A future
//! `file_changed.jsonl` side log in `tau-session` could replay events into
//! the overlay on activation; until then, hosts re-derive cumulative diffs
//! by re-reading the working tree against the project's git base.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};

use tau_agent::AgentEvent;

/// Operation a `FileDiff` represents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FileOp {
    Add,
    Modify,
    Delete,
    Rename { from: PathBuf },
}

/// Cumulative diff for a single file across the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiff {
    pub path: PathBuf,
    pub op: FileOp,
    pub adds: u32,
    pub dels: u32,
    pub hunks: Vec<DiffHunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    pub old_start: u32,
    pub old_count: u32,
    pub new_start: u32,
    pub new_count: u32,
    pub lines: Vec<DiffLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffLine {
    pub old_num: Option<u32>,
    pub new_num: Option<u32>,
    pub kind: DiffLineKind,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffLineKind {
    Context,
    Add,
    Del,
}

/// Per-path tracker: the baseline (snapshot at first observation) and the
/// current content (after the most recent `FileChanged`).
#[derive(Debug, Clone)]
struct FileEntry {
    baseline: Option<String>,
    current: Option<String>,
}

/// Aggregates `FileChanged` events into per-path cumulative diffs.
///
/// Order: paths are returned sorted by `PathBuf` (lexicographic). Stable
/// across calls; useful for UI lists that don't want to re-sort.
#[derive(Debug, Default, Clone)]
pub struct SessionDiffOverlay {
    files: BTreeMap<PathBuf, FileEntry>,
}

impl SessionDiffOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe an agent event. Non-`FileChanged` events are ignored.
    /// `Subagent { event: FileChanged { … }, … }` is unwrapped recursively
    /// so file changes from any depth in the subagent tree are aggregated
    /// into the same overlay.
    ///
    /// Returns the cumulative diff for the affected path on `FileChanged`,
    /// or `None` for any other event or when the cumulative diff is empty
    /// (e.g. the file's current content reverted to baseline).
    pub fn observe(&mut self, event: &AgentEvent) -> Option<FileDiff> {
        let (path, before, after) = match event {
            AgentEvent::FileChanged {
                path,
                before,
                after,
                ..
            } => (path.clone(), before.clone(), after.clone()),
            AgentEvent::Subagent { event, .. } => return self.observe(event),
            _ => return None,
        };

        let entry = self.files.entry(path.clone()).or_insert_with(|| FileEntry {
            // Baseline = the file's content at the FIRST observation (so a
            // tool that edits then re-edits the same file doesn't lose the
            // pre-session state).
            baseline: before.clone(),
            current: before,
        });
        entry.current = after;

        compute_diff(&path, entry)
    }

    /// Snapshot of every tracked file's cumulative diff. Skips entries
    /// whose current content equals the baseline (a "no net change" case
    /// after a revert).
    pub fn snapshot(&self) -> Vec<FileDiff> {
        self.files
            .iter()
            .filter_map(|(path, entry)| compute_diff(path, entry))
            .collect()
    }

    /// Cumulative diff for a single path, or `None` if the file is not
    /// tracked or has no net change vs. baseline.
    pub fn file_diff(&self, path: &Path) -> Option<FileDiff> {
        self.files
            .get(path)
            .and_then(|entry| compute_diff(path, entry))
    }

    /// Forget every tracked file. Use after `/commit` or equivalent.
    pub fn reset(&mut self) {
        self.files.clear();
    }

    /// Number of tracked files (including ones whose net diff is empty).
    pub fn tracked_count(&self) -> usize {
        self.files.len()
    }
}

fn compute_diff(path: &Path, entry: &FileEntry) -> Option<FileDiff> {
    let op = match (&entry.baseline, &entry.current) {
        (None, None) => return None,
        (None, Some(_)) => FileOp::Add,
        (Some(_), None) => FileOp::Delete,
        (Some(a), Some(b)) if a == b => return None, // net no-change
        (Some(_), Some(_)) => FileOp::Modify,
    };

    let baseline = entry.baseline.as_deref().unwrap_or("");
    let current = entry.current.as_deref().unwrap_or("");
    let (hunks, adds, dels) = build_hunks(baseline, current);

    Some(FileDiff {
        path: path.to_path_buf(),
        op,
        adds,
        dels,
        hunks,
    })
}

/// Convert a unified diff (via `similar`) into structured hunks.
/// Context window: 3 lines either side of every change cluster.
fn build_hunks(old: &str, new: &str) -> (Vec<DiffHunk>, u32, u32) {
    let diff = TextDiff::from_lines(old, new);
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut adds = 0u32;
    let mut dels = 0u32;

    for group in diff.grouped_ops(3) {
        if group.is_empty() {
            continue;
        }

        let mut lines = Vec::new();
        let (mut old_start, mut new_start) = (None::<u32>, None::<u32>);
        let (mut old_count, mut new_count) = (0u32, 0u32);

        for op in group {
            for change in diff.iter_changes(&op) {
                let old_num: Option<u32> = change.old_index().map(|i| i as u32 + 1);
                let new_num: Option<u32> = change.new_index().map(|i| i as u32 + 1);
                if old_start.is_none() && old_num.is_some() {
                    old_start = old_num;
                }
                if new_start.is_none() && new_num.is_some() {
                    new_start = new_num;
                }
                let kind = match change.tag() {
                    ChangeTag::Equal => {
                        old_count += 1;
                        new_count += 1;
                        DiffLineKind::Context
                    }
                    ChangeTag::Insert => {
                        new_count += 1;
                        adds += 1;
                        DiffLineKind::Add
                    }
                    ChangeTag::Delete => {
                        old_count += 1;
                        dels += 1;
                        DiffLineKind::Del
                    }
                };
                lines.push(DiffLine {
                    old_num,
                    new_num,
                    kind,
                    content: change.to_string(),
                });
            }
        }

        hunks.push(DiffHunk {
            old_start: old_start.unwrap_or(1),
            old_count,
            new_start: new_start.unwrap_or(1),
            new_count,
            lines,
        });
    }

    (hunks, adds, dels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fc(path: &str, before: Option<&str>, after: Option<&str>) -> AgentEvent {
        AgentEvent::FileChanged {
            path: PathBuf::from(path),
            before: before.map(String::from),
            after: after.map(String::from),
            tool_call_id: "t".into(),
        }
    }

    #[test]
    fn add_op_when_baseline_is_none() {
        let mut o = SessionDiffOverlay::new();
        let d = o.observe(&fc("a.rs", None, Some("hello\n"))).unwrap();
        assert_eq!(d.op, FileOp::Add);
        assert_eq!(d.adds, 1);
        assert_eq!(d.dels, 0);
    }

    #[test]
    fn modify_op_when_both_present() {
        let mut o = SessionDiffOverlay::new();
        let d = o
            .observe(&fc("a.rs", Some("a\nb\n"), Some("a\nB\n")))
            .unwrap();
        assert_eq!(d.op, FileOp::Modify);
        assert_eq!(d.adds, 1);
        assert_eq!(d.dels, 1);
    }

    #[test]
    fn delete_op_when_after_is_none() {
        let mut o = SessionDiffOverlay::new();
        let d = o.observe(&fc("a.rs", Some("x\n"), None)).unwrap();
        assert_eq!(d.op, FileOp::Delete);
    }

    #[test]
    fn baseline_preserved_across_multiple_edits() {
        let mut o = SessionDiffOverlay::new();
        // First edit: original "a\n" -> "b\n"
        o.observe(&fc("a.rs", Some("a\n"), Some("b\n")));
        // Second edit: from "b\n" -> "c\n". Cumulative should show a -> c.
        let d = o
            .observe(&fc("a.rs", Some("b\n"), Some("c\n")))
            .expect("net change");
        assert_eq!(d.op, FileOp::Modify);
        // baseline 'a' replaced with 'c' = 1 add, 1 del
        assert_eq!(d.adds, 1);
        assert_eq!(d.dels, 1);
        assert!(
            d.hunks.iter().any(|h| h
                .lines
                .iter()
                .any(|l| { l.kind == DiffLineKind::Del && l.content.contains('a') })),
            "diff still references the original baseline 'a'"
        );
    }

    #[test]
    fn revert_to_baseline_yields_no_diff() {
        let mut o = SessionDiffOverlay::new();
        o.observe(&fc("a.rs", Some("a\n"), Some("b\n")));
        let d = o.observe(&fc("a.rs", Some("b\n"), Some("a\n")));
        assert!(d.is_none(), "current == baseline => no net diff");
        assert!(o.snapshot().is_empty(), "snapshot omits no-change entries");
    }

    #[test]
    fn snapshot_returns_paths_sorted() {
        let mut o = SessionDiffOverlay::new();
        o.observe(&fc("z.rs", None, Some("z\n")));
        o.observe(&fc("a.rs", None, Some("a\n")));
        o.observe(&fc("m.rs", None, Some("m\n")));
        let snap = o.snapshot();
        let paths: Vec<&str> = snap.iter().map(|d| d.path.to_str().unwrap()).collect();
        assert_eq!(paths, vec!["a.rs", "m.rs", "z.rs"]);
    }

    #[test]
    fn file_diff_returns_per_path() {
        let mut o = SessionDiffOverlay::new();
        o.observe(&fc("a.rs", None, Some("hi\n")));
        let d = o.file_diff(Path::new("a.rs")).expect("present");
        assert_eq!(d.op, FileOp::Add);
        assert!(o.file_diff(Path::new("missing.rs")).is_none());
    }

    #[test]
    fn reset_clears() {
        let mut o = SessionDiffOverlay::new();
        o.observe(&fc("a.rs", None, Some("hi\n")));
        assert_eq!(o.tracked_count(), 1);
        o.reset();
        assert_eq!(o.tracked_count(), 0);
    }

    #[test]
    fn unwraps_subagent_wrapped_file_changed() {
        let mut o = SessionDiffOverlay::new();
        let inner = fc("nested.rs", None, Some("hi\n"));
        let wrapped = AgentEvent::Subagent {
            agent_id: "sub-1".into(),
            description: "test".into(),
            event: Box::new(inner),
        };
        let d = o.observe(&wrapped).expect("subagent file change observed");
        assert_eq!(d.op, FileOp::Add);
        assert_eq!(d.path, PathBuf::from("nested.rs"));
        assert_eq!(o.tracked_count(), 1);
    }

    #[test]
    fn unwraps_subagent_recursively() {
        // depth-2 subagent emits FileChanged
        let mut o = SessionDiffOverlay::new();
        let inner = fc("deep.rs", None, Some("deep\n"));
        let depth_2 = AgentEvent::Subagent {
            agent_id: "sub-2".into(),
            description: "depth 2".into(),
            event: Box::new(inner),
        };
        let depth_1 = AgentEvent::Subagent {
            agent_id: "sub-1".into(),
            description: "depth 1".into(),
            event: Box::new(depth_2),
        };
        let d = o.observe(&depth_1).expect("nested subagent observed");
        assert_eq!(d.op, FileOp::Add);
    }

    #[test]
    fn ignores_non_file_changed_events() {
        let mut o = SessionDiffOverlay::new();
        let d = o.observe(&AgentEvent::AgentStart);
        assert!(d.is_none());
        assert_eq!(o.tracked_count(), 0);
    }
}
