//! [`Repository`] — in-memory content-addressed store for the three
//! object kinds (blobs, trees, commits).
//!
//! Three separate maps keyed by [`ObjectHash`]. Each kind stores
//! `Arc<T>` so callers reading objects share immutable views without
//! cloning the underlying bytes.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::branch::Branch;
use crate::objects::{Blob, Commit, ObjectHash, Tree};

/// In-memory content-addressed object store. Issues branches via
/// [`new_branch`](Self::new_branch) (empty tip) or
/// [`branch_at`](Self::branch_at) (resume from a known commit hash).
///
/// Three object kinds in three internal maps:
/// - blobs: opaque content the host serialized
/// - trees: named directories of blob and subtree references
/// - commits: tree pointers with parent/extra_parents
pub struct Repository {
    blobs: RwLock<HashMap<ObjectHash, Arc<Blob>>>,
    trees: RwLock<HashMap<ObjectHash, Arc<Tree>>>,
    commits: RwLock<HashMap<ObjectHash, Arc<Commit>>>,
    /// Named pointers to commits — git's `refs/tags/` analog. Used
    /// to bookmark canonical commits: agent type templates, session
    /// snapshots, audit markers. Hosts pick the naming convention
    /// (typically a path-like form: `"templates/research-agent"`,
    /// `"snapshots/2024-11-15"`, `"audit/approved-by-david"`).
    ///
    /// Tags are *lightweight* — just a name pointing at a commit
    /// hash, like git's lightweight tags. They're stored on the
    /// repository, not the branch, so they survive any particular
    /// branch's lifecycle. If you want metadata attached to a tag
    /// (description, author, etc.), put it in the tagged commit's
    /// tree.
    ///
    /// Tags are mutable: `set_tag` overwrites. Hosts that want
    /// immutability check with `resolve_tag` first. (Git's default
    /// is reject-if-exists; we leave that policy to the host.)
    tags: RwLock<BTreeMap<String, ObjectHash>>,
}

impl Repository {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            blobs: RwLock::new(HashMap::new()),
            trees: RwLock::new(HashMap::new()),
            commits: RwLock::new(HashMap::new()),
            tags: RwLock::new(BTreeMap::new()),
        })
    }

    // ─── Object store: put ───────────────────────────────────────────

    /// Insert a blob, returning its hash. No-op if a blob with the
    /// same hash is already present.
    pub fn put_blob(&self, blob: Blob) -> ObjectHash {
        let hash = blob.hash();
        if self.blobs.read().contains_key(&hash) {
            return hash;
        }
        let mut g = self.blobs.write();
        g.entry(hash).or_insert_with(|| Arc::new(blob));
        hash
    }

    /// Insert a tree, returning its hash. Idempotent.
    pub fn put_tree(&self, tree: Tree) -> ObjectHash {
        let hash = tree.hash();
        if self.trees.read().contains_key(&hash) {
            return hash;
        }
        let mut g = self.trees.write();
        g.entry(hash).or_insert_with(|| Arc::new(tree));
        hash
    }

    /// Insert a commit, returning its hash. Idempotent.
    pub fn put_commit(&self, commit: Commit) -> ObjectHash {
        let hash = commit.hash();
        if self.commits.read().contains_key(&hash) {
            return hash;
        }
        let mut g = self.commits.write();
        g.entry(hash).or_insert_with(|| Arc::new(commit));
        hash
    }

    // ─── Object store: get ───────────────────────────────────────────

    pub fn get_blob(&self, hash: &ObjectHash) -> Option<Arc<Blob>> {
        self.blobs.read().get(hash).cloned()
    }

    pub fn get_tree(&self, hash: &ObjectHash) -> Option<Arc<Tree>> {
        self.trees.read().get(hash).cloned()
    }

    pub fn get_commit(&self, hash: &ObjectHash) -> Option<Arc<Commit>> {
        self.commits.read().get(hash).cloned()
    }

    // ─── Counts (for tests / observability) ──────────────────────────

    pub fn blob_count(&self) -> usize {
        self.blobs.read().len()
    }

    pub fn tree_count(&self) -> usize {
        self.trees.read().len()
    }

    pub fn commit_count(&self) -> usize {
        self.commits.read().len()
    }

    /// Total objects across all three kinds. Useful for tests that
    /// want to assert "fork didn't duplicate the prefix."
    ///
    /// Note: this snapshot isn't atomic across the three kinds — a
    /// concurrent insert during the call can produce a count anywhere
    /// between the pre- and post-insert totals.
    pub fn object_count(&self) -> usize {
        self.blob_count() + self.tree_count() + self.commit_count()
    }

    // ─── Branch construction ─────────────────────────────────────────

    /// `git checkout --orphan` — a brand-new branch with no commits.
    /// Useful for tests and for hosts that want to build the prefix
    /// piecewise. The first commit on this branch will have
    /// `parent: None`.
    pub fn new_branch(self: &Arc<Self>) -> Arc<Branch> {
        Branch::empty(Arc::clone(self))
    }

    /// `git checkout <commit>` — open a branch positioned at an
    /// existing commit. Lazy: doesn't validate that the commit is
    /// actually in this repository. The first read will surface an
    /// error if it isn't.
    pub fn branch_at(self: &Arc<Self>, tip: ObjectHash) -> Arc<Branch> {
        Branch::at(Arc::clone(self), tip)
    }

    // ─── Tags ────────────────────────────────────────────────────────

    /// `git tag -f <name> <commit>` — create or overwrite a tag.
    /// Lazy: doesn't validate that the commit is in this repository
    /// (matches [`branch_at`](Self::branch_at)'s behavior). If the
    /// tag already exists it's silently replaced; hosts that want
    /// "create only if absent" semantics check
    /// [`resolve_tag`](Self::resolve_tag) first.
    pub fn set_tag(&self, name: impl Into<String>, commit: ObjectHash) {
        self.tags.write().insert(name.into(), commit);
    }

    /// `git rev-parse <tag>` — look up the commit a tag points at.
    /// Returns `None` if the tag doesn't exist.
    pub fn resolve_tag(&self, name: &str) -> Option<ObjectHash> {
        self.tags.read().get(name).copied()
    }

    /// `git tag -d <name>` — remove a tag. Returns the commit hash
    /// it pointed at, or `None` if no such tag existed.
    pub fn remove_tag(&self, name: &str) -> Option<ObjectHash> {
        self.tags.write().remove(name)
    }

    /// Enumerate all tags as `(name, commit)` pairs, sorted by
    /// name. Useful for host UIs that want to display "the agent
    /// types this fleet knows about."
    pub fn tags(&self) -> Vec<(String, ObjectHash)> {
        self.tags
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// `git checkout <tag>` — open a branch positioned at the
    /// commit a tag names. Returns `None` if the tag doesn't exist.
    /// Equivalent to `branch_at(resolve_tag(name)?)`.
    ///
    /// Common pattern for "spawn a new agent of type X":
    /// ```ignore
    /// let template = repo.branch_at_tag("templates/research-agent")?;
    /// let working = template.fork();  // working copy diverges from here
    /// ```
    pub fn branch_at_tag(self: &Arc<Self>, name: &str) -> Option<Arc<Branch>> {
        self.resolve_tag(name).map(|h| self.branch_at(h))
    }
}
