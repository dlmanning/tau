//! The three git object types: [`Blob`], [`Tree`], [`Commit`].
//!
//! Everything in the repository is one of these. All three are
//! content-addressed via [`Hash`] (SHA-256 of a type-tagged
//! serialization). The type tag in the hash input means a blob's
//! hash is distinct from a tree's hash even when the underlying
//! bytes coincide.

use std::collections::BTreeMap;
use std::fmt;
use std::io;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Adapter that lets `serde_json::to_writer` stream bytes directly
/// into a `Sha256` hasher, avoiding an intermediate `Vec<u8>`
/// allocation when hashing trees and commits.
struct SinkAdapter<'a> {
    hasher: &'a mut Sha256,
}

impl<'a> SinkAdapter<'a> {
    fn new(hasher: &'a mut Sha256) -> Self {
        Self { hasher }
    }
}

impl<'a> io::Write for SinkAdapter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.hasher.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Content hash. One uniform type for all three object kinds (blob,
/// tree, commit) — the type tag is folded into the hash input so
/// objects of different kinds with otherwise-identical bytes hash
/// distinctly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectHash([u8; 32]);

impl ObjectHash {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Reconstruct from raw bytes — for snapshot/restore flows where
    /// the caller has persisted a hash and wants to reopen the
    /// corresponding object via [`Repository::get_*`](crate::Repository).
    /// Bytes are taken at face value; no validation against any
    /// particular repository.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        ObjectHash(bytes)
    }

    pub(crate) fn compute_blob(bytes: &[u8]) -> Self {
        Self::compute_with_tag(b"blob\0", bytes)
    }

    fn compute_with_tag(tag: &[u8], bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(tag);
        h.update(bytes);
        ObjectHash(h.finalize().into())
    }
}

impl AsRef<[u8]> for ObjectHash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for ObjectHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Short prefix for log readability, like git's short hashes.
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}…",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

impl fmt::Display for ObjectHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Opaque content. Hosts wrap structured data (a serialized
/// [`Message`](tau_ai::Message), a tool's parameter schema, a system
/// prompt's content blocks) by serializing it to bytes and putting
/// the result in a `Blob`. The repository doesn't know or care what
/// the bytes mean.
#[derive(Clone, Debug)]
pub struct Blob {
    pub bytes: Vec<u8>,
}

impl Blob {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    pub fn hash(&self) -> ObjectHash {
        ObjectHash::compute_blob(&self.bytes)
    }
}

/// One entry in a [`Tree`]: either a blob (a leaf) or a subtree.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TreeEntry {
    Blob(ObjectHash),
    Tree(ObjectHash),
}

impl TreeEntry {
    pub fn hash(&self) -> &ObjectHash {
        match self {
            TreeEntry::Blob(h) | TreeEntry::Tree(h) => h,
        }
    }
}

/// A directory of named entries. The hash of a tree is a function of
/// its entries' (name, kind, hash) triples — so two trees with the
/// same entries in the same order share a hash, and a subtree
/// whose contents didn't change between parent and child commits
/// is referenced by the same hash in both.
///
/// `entries` is a `BTreeMap` so iteration (and thus serialization
/// for hashing) is deterministic by key order.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Tree {
    pub entries: BTreeMap<String, TreeEntry>,
}

impl Tree {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn hash(&self) -> ObjectHash {
        let mut hasher = Sha256::new();
        hasher.update(b"tree\0");
        serde_json::to_writer(SinkAdapter::new(&mut hasher), &self.entries)
            .expect("Tree entries serialize to JSON");
        ObjectHash(hasher.finalize().into())
    }

    pub fn get(&self, name: &str) -> Option<&TreeEntry> {
        self.entries.get(name)
    }

    pub fn insert(&mut self, name: impl Into<String>, entry: TreeEntry) -> Option<TreeEntry> {
        self.entries.insert(name.into(), entry)
    }

    pub fn remove(&mut self, name: &str) -> Option<TreeEntry> {
        self.entries.remove(name)
    }
}

/// A commit pins one root tree to a point in branch history. Has a
/// primary parent (or `None` at the start of an orphan branch) and
/// zero or more `extra_parents` for merges.
///
/// The tree describes the *entire state* at this commit — system
/// prompt, tools, messages. Walking back through parents shows how
/// the state evolved; the tree at any commit is the snapshot at that
/// moment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commit {
    pub parent: Option<ObjectHash>,
    /// Additional parents for merge commits (e.g., a tool result
    /// citing a subagent's tip). Empty for linear commits.
    /// Preserves order — `[A, B]` and `[B, A]` are different
    /// commits, matching git's parent-order convention.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_parents: Vec<ObjectHash>,
    pub tree: ObjectHash,
}

impl Commit {
    pub fn hash(&self) -> ObjectHash {
        let mut hasher = Sha256::new();
        hasher.update(b"commit\0");
        serde_json::to_writer(SinkAdapter::new(&mut hasher), self)
            .expect("Commit serializes to JSON");
        ObjectHash(hasher.finalize().into())
    }
}

/// Minimal tool definition — the prompt-visible surface the model
/// sees in the API's `tools` field.
///
/// Hosts may carry additional per-tool data (risk classification,
/// approval policy hints, the actual `execute` impl) outside the
/// graph; this struct is only what gets serialized into a blob in
/// the `/tools/<name>` tree entry. Two tools with identical
/// `(name, description, parameters_schema)` hash to the same blob
/// and dedupe across branches.
// `PartialEq` but not `Eq`: `serde_json::Value` doesn't implement `Eq`
// because it can contain floats, which aren't `Eq` (NaN != NaN).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters_schema: serde_json::Value,
}

/// What kind of object the repository was asked for vs what it
/// found. Used by [`StoreError`] to disambiguate hash-not-found
/// cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectKind {
    Blob,
    Tree,
    Commit,
}

impl fmt::Display for ObjectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectKind::Blob => write!(f, "blob"),
            ObjectKind::Tree => write!(f, "tree"),
            ObjectKind::Commit => write!(f, "commit"),
        }
    }
}

/// Errors surfaced by the in-crate [`Repository`](crate::Repository).
/// Translated to the boxed form when crossing the
/// [`History`](crate::History) trait boundary so host-side backend
/// impls don't need to know our type.
#[derive(Debug, Error)]
pub enum StoreError {
    /// The requested hash isn't in the repository as the requested
    /// object kind. Typically means a branch was constructed at a
    /// hash that was never stored here, or it was garbage-collected.
    #[error("{kind} {hash} not found in repository")]
    NotFound { hash: ObjectHash, kind: ObjectKind },

    /// A commit asked to cut a prefix beyond the actual message
    /// count. Caller passed a bad cut index to [`compact_prefix`](crate::History::compact_prefix).
    #[error("compact_prefix called with end={end} on a branch of length {len}")]
    CutOutOfBounds { end: usize, len: usize },

    /// A tree entry pointed at a hash of the wrong kind — e.g., a
    /// subtree entry resolves to a blob. Indicates a corrupt tree.
    #[error("tree entry at '{path}' expected {expected} but found {actual}")]
    TypeMismatch {
        path: String,
        expected: ObjectKind,
        actual: ObjectKind,
    },
}
