//! Pluggable storage backend for sessions.
//!
//! The default [`FsStorage`] writes one directory per session:
//!
//! ```text
//! <root>/<session_id>/
//!     info.json          // SessionInfo (rewritten on every update)
//!     messages.jsonl     // appended on every MessageEnd
//!     snapshot.json      // full snapshot, written on hibernate + debounced
//!     ui_state.json      // last UI state from save_ui_state
//! ```
//!
//! v1 is single-process. Per-session file locking (e.g. via `fd-lock`)
//! is left to a follow-up — concurrent processes mutating the same
//! session directory will corrupt each other.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::Value;
use tau_ai::Message;
use tokio::io::AsyncWriteExt;

use crate::info::{SessionId, SessionInfo};
use crate::snapshot::SessionSnapshot;
use crate::{Error, Result};

/// Storage backend for sessions. All operations are async and may block on
/// I/O; the manager calls into them from its persister task.
#[async_trait]
pub trait SessionStorage: Send + Sync {
    /// List every session id that has an `info.json` on disk.
    async fn list(&self) -> Result<Vec<SessionId>>;

    /// Read just the metadata. Cheap; used to populate the sidebar.
    async fn read_info(&self, id: &SessionId) -> Result<SessionInfo>;

    /// Overwrite the metadata file.
    async fn write_info(&self, info: &SessionInfo) -> Result<()>;

    /// Append one message to the session's JSONL log.
    async fn append_message(&self, id: &SessionId, message: &Message) -> Result<()>;

    /// Atomically replace the JSONL log with `messages`. Implementations
    /// must guarantee no partial-write window — either the full new log is
    /// visible or the old one is.
    async fn write_messages(&self, id: &SessionId, messages: &[Message]) -> Result<()>;

    /// Read all messages from the JSONL log.
    async fn read_messages(&self, id: &SessionId) -> Result<Vec<Message>>;

    /// Drop the JSONL log entirely.
    async fn delete_messages(&self, id: &SessionId) -> Result<()>;

    /// Overwrite the full snapshot file.
    async fn write_snapshot(&self, snapshot: &SessionSnapshot) -> Result<()>;

    /// Read the full snapshot, or `Err(NotFound)` if missing.
    async fn read_snapshot(&self, id: &SessionId) -> Result<SessionSnapshot>;

    /// Persist the host's opaque UI state.
    async fn write_ui_state(&self, id: &SessionId, ui: &Value) -> Result<()>;

    /// Read the host's opaque UI state, if present.
    async fn read_ui_state(&self, id: &SessionId) -> Result<Option<Value>>;

    /// Hard-delete every artefact for `id`.
    async fn delete(&self, id: &SessionId) -> Result<()>;
}

/// Default filesystem-backed storage.
pub struct FsStorage {
    root: PathBuf,
}

impl FsStorage {
    /// Create a storage rooted at `root`. The directory is created lazily
    /// on first write.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn dir(&self, id: &SessionId) -> PathBuf {
        self.root.join(id)
    }

    fn info_path(&self, id: &SessionId) -> PathBuf {
        self.dir(id).join("info.json")
    }

    fn messages_path(&self, id: &SessionId) -> PathBuf {
        self.dir(id).join("messages.jsonl")
    }

    fn snapshot_path(&self, id: &SessionId) -> PathBuf {
        self.dir(id).join("snapshot.json")
    }

    fn ui_state_path(&self, id: &SessionId) -> PathBuf {
        self.dir(id).join("ui_state.json")
    }

    async fn ensure_dir(&self, id: &SessionId) -> Result<()> {
        tokio::fs::create_dir_all(self.dir(id)).await?;
        Ok(())
    }
}

#[async_trait]
impl SessionStorage for FsStorage {
    async fn list(&self) -> Result<Vec<SessionId>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut entries = tokio::fs::read_dir(&self.root).await?;
        let mut ids = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            if !entry.file_type().await?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            // Only directories that actually have an info.json count.
            if Path::new(&entry.path()).join("info.json").exists() {
                ids.push(name);
            }
        }
        Ok(ids)
    }

    async fn read_info(&self, id: &SessionId) -> Result<SessionInfo> {
        let path = self.info_path(id);
        let bytes = tokio::fs::read(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::NotFound(id.clone())
            } else {
                Error::Io(e)
            }
        })?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    async fn write_info(&self, info: &SessionInfo) -> Result<()> {
        self.ensure_dir(&info.id).await?;
        let json = serde_json::to_vec_pretty(info)?;
        let path = self.info_path(&info.id);
        atomic_write(&path, &json).await
    }

    async fn append_message(&self, id: &SessionId, message: &Message) -> Result<()> {
        self.ensure_dir(id).await?;
        let line = serde_json::to_string(message)?;
        let path = self.messages_path(id);
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        Ok(())
    }

    async fn read_messages(&self, id: &SessionId) -> Result<Vec<Message>> {
        let path = self.messages_path(id);
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::Io(e)),
        };
        let text = String::from_utf8_lossy(&bytes);
        let mut out = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(line)?);
        }
        Ok(out)
    }

    async fn write_messages(&self, id: &SessionId, messages: &[Message]) -> Result<()> {
        self.ensure_dir(id).await?;
        let mut bytes = Vec::with_capacity(messages.len() * 256);
        for m in messages {
            let line = serde_json::to_string(m)?;
            bytes.extend_from_slice(line.as_bytes());
            bytes.push(b'\n');
        }
        atomic_write(&self.messages_path(id), &bytes).await
    }

    async fn delete_messages(&self, id: &SessionId) -> Result<()> {
        let path = self.messages_path(id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::Io(e)),
        }
    }

    async fn write_snapshot(&self, snapshot: &SessionSnapshot) -> Result<()> {
        self.ensure_dir(&snapshot.info.id).await?;
        let json = serde_json::to_vec_pretty(snapshot)?;
        let path = self.snapshot_path(&snapshot.info.id);
        atomic_write(&path, &json).await
    }

    async fn read_snapshot(&self, id: &SessionId) -> Result<SessionSnapshot> {
        let path = self.snapshot_path(id);
        let bytes = tokio::fs::read(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::NotFound(id.clone())
            } else {
                Error::Io(e)
            }
        })?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    async fn write_ui_state(&self, id: &SessionId, ui: &Value) -> Result<()> {
        self.ensure_dir(id).await?;
        let json = serde_json::to_vec_pretty(ui)?;
        let path = self.ui_state_path(id);
        atomic_write(&path, &json).await
    }

    async fn read_ui_state(&self, id: &SessionId) -> Result<Option<Value>> {
        let path = self.ui_state_path(id);
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::Io(e)),
        };
        Ok(Some(serde_json::from_slice(&bytes)?))
    }

    async fn delete(&self, id: &SessionId) -> Result<()> {
        let dir = self.dir(id);
        match tokio::fs::remove_dir_all(&dir).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(Error::NotFound(id.clone())),
            Err(e) => Err(Error::Io(e)),
        }
    }
}

/// Write `bytes` to `path` atomically: write to a sibling tmp file, then
/// rename. Avoids partial-write corruption if the process dies mid-write.
async fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    tokio::fs::write(&tmp, bytes).await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}
