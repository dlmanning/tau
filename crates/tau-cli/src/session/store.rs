//! Session management for saving and loading conversations

use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use serde::{Deserialize, Serialize};
use tau_ai::Message;

/// Session entry types for JSONL format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    /// Session metadata
    Metadata {
        id: String,
        created_at: i64,
        model: String,
        working_dir: String,
    },
    /// A message in the conversation
    Message { message: Message, timestamp: i64 },
    /// Usage information for a turn
    Usage {
        input: u64,
        output: u64,
        cache_read: u64,
        cache_write: u64,
        timestamp: i64,
    },
    /// Context compaction marker: every `Message` entry before this
    /// point is summarized by `summary`; entries from
    /// `first_kept_message_index` onward are the kept tail (re-appended
    /// immediately after this marker by
    /// [`SessionManager::append_compaction_snapshot`]).
    Compaction {
        summary: String,
        first_kept_message_index: usize,
        timestamp: i64,
    },
}

/// Session manager for persisting conversations
pub struct SessionManager {
    /// Session ID
    id: String,
    /// Writer for appending entries
    writer: Option<BufWriter<File>>,
    /// Count of `Message` entries written to (or loaded from) the
    /// file. A compaction marker's `first_kept_message_index` is this
    /// count at write time.
    message_entries: usize,
}

impl SessionManager {
    /// Construct from an already-opened writer. Used by
    /// [`crate::session::branch::branch_from`] which builds the file
    /// itself; the new manager picks up appends from here.
    /// `message_entries` is the number of `Message` entries the caller
    /// already wrote.
    pub(super) fn from_open_writer(
        id: String,
        writer: BufWriter<File>,
        message_entries: usize,
    ) -> Self {
        Self {
            id,
            writer: Some(writer),
            message_entries,
        }
    }

    /// Get the sessions directory
    pub fn sessions_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("tau")
            .join("sessions")
    }

    /// Create a new session
    pub fn new(model: &str) -> std::io::Result<Self> {
        let id = uuid::Uuid::new_v4().to_string();
        let sessions_dir = Self::sessions_dir();
        fs::create_dir_all(&sessions_dir)?;

        let path = sessions_dir.join(format!("{}.jsonl", id));
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        let metadata = SessionEntry::Metadata {
            id: id.clone(),
            created_at: chrono::Utc::now().timestamp_millis(),
            model: model.to_string(),
            working_dir: std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| ".".to_string()),
        };
        writeln!(writer, "{}", serde_json::to_string(&metadata)?)?;
        writer.flush()?;

        Ok(Self {
            id,
            writer: Some(writer),
            message_entries: 0,
        })
    }

    /// Load an existing session.
    /// Returns (manager, messages, previous_summary).
    /// If a compaction entry exists, messages are rebuilt from the summary + messages after the compaction point.
    pub fn load(id: &str) -> std::io::Result<(Self, Vec<Message>, Option<String>)> {
        let sessions_dir = Self::sessions_dir();
        let path = sessions_dir.join(format!("{}.jsonl", id));

        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Session not found: {}", id),
            ));
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        let mut all_messages = Vec::new();
        let mut last_compaction: Option<(String, usize)> = None;

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            match serde_json::from_str::<SessionEntry>(&line) {
                Ok(SessionEntry::Message { message, .. }) => {
                    all_messages.push(message);
                }
                Ok(SessionEntry::Compaction {
                    summary,
                    first_kept_message_index,
                    ..
                }) => {
                    last_compaction = Some((summary, first_kept_message_index));
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!("Skipping corrupted session line: {}", e);
                }
            }
        }

        let message_entries = all_messages.len();
        let (messages, previous_summary) = rebuild_messages(all_messages, last_compaction);

        let file = File::options().append(true).open(&path)?;
        let writer = BufWriter::new(file);

        Ok((
            Self {
                id: id.to_string(),
                writer: Some(writer),
                message_entries,
            },
            messages,
            previous_summary,
        ))
    }

    /// Get session ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Append a message to the session
    pub fn append_message(&mut self, message: &Message) -> std::io::Result<()> {
        if let Some(ref mut writer) = self.writer {
            let entry = SessionEntry::Message {
                message: message.clone(),
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
            writer.flush()?;
            self.message_entries += 1;
        }
        Ok(())
    }

    /// Record a compaction: a marker stating that everything logged so
    /// far is covered by `summary`, followed by the kept tail
    /// (the agent's post-compaction messages minus the synthetic
    /// summary head) re-appended as fresh entries.
    ///
    /// Re-appending the tail trades a little log redundancy for
    /// correctness without coordinate bookkeeping: `load` rebuilds as
    /// `[summary_message] + entries[first_kept..]` no matter how many
    /// compactions the session has been through, and messages that
    /// were summarized before they were ever logged (mid-turn overflow
    /// compaction) can't be lost.
    pub fn append_compaction_snapshot(
        &mut self,
        summary: &str,
        kept_messages: &[Message],
    ) -> std::io::Result<()> {
        if self.writer.is_none() {
            return Ok(());
        }
        let entry = SessionEntry::Compaction {
            summary: summary.to_string(),
            first_kept_message_index: self.message_entries,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        if let Some(ref mut writer) = self.writer {
            writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
            writer.flush()?;
        }
        for msg in kept_messages {
            self.append_message(msg)?;
        }
        Ok(())
    }

    /// Append usage information
    pub fn append_usage(&mut self, usage: &tau_ai::Usage) -> std::io::Result<()> {
        if let Some(ref mut writer) = self.writer {
            let entry = SessionEntry::Usage {
                input: usage.input,
                output: usage.output,
                cache_read: usage.cache_read,
                cache_write: usage.cache_write,
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
            writer.flush()?;
        }
        Ok(())
    }

    /// List all sessions
    pub fn list_sessions() -> std::io::Result<Vec<SessionInfo>> {
        let sessions_dir = Self::sessions_dir();
        if !sessions_dir.exists() {
            return Ok(vec![]);
        }

        let mut sessions = Vec::new();

        for entry in fs::read_dir(&sessions_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                if let Some(info) = Self::read_session_info(&path) {
                    sessions.push(info);
                }
            }
        }

        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(sessions)
    }

    fn read_session_info(path: &PathBuf) -> Option<SessionInfo> {
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let first_line = lines.next()?.ok()?;
        let Ok(SessionEntry::Metadata {
            id,
            created_at,
            model,
            working_dir,
        }) = serde_json::from_str(&first_line)
        else {
            return None;
        };

        let mut message_count = 0;
        let mut preview = String::new();
        for line in lines.map_while(Result::ok) {
            if let Ok(SessionEntry::Message { message, .. }) =
                serde_json::from_str::<SessionEntry>(&line)
            {
                message_count += 1;
                // First user message = what the session was about.
                if preview.is_empty() && message.role() == "user" {
                    preview = message
                        .text()
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .to_string();
                }
            }
        }

        Some(SessionInfo {
            id,
            created_at,
            model,
            working_dir,
            message_count,
            preview,
        })
    }

    /// Resolve a session id or unique prefix to a full session id.
    pub fn resolve_id(prefix: &str) -> std::io::Result<String> {
        let sessions_dir = Self::sessions_dir();
        if sessions_dir.join(format!("{prefix}.jsonl")).exists() {
            return Ok(prefix.to_string());
        }
        let mut matches: Vec<String> = Vec::new();
        if sessions_dir.exists() {
            for entry in fs::read_dir(&sessions_dir)? {
                let path = entry?.path();
                if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                    continue;
                }
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                    && stem.starts_with(prefix)
                {
                    matches.push(stem.to_string());
                }
            }
        }
        match matches.len() {
            1 => Ok(matches.pop().unwrap()),
            0 => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("No session matches '{prefix}'. Run `tau sessions ls` to see saved sessions."),
            )),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Session prefix '{prefix}' is ambiguous; matches:\n  {}",
                    matches.join("\n  ")
                ),
            )),
        }
    }
}

/// Rebuild the in-memory message list from the log's `Message` entries
/// and the last compaction marker, mirroring exactly what
/// `apply_compaction_result` left in memory at save time:
/// `[summary_message] + entries[first_kept..]`.
fn rebuild_messages(
    all_messages: Vec<Message>,
    last_compaction: Option<(String, usize)>,
) -> (Vec<Message>, Option<String>) {
    if let Some((summary, kept_index)) = last_compaction {
        let mut rebuilt = vec![tau_agent::summary_message(&summary)];
        if kept_index < all_messages.len() {
            rebuilt.extend_from_slice(&all_messages[kept_index..]);
        }
        (rebuilt, Some(summary))
    } else {
        (all_messages, None)
    }
}

/// Information about a saved session
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SessionInfo {
    pub id: String,
    pub created_at: i64,
    pub model: String,
    pub working_dir: String,
    pub message_count: usize,
    /// First line of the first user message — what the session was
    /// about, so `sessions ls` is more than a wall of UUIDs.
    pub preview: String,
}

impl SessionInfo {
    /// Format the created_at timestamp for display
    pub fn created_at_display(&self) -> String {
        use chrono::{TimeZone, Utc};
        Utc.timestamp_millis_opt(self.created_at)
            .single()
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    #[test]
    fn rebuild_without_compaction_returns_messages_verbatim() {
        let msgs = vec![user("a"), user("b")];
        let (rebuilt, summary) = rebuild_messages(msgs.clone(), None);
        assert_eq!(rebuilt.len(), 2);
        assert!(summary.is_none());
    }

    /// The rebuilt head must be byte-identical to what
    /// `apply_compaction_result` left in memory at save time.
    #[test]
    fn rebuild_with_compaction_matches_in_memory_shape() {
        let msgs = vec![user("old1"), user("old2"), user("kept1"), user("kept2")];
        let (rebuilt, summary) = rebuild_messages(msgs, Some(("the summary".into(), 2)));
        assert_eq!(summary.as_deref(), Some("the summary"));
        assert_eq!(rebuilt.len(), 3);
        assert_eq!(rebuilt[0].text(), tau_agent::summary_message("the summary").text());
        assert_eq!(rebuilt[1].text(), "kept1");
        assert_eq!(rebuilt[2].text(), "kept2");
    }

    /// Snapshot semantics: the marker's kept-index points just past
    /// everything logged before it, so a tail re-appended after the
    /// marker is exactly what rebuild picks up — including for a
    /// second compaction whose tail was itself re-appended once
    /// already.
    #[test]
    fn rebuild_uses_only_the_last_compaction() {
        // Log: m0 m1 | C1(kept=2) k0 k1 | C2(kept=4) j0
        let msgs = vec![user("m0"), user("m1"), user("k0"), user("k1"), user("j0")];
        let (rebuilt, summary) = rebuild_messages(msgs, Some(("s2".into(), 4)));
        assert_eq!(summary.as_deref(), Some("s2"));
        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[1].text(), "j0");
    }

    /// A kept-index at or past the end (compaction that summarized
    /// everything, tail not yet appended at read time) must not panic.
    #[test]
    fn rebuild_with_out_of_range_kept_index_keeps_only_summary() {
        let msgs = vec![user("m0")];
        let (rebuilt, _) = rebuild_messages(msgs, Some(("s".into(), 5)));
        assert_eq!(rebuilt.len(), 1);
    }
}
