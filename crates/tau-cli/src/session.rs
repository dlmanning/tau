//! Session management for saving and loading conversations

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
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
        input: u32,
        output: u32,
        cache_read: u32,
        cache_write: u32,
        timestamp: i64,
    },
}

/// Session manager for persisting conversations
pub struct SessionManager {
    /// Session ID
    id: String,
    /// Path to the session file
    #[allow(dead_code)]
    path: PathBuf,
    /// Writer for appending entries
    writer: Option<BufWriter<File>>,
}

impl SessionManager {
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

        // Write metadata
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
            path,
            writer: Some(writer),
        })
    }

    /// Load an existing session
    pub fn load(id: &str) -> std::io::Result<(Self, Vec<Message>)> {
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

        let mut messages = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            if let Ok(SessionEntry::Message { message, .. }) =
                serde_json::from_str::<SessionEntry>(&line)
            {
                messages.push(message);
            }
        }

        // Open for appending
        let file = File::options().append(true).open(&path)?;
        let writer = BufWriter::new(file);

        Ok((
            Self {
                id: id.to_string(),
                path,
                writer: Some(writer),
            },
            messages,
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

        // Sort by created_at descending (newest first)
        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(sessions)
    }

    fn read_session_info(path: &PathBuf) -> Option<SessionInfo> {
        let file = File::open(path).ok()?;
        let reader = BufReader::new(file);
        let first_line = reader.lines().next()?.ok()?;

        if let Ok(SessionEntry::Metadata {
            id,
            created_at,
            model,
            working_dir,
        }) = serde_json::from_str(&first_line)
        {
            // Count messages
            let file = File::open(path).ok()?;
            let reader = BufReader::new(file);
            let message_count = reader
                .lines()
                .map_while(Result::ok)
                .filter(|l| l.contains("\"type\":\"message\""))
                .count();

            Some(SessionInfo {
                id,
                created_at,
                model,
                working_dir,
                message_count,
            })
        } else {
            None
        }
    }

    /// Delete a session
    #[allow(dead_code)]
    pub fn delete(id: &str) -> std::io::Result<()> {
        let sessions_dir = Self::sessions_dir();
        let path = sessions_dir.join(format!("{}.jsonl", id));
        fs::remove_file(path)
    }

    /// Create a branched session from messages up to (and including) branch_index.
    /// If branch_index is None, creates an empty session.
    /// Returns the new SessionManager.
    pub fn branch_from(
        messages: &[Message],
        branch_index: Option<usize>,
        model: &str,
    ) -> std::io::Result<Self> {
        let id = uuid::Uuid::new_v4().to_string();
        let sessions_dir = Self::sessions_dir();
        fs::create_dir_all(&sessions_dir)?;

        let path = sessions_dir.join(format!("{}.jsonl", id));
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write metadata
        let metadata = SessionEntry::Metadata {
            id: id.clone(),
            created_at: chrono::Utc::now().timestamp_millis(),
            model: model.to_string(),
            working_dir: std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| ".".to_string()),
        };
        writeln!(writer, "{}", serde_json::to_string(&metadata)?)?;

        // Write messages up to branch point
        if let Some(idx) = branch_index {
            for msg in messages.iter().take(idx + 1) {
                let entry = SessionEntry::Message {
                    message: msg.clone(),
                    timestamp: chrono::Utc::now().timestamp_millis(),
                };
                writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
            }
        }

        writer.flush()?;

        Ok(Self {
            id,
            path,
            writer: Some(writer),
        })
    }
}

/// Information about a saved session
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub created_at: i64,
    #[allow(dead_code)]
    pub model: String,
    pub working_dir: String,
    pub message_count: usize,
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
