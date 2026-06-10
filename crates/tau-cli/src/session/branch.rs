//! Branching: create a new session whose history is a prefix of an
//! existing conversation.

use std::fs::{self, File};
use std::io::{BufWriter, Write};

use tau_ai::Message;

use super::store::{SessionEntry, SessionManager};

/// Create a branched session from `messages` up to and including
/// `branch_index`. `None` produces an empty session. The returned
/// `SessionManager` is open for further appends; its file lives in the
/// shared sessions directory under a fresh UUID.
pub fn branch_from(
    messages: &[Message],
    branch_index: Option<usize>,
    model: &str,
) -> std::io::Result<SessionManager> {
    let id = uuid::Uuid::new_v4().to_string();
    let sessions_dir = SessionManager::sessions_dir();
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

    let mut written = 0;
    if let Some(idx) = branch_index {
        for msg in messages.iter().take(idx + 1) {
            let entry = SessionEntry::Message {
                message: msg.clone(),
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            writeln!(writer, "{}", serde_json::to_string(&entry)?)?;
            written += 1;
        }
    }

    writer.flush()?;

    Ok(SessionManager::from_open_writer(id, writer, written))
}
