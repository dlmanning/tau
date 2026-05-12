//! JSONL transcript recording for subagents.
//!
//! Writes one message per line to
//! `~/.local/share/tau/agent-transcripts/{agent_id}.jsonl`. Failures
//! are logged and `None` is returned — never raised, since the
//! transcript is a diagnostic, not a correctness concern.

use std::path::PathBuf;

use tau_ai::Message;
use tokio::io::AsyncWriteExt;

pub async fn record_transcript(agent_id: &str, messages: &[Message]) -> Option<PathBuf> {
    let dir = dirs::data_dir().map(|d| d.join("tau/agent-transcripts"))?;

    if tokio::fs::create_dir_all(&dir).await.is_err() {
        return None;
    }

    let path = dir.join(format!("{agent_id}.jsonl"));
    let mut file = match tokio::fs::File::create(&path).await {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!("Failed to create transcript file: {e}");
            return None;
        }
    };

    for msg in messages {
        if let Ok(json) = serde_json::to_string(msg) {
            if file
                .write_all(format!("{json}\n").as_bytes())
                .await
                .is_err()
            {
                return None;
            }
        }
    }

    Some(path)
}
