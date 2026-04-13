//! Subagent transcript recording for debugging.

use tau_ai::Message;
use tokio::io::AsyncWriteExt;

/// Record a subagent's conversation to disk for debugging.
/// Writes JSONL to `~/.local/share/tau/agent-transcripts/{agent_id}.jsonl`.
/// Overwrites any previous transcript for this agent.
/// Failures are logged and silently ignored.
pub async fn record_transcript(agent_id: &str, messages: &[Message]) {
    let dir = match dirs::data_dir() {
        Some(d) => d.join("tau/agent-transcripts"),
        None => return,
    };

    if tokio::fs::create_dir_all(&dir).await.is_err() {
        return;
    }

    let path = dir.join(format!("{}.jsonl", agent_id));
    let mut file = match tokio::fs::File::create(&path).await {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!("Failed to create transcript file: {}", e);
            return;
        }
    };

    for msg in messages {
        if let Ok(json) = serde_json::to_string(msg) {
            let _ = file.write_all(format!("{}\n", json).as_bytes()).await;
        }
    }
}
