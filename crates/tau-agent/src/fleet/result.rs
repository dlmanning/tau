//! `SubagentResult` — what `spawn` / `send` return to the caller.

#[derive(Debug, Clone)]
pub struct SubagentResult {
    pub agent_id: String,
    /// Final assistant text (the literal last assistant turn's text;
    /// empty if the last turn was tool-calls only or the run errored
    /// before producing text).
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_use_count: u32,
    pub duration_ms: u64,
    /// Set only if the subagent ran in a worktree and the worktree
    /// was *not* cleanly removed (i.e., left changes behind).
    pub worktree_path: Option<String>,
    pub worktree_branch: Option<String>,
    /// Path to the JSONL transcript on disk, when recording succeeded.
    pub transcript_path: Option<String>,
}
