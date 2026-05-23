//! `SubagentResult` — what `spawn` / `send` return to the caller.

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SubagentResult {
    pub agent_id: String,
    /// Final assistant text (the literal last assistant turn's text;
    /// empty if the last turn was tool-calls only or the run errored
    /// before producing text).
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Number of `Content::ToolCall` blocks observed in the agent's
    /// message log over this run. Computed by scanning the final
    /// message log, so it counts tool *invocations* the model emitted,
    /// regardless of whether each invocation produced a
    /// `ToolExecutionEnd` event.
    ///
    /// Cross-reference: [`crate::fleet::AgentSnapshot::tool_use_count`]
    /// counts the same concept but from the other side of the wire —
    /// it increments on every `ToolExecutionEnd` event seen by the
    /// fleet bus. The two should usually agree but can diverge if a
    /// tool errors before emitting `ToolExecutionEnd` (or never starts
    /// executing at all).
    pub tool_use_count: u32,
    pub duration_ms: u64,
    /// Set only if the subagent ran in a worktree and the worktree
    /// was *not* cleanly removed (i.e., left changes behind).
    pub worktree_path: Option<String>,
    pub worktree_branch: Option<String>,
    /// Path to the JSONL transcript on disk, when recording succeeded.
    pub transcript_path: Option<String>,
}
