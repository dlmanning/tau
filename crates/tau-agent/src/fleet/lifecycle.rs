//! Spawn / send (resume) / respec / adopt operations.
//!
//! These are the user-facing fleet methods. They compose the registry
//! (invariant-preserving storage), the bus (event forwarding +
//! interaction routing), the worktree module (filesystem isolation),
//! and the transcript module (JSONL dump) into the actual lifecycle
//! behaviors.
//!
//! Cross-cutting concerns:
//!
//! - Every spawn/run is bracketed by `FleetEvent::AgentStarted` and
//!   `FleetEvent::AgentCompleted` on the manager's fleet channel — even when
//!   setup itself fails (worktree creation, history inheritance).
//! - On any exit (success or failure), the transcript is recorded if
//!   a handle exists.
//! - Cancellation: each spawn receives a `CancellationToken`; the
//!   lifecycle bridges parent cancellation onto the subagent's
//!   internal cancel.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use tau_ai::{Content, Message, Usage};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::core::approval::ApprovalPolicy;
use crate::core::builder::AgentBuilder;
use crate::core::config::AgentConfig;
use crate::core::handle::AgentHandle;
use crate::core::interaction::InteractionRequest;
use crate::core::tool::send_event;
use crate::core::transport::Transport;
use crate::fleet::SubagentMessageExt;
use crate::fleet::bus;
use crate::fleet::manager::{AgentSpec, Isolation, SpawnOpts};
use crate::fleet::registry::{AgentEntry, Registry};
use crate::fleet::result::SubagentResult;
use crate::fleet::transcript::record_transcript;
use crate::fleet::worktree::{WorktreeInfo, cleanup_worktree, create_worktree};
use crate::types::error::{Error, Result};
use crate::types::events::{FleetEvent, SubagentOutcome};

/// All the dependencies a lifecycle operation needs. The manager
/// composes one of these per-call rather than holding a single
/// "context" field — keeps each lifecycle function honest about what
/// it touches.
pub struct LifecycleCtx {
    pub registry: Arc<Registry>,
    pub transport: Arc<dyn Transport>,
    pub parent_config: AgentConfig,
    pub fleet_event_tx: broadcast::Sender<FleetEvent>,
    pub parent_interaction_tx: Option<mpsc::Sender<InteractionRequest>>,
    pub default_approval: Arc<dyn ApprovalPolicy>,
    pub interaction_router_capacity: usize,
    /// Deadline applied to each spawned subagent's interaction-gate
    /// awaits. Forwarded to the subagent's builder via
    /// [`AgentBuilder::set_interaction_timeout`]. See
    /// [`AgentManager::with_interaction_timeout`](crate::fleet::manager::AgentManager::with_interaction_timeout).
    pub interaction_timeout: Option<Duration>,
}

// ─── Foreground spawn ────────────────────────────────────────────────

pub async fn spawn(
    ctx: &LifecycleCtx,
    spec: impl Into<Arc<AgentSpec>>,
    initial_prompt: String,
    opts: SpawnOpts,
    cancel: CancellationToken,
) -> Result<SubagentResult> {
    let agent_id = uuid::Uuid::new_v4().to_string();
    let spec = spec.into();

    ctx.registry.begin_spawn(&agent_id, Arc::clone(&spec));
    let result = run_one(ctx, &spec, &initial_prompt, &opts, cancel, &agent_id).await;

    match result {
        Ok((subresult, entry)) => {
            // Snapshot usage + messages-count before moving to idle.
            let final_entry = enrich_entry_for_idle(entry).await;
            ctx.registry.finish_to_idle(&agent_id, final_entry);
            Ok(subresult)
        }
        Err(e) => {
            // `run_one` may have already committed the agent into the
            // running set (via `commit_running`) before the prompt
            // failed. `drop_running` handles both cases: it removes the
            // running entry if commit happened (and the spec either
            // way), and no-ops on the running map if setup failed
            // before commit.
            ctx.registry.drop_running(&agent_id);
            Err(e)
        }
    }
}

/// Snapshot the handle's current usage + message count for delta
/// computation on the next resume.
async fn enrich_entry_for_idle(mut entry: AgentEntry) -> AgentEntry {
    let state = entry.handle.state().await.unwrap_or_default();
    entry.usage_at_pause = state.total_usage.clone();
    entry.messages_at_pause = state.messages.len();
    entry
}

// ─── Background spawn ────────────────────────────────────────────────

/// Spawn a subagent that runs in the background. Returns immediately
/// with the agent's id. On completion (success or failure), a
/// `FollowUp` message is posted to `parent_handle` so the parent's
/// actor can pick it up at its next `DrainFollowUps` opportunity.
///
/// Cancellation is forwarded from the parent's token to a bg-local
/// token. The forwarder is aborted once `run_one` returns so cleanup
/// (event forwarder shutdown, registry untracking) finishes cleanly.
pub async fn spawn_background(
    ctx: &LifecycleCtx,
    spec: impl Into<Arc<AgentSpec>>,
    initial_prompt: String,
    opts: SpawnOpts,
    parent_handle: AgentHandle,
    parent_cancel: CancellationToken,
) -> String {
    let agent_id = uuid::Uuid::new_v4().to_string();
    let description = opts.description.clone();
    let spec = spec.into();
    let bg_cancel = CancellationToken::new();

    parent_handle.expect_follow_up();
    ctx.registry.begin_spawn(&agent_id, Arc::clone(&spec));

    // Build a static copy of everything the spawned future needs;
    // `LifecycleCtx` itself isn't `Clone`.
    let inner_ctx = LifecycleCtx {
        registry: Arc::clone(&ctx.registry),
        transport: Arc::clone(&ctx.transport),
        parent_config: ctx.parent_config.clone(),
        fleet_event_tx: ctx.fleet_event_tx.clone(),
        parent_interaction_tx: ctx.parent_interaction_tx.clone(),
        default_approval: Arc::clone(&ctx.default_approval),
        interaction_router_capacity: ctx.interaction_router_capacity,
        interaction_timeout: ctx.interaction_timeout,
    };
    let aid = agent_id.clone();
    let desc = description.clone();
    let bg_cancel_inner = bg_cancel.clone();

    tokio::spawn(async move {
        let cancel_forwarder = {
            let bg_cancel = bg_cancel.clone();
            tokio::spawn(async move {
                parent_cancel.cancelled().await;
                bg_cancel.cancel();
            })
        };

        let result = run_one(
            &inner_ctx,
            &spec,
            &initial_prompt,
            &opts,
            bg_cancel_inner,
            &aid,
        )
        .await;
        cancel_forwarder.abort();

        match result {
            Ok((subresult, entry)) => {
                let final_entry = enrich_entry_for_idle(entry).await;
                inner_ctx.registry.finish_to_idle(&aid, final_entry);

                let _ = parent_handle
                    .follow_up(Message::subagent_completed(
                        &subresult.agent_id,
                        &desc,
                        format!(
                            "{}\n[Agent {} | {} in + {} out tokens | {} tool calls | {}ms]",
                            subresult.text,
                            subresult.agent_id,
                            subresult.input_tokens,
                            subresult.output_tokens,
                            subresult.tool_use_count,
                            subresult.duration_ms,
                        ),
                    ))
                    .await;
            }
            Err(e) => {
                // See `spawn`: the agent may already be in the running
                // set; `drop_running` cleans up whether or not the
                // commit happened.
                inner_ctx.registry.drop_running(&aid);
                let _ = parent_handle
                    .follow_up(Message::subagent_failed(&aid, &desc, format!("Error: {e}")))
                    .await;
            }
        }
    });

    agent_id
}

// ─── Resume ─────────────────────────────────────────────────────────

pub async fn send(
    ctx: &LifecycleCtx,
    agent_id: &str,
    message: &str,
    parent_cancel: CancellationToken,
) -> Result<SubagentResult> {
    let started_at = Utc::now();
    let start = Instant::now();

    // Atomic idle→running transition. Closes the race where a
    // concurrent respec would see the agent in neither map between a
    // bare `take_idle` and a follow-up `commit_running`. Also puts
    // the real `usage_at_pause` / `messages_at_pause` into the running
    // entry rather than zero placeholders.
    let mut entry = ctx
        .registry
        .take_idle_into_running(agent_id)
        .ok_or_else(|| {
            // Distinguish "mid-respec" (detached, awaiting its new actor)
            // from genuinely gone, so a resume racing a respec reports a
            // retryable busy condition rather than a misleading not-found.
            if ctx.registry.respec_in_progress(agent_id) {
                Error::AgentBusy {
                    id: agent_id.to_string(),
                }
            } else {
                Error::AgentNotFound {
                    id: agent_id.to_string(),
                }
            }
        })?;

    let usage_before = entry.usage_at_pause.clone();
    let messages_at_pause = entry.messages_at_pause;
    let description = entry.description.clone();

    send_event(
        &ctx.fleet_event_tx,
        FleetEvent::AgentResumed {
            agent_id: agent_id.into(),
            description: description.clone(),
            prompt: message.into(),
            resumed_at: started_at,
        },
    );

    let forwarder_shutdown = CancellationToken::new();
    let event_task = bus::spawn_event_forwarder(
        entry.handle.subscribe(),
        ctx.fleet_event_tx.clone(),
        agent_id.into(),
        description.clone(),
        Some(Arc::clone(&ctx.registry)),
        forwarder_shutdown.clone(),
    );
    let cancel_bridge = spawn_cancel_bridge(entry.handle.clone(), parent_cancel.clone());

    let prompt_result = entry.handle.prompt_and_wait(message).await;
    cancel_bridge.abort();
    // Signal then await the forwarder so it can drain any events
    // (final `TurnEnd` / `ToolExecutionEnd`) still buffered in its
    // broadcast receiver. See `spawn_event_forwarder` for the race
    // this closes.
    forwarder_shutdown.cancel();
    let _ = event_task.await;

    let messages = entry.handle.messages().await.unwrap_or_default();
    let current_state = entry.handle.state().await.unwrap_or_default();
    let (delta_input, delta_output) = usage_delta(&usage_before, &current_state.total_usage);
    let tool_use_count = count_tool_uses_since(&messages, messages_at_pause);
    let text = extract_final_text(&messages);

    let transcript_path = record_transcript(agent_id, &messages)
        .await
        .map(|p| p.display().to_string());
    entry.usage_at_pause = current_state.total_usage.clone();
    entry.messages_at_pause = messages.len();

    // Idle bookkeeping: push back, then drop running.
    ctx.registry.finish_to_idle(agent_id, entry);

    let completed_at = Utc::now();
    let duration_ms = start.elapsed().as_millis() as u64;
    let outcome = outcome_from(&prompt_result, &parent_cancel);

    send_event(
        &ctx.fleet_event_tx,
        FleetEvent::AgentCompleted {
            agent_id: agent_id.into(),
            description,
            outcome,
            started_at,
            completed_at,
            duration_ms,
            usage: Usage {
                input: delta_input,
                output: delta_output,
                ..Default::default()
            },
            tool_use_count,
            worktree_path: None,
            worktree_branch: None,
        },
    );

    prompt_result?;

    Ok(SubagentResult {
        agent_id: agent_id.into(),
        text,
        input_tokens: delta_input,
        output_tokens: delta_output,
        tool_use_count,
        duration_ms,
        worktree_path: None,
        worktree_branch: None,
        transcript_path,
    })
}

// ─── Respec ─────────────────────────────────────────────────────────

pub async fn respec(
    ctx: &LifecycleCtx,
    agent_id: &str,
    new_spec: impl Into<Arc<AgentSpec>>,
) -> Result<AgentHandle> {
    // Atomic verify-and-detach under the registry's single lock.
    let entry = ctx
        .registry
        .detach_for_respec(agent_id)
        .map_err(|reason| match reason {
            "running" => Error::AgentBusy {
                id: agent_id.to_string(),
            },
            _ => Error::AgentNotFound {
                id: agent_id.to_string(),
            },
        })?;

    // Fetch history from the still-live handle (we hold a strong
    // reference via `entry.handle`).
    let messages = entry.handle.messages().await.unwrap_or_default();

    // Preserve the original description across the respec transition.
    // A respec is "same agent, new spec" from the user's perspective —
    // clobbering the description with a debug breadcrumb (the old
    // behavior) hid useful info from anyone reading the registry.
    let opts = SpawnOpts {
        description: entry.description.clone(),
        seed: crate::core::builder::AgentSeed::Messages {
            messages,
            previous_summary: None,
        },
        ..Default::default()
    };

    match spawn_interactive(ctx, new_spec, opts).await {
        Ok((handle, _new_id)) => {
            ctx.registry.drop_respec_source(agent_id);
            Ok(handle)
        }
        Err(e) => {
            ctx.registry.restore_respec_source(agent_id, entry);
            // Chain the underlying error rather than stringifying it,
            // so callers branching on `RespecRolledBack` can also
            // inspect the inner cause (e.g. matching on
            // `Error::ActorPanic` from a broken new spec).
            Err(Error::RespecRolledBack {
                id: agent_id.to_string(),
                source: Box::new(e),
            })
        }
    }
}

// ─── Interactive spawn (used by respec internally; also public) ──────

pub async fn spawn_interactive(
    ctx: &LifecycleCtx,
    spec: impl Into<Arc<AgentSpec>>,
    opts: SpawnOpts,
) -> Result<(AgentHandle, String)> {
    let agent_id = uuid::Uuid::new_v4().to_string();
    let spec = spec.into();
    ctx.registry.begin_spawn(&agent_id, Arc::clone(&spec));

    let builder = match configure_builder(ctx, &spec, &opts, &agent_id, None).await {
        Ok(b) => b,
        Err(e) => {
            ctx.registry.drop_running(&agent_id);
            return Err(e);
        }
    };
    let handle = match builder.spawn().await {
        Ok(h) => h,
        Err(e) => {
            ctx.registry.drop_running(&agent_id);
            return Err(e);
        }
    };
    ctx.registry.commit_running(
        &agent_id,
        AgentEntry::new(handle.clone(), opts.description.clone()),
    );
    Ok((handle, agent_id))
}

// ─── Adopt: register an externally-built handle ──────────────────────

/// Stamp `agent_id` on the handle's shared cell and record the spec
/// in the registry. Returns the assigned id (or the existing one if
/// the handle was already adopted).
///
/// The handle is placed in the registry's `adopted` bucket so the
/// invariant — spec exists ⟺ id ∈ idle ∪ running ∪ adopted — holds.
/// `respec` works on adopted handles via the same `detach_for_respec`
/// path as idle entries.
///
/// Concurrency: the id-cell stamp races with a hypothetical concurrent
/// `adopt` of the same handle. We stamp first, then *read back* the
/// winning id through `agent_id()` before recording the spec — so even
/// if two adopters race, both record the spec under the same winning
/// id, not under their own losing UUIDs.
pub fn adopt(
    registry: &Registry,
    handle: &AgentHandle,
    description: impl Into<String>,
    spec: impl Into<Arc<AgentSpec>>,
) -> String {
    let spec = spec.into();
    let description = description.into();

    // Stamp speculatively. `set` is a `OnceLock`; if another adopt
    // (or a fleet-internal spawn) already wrote, ours fails silently.
    // The next `agent_id()` read tells us who actually owns this
    // handle.
    let speculative_id = uuid::Uuid::new_v4().to_string();
    let _ = handle.set_agent_id(speculative_id);
    let actual_id = handle
        .agent_id()
        .expect("agent_id is set after speculative stamp")
        .to_string();

    registry.adopt(
        &actual_id,
        AgentEntry::new(handle.clone(), description),
        spec,
    );
    actual_id
}

// ─── Internal: configure_builder + run_one ───────────────────────────

async fn configure_builder(
    ctx: &LifecycleCtx,
    spec: &AgentSpec,
    opts: &SpawnOpts,
    agent_id: &str,
    fallback_cwd: Option<&str>,
) -> Result<AgentBuilder> {
    let cwd = opts
        .cwd
        .clone()
        .or_else(|| fallback_cwd.map(String::from))
        .unwrap_or_else(|| {
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| ".".into())
        });

    let mut agent_cfg = ctx.parent_config.clone();
    agent_cfg.system_prompt = None;
    agent_cfg.max_turns = Some(spec.max_turns);
    agent_cfg.max_tokens = None;
    // Disable prompt caching for subagents — short-lived, never read back.
    agent_cfg.cache_scope = None;
    agent_cfg.cache_ttl = None;
    if let Some(ref model) = opts.model {
        agent_cfg.model = model.clone();
    }

    let mut builder = AgentBuilder::new(agent_cfg, ctx.transport.clone());
    builder.set_cwd(&cwd);
    builder.set_subagent_depth(opts.subagent_depth);

    let policy = opts
        .approval_policy
        .clone()
        .unwrap_or_else(|| ctx.default_approval.clone());
    builder.set_approval_policy(policy);

    if let Some(sub_tx) = bus::spawn_interaction_router(
        ctx.parent_interaction_tx.clone(),
        agent_id.to_string(),
        ctx.interaction_router_capacity,
    ) {
        builder.set_interaction_sender(sub_tx);
    }
    if let Some(timeout) = ctx.interaction_timeout {
        builder.set_interaction_timeout(timeout);
    }

    for tool in &spec.tools {
        builder.add_tool(tool.clone());
    }

    // System prompt: hosts pass the bare instruction; we leave it as-is.
    // (Original `tau-agent` wraps it with env + tool-list boilerplate;
    // that's a host concern, not a runtime concern — v2 keeps it simple.)
    builder.set_system_prompt(spec.system_prompt.clone());

    // Seed history. `Inherit` is resolved here via the registry;
    // `Messages` is applied directly; `Empty` is a no-op.
    let resolved_seed = match opts.seed.clone() {
        crate::core::builder::AgentSeed::Inherit { agent_id: ref source_id } => {
            let handle = ctx.registry.handle_any(source_id).ok_or_else(|| {
                Error::AgentNotFound {
                    id: source_id.clone(),
                }
            })?;
            // Pull the full conversation, not just the messages: if the
            // source was compacted it carries a `previous_summary` that
            // threads the *next* compaction's update prompt. Dropping it
            // would make the inheritor's first compaction fall back to the
            // initial-summarization path and lose continuity.
            let state = handle.state().await.ok_or_else(|| {
                // The agent exists in the registry but its actor
                // didn't respond — typically because the task is
                // shutting down between idle and dead. No dedicated
                // variant: the condition is rare and "actor died
                // while we were holding a handle" is more naturally
                // expressed as a generic Other than as its own kind.
                Error::Other(format!(
                    "AgentSeed::Inherit: agent '{source_id}' did not return its conversation state \
                     (actor unresponsive or shutting down)"
                ))
            })?;
            crate::core::builder::AgentSeed::Messages {
                messages: state.messages,
                previous_summary: state.previous_summary,
            }
        }
        other => other,
    };
    builder.seed(resolved_seed);

    // Stamp the agent id on the pre-spawn handle so `ExecutionContext::agent_id`
    // is populated from the first tool call.
    let pre = builder.handle();
    let _ = pre.set_agent_id(agent_id.to_string());

    Ok(builder)
}

struct RunOutcome {
    handle: AgentHandle,
    prompt_result: Result<()>,
    result: SubagentResult,
}

async fn run_one(
    ctx: &LifecycleCtx,
    spec: &AgentSpec,
    initial_prompt: &str,
    opts: &SpawnOpts,
    cancel: CancellationToken,
    agent_id: &str,
) -> Result<(SubagentResult, AgentEntry)> {
    let started_at = Utc::now();
    let start = Instant::now();

    // Emit Started before any setup work that could fail.
    send_event(
        &ctx.fleet_event_tx,
        FleetEvent::AgentStarted {
            agent_id: agent_id.into(),
            spec_name: opts.spec_name.clone(),
            description: opts.description.clone(),
            prompt: initial_prompt.into(),
            started_at,
        },
    );

    let emit_setup_failure = |e: Error, wt_path: Option<String>, wt_branch: Option<String>| async {
        let completed_at = Utc::now();
        let duration_ms = start.elapsed().as_millis() as u64;
        send_event(
            &ctx.fleet_event_tx,
            FleetEvent::AgentCompleted {
                agent_id: agent_id.into(),
                description: opts.description.clone(),
                outcome: SubagentOutcome::Failed {
                    reason: e.to_string(),
                },
                started_at,
                completed_at,
                duration_ms,
                usage: Usage::default(),
                tool_use_count: 0,
                worktree_path: wt_path,
                worktree_branch: wt_branch,
            },
        );
        e
    };

    let worktree = if opts.isolation == Some(Isolation::Worktree) {
        match create_worktree(agent_id).await {
            Ok(wt) => Some(wt),
            Err(e) => {
                let err = Error::WorktreeSetupFailed {
                    reason: e.to_string(),
                };
                return Err(emit_setup_failure(err, None, None).await);
            }
        }
    } else {
        None
    };

    let inner = run_agent_inner(
        ctx,
        spec,
        initial_prompt,
        opts,
        agent_id,
        &worktree,
        cancel.clone(),
    )
    .await;
    let (wt_path, wt_branch) = teardown_worktree(&worktree).await;

    let completed_at = Utc::now();
    let duration_ms = start.elapsed().as_millis() as u64;

    let RunOutcome {
        handle,
        prompt_result,
        mut result,
    } = match inner {
        Ok(o) => o,
        Err(e) => {
            return Err(emit_setup_failure(e, wt_path, wt_branch).await);
        }
    };

    let messages = handle.messages().await.unwrap_or_default();
    let transcript_path = record_transcript(agent_id, &messages)
        .await
        .map(|p| p.display().to_string());

    let outcome = outcome_from(&prompt_result, &cancel);
    result.transcript_path = transcript_path;
    result.worktree_path = wt_path.clone();
    result.worktree_branch = wt_branch.clone();
    result.duration_ms = duration_ms;

    let usage = Usage {
        input: result.input_tokens,
        output: result.output_tokens,
        ..Default::default()
    };
    let tool_use_count = result.tool_use_count;

    send_event(
        &ctx.fleet_event_tx,
        FleetEvent::AgentCompleted {
            agent_id: agent_id.into(),
            description: opts.description.clone(),
            outcome,
            started_at,
            completed_at,
            duration_ms,
            usage,
            tool_use_count,
            worktree_path: wt_path,
            worktree_branch: wt_branch,
        },
    );

    match prompt_result {
        Ok(()) => Ok((result, AgentEntry::new(handle, opts.description.clone()))),
        Err(e) => Err(e),
    }
}

async fn run_agent_inner(
    ctx: &LifecycleCtx,
    spec: &AgentSpec,
    initial_prompt: &str,
    opts: &SpawnOpts,
    agent_id: &str,
    worktree: &Option<WorktreeInfo>,
    cancel: CancellationToken,
) -> Result<RunOutcome> {
    let wt_cwd = worktree.as_ref().map(|w| w.path.display().to_string());
    let builder = configure_builder(ctx, spec, opts, agent_id, wt_cwd.as_deref()).await?;
    let handle = builder.spawn().await?;

    let forwarder_shutdown = CancellationToken::new();
    let event_task = bus::spawn_event_forwarder(
        handle.subscribe(),
        ctx.fleet_event_tx.clone(),
        agent_id.into(),
        opts.description.clone(),
        Some(Arc::clone(&ctx.registry)),
        forwarder_shutdown.clone(),
    );
    let cancel_bridge = spawn_cancel_bridge(handle.clone(), cancel.clone());

    ctx.registry.commit_running(
        agent_id,
        AgentEntry::new(handle.clone(), opts.description.clone()),
    );

    let prompt_result = handle.prompt_and_wait(initial_prompt).await;
    cancel_bridge.abort();
    // Signal then await the forwarder so it can drain any events
    // (final `TurnEnd` / `ToolExecutionEnd`) still buffered in its
    // broadcast receiver. See `spawn_event_forwarder` for the race
    // this closes.
    forwarder_shutdown.cancel();
    let _ = event_task.await;

    let messages = handle.messages().await.unwrap_or_default();
    let text = extract_final_text(&messages);
    let state = handle.state().await.unwrap_or_default();
    let tool_use_count = count_tool_uses_since(&messages, 0);

    let result = SubagentResult {
        agent_id: agent_id.into(),
        text,
        input_tokens: state.total_usage.input,
        output_tokens: state.total_usage.output,
        tool_use_count,
        duration_ms: 0,
        worktree_path: None,
        worktree_branch: None,
        transcript_path: None,
    };
    Ok(RunOutcome {
        handle,
        prompt_result,
        result,
    })
}

// ─── Helpers ────────────────────────────────────────────────────────

pub(crate) fn spawn_cancel_bridge(
    handle: AgentHandle,
    parent_cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        parent_cancel.cancelled().await;
        handle.abort();
    })
}

async fn teardown_worktree(worktree: &Option<WorktreeInfo>) -> (Option<String>, Option<String>) {
    match worktree {
        Some(wt) => match cleanup_worktree(wt).await {
            Ok(true) => (None, None),
            _ => (Some(wt.path.display().to_string()), Some(wt.branch.clone())),
        },
        None => (None, None),
    }
}

fn outcome_from(prompt_result: &Result<()>, cancel: &CancellationToken) -> SubagentOutcome {
    match prompt_result {
        Ok(()) if cancel.is_cancelled() => SubagentOutcome::Aborted {
            reason: "cancelled by parent".into(),
        },
        Ok(()) => SubagentOutcome::Completed,
        Err(e) if cancel.is_cancelled() => SubagentOutcome::Aborted {
            reason: e.to_string(),
        },
        Err(e) => SubagentOutcome::Failed {
            reason: e.to_string(),
        },
    }
}

fn usage_delta(before: &Usage, after: &Usage) -> (u64, u64) {
    (
        after.input.saturating_sub(before.input),
        after.output.saturating_sub(before.output),
    )
}

fn count_tool_uses_since(messages: &[Message], from: usize) -> u32 {
    let start = from.min(messages.len());
    messages[start..]
        .iter()
        .map(|m| match m {
            Message::Assistant { content, .. } => content
                .iter()
                .filter(|c| matches!(c, Content::ToolCall { .. }))
                .count(),
            _ => 0,
        })
        .sum::<usize>() as u32
}

/// Text of the literal *last* assistant message. Empty when the last
/// assistant turn was tool-calls only or there's no assistant message
/// at all. We don't fall back to earlier turns: stale text from
/// before the actual work would mislead the parent into thinking it
/// had the agent's final answer.
fn extract_final_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::Assistant { content, .. } => Some(
                content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            ),
            _ => None,
        })
        .unwrap_or_default()
}
