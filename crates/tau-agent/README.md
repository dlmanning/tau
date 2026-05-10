# tau-agent

Actor-based agent runtime with tool execution, context compaction, and subagent management.

## Architecture

The crate is split along an I/O boundary:

```
AgentHandle ──channels──▶ actor.rs (async I/O loop)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              logic.rs   transport.rs  compaction.rs
              (sync)      (async)       (async)
                    │
                    ▼
                state.rs (data)
```

- **`actor.rs`** — Stepped state machine (`StepPhase`) that owns `AgentState` exclusively. Processes commands from dual channels (urgent/normal) and `select!`s on async I/O.
- **`logic.rs`** — Synchronous decision logic. Methods on `AgentState` that never do I/O, making them testable without a tokio runtime.
- **`state.rs`** — Pure data types. `AgentState` holds config, conversation, tools, queues, and shared atomics.
- **`transport.rs`** — LLM provider abstraction with retry, stream timeout/stall detection, and cancellation.
- **`compaction.rs`** — Context window management: summarization, cut-point finding, token estimation.

## Key types

| Type | Module | Role |
|------|--------|------|
| `AgentBuilder` | `builder` | Configure and spawn an agent. `pre_handle()` available before `spawn()` to break circular deps. |
| `AgentHandle` | `handle` | Clone + Send + Sync handle. All interaction goes through this. Spec changes consume `self` and return a new handle (`respec`, `with_system_prompt`, `with_tools`). |
| `AgentConfig` | `config` | Model, reasoning level, max turns, compaction settings. |
| `Tool` | `tool` | Trait for tool implementations: name, schema, concurrency, execute. Optional `bind_to_agent` hook for tools that need their owning agent's handle (e.g. recursive `AgentTool`). |
| `AgentEvent` | `events` | Broadcast event enum: lifecycle, messages, tools, compaction, errors, subagents, file changes. |
| `AgentSpec` | `manager` | Immutable per-agent input the runtime treats as fixed: system prompt, tools, max turns, worktree allowance, allowed-subagent-spec names. Stored as `Arc<AgentSpec>` in the manager registry; spawn/respec methods accept `impl Into<Arc<AgentSpec>>`. To change any field, spawn a new agent (typically via `respec`). |
| `SpawnOpts` | `manager` | Per-spawn options that aren't part of the spec: description, model override, cwd, `Isolation`, `inherit_history_from`, `seed_messages`, approval-policy override, `spec_name`. |
| `Isolation` | `manager` | Filesystem isolation mode for a subagent. Currently `Worktree`. Serializes as snake_case so the same enum drives both Rust APIs and tool-arg JSON schemas. |
| `InteractionRequest` | `interaction` | Tool → host UI round-trip. Two shapes: `AskQuestion` (untyped, returns `Answer`) and `Typed { schema_id, payload }` (host renders by `schema_id`, replies with `Approved { payload }` / `Rejected` / `Cancelled`). The runtime treats `payload` as opaque JSON; schemas (e.g. `plan.submit`) and their typed payload structs live with the tools that emit them (e.g. `tau_tools::Plan`). |
| `ApprovalPolicy` | `approval` | Trait that classifies a pending tool call into `Auto` / `Gate` / `Reject`. Built-in policies: `DefaultApprovalPolicy`, `AutoAcceptAllPolicy`, `RulePolicy`. |
| `AgentManager` | `manager` | Subagent lifecycle: spawn foreground/background/interactive, resume, respec, evict. Holds the spec registry; enforces a single invariant ("spec exists ⟺ id is in idle storage or running"). |
| `Transport` | `transport` | Trait abstracting LLM calls. `ProviderTransport` handles Anthropic/OpenAI/Google. |

## Agent loop

1. `AgentHandle::prompt()` sends a `Command::Prompt` over the normal channel.
2. The actor transitions through phases: `Idle` → `PrepareTurn` → `AwaitingModel` → `ProcessResponse`.
3. `logic.rs` decides the next action: run tools, compact, or finish.
4. Tool calls are grouped by concurrency and executed on a `JoinSet`.
5. After tools complete, the loop returns to `PrepareTurn` for the next LLM call.
6. Steering messages (via `handle.steer()`) arrive on the urgent channel and interrupt between tool groups.

## Subagents

`AgentManager` spawns independent agent instances, each with their own actor loop. Each subagent is itself a complete agent satisfying the same boundary contract as the root — events flow up to the parent's broadcast wrapped as `Subagent { event }`, alongside parent-timeline events (`SubagentStarted`, `SubagentCompleted`).

Hosts construct an `AgentSpec` (system prompt, tools, max turns, worktree allowance, allowed-subagent-spec names) and pass it to `AgentManager::spawn` along with the initial prompt and a `SpawnOpts`. The runtime treats spec names as opaque strings owned by the host — `general-purpose`, `explore`, `plan`, etc., are conventions defined in the host's spec map / resolver, not in the runtime.

### Lifecycle methods

- **Foreground** — `spawn(spec, prompt, opts, cancel)` blocks the parent until completion.
- **Background** — `spawn_background(...)` returns immediately; posts a `FollowUp` message to the parent on completion. Cancellation is forwarded via a token bridge so the subagent's internal cleanup runs to completion (event forwarder abort, registry untracking) before the task ends.
- **Interactive** — `spawn_interactive(...)` returns the `AgentHandle` for caller-driven prompting. The caller is responsible for `remove_interactive` when done.
- **Resume** — `send(id, message, cancel)` rehydrates a stored idle agent with a new prompt. The agent is tracked in `running_handles` for the duration of the resume so `find_agent` / `send_to_running` continue to locate it.
- **Respec** — `respec(agent_id, new_spec)` is a *transition*: spawns a fresh agent under `new_spec` with `inherit_history_from = agent_id`, then evicts the original idle entry. The old id stops resolving. To fork instead of transition, call `spawn` directly with `inherit_history_from`.
- **Adopt** — `adopt(handle, spec)` registers a builder-spawned root agent so it can `respec` / `with_system_prompt` / `with_tools`.

### History seeding

Two ways to seed a new subagent's conversation, in precedence order:

1. **`SpawnOpts::seed_messages: Option<Vec<Message>>`** — explicit message vector from the host. Used for `/branch`-style flows that fork from an arbitrary index in the parent's history.
2. **`SpawnOpts::inherit_history_from: Option<String>`** — agent id whose full history is fetched from the registry. Used for plan → execute handoffs.

`seed_messages` wins when both are set: the host already has the messages, no lookup needed.

### Plan execution

One application of `respec` / `inherit_history_from`. A `plan` subagent investigates and submits a plan via the `plan.submit` typed interaction; the host approves (optionally editing the body); the host then spawns an executor subagent with `SpawnOpts.inherit_history_from = <planner_agent_id>` so the executor sees the planner's investigation and the approved plan as its own conversation history. There is no separate plan-execution mechanism in the runtime.

### Tool sharing across spawns

`BoxedTool` is `Arc<dyn Tool>`, so reusing the same `AgentSpec` for multiple concurrent spawns shares the same underlying tool objects. Tools that capture per-agent state via `Tool::bind_to_agent` (e.g. `AgentTool`'s `OnceLock<AgentHandle>`) bind to whichever spawn happens first and silently mis-route for subsequent ones. Hosts that spawn the same spec concurrently must construct fresh tool instances per spawn — typically via a resolver closure that builds a new `AgentSpec` each call — rather than cloning a shared tool vector.

## Events on subagent lifecycle

Every subagent run is bracketed by `SubagentStarted` and `SubagentCompleted`, even on setup failure (worktree creation, history-inherit lookup). Failed runs preserve token usage, tool-call count, any partial assistant text, and write a transcript to disk — the post-mortem signal for diagnosing model errors. `SubagentResumed` brackets a resume; `SubagentReport` is a self-label the subagent emits before terminating (host correlates by `agent_id`).

## Testing

The `test-utils` feature exports mock transports, test tools, and an `EventCollector` for deterministic async testing:

```toml
[dev-dependencies]
tau-agent = { workspace = true, features = ["test-utils"] }
```

```rust
use tau_agent::test_utils::*;

let transport = MockTransport::new()
    .with_text_response("Hello!");
let (handle, collector) = spawn_test_agent(transport, vec![]);

handle.prompt("hi").await.unwrap();
collector.wait_for_end().await;

assert_eq!(collector.assistant_messages()[0].text(), "Hello!");
```

Available mocks: `MockTransport`, `TextTransport`, `ToolCallTransport`, `SlowTransport`, `CapturingTransport`, `ErrorTransport`. Test tools: `EchoTool`, `FailTool`, `SlowTool`.

## Module index

| Module | Visibility | Purpose |
|--------|-----------|---------|
| `actor` | crate | Async event loop, `StepPhase` state machine |
| `approval` | pub | `ApprovalPolicy` trait, `ToolRisk`, `ApprovalDecision`, built-in policies |
| `builder` | pub | `AgentBuilder` setup and spawn |
| `command` | crate | `Command` enum (handle → actor protocol) |
| `compaction` | pub | Context summarization and token management |
| `config` | pub | `AgentConfig`, `DequeueMode` |
| `context` | pub | Hierarchical context file loading (AGENTS.md / CLAUDE.md) |
| `conversation` | pub | `Conversation` state container |
| `error` | pub | Error types (`Error::Unmanaged` for builder-spawned handles trying to `respec`) |
| `events` | pub | `AgentEvent` enum (broadcast events) |
| `handle` | pub | `AgentHandle` — public API; `respec` / `with_system_prompt` / `with_tools` consume `self` |
| `interaction` | pub | Tool ↔ UI request/reply protocol (`AskQuestion`, `Typed { schema_id, payload }`) |
| `logic` | crate | Sync decision logic (no I/O) |
| `manager` | pub | `AgentManager` — subagent lifecycle, `AgentSpec`, `SpawnOpts`, `Isolation` |
| `overflow` | crate | Context overflow detection (30+ provider patterns) |
| `prompts` | pub | System prompt assembly |
| `state` | crate | `AgentState`, `ToolCall` data types |
| `stream` | pub | `StreamReducer` — event stream aggregation |
| `tool` | pub | `Tool` trait, `ExecutionContext`, `ToolResult`, `bind_to_agent` hook |
| `tool_executor` | crate | Single tool execution harness |
| `transcript` | pub | JSONL logging for subagent conversations (recorded on success and failure) |
| `transport` | pub | `Transport` trait, `ProviderTransport` |
| `worktree` | crate | Git worktree isolation for subagents |
| `test_utils` | pub (feature-gated) | Mocks, test tools, `EventCollector` |
