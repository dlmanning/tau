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
| `AgentHandle` | `handle` | Clone + Send + Sync handle. All interaction goes through this. |
| `AgentConfig` | `config` | Model, reasoning level, max turns, compaction settings. |
| `Tool` | `tool` | Trait for tool implementations: name, schema, concurrency, execute. |
| `AgentEvent` | `events` | Broadcast event enum (~21 variants): lifecycle, messages, tools, compaction, errors, subagents, file changes, deferred conversation ops. |
| `InteractionRequest` | `interaction` | Tool → host UI round-trip. Two shapes: `AskQuestion` (untyped, returns `Answer`) and `Typed { schema_id, payload }` (host renders by `schema_id`, replies with `Approved { payload }` / `Rejected` / `Cancelled`). The runtime treats `payload` as opaque JSON; schemas (e.g. `plan.submit`) and their typed payload structs live with the tools that emit them (e.g. `tau_tools::Plan`). |
| `ApprovalPolicy` | `approval` | Trait that classifies a pending tool call into `Auto` / `Gate` / `Reject`. Built-in policies: `DefaultApprovalPolicy`, `AutoAcceptAllPolicy`, `RulePolicy`. |
| `AgentManager` | `manager` | Subagent lifecycle: spawn foreground/background, resume, evict. |
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

- **Foreground** — blocks the parent until completion.
- **Background** — returns immediately; posts a `FollowUp` message to the parent on completion.
- **Resume** — rehydrate a stored idle agent with a new prompt via `send()`.

Agent types: `GeneralPurpose`, `Explore` (read-only tools), `Plan` (read-only tools). Max nesting depth: 3. Max turns per subagent: 200.

**Plan execution** uses the recursive-agent pattern: a `Plan` subagent investigates and submits a plan via the `plan.submit` typed interaction; the host approves (optionally editing the body); a `GeneralPurpose` subagent is spawned with `SpawnRequest::inherit_history_from = <planner_agent_id>` so it sees the planner's investigation and the approved plan as its own conversation history. There is no separate plan-execution mechanism in the runtime.

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
| `error` | pub | Error types |
| `events` | pub | `AgentEvent` enum (broadcast events) |
| `handle` | pub | `AgentHandle` — public API |
| `interaction` | pub | Tool ↔ UI request/reply protocol (`AskQuestion`, `Typed { schema_id, payload }`) |
| `logic` | crate | Sync decision logic (no I/O) |
| `manager` | pub | `AgentManager` — subagent lifecycle |
| `overflow` | crate | Context overflow detection (30+ provider patterns) |
| `prompts` | pub | System prompt assembly |
| `state` | crate | `AgentState`, `ToolCall` data types |
| `stream` | pub | `StreamReducer` — event stream aggregation |
| `tool` | pub | `Tool` trait, `ExecutionContext`, `ToolResult` |
| `tool_executor` | crate | Single tool execution harness |
| `transcript` | pub | Debug JSONL logging for subagent conversations |
| `transport` | pub | `Transport` trait, `ProviderTransport` |
| `worktree` | crate | Git worktree isolation for subagents |
| `test_utils` | pub (feature-gated) | Mocks, test tools, `EventCollector` |
