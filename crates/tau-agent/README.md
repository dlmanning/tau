# tau-agent

Actor-based agent runtime with tool execution, context compaction, and
subagent management. Clean-room rewrite of `tau-agent`.

## What this is

A library for building agents that talk to LLM providers (Anthropic,
OpenAI, Google, Ollama), execute tools, manage long conversations, and
spawn subagents for parallel work. The runtime is provider-agnostic;
all I/O goes through a `Transport` trait, and tools are user-defined.

The agent runs on a background tokio task; consumers interact with it
through a `Clone + Send + Sync` handle. Events stream out on a
broadcast channel.

## Architecture

Strict three-tier layering, top to bottom:

```text
fleet/    — multi-agent management (registry, lifecycle, bus)
  ↓
core/     — single-agent actor: handle, state, transitions, I/O
  ↓
types/    — leaf data (events, errors, conversation)
```

Lower tiers do not import from higher tiers. `core/` knows nothing
about subagents. `fleet/` composes `core` agents.

### State split

The actor's state is split into three structs by mutation discipline:

```rust
struct Frame {  // wiring: tools, transport, schema cache, policies
    /* Read-only inside transitions. Mutated only by the actor's
       command handler in response to SetModel / SetReasoning /
       SetCompactionConfig / SetApprovalPolicy. */
}

struct Conv {   // mutable per-turn state
    conversation: Conversation,
    steering_queue: Vec<Message>,
    follow_up_queue: Vec<Message>,
    cwd: Option<PathBuf>,
}

struct Shared { // atomics shared with the handle
    is_running: Arc<AtomicBool>,
    pending_follow_ups: Arc<AtomicU32>,
    cancel: Arc<Mutex<CancellationToken>>,
    agent_id: Arc<OnceLock<String>>,
    /* … */
}
```

### `decide_*` / `apply_*` discipline

`core::transitions` is purely synchronous, no I/O. Function names
encode whether a function decides or mutates:

- `decide_*` — pure decisions, take `&Frame, &Conv`, return an
  action enum.
- `apply_*` — state transitions, take `&Frame, &mut Conv`, mutate.
  Never re-decide.
- `build_*` and other plain helpers — pure utilities.

The actor reads a `decide_*` result, performs any I/O the result
demands, then calls one or more `apply_*` to commit.

### Phase machine

Nested. Outer `Phase` is four variants; sub-machines for the prompt
body, tool batches, queue drains, and compaction live in their own
enums:

```text
Phase
├── Idle
├── Turn(Turn { first_user_message, sub: TurnSub })
│   ├── Prepare { pending }
│   ├── AwaitingModel { stream, pending }
│   ├── Processing { outcome, pending }
│   ├── Tool(ToolPhase)
│   │   ├── AwaitingApproval { tool_calls, groups, pending_gates, … }
│   │   ├── Executing { join_set, remaining_groups, … }
│   │   └── Applying { tool_calls, results_map }
│   └── Drain(DrainPhase)
│       ├── CheckQueues
│       └── WaitingForBackground
├── Compaction(CompactionTrigger)
│   ├── Manual { reply }
│   └── Overflow { resume_pending }
└── Done(Result)
```

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
tau-agent = { path = "../path/to/tau-agent" }
tokio = { version = "1", features = ["full"] }
```

### Minimal example

```rust
use std::sync::Arc;
use tau_agent::{AgentBuilder, AgentConfig, ProviderTransport};

#[tokio::main]
async fn main() -> tau_agent::Result<()> {
    let config = AgentConfig {
        system_prompt: Some("You are a helpful assistant.".into()),
        /* model, reasoning, compaction, etc. — see config.rs */
        .. /* fields */
    };

    let transport = Arc::new(ProviderTransport::new());
    let handle = AgentBuilder::new(config, transport).spawn();

    // Send a prompt and wait for completion.
    handle.prompt_and_wait("What's the capital of France?").await?;

    // Read back the conversation.
    if let Some(state) = handle.state().await {
        for msg in &state.messages {
            println!("[{}] {}", msg.role(), msg.text());
        }
    }
    Ok(())
}
```

### Subscribing to events

The handle exposes a broadcast subscription. Events fire for every
agent lifecycle moment — prompts, tool calls, message deltas,
compaction, errors.

```rust
let handle = AgentBuilder::new(config, transport).spawn();
let mut events = handle.subscribe();

handle.prompt("hello").await?;

while let Ok(event) = events.recv().await {
    use tau_agent::AgentEvent;
    match event {
        AgentEvent::MessageEnd { message } => println!("→ {}", message.text()),
        AgentEvent::ToolExecutionStart { tool_name, activity, .. } => {
            println!("  ↳ {tool_name}: {activity}");
        }
        AgentEvent::AgentEnd { total_turns, total_usage } => {
            println!("[done in {total_turns} turns, {} in / {} out tokens]",
                     total_usage.input, total_usage.output);
            break;
        }
        _ => {}
    }
}
```

### Custom tools

Implement the `Tool` trait. Each tool gets an `ExecutionContext` with
the agent's id, the cancellation token, a progress sender, the file
access tracker, and the interaction channel (if the host wired one):

```rust
use async_trait::async_trait;
use serde_json::Value;
use tau_agent::{ExecutionContext, Tool, ToolResult};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echo the text argument back." }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        })
    }
    async fn execute(&self, args: Value, _ctx: ExecutionContext) -> ToolResult {
        let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
        ToolResult::text(text)
    }
}

// then:
let mut builder = AgentBuilder::new(config, transport);
builder.add_tool(Arc::new(EchoTool));
let handle = builder.spawn();
```

The `ExecutionContext::agent_id` field carries the owning agent's
id (when the host or fleet has stamped one). Tools that need to
identify their caller — e.g. a recursive-spawn tool that needs the
parent handle — read this rather than capturing state at construction.

### Approval policy

Every tool call is classified by an `ApprovalPolicy` before it runs:

```rust
use tau_agent::{AutoAcceptAll, DefaultPolicy, RulePolicy, ToolRule};

// "Auto" everything (CI / scripts).
builder.set_approval_policy(Arc::new(AutoAcceptAll));

// "Gate" tools whose risk is Elevated (Bash, network posts).
builder.set_approval_policy(Arc::new(DefaultPolicy));

// Per-tool / per-argument rules with a fallback.
let policy = RulePolicy::new(Arc::new(DefaultPolicy))
    .allow(ToolRule { tool: "bash".into(), arg_substrings: vec!["git status".into()] })
    .deny(ToolRule { tool: "bash".into(), arg_substrings: vec!["rm -rf".into()] });
builder.set_approval_policy(Arc::new(policy));
```

When the policy returns `Gate`, the runtime emits a `Typed
{ schema_id: "tool.confirm" }` interaction request and waits for the
host to approve, reject, or cancel. Configure the interaction sender
with `builder.set_interaction_sender(mpsc_tx)` to receive these.

### Mutating an agent after spawn

The handle's mutators take `&self` and queue commands to the actor:

```rust
handle.set_model(new_model).await?;
handle.set_reasoning(ReasoningLevel::High).await?;
handle.set_approval_policy(Arc::new(AutoAcceptAll)).await?;
handle.set_compaction_config(CompactionConfig::default()).await?;

// Steer mid-prompt (interrupts between tool batches).
handle.steer(Message::user("actually, focus on X")).await?;
```

To change the *spec* (tools or system prompt), spawn a new agent. The
runtime treats `AgentSpec` as immutable for the agent's lifetime.

### Multi-agent: `AgentManager`

For hosts that need to spawn subagents, `AgentManager` is a thin
composition of a registry, a lifecycle, and an event bus:

```rust
use tau_agent::{AgentManager, AgentSpec, SpawnOpts};

let (event_tx, _) = tokio::sync::broadcast::channel(256);
let manager = Arc::new(AgentManager::new(event_tx, config, transport, /* max_agents */ 20));

let spec = AgentSpec {
    system_prompt: "You are a focused research agent.".into(),
    tools: vec![Arc::new(EchoTool)],
    max_turns: 50,
    allows_worktree: false,
    allowed_subagent_specs: None,
};

let result = manager
    .spawn(spec, "find recent commits".into(), SpawnOpts::default(), CancellationToken::new())
    .await?;
println!("subagent says: {}", result.text);

// Later, resume:
let resumed = manager
    .send(&result.agent_id, "anything else?", CancellationToken::new())
    .await?;

// Or transition to a new spec (e.g. swap toolset):
let new_handle = manager.respec(&result.agent_id, new_spec).await?;
```

`AgentManager` provides:

- `spawn` — foreground subagent; blocks until completion.
- `spawn_background` — fire-and-forget; posts a follow-up to the
  parent handle on completion.
- `spawn_interactive` — caller drives the returned handle.
- `send` — resume a stored agent with a follow-up prompt.
- `respec` — atomic spec transition: detach the agent (verifying
  it's idle), spawn a new one inheriting its history, drop the old
  spec. Rolls back on failure.
- `adopt` — register an externally-built handle (e.g. the host's
  root agent) so `spec_for` / `respec` work.
- `find_agent` / `handle_for` — registry lookups.

### Registry invariant

The fleet's `Registry` maintains a single load-bearing invariant:

> A spec exists in the registry **iff** its id is in idle ∪ running ∪ adopted.

This is enforced by the registry's method surface — every mutation
goes through a method that preserves the invariant. Five registry
unit tests pin the edges. The `respec` flow uses an atomic
verify-and-detach to close the race where a concurrent `send` could
slip in between the "not running" check and the spec drop.

## Testing

The `test-utils` feature exports mock transports, fake tools, and an
`EventCollector` for deterministic async testing:

```toml
[dev-dependencies]
tau-agent = { path = "...", features = ["test-utils"] }
```

```rust
use tau_agent::test_utils::*;

let transport = MockTransport::new().with_text_response("Hello!");
let (handle, collector) = spawn_test_agent(transport, vec![]);

handle.prompt_and_wait("hi").await.unwrap();
collector.wait_for_end().await;

assert_eq!(collector.assistant_messages()[0].text(), "Hello!");
```

Provided fixtures: `MockTransport`, `TextTransport`, `ToolCallTransport`,
`SlowTransport`, `CapturingTransport`, `ErrorTransport`, `PanicTransport`,
`EchoTool`, `FailTool`, `SlowTool`, `PanicTool`, `EventCollector`,
`spawn_test_agent`.

## Module map

| Module | Role |
|---|---|
| `types::events` | `AgentEvent` enum + `ConsoleLine` / `ToolApprovalOutcome` / `SubagentOutcome` payloads |
| `types::error` | `Error` + `Result` |
| `types::conversation` | `Conversation` (messages, usage, streaming, previous summary) |
| `core::actor` | `run_actor`, the phase machine, the command/event loop |
| `core::approval` | `ApprovalPolicy` trait + built-in policies |
| `core::builder` | `AgentBuilder` setup → spawn |
| `core::command` | `Command` (handle ↔ actor protocol) |
| `core::compaction` | Cut-point finding, summarization, `apply_compaction_result` |
| `core::config` | `AgentConfig`, `DequeueMode` |
| `core::handle` | `AgentHandle` — `Clone + Send + Sync` API surface |
| `core::interaction` | Tool ↔ UI round-trip protocol |
| `core::overflow` | Context-overflow regex detection across providers |
| `core::state` | `Frame` / `Conv` / `Shared` / `State` |
| `core::stream` | `StreamReducer` — aggregate transport events into a turn outcome |
| `core::tool` | `Tool` trait, `ExecutionContext`, `ToolResult`, `FileAccessTracker` |
| `core::transitions` | Sync decision + apply functions (no I/O) |
| `core::transport` | `Transport` trait + `ProviderTransport` |
| `fleet::bus` | Child→parent event forwarding, interaction routing |
| `fleet::lifecycle` | `spawn` / `send` / `respec` / `adopt` / `spawn_background` |
| `fleet::manager` | `AgentManager` composition root + `AgentSpec` / `SpawnOpts` |
| `fleet::registry` | Spec / idle / running / adopted maps with invariant baked in |
| `fleet::result` | `SubagentResult` |
| `fleet::transcript` | JSONL transcript recording |
| `fleet::worktree` | Git worktree isolation for subagents |

## Differences from `tau-agent`

If you're coming from `tau-agent` (v1):

- **No `bind_to_agent` hook** on `Tool`. Tools read `ctx.agent_id`
  instead of capturing a handle at construction time.
- **`AgentBuilder::handle()`** replaces `pre_handle()`.
- **`AgentHandle::respec_with(|spec| { ... })`** replaces the
  `with_system_prompt` / `with_tools` convenience methods.
- **`respec` lives on `AgentManager` only.** Use
  `manager.respec(handle.agent_id().unwrap(), new_spec)` instead of
  `handle.respec(spec)`.
- **No `prompts/` module.** v1 shipped opinionated system-prompt
  fragments inside the runtime crate; v2 leaves prompt assembly to
  the host.
- **No `context::load_context`.** Same reason — host concern.
- **`AutoAcceptAllPolicy` → `AutoAcceptAll`**; `DefaultApprovalPolicy`
  → `DefaultPolicy`.
- **`Registry` invariant** is method-enforced, not doc-enforced.
- **State split** (`Frame` / `Conv` / `Shared`) replaces v1's flat
  `AgentState` god struct.

## Test count

`cargo test -p tau-agent` runs 116 tests:

- 25 unit (approval, overflow, stream, transitions, compaction, registry)
- 91 integration across 14 files

All passing.
