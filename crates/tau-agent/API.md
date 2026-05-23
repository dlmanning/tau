# tau-agent API Reference

A practical, example-driven guide to the public surface of `tau-agent`.
Pair with `README.md` (architecture) and the integration tests under
`tests/` (live examples).

## Getting started

Copy-paste flow for a first agent — spawn, subscribe, prompt, drain
events, shut down:

```rust
use std::sync::Arc;
use tau_agent::{AgentBuilder, AgentConfig, AgentEvent, ProviderTransport};

#[tokio::main]
async fn main() -> tau_agent::Result<()> {
    // 1. Configure.
    let model = tau_ai::get_model_by_id("claude-opus-4-7").expect("model exists");
    let config = AgentConfig::builder(model)
        .system_prompt("You are a helpful assistant.")
        .build();
    let transport = Arc::new(ProviderTransport::new());

    // 2. Build, subscribe before spawn (catches AgentStart),
    //    then spawn the actor task. spawn() awaits the actor's
    //    readiness signal, so the handle is live at the point of
    //    return. (The actor can still die later — re-check via
    //    handle.health() if you need to confirm mid-flight.)
    let builder = AgentBuilder::new(config, transport);
    let mut events = builder.subscribe();
    let handle = builder.spawn().await?;

    // 3. Send a prompt and drain events until completion.
    handle.prompt("What's the capital of France?").await?;
    while let Ok(ev) = events.recv().await {
        match ev {
            AgentEvent::MessageEnd { message } => {
                println!("{}", message.text());
            }
            AgentEvent::AgentEnd { .. } => break,
            AgentEvent::Error { message } => {
                eprintln!("error: {message}");
                break;
            }
            _ => {}
        }
    }

    // 4. Optional: drop the handle to release the actor task. After
    //    AgentEnd the actor returns to Idle and you can prompt again
    //    instead of dropping.
    Ok(())
}
```

The 16 sections below are organized as a reference. If you want
worked patterns instead of subsystem-by-subsystem coverage, jump to
[§16 Common patterns](#16-common-patterns).

## Stability

`tau-agent` is pre-1.0; nothing is API-frozen. That said, the
following rough stability tiers reflect how often each subsystem
changes in practice:

| Tier         | Surface                                                                                                                                                                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Stable**   | `AgentBuilder` core methods (`new`, `add_tool`, `set_*`, `spawn`), `AgentHandle` (prompts, queries, lifecycle), `Tool` trait, `AgentEvent` core variants (`TurnStart`/`MessageEnd`/`ToolExecution*`/`TurnEnd`/`AgentEnd`), `Error` variants, `Result` alias. |
| **Settling** | `ApprovalPolicy` + `RulePolicy` shape, `InteractionRequest`/`InteractionResponse` and the `"tool.confirm"` schema, `CompactionConfig` fields, `AgentEvent::FileChanged`.                                                                                     |
| **Evolving** | `AgentManager` lifecycle methods, `AgentSpec` and `SpawnOpts` field set, snapshot shapes (`FleetSnapshot`/`AgentSnapshot`), `FleetEvent` variants, `AgentEvent::AgentReport` (tool-emitted, translated to `FleetEvent::AgentReport` by the bus).             |
| **Internal** | `core/`, `fleet/`, `types/` modules are crate-private; only the items re-exported at the crate root (`tau_agent::Foo`) are supported. Anything reachable via deeper paths in older snapshots is gone — `tau_agent::PromptResult` etc. now live at the root.  |

Treat anything in "Evolving" as a moving target across minor
versions. The fleet surface in particular still picks up methods and
fields as new host requirements surface.

Layout:

- [tau-agent API Reference](#tau-agent-api-reference)
  - [Getting started](#getting-started)
  - [Stability](#stability)
  - [1. Building \& spawning an agent](#1-building--spawning-an-agent)
    - [`AgentBuilder` methods](#agentbuilder-methods)
      - [Lesser-used builder methods](#lesser-used-builder-methods)
    - [`AgentConfig` fields](#agentconfig-fields)
      - [`AgentConfig` accessors](#agentconfig-accessors)
      - [`DequeueMode` semantics](#dequeuemode-semantics)
    - [Re-exported `tau_ai` types](#re-exported-tau_ai-types)
      - [`Model`](#model)
      - [`ReasoningLevel`](#reasoninglevel)
      - [`Usage`](#usage)
      - [`Message`](#message)
      - [`Content`](#content)
  - [2. Prompting](#2-prompting)
    - [Queueing helpers](#queueing-helpers)
  - [3. Events](#3-events)
    - [`AgentEvent` taxonomy (per-agent stream)](#agentevent-taxonomy-per-agent-stream)
    - [`FleetEvent` taxonomy (manager stream)](#fleetevent-taxonomy-manager-stream)
    - [Ordering guarantees](#ordering-guarantees)
  - [4. Tools](#4-tools)
    - [`Tool` defaults](#tool-defaults)
    - [`ToolResult`](#toolresult)
    - [`ExecutionContext`](#executioncontext)
    - [Concurrency \& categories](#concurrency--categories)
    - [`ToolRisk`](#toolrisk)
    - [`BoxedTool`](#boxedtool)
  - [5. Approval policies](#5-approval-policies)
    - [Built-in policies](#built-in-policies)
    - [Custom policies](#custom-policies)
    - [Approval edge cases](#approval-edge-cases)
  - [6. Interactions (UI round-trip)](#6-interactions-ui-round-trip)
    - [Built-in schema](#built-in-schema)
    - [Response types](#response-types)
    - [Round-trip and failure modes](#round-trip-and-failure-modes)
    - [Other interaction questions](#other-interaction-questions)
  - [7. Compaction](#7-compaction)
    - [Compaction details](#compaction-details)
  - [8. Mutating a running agent](#8-mutating-a-running-agent)
    - [What can't be changed mid-flight](#what-cant-be-changed-mid-flight)
  - [9. Inspecting an agent](#9-inspecting-an-agent)
    - [Lifecycle states](#lifecycle-states)
  - [10. Interrupt vs abort](#10-interrupt-vs-abort)
    - [Phase machine and safe boundaries](#phase-machine-and-safe-boundaries)
    - [Interrupt vs abort](#interrupt-vs-abort)
      - [Tool cancellation contract](#tool-cancellation-contract)
  - [11. Transport](#11-transport)
    - [Custom transports](#custom-transports)
    - [`AgentRunConfig`](#agentrunconfig)
    - [Transport authoring notes](#transport-authoring-notes)
    - [`ProviderTransport` provider selection](#providertransport-provider-selection)
  - [12. Fleet (multi-agent)](#12-fleet-multi-agent)
    - [Spawn modes](#spawn-modes)
    - [Resuming, mutating, and adopting](#resuming-mutating-and-adopting)
    - [`AgentSpec`](#agentspec)
    - [`SpawnOpts`](#spawnopts)
    - [`Isolation`](#isolation)
    - [`SubagentResult`](#subagentresult)
    - [Snapshots](#snapshots)
    - [Registry invariant](#registry-invariant)
    - [Fleet edge cases](#fleet-edge-cases)
  - [13. Errors](#13-errors)
    - [Return-type quick reference](#return-type-quick-reference)
    - [Recovery guidance](#recovery-guidance)
  - [14. Testing](#14-testing)
    - [Test-utils notes](#test-utils-notes)
  - [15. Type cheatsheet](#15-type-cheatsheet)
    - [Trait \& type quick reference](#trait--type-quick-reference)
  - [16. Common patterns](#16-common-patterns)
    - [A. Single-shot prompt with tool approval](#a-single-shot-prompt-with-tool-approval)
    - [B. Long-running agent with compaction](#b-long-running-agent-with-compaction)
    - [C. Parent spawning a research subagent](#c-parent-spawning-a-research-subagent)

---

## 1. Building & spawning an agent

`AgentBuilder` is the entry point. Configure, add tools, then `spawn()`
to get an `AgentHandle`.

```rust
use std::sync::Arc;
use tau_agent::{AgentBuilder, AgentConfig, ProviderTransport};

let model = tau_ai::get_model_by_id("claude-opus-4-7").expect("model exists");
let config = AgentConfig::builder(model)
    .system_prompt("You are a helpful assistant.")
    .build();

let transport = Arc::new(ProviderTransport::new());
let handle = AgentBuilder::new(config, transport).spawn().await?;
```

`spawn` is **async and fallible**: it awaits the actor's readiness
signal before returning. On success the handle is live **at the
point of return** (the actor task can still die later — query
`health()` if you need to confirm liveness mid-flight). On
`Err(Error::ActorPanic)` the actor died during startup and there
is no half-alive handle to clean up. (Schema-compilation failures
for tool argument schemas are logged via `tracing::warn!` and
non-fatal — the tool is registered without argument validation.)

### `AgentBuilder` methods

| Method                                                               | Purpose                                                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `new(config, transport)`                                             | Default channel capacities                                                                       |
| `with_channel_capacities(config, transport, urgent_cap, normal_cap)` | Custom backpressure                                                                              |
| `add_tool(tool)`                                                     | Register a single tool                                                                           |
| `set_tools(vec)`                                                     | Replace tool set                                                                                 |
| `add_server_tool(tool)`                                              | Register a `tau_ai::ServerTool` (provider-hosted)                                                |
| `set_system_prompt(s)`                                               | Override `config.system_prompt`                                                                  |
| `set_interaction_sender(tx)`                                         | Receive `InteractionRequest`s from tools                                                         |
| `set_interaction_timeout(d)` / `clear_interaction_timeout()`         | Deadline (or none) for runtime-side `tool.confirm` gates; see §6                                 |
| `set_approval_policy(policy)`                                        | Override the default approval policy                                                             |
| `set_cwd(path)`                                                      | Working directory the agent reports to tools                                                     |
| `seed(AgentSeed)`                                                    | Load history (messages + optional compaction tail). See §7.                                      |
| `set_subagent_depth(n)`                                              | Tree depth (0 for root)                                                                          |
| `set_transform_context(fn)`                                          | Install a hook that rewrites context before each model call                                      |
| `config()` / `tools()` / `tool_names()`                              | Read-only accessors for fleet setup                                                              |
| `event_sender()`                                                     | Get the broadcast sender before `spawn`                                                          |
| `subscribe()`                                                        | Get a `broadcast::Receiver<AgentEvent>` before `spawn` (the idiomatic way to catch `AgentStart`) |
| `handle()`                                                           | Get a fully-wired handle **before** `spawn` (mainly for the fleet manager)                       |
| `spawn().await`                                                      | Consume, start the actor task, await the actor's readiness signal, return `Result<AgentHandle>`  |

`AgentBuilder::subscribe()` is the idiomatic way to grab an event
receiver before the actor starts — it mirrors
`broadcast::Sender::subscribe` and guarantees you'll see
`AgentStart`. `handle()` is also available for the rare case where
you need the full handle pre-spawn (the fleet manager uses it to
stamp `agent_id` before consuming the builder); the returned handle
is interchangeable with the one `spawn()` produces.

#### Lesser-used builder methods

- **`set_transform_context(fn)`** — installs a hook with signature
  `Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>`. The
  runtime calls it on every model call, **after** the conversation
  history + queued steers/follow-ups are assembled and **before** the
  result is handed to the transport. Use it to inject ephemeral
  context (date, cwd, env state) without polluting the stored
  conversation. The function must be pure (no I/O) — it runs in the
  actor's sync transition path.

- **`set_subagent_depth(n)`** — stamps the depth value tools see in
  `ctx.subagent_depth`. There is no validation: setting `n` on a root
  agent is legal and just makes the root think it's at depth `n`.
  When the fleet spawns a child, it sets `SpawnOpts.subagent_depth`
  on the new builder, overriding any value the spec carried.

- **`event_sender()`** — returns a clone of the broadcast
  `Sender<AgentEvent>` the actor will use to publish events. Useful
  when you want to **inject** events from another task. For just
  *receiving* events, use `builder.subscribe()` — same channel,
  returns a `Receiver`.

- **`config()` / `tools()` / `tool_names()`** — read-only accessors
  used by code that builds *around* an agent. Most callers don't
  need them. The fleet uses them at spawn time to derive a
  per-subagent `AgentRunConfig` from the parent's config and the
  spec's tool list.

### `AgentConfig` fields

Construct an `AgentConfig` via the builder:

```rust
use tau_agent::{AgentConfig, DequeueMode};
use tau_ai::{get_model_by_id, ReasoningLevel};

let opus = get_model_by_id("claude-opus-4-7").expect("model exists");
let config = AgentConfig::builder(opus)
    .system_prompt("You are a helpful assistant.")
    .reasoning(ReasoningLevel::Medium)
    .max_turns(200)
    .follow_up_mode(DequeueMode::All)
    .build();
```

The builder is the only way to construct an `AgentConfig` from outside
the crate; the fields are private. Use [accessor methods](#agentconfig-accessors)
to read fields off an existing config.

`AgentConfigBuilder` methods (all `self -> Self`, defaults shown):

| Method                                      | Default                       | Notes                                                           |
| ------------------------------------------- | ----------------------------- | --------------------------------------------------------------- |
| `system_prompt(impl Into<String>)`          | `None`                        | Prepended to every model call                                   |
| `model(Model)`                              | (required at `builder`)       | Replaces the model passed to `builder(_)`                       |
| `reasoning(ReasoningLevel)`                 | `ReasoningLevel::Off`         | Extended-thinking budget; provider-specific                     |
| `thinking_adaptive(bool)`                   | `false`                       | **Anthropic-only**: model picks its own thinking budget         |
| `max_tokens(u32)`                           | `None`                        | Max output tokens per model call                                |
| `max_turns(u32)`                            | `None` (unlimited)            | Final summary turn still flushes when limit hits mid-tool-batch |
| `compaction(CompactionConfig)`              | `CompactionConfig::default()` | See §7                                                          |
| `steering_mode(DequeueMode)`                | `DequeueMode::All`            | Urgent-queue drain mode                                         |
| `follow_up_mode(DequeueMode)`               | `DequeueMode::All`            | Follow-up-queue drain mode                                      |
| `cache_scope(impl Into<String>)`            | `None`                        | **Anthropic-only** prompt-cache scope (`"global"` / `"org"`)    |
| `cache_ttl(impl Into<String>)`              | `None`                        | **Anthropic-only** TTL (e.g. `"5m"`, `"1h"`)                    |
| `system_prompt_boundary(impl Into<String>)` | `None`                        | **Anthropic-only** split marker for prompt caching (see below)  |
| `build() -> AgentConfig`                    | —                             | Consume the builder                                             |

`system_prompt_boundary` placement rules: split is by `str::find`, so the
**first occurrence wins** if the marker repeats. Choose a string that (a)
appears exactly once in the prompt, (b) cannot occur in any dynamic
substring you append after assembly, and (c) is distinctive enough not
to collide with model output (e.g. `"\n<!-- cache-boundary -->\n"` rather
than `"---"`). If the marker isn't found, the whole prompt is treated as
a single cacheable block.

#### `AgentConfig` accessors

Each builder field has a matching reader (returns `&T` for non-`Copy`
types, `T` for `Copy` types, `Option<&str>` for `Option<String>`):

```rust
config.system_prompt()             -> Option<&str>
config.model()                     -> &Model
config.reasoning()                 -> ReasoningLevel
config.thinking_adaptive()         -> bool
config.max_tokens()                -> Option<u32>
config.max_turns()                 -> Option<u32>
config.compaction()                -> &CompactionConfig
config.steering_mode()             -> DequeueMode
config.follow_up_mode()            -> DequeueMode
config.cache_scope()               -> Option<&str>
config.cache_ttl()                 -> Option<&str>
config.system_prompt_boundary()    -> Option<&str>
```

To tweak an existing config, use [`AgentConfig::into_builder`]:

```rust
let tweaked = base_config.into_builder().max_turns(50).build();
```

#### `DequeueMode` semantics

Both `steering_mode` and `follow_up_mode` control how the actor pulls
from its respective queue at the next turn boundary:

- `DequeueMode::All` — drain **everything** queued into one combined
  user message before the next model call. If three steers are queued,
  the agent sees all three at once.
- `DequeueMode::OneAtATime` — pull a single message per turn and leave
  the rest queued. The agent responds to each in sequence, with the
  remainder still waiting.

Pick `All` when steers are corrections to the same goal and should be
combined. Pick `OneAtATime` when each steer is a distinct sub-task you
want the agent to acknowledge individually.

### Re-exported `tau_ai` types

Several `AgentConfig` fields and `AgentHandle` methods take types that
live in `tau_ai`. The runtime re-uses them as-is. The key ones:

#### `Model`

A runtime struct (not an enum) describing one LLM:

```rust
pub struct Model {
    pub id: String,             // e.g. "claude-opus-4-7"
    pub name: String,           // human-readable
    pub api: Api,
    pub provider: Provider,     // Anthropic | OpenAI | Google | Ollama | Custom
    pub base_url: String,
    pub reasoning: bool,        // does it support extended thinking?
    pub input_types: Vec<InputType>,
    pub cost: CostInfo,
    pub context_window: u32,    // used by compaction
    pub max_tokens: u32,
    pub headers: HashMap<String, String>,
}
```

Build one via `tau_ai::get_model_by_id("claude-opus-4-7")` or
`tau_ai::get_model(Provider::Anthropic, "claude-opus-4-7")` — both
return `Option<Model>`. `Model` is a struct, not an enum — there are
no `Model::Foo` variants. In examples below we hold the lookup
result in a local (`let opus = get_model_by_id("claude-opus-4-7").unwrap();`)
and pass it by value.

#### `ReasoningLevel`

```rust
pub enum ReasoningLevel { Off, Minimal, Low, Medium, High }
```

Defaults to `Off`. Provider mapping (Anthropic, today):

| Level     | Thinking-token budget |
| --------- | --------------------- |
| `Off`     | thinking disabled     |
| `Minimal` | 1 024                 |
| `Low`     | 4 096                 |
| `Medium`  | 10 000                |
| `High`    | 32 000                |

When `AgentConfig.thinking_adaptive = true`, the budget number is
ignored and the model decides per call.

#### `Usage`

Cumulative token counters carried on `AgentEvent::TurnEnd`,
`AgentEvent::AgentEnd`, and `Conversation.total_usage`:

```rust
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub thinking: u64,            // extended-thinking tokens
    pub cache_creation_1h: u64,
    pub cache_creation_5m: u64,
    pub service_tier: Option<String>,
}
```

Compute cost with `usage.calculate_cost(&model)`.

#### `Message`

The conversation primitive. Tagged on `role`:

```rust
pub enum Message {
    User { content: Vec<Content>, timestamp: i64 },
    Assistant { content: Vec<Content>, metadata: AssistantMetadata },
    ToolResult { tool_call_id, tool_name, content: Vec<Content>, is_error, timestamp },
    SystemInjection { content: Vec<Content>, source: InjectionSource },
}
```

Constructors you'll use most often:

```rust
Message::user("hello")
Message::user_with_content(vec![Content::text("…"), Content::image_url("…")])
Message::tool_result(call_id, tool_name, vec![Content::text("…")], is_error)
Message::subagent_completed(agent_id, description, text)  // SystemInjection
Message::subagent_failed(agent_id, description, error)    // SystemInjection
```

`SystemInjection` messages are not from the user or the model — the
runtime converts them to user-role messages before sending to LLM
APIs. The fleet's background-spawn flow uses these to deliver
subagent results to the parent.

#### `Content`

Building block for `Message.content`. The shapes you'll touch from a host:

```rust
Content::text("hi")
Content::image_url("https://…")
Content::ToolCall { id, name, arguments }   // emitted by assistant
```

---

## 2. Prompting

Two flavors, depending on how much you care about completion:

```rust
// Fire-and-forget; returns a oneshot you can await for the PromptResult.
let rx = handle.prompt("hello").await?;

// Convenience: send + block until the agent reaches Idle.
handle.prompt_and_wait("What's 2+2?").await?;
```

If a prompt is already in flight, a second `prompt()` returns
`Error::Busy`. Use `steer()` to inject a message mid-prompt instead.

### Queueing helpers

```rust
use tau_ai::Message;

// Steer: urgent. Delivered at the next safe phase boundary (between
// tool batches or before the next model call).
handle.steer(Message::user("focus on X instead")).await?;

// Follow-up: normal-priority continuation queued for after the current
// turn completes.
handle.follow_up(Message::user("also do Y")).await?;

// Non-blocking variants (returns Error::ChannelFull on backpressure):
handle.try_steer(Message::user("…"))?;
handle.try_follow_up(Message::user("…"))?;
```

---

## 3. Events

Two streams: an **`AgentEvent`** stream per agent (subscribe via
`handle.subscribe()` or `builder.subscribe()`), and a **`FleetEvent`**
stream per `AgentManager` (subscribe via `manager.subscribe()`). The
per-agent stream carries only that agent's own activity; the fleet
stream aggregates lifecycle events and forwarded child events across
every agent the manager tracks.

```rust
let mut events = handle.subscribe();
handle.prompt("hi").await?;

use tau_agent::AgentEvent;
while let Ok(ev) = events.recv().await {
    match ev {
        AgentEvent::MessageEnd { message } => println!("→ {}", message.text()),
        AgentEvent::ToolExecutionStart { tool_name, activity, .. } => {
            println!("  ↳ {tool_name}: {activity}");
        }
        AgentEvent::AgentEnd { total_turns, total_usage, interrupted } => {
            println!("done in {total_turns} turns (interrupted={interrupted})");
            break;
        }
        _ => {}
    }
}
```

### `AgentEvent` taxonomy (per-agent stream)

| Variant                                                               | When                                  |
| --------------------------------------------------------------------- | ------------------------------------- |
| `AgentStart`                                                          | Agent actor task started              |
| `TurnStart { turn_number }`                                           | Beginning of a model call             |
| `MessageStart { message }`                                            | Streaming response began              |
| `MessageUpdate { message }`                                           | Incremental message content           |
| `MessageEnd { message }`                                              | Assistant turn complete               |
| `ToolExecutionStart { tool_call_id, tool_name, arguments, activity }` | Tool invoked                          |
| `ToolExecutionUpdate { tool_call_id, tool_name, lines }`              | Tool progress (from `ProgressSender`) |
| `ToolExecutionEnd { tool_call_id, tool_name, result, is_error }`      | Tool finished                         |
| `ToolApprovalResolved { tool_call_id, tool_name, outcome }`           | Approval gate resolved                |
| `TurnEnd { turn_number, message, usage }`                             | Turn complete (post-tools)            |
| `AgentEnd { total_turns, total_usage, interrupted }`                  | Agent idled / stopped                 |
| `CompactionStart { reason }`                                          | Summarization began                   |
| `CompactionEnd { tokens_before, tokens_after }`                       | Summarization done                    |
| `Error { message }`                                                   | Unrecoverable error                   |
| `FileChanged { path, before, after, tool_call_id }`                   | Tool wrote a file                     |
| `AgentReport { tag, summary }`                                        | Tool self-labels this agent's outcome |

Helpers: `event.is_terminal()` is true for `AgentEnd` and `Error`.

### `FleetEvent` taxonomy (manager stream)

Subscribe via `AgentManager::subscribe() -> Receiver<FleetEvent>`.

| Variant                                                                                                                                           | When                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `AgentStarted { agent_id, spec_name, description, prompt, started_at }`                                                                           | Manager dispatched a spawn                                         |
| `AgentResumed { agent_id, description, prompt, resumed_at }`                                                                                      | Manager resumed an idle agent                                      |
| `AgentCompleted { agent_id, description, outcome, started_at, completed_at, duration_ms, usage, tool_use_count, worktree_path, worktree_branch }` | Agent terminated (`SubagentOutcome::Completed`/`Aborted`/`Failed`) |
| `AgentReport { agent_id, description, tag, summary }`                                                                                             | Translated from a tracked agent's `AgentEvent::AgentReport`        |
| `Forwarded { agent_id, description, event: AgentEvent }`                                                                                          | Every other event from a tracked agent, stamped with its id        |

`Forwarded::event` is a flat `AgentEvent` — never another `FleetEvent`,
so depth-2 nesting is structurally impossible. A grandchild's events
arrive on the same fleet channel with the grandchild's `agent_id`.

`ToolApprovalOutcome` (event payload): `AutoApproved | Approved |
Rejected { reason }`.

`ConsoleLine`:

```rust
pub struct ConsoleLine {
    pub content: String,
    pub level: ConsoleLevel,
}

pub enum ConsoleLevel {
    Muted,    // de-emphasized (e.g. progress noise)
    Normal,   // default
    Warning,  // attention but not error
    Success,  // completed step
    Danger,   // failed step / dangerous state
}
```

Constructors: `ConsoleLine::new(text, level)` and the shorthand
`ConsoleLine::normal(text)`.

`SubagentOutcome` (carried in `FleetEvent::AgentCompleted`):

```rust
pub enum SubagentOutcome {
    Completed,
    Aborted { reason: String },
    Failed { reason: String },
}
```

### Ordering guarantees

Each stream is ordered independently in the order the producer writes
events. From a subscriber's perspective:

- **`AgentEvent` stream**:
  - `AgentStart` is the first event on a fresh agent.
  - `AgentEnd` and `Error` are terminal — no further events for that
    agent will arrive after one of them. (`is_terminal()` tests this.)
  - For each turn: `TurnStart` precedes its `MessageStart` /
    `MessageUpdate*` / `MessageEnd`; `MessageEnd` precedes any tool
    events from that turn; `TurnEnd` follows the last
    `ToolExecutionEnd` of the turn.
  - For each tool call: `ToolApprovalResolved` (if it was gated) →
    `ToolExecutionStart` → `ToolExecutionUpdate*` → `ToolExecutionEnd`,
    always in that order **for the same `tool_call_id`**. Tools in a
    parallel batch interleave with each other arbitrarily, but each
    tool's own four-event sequence is strict.
  - `CompactionStart` always pairs with a later `CompactionEnd` (or an
    `Error` if compaction itself failed).
  - `FileChanged` is emitted by tools — its position relative to
    `ToolExecutionUpdate` for the same `tool_call_id` is whatever
    order the tool chose to emit them.

- **`FleetEvent` stream**:
  - `AgentStarted` / `AgentResumed` for a given `agent_id` is followed
    by zero or more `Forwarded` events for that id, then an
    `AgentCompleted` — but events from concurrent agents interleave
    arbitrarily, so do not assume a contiguous block per agent.
  - `AgentReport` is emitted only when a tool inside the agent calls
    `ctx.progress.emit(AgentEvent::AgentReport { … })`. It is *not*
    additionally forwarded as a `Forwarded { event: AgentReport }` — the
    bus picks one form.

Both channels have a fixed capacity (256 by default; override the
fleet channel via `AgentManager::with_event_capacity`). Slow
subscribers see `RecvError::Lagged(skipped)`; events between are
dropped for that subscriber but still delivered to others. Build
state machines that tolerate lag, or subscribe before
`spawn()` / before construction and keep the receiver drained.

---

## 4. Tools

Implement `Tool`. The trait is small — only `name`, `description`,
`parameters_schema`, and `execute` are required:

```rust
use async_trait::async_trait;
use serde_json::Value;
use tau_agent::{ExecutionContext, Tool, ToolCategory, ToolResult};

struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str { "read_file" }
    fn description(&self) -> &str { "Read a UTF-8 file from the local filesystem." }
    fn category(&self) -> ToolCategory { ToolCategory::Read }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
        })
    }

    fn activity_description(&self, args: &Value) -> String {
        match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => format!("Reading {p}"),
            None => "Reading file".into(),
        }
    }

    async fn execute(&self, args: Value, ctx: ExecutionContext) -> ToolResult {
        let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let abs = ctx.resolve_path(path_str);

        match tokio::fs::read_to_string(&abs).await {
            Ok(text) => {
                ctx.mark_read(&abs);
                ToolResult::text(text)
            }
            Err(e) => ToolResult::error(format!("read failed: {e}")),
        }
    }
}
```

### `Tool` defaults

If you don't override them, you get sensible defaults:

| Method                    | Default                 |
| ------------------------- | ----------------------- |
| `label()`                 | Same as `name()`        |
| `concurrency()`           | `Concurrency::Parallel` |
| `activity_description(_)` | `"Running {name}"`      |
| `risk(_)`                 | `ToolRisk::Local`       |
| `category()`              | `ToolCategory::Other`   |

### `ToolResult`

```rust
ToolResult::text("ok");                      // text success
ToolResult::error("oops");                   // error
ToolResult::with_content(vec![Content::…]);  // structured success
ToolResult::text("ok").with_details(json!({"bytes": 42}));  // attach metadata
```

`ToolResult::error(msg)` causes `ToolExecutionEnd.is_error = true`. The
agent sees the error message and can recover.

`with_details(value)` attaches metadata to the result. **The model
never sees `details`** — only the `content` blocks are fed back into
the conversation. `details` surfaces only in serialized form
(transcripts, snapshots, host UIs that introspect tool results). Use
it for structured side-channel information the host UI should
render — diff summaries, file paths, exit codes — that you don't want
the model conditioning on.

**`details` is not carried on events or across the fleet boundary.**
`ToolExecutionEnd` only includes `result: String` and `is_error: bool`,
and `SubagentResult` carries only `text`, token counts, duration, and
worktree paths. Anything you want a parent to see from a child tool
must travel through `ToolResult.content` (which the model sees) or
through a side-channel the host owns (a custom interaction, a shared
store keyed by `tool_call_id`, a wrapping `Transport`). `details` is
best treated as a **local-only** metadata channel — useful for the
agent's own host UI, not for inter-agent communication.

### `ExecutionContext`

What the tool receives:

```rust
pub struct ExecutionContext {
    pub cwd: PathBuf,                                 // working dir
    pub cancel: CancellationToken,                    // cooperative cancel
    pub progress: ProgressSender,                     // stream updates
    pub interaction: Option<mpsc::Sender<InteractionRequest>>,
    /// Mirror of AgentBuilder::set_interaction_timeout. Tools that
    /// originate their own InteractionRequests should wrap their
    /// `response_rx.await` in `tokio::time::timeout` using this value
    /// so they honor the same deadline the actor uses for tool.confirm.
    /// `None` = unbounded wait. See §6.
    pub interaction_timeout: Option<Duration>,
    pub file_access: Arc<Mutex<FileAccessTracker>>,
    pub agent_id: Option<String>,                     // owning agent id
    pub subagent_depth: u32,                          // 0 = root
}
```

Useful methods:

- `ctx.resolve_path("~/foo")` — expands `~/` and resolves against `cwd`.
- `ctx.mark_read(&path)` — record a successful read.
- `ctx.require_read(&path)` — fail unless the file was read first
  (read-before-write enforcement). The tracker is shared across all
  tools in the batch via `Arc<Mutex<…>>`, so a `read` tool's
  `mark_read` is immediately visible to a parallel `edit` tool's
  `require_read`. The mutex is `parking_lot::Mutex` and critical
  sections are trivial (HashSet insert / contains), so contention is
  negligible in practice — but tools that touch the tracker
  thousands of times per call should batch their updates.
- `ctx.progress.send("doing the thing")` — emit a `ConsoleLine`
  attributed to this tool call (becomes a `ToolExecutionUpdate`
  event at `ConsoleLevel::Normal`).
- `ctx.progress.send_at("careful", ConsoleLevel::Warning)` — same with
  a level.
- `ctx.progress.send_lines(vec![ConsoleLine::new("a", Normal), …])` —
  batch emit.
- `ctx.progress.emit(AgentEvent::…)` — send a **raw** `AgentEvent` on
  the agent's broadcast channel. Use this to emit
  `AgentEvent::AgentReport { tag, summary }` for the agent's
  self-label (which the fleet bus translates to
  `FleetEvent::AgentReport` when forwarding). The event is broadcast
  as-is; it does not run through the per-tool `ToolExecutionUpdate`
  wrapping.
- `ctx.cancel.is_cancelled()` — bail out cooperatively on interrupt.

The agent id lets tools that need to identify their caller (e.g.
recursive spawn tools) look themselves up in a registry. Don't capture
a handle at construction time — read it from `ctx`.

### Concurrency & categories

```rust
fn concurrency(&self) -> Concurrency { Concurrency::Sequential }  // runs alone
fn category(&self) -> ToolCategory { ToolCategory::Execute }      // UI hint
fn risk(&self, _args: &Value) -> ToolRisk { ToolRisk::Elevated }  // gate me
```

`Concurrency::Sequential` tools form a group that runs alone, even when
the model emits a parallel batch. `Parallel` tools (the default) run
concurrently within a batch.

`ToolCategory` is **purely informational** — it surfaces in
`ToolInfo.category` and in event payloads so hosts can group/colorize
tools in their UI. It does **not** affect approval, concurrency, or
any other runtime behavior. The four variants are:

| Variant   | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| `Read`    | Reads filesystem / project state (e.g. `read`, `grep`)   |
| `Edit`    | Mutates files in the workspace (e.g. `write`, `edit`)    |
| `Execute` | Runs processes / external commands (e.g. `bash`)         |
| `Other`   | Anything else — UI prompts, plan submission, web fetches |

Risk (which **does** affect runtime behavior, via approval) is a
separate axis: a `Read` tool is usually `Safe`, but the two are
independent in principle and hosts that surface them do so separately.

### `ToolRisk`

Self-reported by each tool's `risk(&Value) -> ToolRisk` method. The
runtime hands this to the active `ApprovalPolicy::classify` along
with the tool name and arguments. Three variants — this is the
complete list:

| Variant    | Meaning                                                           | `DefaultPolicy` |
| ---------- | ----------------------------------------------------------------- | --------------- |
| `Safe`     | Read-only operations (file reads, listings, searches)             | `Auto`          |
| `Local`    | Local mutation the user normally allows (file edits, in-process)  | `Auto`          |
| `Elevated` | Shell, network posts, sending drafts — things the user should see | `Gate`          |

`risk()` takes `&Value` (the arguments) so a tool can return
different risk for different invocations — `bash` returns `Elevated`
for `rm -rf /` and `Safe` for `git status`, for example.

### `BoxedTool`

Type alias: `pub type BoxedTool = Arc<dyn Tool>;`. Tools are
stateless w.r.t. the agent, so sharing one `BoxedTool` across many
agents (different `AgentSpec`s, different `AgentManager` spawns) is
safe and expected.

---

## 5. Approval policies

Every tool call is classified before it runs. The classifier returns an
`ApprovalDecision`:

- `Auto` — run immediately
- `Gate` — emit a `tool.confirm` interaction; wait for the host
- `Reject(reason)` — synthesize an error result; do not run

### Built-in policies

```rust
use std::sync::Arc;
use tau_agent::{AutoAcceptAll, DefaultPolicy, RulePolicy, ToolRule};

// CI / scripts.
builder.set_approval_policy(Arc::new(AutoAcceptAll));

// Default: auto-approve Safe & Local, gate Elevated.
builder.set_approval_policy(Arc::new(DefaultPolicy));

// Per-tool / per-argument rules with a fallback.
let policy = RulePolicy::new(Arc::new(DefaultPolicy))
    .allow(ToolRule::contains_at("bash", "command", "git status"))
    .allow(ToolRule::contains_at("bash", "command", "git diff"))
    .deny(ToolRule::contains_at("bash", "command", "rm -rf"))
    .allow(ToolRule::any("read_file"));
builder.set_approval_policy(Arc::new(policy));
```

Rule precedence: **deny > allow > fallback**.

### `ToolRule` matching

A rule matches when (1) `tool` equals the call's tool name AND (2)
either `matches` is empty (any args) or **any** of its `ArgMatch`es
matches (OR semantics — combine via separate rules for AND).

```rust
pub struct ToolRule {
    pub tool: String,
    pub matches: Vec<ArgMatch>,
}

pub struct ArgMatch {
    /// JSON path scoping the match. `None` walks every string-typed
    /// leaf; `Some("command")` targets `args.command`;
    /// `Some("options.argv.0")` walks nested objects and arrays.
    /// Components are exact (no globbing). A non-string value at the
    /// path causes the match to fail.
    pub path: Option<String>,
    pub pattern: ArgPattern,
}

pub enum ArgPattern {
    /// Normalized substring search.
    Contains(String),
    /// Exact equality after normalization.
    Equals(String),
}
```

**Comparison is against typed JSON string values, not the serialized
JSON.** Earlier versions matched substrings against
`serde_json::to_string(args)`, which made whitespace, key order, and
escaping part of the security boundary — a `"rm -rf"` rule could be
bypassed by `"rm  -rf"` (two spaces). Today both the needle and the
value are normalized first: runs of ASCII whitespace collapse to a
single space, then both ends are trimmed. Matching is case-sensitive.

Helpers for the common shapes:

```rust
ToolRule::any("read_file");                       // any args for this tool
ToolRule::contains("bash", "rm -rf");             // any string leaf
ToolRule::contains_at("bash", "command", "rm");   // scoped to args.command
ToolRule::equals_at("bash", "command", "ls -la"); // exact match at path
```

### Custom policies

```rust
use serde_json::Value;
use tau_agent::{ApprovalDecision, ApprovalPolicy, ToolRisk};

struct OfficeHoursPolicy;

impl ApprovalPolicy for OfficeHoursPolicy {
    fn classify(&self, tool: &str, _args: &Value, risk: ToolRisk) -> ApprovalDecision {
        if !is_business_hours() {
            ApprovalDecision::Reject(format!("{tool} blocked outside business hours"))
        } else if matches!(risk, ToolRisk::Elevated) {
            ApprovalDecision::Gate
        } else {
            ApprovalDecision::Auto
        }
    }
}
```

### Approval edge cases

- **Order of calls**: the runtime calls `tool.risk(&args)` first,
  then passes the resulting `ToolRisk` into `policy.classify(name,
  &args, risk)`. The policy sees the post-risk classification — it
  can use the risk, the tool name, the arguments, or any combination.

- **`ToolRule` matching semantics**: rules walk the typed JSON tree
  and compare against string-typed leaves (or against the value at
  `ArgMatch.path`, if set). Both sides are whitespace-normalized —
  runs of ASCII whitespace collapse to a single space, then both ends
  are trimmed. So `"rm -rf"`, `"rm  -rf"` (two spaces), and
  `"rm\t-rf"` are equivalent. Matching is case-sensitive. **Never
  matched against the serialized JSON**, so JSON encoding, key order,
  and escaping do not affect rule matches.

- **`ToolRule::any(name)`**: equivalent to `ToolRule { tool: name,
  matches: vec![] }`. An empty `matches` list is special-cased to
  match **any** invocation of the named tool, regardless of
  arguments.

- **Swapping policy mid-flight**: calling
  `handle.set_approval_policy(new)` updates the policy on the actor's
  `Frame` at its next safe boundary. Tool calls that have already
  passed the approval gate (whether `Auto`, `Approved`, or `Gate`d
  and resolved) continue to run under the old decision. The new
  policy applies to subsequent batches only — there is no
  retroactive re-classification.

---

## 6. Interactions (UI round-trip)

When a policy returns `Gate`, the runtime sends an `InteractionRequest`
to the channel registered via `builder.set_interaction_sender(tx)`.
The same channel is exposed to tools as `ctx.interaction` for
custom prompts (plan submission, free-form questions, …).

> **Design note.** Agents in this codebase model UI round-trips as
> extensions to `InteractionRequest` (e.g. new `schema_id`s), **not**
> new actor modes or phases. Add a schema, render it in the host, and
> the tool/runtime stays linear.

```rust
use tokio::sync::mpsc;
use tau_agent::{InteractionKind, InteractionRequest, InteractionResponse};

let (tx, mut rx) = mpsc::channel::<InteractionRequest>(32);
builder.set_interaction_sender(tx);

// In the host UI task:
tokio::spawn(async move {
    while let Some(req) = rx.recv().await {
        match req.kind {
            InteractionKind::Typed { schema_id, payload } if schema_id == "tool.confirm" => {
                // Render confirmation UI; collect user choice.
                let approved = ask_user(&payload).await;
                let resp = if approved {
                    InteractionResponse::Approved { payload: None }
                } else {
                    InteractionResponse::Rejected { reason: "user declined".into() }
                };
                let _ = req.response_tx.send(resp);
            }
            InteractionKind::Typed { schema_id, payload } => {
                // Host- or tool-defined schema. Render and reply.
                let _ = req.response_tx.send(handle_typed(&schema_id, payload).await);
            }
            InteractionKind::AskQuestion { question, options } => {
                let choice = pick_option(&question, &options).await;
                let _ = req.response_tx.send(InteractionResponse::Answer(choice));
            }
        }
    }
});
```

### Built-in schema

The runtime defines exactly one `schema_id`:

- `"tool.confirm"` — approval gate for risky tools.
  Payload:
  ```json
  {
    "tool_call_id": "…",
    "tool_name": "bash",
    "arguments": { … },
    "activity": "Running git push",
    "risk": "Elevated"
  }
  ```

All other `schema_id`s are opaque to the runtime — define your own.

### Response types

| Response                              | Use for                                           |
| ------------------------------------- | ------------------------------------------------- |
| `Answer(String)`                      | Reply to `AskQuestion` (carries chosen label)     |
| `Approved { payload: Option<Value> }` | Reply to `Typed`; `Some(edited)` if user modified |
| `Rejected { reason }`                 | Reply to `Typed`; tool gets error                 |
| `Cancelled`                           | User aborted the prompt                           |

### Round-trip and failure modes

The full flow for a `tool.confirm` request (and any other `Typed`
request the runtime emits) is:

1. The runtime creates an `oneshot::channel` for the response and
   sends `InteractionRequest { kind, response_tx, agent_id }` on the
   `mpsc::Sender<InteractionRequest>` registered via
   `builder.set_interaction_sender(tx)`.
2. The actor `.await`s the matching `response_rx`. The actor is
   blocked on this future inside its approval-gate phase — other
   pending gates for the same batch are awaited concurrently.
3. The host UI takes `req.response_tx` out of the request and calls
   `.send(response)` exactly once.
4. The actor wakes, classifies the response, and either dispatches
   the tool (`Approved`), synthesizes a tool error (`Rejected`,
   `Cancelled`), or treats it as a bug (`Answer` to a `Typed`
   request → "unexpected response to tool.confirm" error result).

**Built-in timeout** (recommended). Configure via
`AgentBuilder::set_interaction_timeout(Duration)`. When set, the actor
wraps every gate's `response_rx.await` in a `tokio::time::timeout`.
On expiry it synthesizes
`InteractionResponse::Rejected { reason: "interaction timed out after …" }`,
emits a `tracing::warn`, and unblocks the tool batch with that one
call rejected. The host's `response_tx.send(...)` after the deadline
fails silently — same as any "host replied after the actor moved on"
race.

```rust
use std::time::Duration;
builder.set_interaction_timeout(Duration::from_secs(30));
// Subagents inherit when configured on the manager:
let manager = AgentManager::new(parent_config, transport, 20)
    .with_interaction_timeout(Duration::from_secs(30));
```

If unset, the actor waits indefinitely (historical behavior). Set it.
A host that never replies otherwise hangs the whole prompt — the
single biggest footgun in the API before this knob existed.

The same `Duration` is mirrored to tools via
[`ExecutionContext::interaction_timeout`](#executioncontext) so
**tool-initiated** interactions (which the tool, not the actor,
awaits) can honor the same deadline. Tools that originate their own
`InteractionRequest`s should wrap their `response_rx.await`:

```rust
async fn execute(&self, args: Value, ctx: ExecutionContext) -> ToolResult {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let req = InteractionRequest { /* … */ response_tx: tx, /* … */ };
    let _ = ctx.interaction.as_ref().unwrap().send(req).await;

    let reply = match ctx.interaction_timeout {
        Some(d) => tokio::time::timeout(d, rx).await
            .unwrap_or_else(|_| Ok(InteractionResponse::Cancelled))
            .unwrap_or(InteractionResponse::Cancelled),
        None => rx.await.unwrap_or(InteractionResponse::Cancelled),
    };
    /* … */
    ToolResult::text("ok")
}
```

(Host-side timeouts on the receiver end still work — see the
`tokio::select!` pattern below — and compose cleanly with the
runtime-side timeout. Either path produces the same downstream
effect: a synthesized error result for the offending tool call.)

```rust
use std::time::Duration;
use tau_agent::{InteractionRequest, InteractionResponse};

async fn handle_with_timeout(req: InteractionRequest) {
    let reply: InteractionResponse = tokio::select! {
        biased;
        r = ask_user(&req.kind) => r,
        _ = tokio::time::sleep(Duration::from_secs(30)) => {
            InteractionResponse::Rejected { reason: "timeout".into() }
        }
    };
    let _ = req.response_tx.send(reply);
}
```

(Use `Cancelled` instead of `Rejected` if you want the *whole prompt*
treated as cancelled rather than just this one tool. For tool-confirm,
both behave the same to the agent — a synthesized error result.)

Failure modes:

- **Host drops `response_tx` without sending**: the actor sees
  `RecvError` and treats it as `"interaction channel closed"` — the
  tool is rejected with that reason and execution continues with the
  rest of the batch.
- **Host's `mpsc` receiver is dropped or full at send time**: the
  runtime's `try_send` fails. The tool is rejected with
  `"interaction channel saturated or closed"`; the agent sees an
  error result for that call.
- **`handle.abort()` while a gate is pending**: the cancellation
  token trips, the actor unwinds the entire prompt, all pending
  gates are dropped without responses. The host's `response_tx`
  send (if attempted afterward) silently fails.
- **`handle.interrupt()` while a gate is pending**: in-flight gates
  still complete normally — `interrupt` only blocks the *next* turn
  from starting.

Tools that initiate their own custom `Typed` interactions follow the
same rules. Inside a tool, treat `ctx.interaction` as `Option` (the
host may not have wired one): fall back to a safe default if
`None`, and on `RecvError` from `response_rx.await` treat the gate
as cancelled rather than panicking.

### Other interaction questions

- **Who emits `InteractionKind::AskQuestion`?** Only **tools** —
  the runtime itself never emits `AskQuestion`. The shape exists for
  user-defined tools that want a generic "pick an option" prompt
  without defining a JSON schema. The runtime emits only
  `Typed { schema_id: "tool.confirm", … }` from the approval gate.

  The `options` field is `Vec<QuestionOption>` where
  `QuestionOption { label: String, description: String }`. The host
  responds with `InteractionResponse::Answer(chosen_label)`; the tool
  looks up the matching `QuestionOption` by `label`.

- **`InteractionRequest.agent_id`** is `Some(id)` when the request
  originated inside a subagent (the fleet bus stamps it on outgoing
  requests so the host can attribute the prompt to a specific
  agent). For the host's root agent — or any builder-spawned agent
  not adopted into a fleet — it is `None`.

- **Multiple requests per `execute`** are fine. A tool can send any
  number of `InteractionRequest`s during a single `execute` call
  (sequentially or concurrently via `futures::join!`). Each carries
  its own oneshot. The actor isn't involved in the tool↔host loop
  beyond having handed the tool the `ctx.interaction` sender — the
  tool blocks on its own awaits, not the actor's.

- **Unknown `schema_id`**: the runtime treats the payload as opaque
  JSON and routes the request to the configured channel as-is. If
  the host doesn't recognize the `schema_id`, it should
  `req.response_tx.send(InteractionResponse::Rejected { reason })`
  (or `Cancelled`) so the tool gets a clean error rather than
  blocking forever. Hosts that maintain a renderer table should
  default-render unknown schemas as a generic JSON-with-approve/reject
  panel rather than dropping them silently.

---

## 7. Compaction

When the conversation approaches the context window the runtime
summarizes the oldest messages. Trigger sources:

- Proactive (`Threshold`): the `reserve` budget is remaining.
- Reactive (`Overflow`): a provider returned an overflow error.
- Manual: explicit `handle.compact(…)` call.

Configure:

```rust
use tau_agent::{AgentConfig, CompactionConfig, CompactionThreshold};

let config = AgentConfig::builder(model)
    .compaction(CompactionConfig {
        enabled: true,
        // Trigger compaction when within 8% of the context window…
        reserve: CompactionThreshold::Fraction(0.08),
        // …and keep the most recent 10% of the window through the cut.
        keep_recent: CompactionThreshold::Fraction(0.10),
    })
    .build();
```

`CompactionConfig::default()` returns:

```rust
CompactionConfig {
    enabled: true,
    reserve: CompactionThreshold::Fraction(0.08),     // 8% of context_window
    keep_recent: CompactionThreshold::Fraction(0.10), // 10% of context_window
}
```

Both budgets are [`CompactionThreshold`](#compactionthreshold) values
— either a fraction of `model.context_window` or an absolute token
count. **Fractions are the default for a reason**: an absolute reserve
of 16,384 tokens is ~8% of Opus's 200K window but ~50% of a
32K-context model, which would leave that model with effectively no
room to work. Fractions scale correctly across model sizes.

Use `CompactionThreshold::Tokens(n)` when you want explicit control
independent of model size — e.g. "always keep at least 5,000 tokens
of recent context regardless of what model is loaded."

```rust
use tau_agent::{CompactionConfig, CompactionThreshold};

// Pin absolute counts (e.g. for tests, or to match a known prompt budget):
CompactionConfig {
    enabled: true,
    reserve: CompactionThreshold::Tokens(8_000),
    keep_recent: CompactionThreshold::Tokens(12_000),
};
```

(Omitting `compaction` from `AgentConfig` gives the fraction default
above, not "no compaction" — to disable, set `enabled: false`.)

#### `CompactionThreshold`

```rust
pub enum CompactionThreshold {
    /// Fraction of `model.context_window`. Clamped to `[0.0, 1.0]`
    /// before scaling; `NaN` resolves to `0`.
    Fraction(f32),
    /// Absolute token count, independent of the model.
    Tokens(u64),
}

impl CompactionThreshold {
    pub fn resolve(&self, context_window: u64) -> u64;
}
```

`resolve` rounds to the nearest token rather than truncating, so
`Fraction(0.08).resolve(200_000) == 16_000` exactly even though
`0.08f32 * 200_000` is `15_999.999…` in IEEE-754.

Trigger manually with optional steering instructions:

```rust
use tau_agent::CompactionReason;

let rx = handle.compact(
    CompactionReason::Manual,
    Some("Focus the summary on outstanding TODOs.".into()),
).await?;
```

You will observe `CompactionStart { reason }` then `CompactionEnd
{ tokens_before, tokens_after }` events.

### Compaction details

- **Where the context window comes from**: read from
  `AgentConfig.model.context_window`. Compaction is a function of
  the model — there is no separate field to override it.

- **Which model summarizes**: the same `AgentConfig.model` runs the
  summarization call (with a fixed system prompt and
  `max_tokens: 4096`). There's no separate "compaction model" knob.

- **`reserve` vs. `keep_recent`**: they describe different sides of
  the cut. `reserve` is the *trigger* — compaction fires when
  `(input + cache_read)` is within `reserve.resolve(context_window)`
  tokens of `context_window`. `keep_recent` is a lower bound on what
  survives — the cut-point search walks back from the end of the
  conversation until it has accumulated at least
  `keep_recent.resolve(context_window)` tokens, then continues to
  the next message boundary so a turn isn't split mid-tool-batch.
  If the resolved `keep_recent` exceeds what's available, the run
  keeps everything and there's nothing to compact; the proactive
  trigger will fire again next turn.

  Both are resolved against the model's current `context_window` at
  each evaluation, so swapping models via `handle.set_model(...)`
  automatically rescales any `Fraction` thresholds — no need to
  reconfigure compaction when the model changes.

- **What `messages()` returns post-compaction**: the *compacted*
  history. The runtime replaces the summarized prefix in place with
  a `SystemInjection` block containing the `<context-summary>` text.
  The `Conversation.previous_summary` field also retains the most
  recent summary so it can be threaded into the next compaction if
  needed.

- **History seeding (`AgentSeed`)**: both `AgentBuilder::seed(…)` and
  `SpawnOpts.seed` take a single `AgentSeed` enum:

  ```rust
  pub enum AgentSeed {
      Empty,
      Messages { messages: Vec<Message>, previous_summary: Option<String> },
      Inherit { agent_id: String },
  }
  ```

  - `Empty` (the default) — start fresh.
  - `Messages` — load the supplied history and (optionally) seed
    `Conversation.previous_summary` for the next compaction pass. The
    summary isn't inserted into messages; it's threaded through the
    summarizer separately.
  - `Inherit` — clone another tracked agent's `messages().await` into
    the new agent at spawn time. The id must still be in the
    [`AgentManager`](#12-fleet-multi-agent) registry. **Only meaningful
    inside `SpawnOpts`** — setting it on `AgentBuilder` is a no-op
    (the builder has no registry) and emits a `tracing::warn!`.

  Typical recipe for restoring a session:
  `builder.seed(AgentSeed::Messages { messages: prior, previous_summary: Some(prior_summary) })`.

- **`max_turns` and compaction**: compaction turns **do not** count
  toward `max_turns`. The summarization call runs on its own
  sub-phase machine and does not increment the actor's
  `turn_number`. After an `Overflow`-triggered compaction the
  counter is even **reset to zero** so the agent can continue past
  what would otherwise be its hard cap. `Threshold` and `Manual`
  compactions leave the counter alone. `max_turns` is therefore a
  budget on *productive* model calls, not total LLM calls.

---

## 8. Mutating a running agent

All mutators return `Result<()>` and queue commands on the actor's
urgent channel; changes apply between safe phase boundaries.

```rust
use tau_ai::{get_model_by_id, ReasoningLevel};

let opus = get_model_by_id("claude-opus-4-7").expect("model exists");
handle.set_model(opus).await?;
handle.set_reasoning(ReasoningLevel::High).await?;
handle.set_approval_policy(Arc::new(AutoAcceptAll)).await?;
handle.set_compaction_config(CompactionConfig::default()).await?;
```

`try_set_compaction_config` is the only non-blocking variant, kept for
backpressure tests; it returns `Error::ChannelFull { channel: "urgent" }`
when the urgent channel is full. The other config setters are async-only.
For steers and follow-ups, both `try_*` and async variants exist (see §2).

### What can't be changed mid-flight

`AgentSpec` is immutable per agent. To change tools or the system
prompt of a managed agent, use `AgentManager::respec` (see §12). For
an unmanaged agent, spawn a fresh one with the desired spec.

---

## 9. Inspecting an agent

```rust
// Full state snapshot (None if actor task has died).
if let Some(state) = handle.state().await {
    println!("{} messages, {} input tokens",
             state.messages.len(),
             state.total_usage.input);
}

// Just the messages.
let msgs = handle.messages().await.unwrap_or_default();

// Context window estimate (heuristic: char/4).
if let Some(stats) = handle.context_stats().await {
    println!("{}/{} tokens used, {} remaining",
             stats.used, stats.limit, stats.remaining);
}

// Tool registry, with current approval status under the live policy.
for tool in handle.list_tools().await.unwrap_or_default() {
    println!("{:8}  default={}  current={}  {}",
             tool.name, tool.default_allowed, tool.currently_allowed,
             tool.description);
}

// Lifecycle.
use tau_agent::AgentHealth;
match handle.health() {
    AgentHealth::Running        => println!("processing a prompt"),
    AgentHealth::Idle           => println!("alive, waiting for work"),
    AgentHealth::Dead { reason } => println!("dead: {reason:?}"),
}
let id    = handle.agent_id();           // Some when managed by a fleet
```

`Conversation` fields available on `state().await`:

```rust
pub struct Conversation {
    pub messages: Vec<Message>,
    pub total_usage: Usage,
    pub error: Option<String>,
    pub previous_summary: Option<String>,
}
```

"Is a prompt in flight?" is **not** carried on `Conversation` —
query [`AgentHandle::health`](#9-inspecting-an-agent) for that.
For event-stream-driven UIs, treat `AgentStart` / `AgentEnd` as the
authoritative edges; use `health()` only for "is anything in flight
right now?" UI hints (e.g. polling for a "thinking" indicator from
a non-async context).

`ContextStats`:

```rust
pub struct ContextStats {
    pub used: u64,
    pub remaining: u64,
    pub limit: u64,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}
```

`ToolInfo`:

```rust
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub default_allowed: bool,
    pub currently_allowed: bool,
}
```

### Lifecycle states

`health()` returns one of three states, defined as an enum re-exported
at the crate root:

```rust
pub enum AgentHealth {
    /// Actor is alive and currently processing a prompt
    /// (between `AgentStart` and `AgentEnd`).
    Running,
    /// Actor is alive and waiting for work. A new `prompt()` will be
    /// accepted.
    Idle,
    /// Actor task has terminated. `reason` is `Some(payload)` on
    /// panic, `None` on clean shutdown.
    Dead { reason: Option<String> },
}
```

- **After `AgentEnd`**: the actor transitions from `Running` back to
  `Idle`. The handle is **not** done — call `prompt()` again to
  start a new turn-loop on the same actor with the existing
  conversation. An agent is restartable for as long as `health()`
  returns `Running` or `Idle`.

- **`Dead { reason }`**:
  - `reason: None` after a clean shutdown / drop (all senders dropped
    and the actor's `recv` loop ended). **Rarely observed in
    practice** — clean shutdown requires every `AgentHandle` clone to
    be dropped first, so a handle that lived long enough to observe
    `Dead { reason: None }` is usually a transitional view from a
    sibling clone in the same task. If you only hold one handle, you
    will almost never see this variant.
  - `reason: Some(payload)` when the actor task panicked. Payload is
    the panic string if it was a `&str`/`String`, or the literal
    `"<non-string panic payload>"` otherwise.

- **`health()` vs `AgentEvent` consistency**: the `prompt_in_flight`
  atomic that drives `Running`/`Idle` flips just *before* the
  corresponding `AgentStart` / `AgentEnd` events are emitted on the
  broadcast channel. A subscriber and a `health()` caller can
  therefore disagree by one event in either direction at the
  transition boundary. For state-machine logic, treat
  `AgentStart` / `AgentEnd` as the authoritative edges; read
  `health()` for *snapshot* questions ("is the agent doing
  anything right now?") where one-event-of-skew is fine.

- **Broadcast channel after the actor exits**: once the actor task
  drops, the channel's senders are gone. Existing subscribers'
  `recv().await` will resolve to `Err(RecvError::Closed)` once the
  in-flight buffer is drained. New subscribers from `handle.subscribe()`
  still produce a `Receiver`, but it will immediately resolve to
  `Closed` on the first `recv()`.

---

## 10. Interrupt vs abort

### Phase machine and safe boundaries

The actor cycles through a handful of phases. Mutators, steers, and
interrupt checks all apply at the boundaries between them:

```text
      ┌──────► Idle ◄───────────────────-───────────────┐
      │         │                                       │
      │         │ prompt()                              │ AgentEnd
      │         ▼                                       │
      │     ┌─ ModelCall ◄─────────-─────────────────┐  │
      │     │     │                                  │  │
      │     │     │ assistant message has tool calls │  │
      │     │     ▼                                  │  │
      │     │   ApprovalGate ── any Reject ──┐       │  │
      │     │     │ all Approved             │       │  │
      │     │     ▼                          ▼       │  │
      │     │   ToolBatch ──────────► ToolResults ───┘  │
      │     │ (parallel batches,                        │
      │     │  sequential tools                         │
      │     │  alone in their own                       │
      │     │  group)                                   │
      │     │                                           │
      │     └─► QueueDrain (steers + follow-ups) ───────┘
      │                  │
      │                  │ no work left
      └──────────────────┘
```

Boundaries (after `ModelCall`, after each `ToolBatch`, before
`QueueDrain`, and at `Idle`) are where the actor:

- Processes urgent-channel commands (`set_model`, `set_reasoning`,
  `set_approval_policy`, `set_compaction_config`, `steer`, `follow_up`).
- Checks the `interrupt_requested` flag.
- Drains queues per `steering_mode` / `follow_up_mode`.
- Decides whether to compact (proactive `Threshold`) or stop on
  `max_turns`.

Compaction inserts itself before the next `ModelCall` and runs its own
sub-phase machine; same boundary semantics resume after it completes.

### Interrupt vs abort

Two ways to stop work, with different semantics:

```rust
handle.interrupt();   // graceful: finish current turn, then idle
handle.abort();       // hard: trip the cancellation token now
```

`interrupt()` sets a flag that's checked at safe boundaries — tools
already running complete; queued tools and the next model call are
skipped; `AgentEnd { interrupted: true }` fires. Use this from a UI
ESC binding.

`abort()` cancels the `CancellationToken` immediately. Tools that
respect `ctx.cancel.is_cancelled()` bail out; in-flight transport
requests are dropped. Use this for shutdown.

#### Tool cancellation contract

Tools are **expected but not required** to be cancellation-aware.
Specifically:

- The runtime drives the tool batch off a `tokio::task::JoinSet`. On
  `abort()`, the JoinSet is dropped, which fires tokio's
  abort-on-drop semantics for every spawned task.
- Tokio aborts take effect at the **next `.await` point** inside the
  task. A tool that awaits I/O, channels, or `tokio::time::sleep`
  will be torn down promptly.
- A tool stuck in **synchronous CPU work between awaits** — a tight
  loop, a blocking syscall in async context, a CPU-bound parser —
  cannot be aborted. The task runs to completion and only sees the
  cancel signal if it checks `ctx.cancel.is_cancelled()` explicitly.
- A tool that calls `block_in_place` or runs on `spawn_blocking` is
  similarly opaque to cancellation until it returns.

**Recommendation**: tools doing long CPU work should sprinkle
`if ctx.cancel.is_cancelled() { return ToolResult::error("cancelled"); }`
checks at natural boundaries (loop iterations, chunk boundaries).
Tools that only `.await` external work get cancellation for free.

The actor itself does **not** wait for non-cancellable tools after
`abort()`: it returns to `Idle` immediately and emits `AgentEnd`.
The tool task continues to run detached but its result is discarded
(no `ToolExecutionEnd` event is emitted because the actor isn't
listening). For tools that mutate shared external state, this means
`abort()` is not a clean rollback signal — design tools to be
re-entrant or to commit only at the end.

You can subscribe to the cancellation token directly:

```rust
let token = handle.cancel_token();
let cancel = token.lock().await.child_token();
```

---

## 11. Transport

`ProviderTransport` is the production implementation — it dispatches to
tau-ai's per-provider clients (Anthropic, OpenAI, Google, Ollama).

```rust
use tau_agent::ProviderTransport;

let transport = Arc::new(ProviderTransport::new());

// Or with an explicit API key (overrides env):
let transport = Arc::new(ProviderTransport::with_api_key("sk-…"));

// Or with a custom retry policy:
use tau_ai::RetryConfig;
let transport = Arc::new(
    ProviderTransport::new().with_retry_config(RetryConfig {
        max_retries: 5,
        initial_delay: std::time::Duration::from_millis(500),
        max_delay: std::time::Duration::from_secs(60),
        backoff_multiplier: 2.0,
    }),
);
```

### Custom transports

Implement `Transport` for routing, mocking, recording, or in-process
LLMs. The trait is:

```rust
#[async_trait]
pub trait Transport: Send + Sync {
    async fn run(
        &self,
        messages: Vec<Message>,
        config: &AgentRunConfig,
        cancel: CancellationToken,
    ) -> Result<AgentEventStream>;
}
```

`AgentEventStream` is `Pin<Box<dyn Stream<Item = AgentEvent> + Send>>`.
Yield `MessageStart` → `MessageUpdate`* → `MessageEnd` → `TurnEnd` at
minimum.

### `AgentRunConfig`

The per-call config passed to `Transport::run`. The actor builds one
from the agent's `Frame` for every turn — transports only *read* it.
Marked `#[non_exhaustive]`, so fields can be added without breaking
transport implementations.

```rust
#[non_exhaustive]
pub struct AgentRunConfig {
    pub system_prompt: Option<String>,
    pub tools: Vec<tau_ai::Tool>,         // converted from BoxedTool
    pub server_tools: Vec<ServerTool>,
    pub model: Model,
    pub reasoning: Option<ReasoningLevel>,
    pub thinking_adaptive: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub cache_scope: Option<String>,
    pub cache_ttl: Option<String>,
    pub system_prompt_boundary: Option<String>,
    pub turn_number: u32,                  // 1-indexed
}
```

`turn_number` is stamped on the `TurnStart` / `TurnEnd` events the
transport emits. Custom transports should propagate it as-is.

### Transport authoring notes

- **Event ordering**: yield `MessageStart` → `MessageUpdate`* →
  `MessageEnd` → `TurnEnd` for each completed turn. For tool calls
  the *model* produced, emit them as `Content::ToolCall { id, name,
  arguments }` blocks inside the assistant `Message` — the actor
  will dispatch them. Do **not** emit `ToolExecutionStart` /
  `ToolExecutionEnd` from the transport; those come from the actor.
- **Out-of-order events**: the actor's `StreamReducer` is lenient
  but expects each turn's `MessageStart` to arrive before its
  `MessageEnd`, and `TurnEnd` to be the last event for that turn.
  Out-of-order updates are tolerated but may produce surprising
  partial-message states for UI subscribers.
- **Errors**: yield `AgentEvent::Error { message }` to signal a
  failed turn the actor can recover from (or compact past, when
  `is_context_overflow` matches). Return `Err(_)` from `run` for
  hard failures (auth, malformed request) — the actor will treat
  these as `Error::Ai`.

### `ProviderTransport` provider selection

`ProviderTransport` dispatches based on `config.model.api`
(`AnthropicMessages`, `OpenAIResponses`, `Google`, `Ollama`,
`Custom`). The model itself names which provider it belongs to —
you don't select the provider separately.

`with_api_key(key)` is honored by every HTTP provider (Anthropic,
OpenAI, Google). When set, it bypasses the provider's
`from_env` lookup and uses the supplied string directly. When not
set, each provider falls back to its environment variable
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, …). The
Ollama provider ignores the key — it uses `model.base_url` for its
local HTTP endpoint instead.

---

## 12. Fleet (multi-agent)

`AgentManager` composes a registry, lifecycle, and event bus for hosts
that spawn subagents.

```rust
use std::sync::Arc;
use tau_agent::{AgentConfig, AgentManager, AgentSpec, ProviderTransport, SpawnOpts};
use tokio_util::sync::CancellationToken;

let transport = Arc::new(ProviderTransport::new());
let parent_config = AgentConfig::builder(model.clone()).build();
let manager = Arc::new(
    AgentManager::new(
        parent_config,            // parent config (template)
        transport,
        20,                       // max concurrent agents
    )
    // Optional, chainable:
    .with_parent_interaction_sender(parent_interaction_tx)
    .with_default_approval_policy(Arc::new(DefaultPolicy))
    .with_interaction_router_capacity(64)
    .with_interaction_timeout(Duration::from_secs(30))  // §6 timeout for every subagent
    .with_event_capacity(512),    // override default FleetEvent channel cap
);

// Subscribe to fleet-level events (subagent lifecycle + forwarded
// child events). See §3 for the event taxonomy.
let mut fleet_events = manager.subscribe();

// Swap the default subagent policy at runtime (in-flight agents keep
// the policy they were spawned with):
manager.set_default_approval_policy(Arc::new(AutoAcceptAll));

let spec = Arc::new(AgentSpec {
    system_prompt: "You are a focused research agent.".into(),
    tools: vec![Arc::new(ReadFileTool)],
    max_turns: 50,
});
```

### Spawn modes

| Method              | Blocks caller?                   | Who drives turns?   | Result delivery                                          | Use when                                                                     |
| ------------------- | -------------------------------- | ------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `spawn`             | Yes                              | Manager             | Return value (`SubagentResult`)                          | Parent tool needs the answer before continuing — typical "agent-as-function" |
| `spawn_interactive` | No (handle returned immediately) | Caller (via handle) | Whatever the caller does with the handle                 | Multi-turn or incremental conversation; host UI drives the loop              |
| `spawn_background`  | No (id returned immediately)     | Manager             | `FollowUp` message posted to parent handle on completion | Fire-and-forget monitoring or async work that should resume the parent       |

All three share the same `AgentSpec`, `SpawnOpts`, and event-bus
plumbing. They differ only in lifecycle: who keeps the handle, who
awaits the result, and how the result gets delivered.

```rust
// 1) Foreground — block until completion.
let result = manager.spawn(
    spec.clone(),
    "Find recent commits".into(),
    SpawnOpts::default(),
    CancellationToken::new(),
).await?;
println!("{}", result.text);

// 2) Interactive — caller drives the handle.
let (handle, agent_id) = manager
    .spawn_interactive(spec.clone(), SpawnOpts::default())
    .await?;
handle.prompt_and_wait("hello").await?;
manager.remove_interactive(&agent_id);

// 3) Background — fire-and-forget; posts a follow-up to the parent.
let parent_handle = /* … */;
let agent_id = manager.spawn_background(
    spec.clone(),
    "Watch the logs and report when CI passes.".into(),
    SpawnOpts::default(),
    parent_handle.clone(),
    CancellationToken::new(),
).await;
```

### Resuming, mutating, and adopting

```rust
// Resume an idle agent with a new prompt.
let result = manager.send(&agent_id, "anything else?", CancellationToken::new()).await?;

// Swap the spec (atomic detach → respawn with inherited history).
let new_spec = Arc::new(AgentSpec { /* new tools */ ..(*spec).clone() });
let new_handle = manager.respec(&agent_id, new_spec).await?;

// Preconditions for respec:
// - Agent must be Idle or Adopted (not Running). To respec a running
//   agent, call handle.abort() or handle.interrupt() first and wait
//   for AgentEnd before respec'ing.
// - The id must still be in the registry (not evicted by max_agents).
//
// Errors (structured variants; see §13):
// - Error::AgentBusy { id }    — agent is currently running; abort or
//                                interrupt and await AgentEnd, then retry
// - Error::AgentNotFound { id } — id was evicted, never stored, or
//                                already respec'd; look up via find_agent
// - Error::RespecRolledBack { id, source } — new spec failed to start;
//                                the agent is back on its previous spec.
//                                Inspect `source` for the underlying cause.
//
// Effects:
// - The old actor task is dropped. Existing AgentHandle clones for the
//   old agent become inert — sends will return channel-closed errors.
//   Subscribers will see AgentEnd before the channel closes.
// - A fresh actor starts under the new spec, seeded with the old
//   agent's full message history. The returned handle drives it.
// - On spawn failure, the old entry is restored — the respec is
//   atomic from the registry's perspective.

// Register an externally-built handle so respec() / spec_for() work.
let root_id = manager.adopt(&root_handle, "root", spec.clone());

// Look up by id or description.
if let Some(located) = manager.find_agent("research") {
    let h = located.handle;
    /* … */
}

// Get the spec a running/idle/adopted agent is currently using.
let spec = manager.spec_for(&agent_id);

// Get a clone of a running agent's handle (None if idle/adopted).
let handle = manager.handle_for(&agent_id);

// Steer a running agent by id (lookup + handle.steer()):
if let Some(handle) = manager.handle_for(&agent_id) {
    let _ = handle.steer(Message::user("focus on X")).await;
}

// Synchronous snapshot of every tracked agent.
let snap = manager.snapshot();
for a in snap.agents {
    println!("{}  {}  {:?}", a.agent_id, a.description, a.status);
}
```

### `AgentSpec`

```rust
pub struct AgentSpec {
    pub system_prompt: String,
    pub tools: Vec<BoxedTool>, // shared, stateless
    pub max_turns: u32,
}
```

`AgentSpec` is now purely the runtime inputs the actor consumes.
Host-side concerns (worktree permission, allowlists for nested spawns)
live outside the spec — the host owns them. See
[Fleet edge cases](#fleet-edge-cases) for the migration notes.

### `SpawnOpts`

```rust
pub struct SpawnOpts {
    pub description: String,                         // human label
    pub model: Option<Model>,                        // override parent
    pub cwd: Option<String>,
    pub isolation: Option<Isolation>,                // see below
    pub approval_policy: Option<Arc<dyn ApprovalPolicy>>,
    pub spec_name: Option<String>,
    pub seed: AgentSeed,                             // history seed; see §7
    pub subagent_depth: u32,
}
```

### `Isolation`

```rust
pub enum Isolation {
    /// Run the agent inside a fresh git worktree on a per-agent branch.
    /// The agent's `cwd` becomes the worktree path. On clean exit
    /// (no uncommitted changes) the worktree is removed; on a dirty
    /// exit the path/branch are returned in `SubagentResult` for the
    /// host to inspect.
    Worktree,
}
```

`Isolation::Worktree` is honored unconditionally — requesting it via
`SpawnOpts.isolation` is what enables it. There is no spec-level
permission gate anymore; hosts that want per-spec restrictions (e.g.
forbidding worktree spawns for read-only specs) enforce this in
their own spawn wrapper or `AgentTool` before calling
`manager.spawn`.

### `SubagentResult`

```rust
pub struct SubagentResult {
    pub agent_id: String,
    pub text: String,                // final assistant text ("" if tool-only / error)
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_use_count: u32,
    pub duration_ms: u64,
    pub worktree_path: Option<String>,
    pub worktree_branch: Option<String>,
    pub transcript_path: Option<String>,
}
```

### Snapshots

```rust
let snap: FleetSnapshot = manager.snapshot();
for a in &snap.agents {
    // a.agent_id, a.description, a.status (Running | Idle | Adopted),
    // a.usage, a.tool_use_count, a.started_at, a.completed_at
}
```

### Registry invariant

> A spec exists in the registry **iff** its id is in
> `idle ∪ running ∪ adopted`.

Enforced by the registry's method surface — see `fleet::registry` for
details. The `respec` flow uses an atomic verify-and-detach to close
the race where a concurrent `send` could slip in between the
"not running" check and the spec drop.

### Fleet edge cases

- **`max_agents` at capacity**: the limit governs the *idle* bucket.
  When `commit_running` for a new agent finds the idle bucket full,
  it evicts the **oldest idle entry** (LRU) and drops its spec.
  `running` and `adopted` buckets are not subject to eviction —
  they have no cap. Idle agents whose ids were just evicted can no
  longer be resumed by `send` or `respec`; attempts return
  `Error::AgentNotFound { id }`. (Eviction itself is currently
  silent — no `FleetEvent` is emitted. Future work.)

- **`remove_interactive(id)` is not strictly required** — the
  registry will eventually evict the agent under LRU pressure. But
  calling it eagerly drops the handle and spec immediately, freeing
  the actor's tokio task. Leaving interactive agents around
  indefinitely will eventually cause one of them to be evicted
  unexpectedly when `max_agents` is hit.

- **`find_agent(name_or_id)` resolution order**:
  1. Exact-id match in `running` → 2. case-insensitive substring
  match on `running.description` → 3. exact-id match in `idle` →
  4. substring match on `idle.description` → 5. exact-id match in
  `adopted` → 6. substring match on `adopted.description`. First
  match wins. `running` and `adopted` are backed by `IndexMap`
  (insertion-ordered) and `idle` by `VecDeque` (LRU-ordered), so
  substring ties resolve deterministically to the oldest entry within
  the bucket — equal-description ties are no longer subject to
  HashMap iteration order. Pass an exact id when you don't want any
  ambiguity at all.

- **`adopt` called twice on the same handle**: the second call is a
  no-op. The handle's id is a `OnceLock` — the first writer wins.
  The second `adopt` reads back the existing id and returns it
  (rather than minting a new UUID), so both callers see the same id
  and the spec is recorded under the winning id, not a losing one.
  Return type: `String` (the agent id).

- **Per-spec spawn allowlists are host-owned.** `AgentSpec` no longer
  carries an `allowed_subagent_specs` field — the runtime had no use
  for it, and hosts already enforce allowlists in their resolver /
  spawn wrapper. Carry the allowlist next to your `AgentSpec` in a
  host-side wrapper (e.g. `tau-cli/src/subagents.rs` keeps a
  `ResolvedSpec { spec, can_spawn }`), then use it to install
  `AgentTool::with_allowed_specs(...)` on the nested tool. The
  runtime sees only the `AgentSpec`.

- **`SpawnOpts.spec_name`** is the audit / lookup tag for the host's
  spawn allowlist. The fleet records it on the
  `FleetEvent::AgentStarted` event (`spec_name: Option<String>`) so
  hosts can reconstruct who spawned what. It does not name the spec
  for storage — agents are keyed by their UUID, not their spec name.

- **`AgentEvent::AgentReport`** is **not emitted by the runtime**. It
  exists so a tool can self-label its outcome via
  `ctx.progress.emit(AgentEvent::AgentReport { tag, summary })`. The
  fleet bus translates this into `FleetEvent::AgentReport { agent_id,
  description, tag, summary }` when forwarding; hosts that want a
  child's own summary (rather than `SubagentResult.text`) subscribe
  to `AgentManager::subscribe()` and match on `FleetEvent::AgentReport`.

- **`FleetEvent::AgentStarted` / `AgentResumed` / `AgentCompleted` fields**:
  see §3 for the full field list. These are emitted by the manager
  on the fleet channel; no equivalent variants exist on the per-agent
  `AgentEvent` stream.

- **Background-spawn `FollowUp` payload**: on completion,
  `spawn_background` posts a `Message::SystemInjection` to the
  parent's follow-up queue. On success the source is
  `InjectionSource::SubagentCompleted { agent_id, description }`
  and the content text is the subagent's final assistant text. On
  failure it's `InjectionSource::SubagentFailed { agent_id,
  description }` with the error message. The parent's actor picks
  this up at its next `DrainFollowUps` boundary and converts the
  injection to a user-role message before the next model call.

---

## 13. Errors

```rust
pub enum Error {
    Ai(tau_ai::Error),                            // provider error
    Compaction(String),
    Busy,                                         // prompt already in flight
    ActorPanic(String),
    ChannelFull { channel: &'static str },        // "urgent" or "normal"

    // Fleet-side structured conditions. Branch on these instead of
    // string-matching the Display output.
    AgentNotFound { id: String },                 // no such agent in registry
    AgentBusy { id: String },                     // operation needs idle, got running
    RespecRolledBack {                            // new spec failed; old spec restored
        id: String,
        source: Box<Error>,                       // chains the inner cause
    },
    WorktreeSetupFailed { reason: String },       // git/filesystem error on isolation setup

    Other(String),                                // unstructured catch-all
}
```

`RespecRolledBack` implements `std::error::Error::source` via the
boxed inner error, so the standard chain walker (e.g.
`anyhow::Error::chain`) surfaces the original failure. Match on the
outer variant for the "respec didn't take effect" branch; inspect
`source` if you need the underlying reason (an `ActorPanic`, an
`Ai(e)`, etc.).

Useful helper: `err.is_context_overflow()` — true when the underlying
`tau_ai::Error` indicates the prompt exceeded the model's context.
Defined directly on `Error` (returns `false` for any non-`Ai`
variant), so you can call it without matching first:

```rust
if err.is_context_overflow() {
    handle.compact(CompactionReason::Overflow, None).await?.await?;
}
```

`Result<T> = std::result::Result<T, Error>`.

### Return-type quick reference

- **`AgentBuilder::spawn`** is **async** and returns
  `Result<AgentHandle>`. It awaits the actor's readiness signal
  before returning, so the handle is guaranteed live on `Ok`. On
  startup panic, returns `Err(Error::ActorPanic(reason))`.
- **`AgentHandle::prompt(input)`** returns
  `Result<oneshot::Receiver<PromptResult>>`. The outer `Result` is
  immediate (e.g. `Error::Busy`, `Error::ActorPanic`, channel
  closed); awaiting the inner receiver resolves to a
  `PromptResult { result: Result<(), Error> }` once the prompt
  completes.
- **`AgentHandle::compact(reason, instructions)`** returns
  `Result<oneshot::Receiver<PromptResult>>`. Same shape — outer
  errors are immediate dispatch failures; awaiting the receiver
  gives you `PromptResult { result: Ok(()) }` on success or
  `Err(Error::Compaction(msg))` on failure.

### Recovery guidance

| Variant                                 | Recoverable? | How                                                                                                                                                                                    |
| --------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ai(e)` where `e.is_context_overflow()` | Yes          | Call `handle.compact(CompactionReason::Overflow, None)`, await it, then resend the prompt                                                                                              |
| `Ai(e)` — other provider errors         | Sometimes    | `ProviderTransport` already retries transient errors (429s, 5xx, timeouts) per `RetryConfig`. What surfaces here has exhausted retries — usually means resend later or fix the request |
| `Compaction(msg)`                       | Sometimes    | The summarization call failed. Try again, or switch to a different `model` for compaction by setting a new config and retrying                                                         |
| `Busy`                                  | Yes          | A prompt is already in flight. Either wait for `AgentEnd`, or call `handle.steer(msg)` to inject mid-prompt                                                                            |
| `ChannelFull { channel }`               | Yes          | The non-blocking `try_*` variant hit backpressure. Retry after a brief delay, or use the awaiting variant                                                                              |
| `ActorPanic(reason)`                    | No           | The actor task is dead. The handle is permanently inert. Spawn a fresh agent (and optionally seed it with the dead one's `messages().await` if you snapshotted them before)            |
| `AgentNotFound { id }`                  | Yes          | The id is unknown to the registry (evicted under LRU, never spawned, or already detached by a respec). Re-spawn or look up the correct id via `manager.find_agent` / `manager.snapshot` |
| `AgentBusy { id }`                      | Yes          | The operation needs an idle agent. Call `handle.abort()` (hard) or `handle.interrupt()` (graceful), wait for `AgentEnd`, then retry                                                    |
| `RespecRolledBack { id, source }`       | Yes          | The new spec failed to start; the agent is still alive under its previous spec. Inspect `source` (often `ActorPanic` from a broken spec) and decide whether to retry, log, or give up   |
| `WorktreeSetupFailed { reason }`        | Sometimes    | Git or filesystem failure setting up an isolated worktree. Verify the repo is in a clean state and the parent path is writable; retry the spawn                                        |
| `Other(msg)`                            | Depends      | Reserved for unstructured conditions (channel-closed-after-shutdown, internal invariants). Read the message; channel-closed is not recoverable                                         |

When in doubt, log the error, surface it to the user, and treat the
agent as compromised — `handle.health()` tells you whether the
underlying actor is still alive and, if dead, why.

---

## 14. Testing

Enable the `test-utils` feature:

```toml
[dev-dependencies]
tau-agent = { path = "…", features = ["test-utils"] }
```

```rust
use tau_agent::test_utils::*;

#[tokio::test]
async fn smoke() {
    let transport = MockTransport::new().with_text_response("Hello!");
    let (handle, collector) = spawn_test_agent(transport, vec![]);

    handle.prompt_and_wait("hi").await.unwrap();
    collector.wait_for_end().await;

    assert_eq!(collector.assistant_messages()[0].text(), "Hello!");
}
```

Available fixtures:

| Fixture                                             | Purpose                                                                                        |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `MockTransport`                                     | Queue canned events                                                                            |
| `TextTransport`                                     | Always returns a fixed text response                                                           |
| `ToolCallTransport`                                 | Emits a single tool call                                                                       |
| `SlowTransport`                                     | Adds artificial latency                                                                        |
| `CapturingTransport`                                | Records every `run()` call                                                                     |
| `ErrorTransport`                                    | Yields a provider error                                                                        |
| `PanicTransport`                                    | Panics in `run()` (tests actor recovery)                                                       |
| `EchoTool`                                          | Returns its arguments verbatim                                                                 |
| `FailTool`                                          | Returns `ToolResult::error`                                                                    |
| `SlowTool`                                          | Sleeps before returning                                                                        |
| `PanicTool`                                         | Panics in `execute`                                                                            |
| `EventCollector`                                    | Drain `AgentEvent`s for assertions                                                             |
| `spawn_test_agent`                                  | One-call agent + event collector                                                               |
| `make_test_config` / `test_config`                  | Default `AgentConfig` (`test_config` adds a stock `system_prompt`)                             |
| `make_execution_context`                            | Dummy `ExecutionContext` for tool unit tests                                                   |
| `make_assistant_message` / `make_tool_call_message` | Message constructors (use `Message::user` / `Message::tool_result` from `tau_ai` for the rest) |

### Test-utils notes

- **`MockTransport` queues multi-turn responses**. `with_text_response`,
  `with_tool_call_response`, `with_error`, and `with_events` each
  push one response onto an internal `VecDeque`. The N-th call to
  `transport.run()` pops the N-th queued response. Chain them for
  multi-turn flows:

  ```rust
  let transport = MockTransport::new()
      .with_tool_call_response("read", "tc1", json!({"path": "foo"}))
      .with_text_response("done");
  ```

  Running out of queued responses panics — set up enough for the
  test's expected turn count.

- **`CapturingTransport`** records every call. Access via
  `.calls() -> Vec<CapturedCall>` where:

  ```rust
  pub struct CapturedCall {
      pub messages: Vec<Message>,
      pub system_prompt: Option<String>,
      pub tool_names: Vec<String>,
      pub model_id: String,
  }
  ```

  The transport always replies with a fixed `text` response so the
  agent makes progress. Use it to assert what the agent sent to the
  LLM (system prompt, message history, tool registry).

- **`make_execution_context()`** returns an `ExecutionContext` with:
  - `cwd = "/tmp"`,
  - `cancel = CancellationToken::new()` (never tripped),
  - `progress` wired to a fresh broadcast channel **whose receiver
    is dropped immediately** — calls to `progress.send(...)` are
    no-ops at runtime (no subscribers). Don't rely on it for
    event-based assertions.
  - `interaction = None`,
  - `file_access` is a real, fresh `FileAccessTracker`,
  - `agent_id = None`, `subagent_depth = 0`.

  Use it for unit-testing a tool's `execute` logic in isolation. For
  event-stream behavior, prefer `spawn_test_agent`.

- **`EventCollector` query methods**:
  - `wait_for_end()` — block until `AgentEnd` or `Error` (5 s timeout, panics on miss).
  - `wait_for_event(pred)` — block until any event matches (5 s timeout).
  - `wait_for_event_timeout(pred, dur)` — explicit timeout.
  - `wait_for_count(n)` — block until at least `n` events collected.
  - `events()` — clone the full event vector.
  - `take_events()` — drain and return the buffer.
  - `assistant_messages()` — filter to `MessageEnd.message` only.
  - `event_names()` — return event type names as `&'static str` for diff-friendly assertions.
  - `count()` — current event count.

  Note: `EventCollector` **panics** if the broadcast channel lags
  (256 events behind). For high-volume tests, drain promptly or
  raise the channel capacity.

---

## 15. Type cheatsheet

Imports for typical hosts:

```rust
use tau_agent::{
    // builder + spawn
    AgentBuilder, AgentConfig, AgentConfigBuilder, AgentHandle, AgentSeed, DequeueMode,
    ProviderTransport, PromptResult, Transport,

    // events
    AgentEvent, ConsoleLevel, ConsoleLine, FleetEvent, SubagentOutcome, ToolApprovalOutcome,

    // tools
    BoxedTool, Concurrency, ExecutionContext, FileAccessTracker, ProgressSender,
    Tool, ToolCategory, ToolResult,

    // approval
    ApprovalDecision, ApprovalPolicy, ArgMatch, ArgPattern, AutoAcceptAll,
    DefaultPolicy, RulePolicy, ToolRisk, ToolRule,

    // interactions
    InteractionKind, InteractionRequest, InteractionResponse, QuestionOption,

    // compaction
    CompactionConfig, CompactionReason, CompactionThreshold,

    // inspection
    AgentHealth, ContextStats, Conversation, ToolInfo,

    // fleet
    AgentManager, AgentSnapshot, AgentSpec, AgentStatus, FleetSnapshot,
    Isolation, SpawnOpts, SubagentResult,

    // errors
    Error, Result,
};
```

### Trait & type quick reference

| Type                     | Module | Role                                     |
| ------------------------ | ------ | ---------------------------------------- |
| `AgentBuilder`           | core   | Configure + spawn                        |
| `AgentHandle`            | core   | Channel-based control surface (Clone)    |
| `AgentConfig`            | core   | Per-agent immutable-after-spawn config   |
| `Tool` (trait)           | core   | User-defined capability                  |
| `Transport` (trait)      | core   | Provider abstraction                     |
| `ApprovalPolicy` (trait) | core   | Decide gate/auto/reject                  |
| `AgentEvent`             | types  | Per-agent broadcast event variant        |
| `FleetEvent`             | types  | Fleet (manager) event variant            |
| `Conversation`           | types  | Mutable per-turn state                   |
| `Error` / `Result`       | types  | Runtime errors                           |
| `ContextStats`           | types  | Context-window snapshot                  |
| `ToolInfo`               | types  | Tool registry snapshot with policy state |
| `AgentManager`           | fleet  | Multi-agent orchestrator                 |
| `AgentSpec`              | fleet  | Reusable subagent definition             |
| `SpawnOpts`              | fleet  | Per-spawn knobs                          |
| `SubagentResult`         | fleet  | Spawn/send completion                    |
| `FleetSnapshot`          | fleet  | Synchronous fleet view                   |

---

## 16. Common patterns

### A. Single-shot prompt with tool approval

A CLI that runs one prompt, gates risky tools through the terminal,
and prints the assistant's final answer.

> Placeholders used below: `MyBashTool` is your `Tool` impl;
> `read_yes_no()` is whatever your CLI uses to read y/n from stdin.

```rust
use std::sync::Arc;
use tau_agent::{
    AgentBuilder, AgentConfig, AgentEvent, DefaultPolicy, InteractionKind,
    InteractionRequest, InteractionResponse, ProviderTransport,
};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> tau_agent::Result<()> {
    let model = tau_ai::get_model_by_id("claude-opus-4-7").expect("model exists");
    let config = AgentConfig::builder(model)
        .system_prompt("You are a careful coding assistant.")
        .build();
    let transport = Arc::new(ProviderTransport::new());

    let (interaction_tx, mut interaction_rx) = mpsc::channel::<InteractionRequest>(32);

    let mut builder = AgentBuilder::new(config, transport);
    builder.add_tool(Arc::new(MyBashTool));
    builder.set_interaction_sender(interaction_tx);
    builder.set_approval_policy(Arc::new(DefaultPolicy));

    let mut events = builder.subscribe();        // before spawn → catches AgentStart
    let handle = builder.spawn().await?;

    // Terminal-side approval loop.
    tokio::spawn(async move {
        while let Some(req) = interaction_rx.recv().await {
            if let InteractionKind::Typed { schema_id, payload } = &req.kind {
                if schema_id == "tool.confirm" {
                    println!("Run {}? [y/N]", payload["tool_name"]);
                    let approved = read_yes_no();
                    let resp = if approved {
                        InteractionResponse::Approved { payload: None }
                    } else {
                        InteractionResponse::Rejected { reason: "user declined".into() }
                    };
                    let _ = req.response_tx.send(resp);
                }
            }
        }
    });

    handle.prompt("clean up the dead code in src/foo.rs").await?;

    while let Ok(ev) = events.recv().await {
        match ev {
            AgentEvent::MessageEnd { message } => println!("{}", message.text()),
            AgentEvent::AgentEnd { .. } => break,
            AgentEvent::Error { message } => {
                eprintln!("error: {message}");
                break;
            }
            _ => {}
        }
    }
    Ok(())
}
```

### B. Long-running agent with compaction

A daemon that holds a single agent for hours, lets it self-compact on
threshold, and recovers from context overflow.

> Placeholder: `next_user_input()` is your input source (Slack, gRPC,
> stdin — whatever drives the daemon).

```rust
use tau_agent::{
    AgentBuilder, AgentConfig, CompactionConfig, CompactionReason,
    CompactionThreshold, Error, ProviderTransport,
};
use std::sync::Arc;

let model = tau_ai::get_model_by_id("claude-opus-4-7").expect("model exists");
let config = AgentConfig::builder(model)
    .compaction(CompactionConfig {
        enabled: true,
        // Start compacting earlier than the default — 16% headroom
        // instead of 8% — on whatever model this agent runs.
        reserve: CompactionThreshold::Fraction(0.16),
        // Preserve a decent recent window (15% of context_window).
        keep_recent: CompactionThreshold::Fraction(0.15),
    })
    .build();

let handle = AgentBuilder::new(config, Arc::new(ProviderTransport::new())).spawn();

loop {
    let prompt = next_user_input().await;

    match handle.prompt_and_wait(&prompt).await {
        Ok(()) => {}
        Err(e) if e.is_context_overflow() => {
            // Reactive compaction (the runtime triggers proactive
            // compaction on its own; this branch handles the case
            // where a single prompt blew past the window).
            eprintln!("compacting after overflow…");
            let rx = handle.compact(CompactionReason::Overflow, None).await?;
            let _ = rx.await;
            handle.prompt_and_wait(&prompt).await?;
        }
        Err(Error::Busy) => {
            // Another caller beat us to it; steer instead.
            handle.steer(tau_ai::Message::user(&prompt)).await?;
        }
        Err(Error::ActorPanic(reason)) => {
            // Dead actor — spawn a fresh one in the next iteration.
            eprintln!("agent died: {reason}");
            return Err(Error::ActorPanic(reason));
        }
        Err(e) => return Err(e),
    }
}
```

### C. Parent spawning a research subagent

A parent tool that spawns a focused research agent, awaits its result,
and folds the answer into its own response.

> Placeholders: `ReadFileTool`, `WebSearchTool`, `parent_config`, and
> `parent_spec` are project-specific values you bring. The pattern
> is the wiring, not the tool implementations.

```rust
use std::sync::Arc;
use tau_agent::{
    AgentManager, AgentSpec, ExecutionContext, ProviderTransport, SpawnOpts,
    Tool, ToolCategory, ToolResult,
};
use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

struct ResearchTool {
    manager: Arc<AgentManager>,
    research_spec: Arc<AgentSpec>,
}

#[async_trait]
impl Tool for ResearchTool {
    fn name(&self) -> &str { "research" }
    fn description(&self) -> &str { "Spawn a research subagent on a focused question." }
    fn category(&self) -> ToolCategory { ToolCategory::Other }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "question": { "type": "string" } },
            "required": ["question"],
        })
    }

    async fn execute(&self, args: Value, ctx: ExecutionContext) -> ToolResult {
        let question = args["question"].as_str().unwrap_or("").to_string();

        let opts = SpawnOpts {
            description: format!("research: {}", truncate(&question, 40)),
            // One deeper than the parent so recursion limits work.
            subagent_depth: ctx.subagent_depth + 1,
            ..SpawnOpts::default()
        };

        match self.manager
            .spawn(self.research_spec.clone(), question, opts, ctx.cancel.clone())
            .await
        {
            Ok(result) => ToolResult::text(result.text)
                .with_details(serde_json::json!({
                    "agent_id": result.agent_id,
                    "tokens": { "in": result.input_tokens, "out": result.output_tokens },
                    "duration_ms": result.duration_ms,
                })),
            Err(e) => ToolResult::error(format!("research failed: {e}")),
        }
    }
}
```

Wire it up at the parent's builder:

```rust
let manager = Arc::new(AgentManager::new(
    parent_config.clone(),
    Arc::new(ProviderTransport::new()),
    20,
));
// Subscribe before spawning anything if the host needs fleet events:
let _fleet_events = manager.subscribe();

let research_spec = Arc::new(AgentSpec {
    system_prompt: "You research a single question and return the answer concisely.".into(),
    tools: vec![Arc::new(ReadFileTool), Arc::new(WebSearchTool)],
    max_turns: 25,
});

let mut builder = AgentBuilder::new(parent_config, Arc::new(ProviderTransport::new()));
builder.add_tool(Arc::new(ResearchTool {
    manager: Arc::clone(&manager),
    research_spec,
}));
let parent_handle = builder.spawn().await?;
manager.adopt(&parent_handle, "root", parent_spec);
```

The `manager.adopt(...)` call registers the parent so its events
appear in `manager.snapshot()` alongside the spawned research agents.

---

For deeper architectural detail (phase machine, state split,
registry invariant) see `README.md`. For runnable examples
see the integration tests under `tests/`.
