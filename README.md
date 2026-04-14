# tau

**tau** is an AI-powered coding agent CLI written in Rust. It provides a powerful interface for interacting with AI models (Anthropic, OpenAI, Google) with specialized tools for software development.

## Features

- **Multiple AI Providers**: Anthropic Claude, OpenAI, Google Gemini
- **12 Built-in Tools**: bash, read, write, edit, glob, grep, list, lsp, web_fetch, agent, send_message, ask_user
- **Web Search**: Server-side web search via Anthropic API (automatic for Anthropic models)
- **LSP Code Intelligence**: Go-to-definition, find-references, hover, document symbols via language servers
- **TUI**: Full terminal UI with inline message arrows, model selector, token/cost tracking
- **Prompt Caching**: Scoped cache control with TTL, dynamic system prompt splitting
- **Adaptive Thinking**: Model-driven reasoning with budget or adaptive mode
- **Stream Watchdog**: Detects and recovers from stalled API connections
- **Subagents**: Spawn foreground or background agents (General Purpose, Explore, Plan) with progress tracking
- **Steering**: Inject messages while the agent is working
- **Context Compaction**: Automatic summarization when approaching context window limits
- **Session Management**: Save, resume, and list conversation sessions
- **Smart Retry**: Exponential backoff for rate limits and transient errors
- **OAuth Support**: Direct login to Anthropic (no API key needed)

## Installation

```bash
cargo install --path crates/tau-cli
```

Or build from source:

```bash
cargo build --release
cp target/release/tau ~/.local/bin/
```

## Quick Start

### 1. Configure API Access

**Option A: OAuth (Anthropic only)**
```bash
tau --login anthropic
```

**Option B: API Keys**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or generate a config file:
```bash
tau --init-config
# Edit ~/.config/tau/config.toml
```

### 2. Start Coding

```bash
tau                    # TUI mode (default)
tau --no-tui           # Simple stdin/stdout mode
tau -c "fix the bug"   # One-shot command
tau -r                 # Enable reasoning (medium)
tau --reasoning-level high  # Deep reasoning
tau --resume <id>      # Resume a saved session
tau --sessions         # List saved sessions
```

## Available Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands with timeout and head+tail truncation |
| `read` | Read file contents with offset/limit |
| `write` | Write content to files |
| `edit` | Precise text replacement in files |
| `glob` | Find files matching patterns (`**/*.rs`) |
| `grep` | Search file contents with regex |
| `list` | List directory contents |
| `lsp` | Code intelligence via language servers (definition, references, hover, symbols) |
| `web_fetch` | Fetch URLs and convert HTML to markdown |
| `agent` | Spawn subagents for parallel or background work |
| `send_message` | Send a message to a running or idle subagent |
| `ask_user` | Present the user with a multiple-choice question |

The LSP tool auto-detects installed language servers: rust-analyzer, typescript-language-server, pyright, gopls, clangd.

Anthropic models also get **server-side web search** — the model can search the web directly during a conversation without using a client tool.

## Configuration

Config file: `~/.config/tau/config.toml`

```toml
model = "claude-sonnet-4-5-20250929"
provider = "anthropic"
reasoning_level = "off"        # off, minimal, low, medium, high
# thinking_adaptive = true     # model decides when to think

tui = true

[api_keys]
# anthropic = "sk-ant-..."
# openai = "sk-..."
# google = "..."

# [compaction]
# enabled = true
# reserve_tokens = 16384
# keep_recent_tokens = 20000

# [cache]
# scope = "org"                # "global" (1P only) or "org"
# ttl = "1h"                   # "1h" or "5m"
# prompt_boundary = "<!-- DYNAMIC_BOUNDARY -->"
```

### Environment Variables
- `ANTHROPIC_API_KEY` — Anthropic API key
- `OPENAI_API_KEY` — OpenAI API key
- `GOOGLE_API_KEY` — Google API key
- `TAU_CONFIG_PATH` — Override config file location

## TUI Mode

TUI is the default. Use `--no-tui` for simple mode.

The interface has four areas:

| Area | Description |
|------|-------------|
| **Header** | τ glyph (rainbow when working, green when idle), cwd, clock |
| **Conversation** | Message thread with inline arrows — `▶` user, `◀` assistant, `⚙` tools, `◇` agents |
| **Status line** | Model name, thinking level, token counts, cache stats, cost |
| **Input** | Text entry (active during both idle and processing for steering) |

```
τ { ~/git/myproject }                                     04/11/2026 12:20:30AM
┌─────────────────────────────────────────────────────────────────────────────┐
│ ▶ Fix the login bug                                                         │
│                                                                             │
│ ◀ I found the issue in auth.rs...                                           │
│                                                                             │
│ ⚙ Editing auth.rs                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
claude-sonnet-4-5 · think:high/a · 1.2k in, 3.4k out · $0.0438
> _
```

**Keyboard shortcuts:**
- `Enter` — Send message (or steer during processing)
- `Ctrl+K` — Model selector
- `Ctrl+L` — Clear conversation
- `Ctrl+C` — Cancel current operation
- `Ctrl+D` — Exit
- `Page Up/Down` — Scroll messages

**Slash commands:**
- `/thinking <level>` — Change reasoning level
- `/model` — Switch model
- `/session` — Session info
- `/clear` — Clear conversation

## Project Structure

```
tau/
├── crates/
│   ├── tau-ai/        # AI provider implementations (Anthropic, OpenAI, Google)
│   ├── tau-agent/     # Actor-based agent runtime, tool execution, compaction, subagent management
│   ├── tau-tools/     # Built-in tool implementations
│   └── tau-cli/       # CLI application, TUI, LSP client, config
└── Cargo.toml         # Workspace configuration
```

## Development

```bash
cargo build --release       # Build
cargo test                  # Test
cargo clippy && cargo fmt --check  # Lint
cargo run --release         # Run
```

## License

MIT
