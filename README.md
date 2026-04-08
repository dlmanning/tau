# tau

**tau** is an AI-powered coding agent CLI written in Rust. It provides a powerful interface for interacting with AI models (Anthropic, OpenAI, Google) with specialized tools for software development.

## Features

- **Multiple AI Providers**: Anthropic Claude, OpenAI, Google Gemini
- **8 Built-in Tools**: bash, read, write, edit, glob, grep, list, lsp
- **LSP Code Intelligence**: Go-to-definition, find-references, hover, document symbols via language servers
- **TUI**: Full terminal UI with model selector, token/cost tracking, cache stats
- **Prompt Caching**: Scoped cache control with TTL, dynamic system prompt splitting
- **Adaptive Thinking**: Model-driven reasoning with budget or adaptive mode
- **Stream Watchdog**: Detects and recovers from stalled API connections
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

The LSP tool auto-detects installed language servers: rust-analyzer, typescript-language-server, pyright, gopls, clangd.

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

**Status bar** shows model, thinking level, token usage, cache stats, and cost:
```
claude-sonnet-4-5 │ thinking: medium │ Ready | 98 in, 1131 out | cache: 28.0kr 10.4kw | $0.0645
```

**Keyboard shortcuts:**
- `Enter` — Send message
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
│   ├── tau-agent/     # Agent loop, tool execution, transport, compaction
│   ├── tau-tui/       # TUI widgets (ratatui)
│   └── tau-cli/       # CLI application, tools, LSP client, config
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
