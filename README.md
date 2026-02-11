# tau

**tau** is an AI-powered coding agent CLI written in Rust. It provides a powerful interface for interacting with AI models (Anthropic, OpenAI, Google) with specialized tools for software development.

## Features

- 🤖 **Multiple AI Providers**: Anthropic Claude, OpenAI GPT, Google Gemini
- 🛠️ **7 Built-in Tools**: bash, read, write, edit, glob, grep, list
- 💬 **Two Modes**: Interactive CLI or full TUI (Terminal User Interface)
- 💾 **Session Management**: Save, resume, and list conversation sessions
- 📊 **Token Tracking**: Real-time token usage and cost estimates
- 🔄 **Smart Retry**: Exponential backoff for rate limits and transient errors
- ⚙️ **Configuration**: TOML-based config with environment variable support
- 🔐 **OAuth Support**: Direct login to Anthropic (no API key needed)

## Installation

### From Source

```bash
cargo build --release
sudo cp target/release/tau /usr/local/bin/
```

### From Cargo

```bash
cargo install --path crates/tau-cli
```

## Quick Start

### 1. Initialize Configuration

```bash
tau --init-config
```

This creates `~/.config/tau/config.toml` with default settings.

### 2. Configure API Access

**Option A: OAuth (Anthropic only)**
```bash
tau --login anthropic
```

**Option B: API Keys**

Edit `~/.config/tau/config.toml`:
```toml
[api_keys]
anthropic = "sk-ant-..."
openai = "sk-..."
google = "..."
```

Or use environment variables:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

### 3. Start Coding!

**Interactive mode:**
```bash
tau
```

**TUI mode:**
```bash
tau -t
```

**One-shot command:**
```bash
tau -c "refactor the main function to use async/await"
```

## Usage Examples

### Basic Conversation
```bash
$ tau
tau> Help me fix the compile errors in src/main.rs

# The agent will:
# 1. Read the file
# 2. Analyze the errors
# 3. Propose fixes
# 4. Execute edits
```

### Using Different Models
```bash
# Use GPT-4
tau -p openai -m gpt-4-turbo

# Use Gemini
tau -p google -m gemini-2.0-flash-exp

# Enable reasoning mode (Claude only)
tau -r --reasoning-level high
```

### Session Management
```bash
# Resume a previous session
tau --resume abc123

# List all sessions
tau --sessions

# Work continues automatically saving
tau -t  # Auto-saves after each turn
```

### Working Directory
```bash
# Run tau in a specific directory
tau -w /path/to/project
```

## Available Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands with timeout support |
| `read` | Read file contents with offset/limit options |
| `write` | Write content to files (creates/overwrites) |
| `edit` | Precise text replacement in files |
| `glob` | Find files matching patterns (`**/*.rs`) |
| `grep` | Search file contents with regex |
| `list` | List directory contents with metadata |

The agent automatically chooses which tools to use based on your request.

## Configuration

### Config File Location
- Linux/macOS: `~/.config/tau/config.toml`
- Windows: `%APPDATA%\tau\config.toml`

### Example Configuration
```toml
[defaults]
model = "claude-sonnet-4-5-20250929"
provider = "anthropic"
reasoning_level = "off"

[api_keys]
anthropic = "sk-ant-..."
# openai = "sk-..."
# google = "..."

[session]
auto_save = true
session_dir = "~/.local/share/tau/sessions"

[tools]
bash_timeout_seconds = 60
max_file_size_bytes = 10485760  # 10MB
```

### Environment Variables
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENAI_API_KEY` - OpenAI API key
- `GOOGLE_API_KEY` - Google API key
- `TAU_CONFIG_PATH` - Override config file location

## TUI Mode

Press `tau -t` to enter full TUI mode with:
- **Message list** with syntax highlighting
- **Scrolling** through conversation history (Page Up/Down)
- **Status bar** with token usage and costs
- **Input box** with multi-line support
- **Keyboard shortcuts**:
  - `Enter` - Send message
  - `Ctrl+C` / `Esc` - Cancel current operation
  - `Ctrl+D` - Exit
  - `Page Up/Down` - Scroll messages

## Project Structure

```
tau/
├── crates/
│   ├── tau-ai/        # AI provider implementations
│   ├── tau-agent/     # Agent loop and tool execution
│   ├── tau-tui/       # TUI widgets (ratatui)
│   └── tau-cli/       # CLI application
├── Cargo.toml         # Workspace configuration
└── PLAN.md           # Development roadmap
```

## Development

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test
```

### Lint
```bash
cargo clippy && cargo fmt --check
```

### Run in Development
```bash
cargo run --release -- -t
```

## Model Support

### Anthropic
- `claude-sonnet-4-5-20250929` (default)
- `claude-sonnet-4-20250514`
- `claude-opus-4-20250514`
- `claude-3-5-sonnet-20241022`
- Extended thinking mode support

### OpenAI
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`
- All chat completion models

### Google
- `gemini-2.0-flash-exp` (default)
- `gemini-1.5-pro-latest`
- `gemini-1.5-flash-latest`
- All Gemini models

## Session Storage

Sessions are saved in JSONL format:
- Linux/macOS: `~/.local/share/tau/sessions/`
- Windows: `%APPDATA%\tau\sessions\`

Each session contains:
- Unique ID and timestamp
- Full conversation history
- Model and provider information
- Token usage and costs

## Cost Tracking

tau tracks token usage in real-time:
- Input tokens (with prompt caching)
- Output tokens
- Reasoning tokens (Claude extended thinking)
- Estimated costs based on current provider pricing

Example output:
```
Total: 1.5K in / 850 out / 0 cache / 2.3K reasoning ($0.0234)
```

## Error Handling

tau includes robust error handling:
- **Rate limits (429)**: Exponential backoff with retry
- **Server errors (5xx)**: Automatic retry with backoff
- **Network errors**: Transient error retry
- **Cancellation**: Ctrl+C to abort current operation
- **Context length**: Clear error messages when exceeded

## License

MIT

## Contributing

Contributions welcome! This project uses:
- Rust 1.85+ (edition 2024)
- `cargo fmt` for formatting
- `cargo clippy` for linting

## Related Projects

Part of the [pi-mono](https://github.com/badlogic/pi-mono) repository.

## Support

For issues, feature requests, or questions, please open an issue on GitHub.
