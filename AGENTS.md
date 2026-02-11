# tau Development Rules

## First Message
If the user did not give you a concrete task in their first message,
read PLAN.md to understand the project status, then ask what they'd like to work on.

## Project Structure
- `crates/tau-ai/` - Core AI types and provider implementations (Anthropic, OpenAI, Google)
- `crates/tau-agent/` - Agent loop, tool execution, events
- `crates/tau-cli/` - CLI application with all tools
- `crates/tau-tui/` - TUI widgets (ratatui-based)

## Code Quality
- Use idiomatic Rust - prefer `Result` and `Option` over panics
- Run `cargo clippy` and `cargo fmt` before committing
- Add tests for new functionality

## Commands
- Build: `cargo build --release`
- Test: `cargo test`
- Check: `cargo clippy && cargo fmt --check`
- NEVER commit unless user asks

## Style
- Keep answers short and concise
