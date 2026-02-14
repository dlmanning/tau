---
name: tau
description: Run tau (the AI coding agent built in this repo) in CLI mode with a single prompt. Use when the user wants to test tau, run it against a task, or see its output.
argument-hint: [prompt]
disable-model-invocation: true
allowed-tools: Bash
---

# Run tau in CLI mode

Run the `tau` binary from this workspace in non-interactive (CLI) mode using the `-c` flag.

## Build first if needed

Before running, ensure the binary is built:

```
cargo build --release -p tau-cli
```

The binary is at `target/release/tau`.

## Run with the user's prompt

```
./target/release/tau -c "$ARGUMENTS"
```

## Flags reference

Use these flags when the user requests specific behavior:

| Flag | Description |
|------|-------------|
| `-c <PROMPT>` | Non-interactive mode (required for CLI usage) |
| `-m <MODEL>` | Model to use (default: claude-sonnet-4-5-20250929) |
| `-p <PROVIDER>` | Provider: anthropic, openai, google, groq, cerebras, xai, openrouter, ollama |
| `-r` | Enable reasoning/thinking (medium level) |
| `--reasoning-level <LEVEL>` | off, minimal, low, medium, high |
| `-w <DIR>` | Working directory |
| `-v` | Verbose/debug output |
| `--resume <ID>` | Resume a previous session |

## Examples

Basic run:
```
./target/release/tau -c "Explain what this project does"
```

With a specific model:
```
./target/release/tau -m claude-opus-4-6 -c "Review the error handling in src/"
```

With reasoning enabled:
```
./target/release/tau -r -c "Find and fix any bugs in the compaction module"
```

With a different working directory:
```
./target/release/tau -w /some/other/project -c "Summarize this codebase"
```

## Output format

In `-c` mode, tau prints:
- The prompt echoed as `tau> <prompt>`
- Streaming assistant text
- Tool executions as `[Running tool_name...]` / `[tool_name: result]`
- Context compaction events if triggered
- Final token usage and cost: `[Tokens: X in, Y out | Cost: $Z.ZZZZ]`

## Important notes

- tau requires an API key. It checks OAuth credentials, config file (`~/.config/tau/config.toml`), and environment variables (e.g. `ANTHROPIC_API_KEY`) in that order.
- If the build fails, run `cargo build --release -p tau-cli 2>&1` and show the user the errors.
- Always use `--release` for a responsive experience.
- The `-c` flag runs a single prompt and exits. Do not use interactive or TUI mode from within this skill.
