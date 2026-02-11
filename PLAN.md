# tau Implementation Plan

## Phase 1: Complete Core Tools

### 1.1 Glob Tool
- [ ] Add `glob` crate dependency (already in workspace)
- [ ] Implement glob tool in tau-cli/src/tools/glob.rs
- [ ] Pattern matching for file discovery
- [ ] Respect .gitignore patterns (optional)

### 1.2 Grep Tool
- [ ] Implement grep tool in tau-cli/src/tools/grep.rs
- [ ] Regex-based content search
- [ ] Return file:line:content matches
- [ ] Limit results to prevent overwhelming context

### 1.3 List Tool
- [ ] Implement list tool in tau-cli/src/tools/list.rs
- [ ] Directory listing with metadata (size, modified time)
- [ ] Tree view option for nested display

## Phase 2: Agent Improvements

### 2.1 Token Tracking
- [ ] Track input/output/cache tokens per turn
- [ ] Accumulate total usage across conversation
- [ ] Display cost estimates based on model pricing

### 2.2 Error Handling & Retry
- [ ] Exponential backoff for rate limits (429)
- [ ] Retry on transient network errors
- [ ] Graceful handling of context length exceeded

### 2.3 Cancellation
- [ ] Ctrl+C to cancel current generation
- [ ] Clean abort of streaming responses
- [ ] Cancel long-running tool executions

## Phase 3: Full TUI

### 3.1 Wire Up Existing Widgets
- [ ] Create tau-tui app state structure
- [ ] Integrate MessageList widget
- [ ] Integrate InputBox widget
- [ ] Integrate StatusBar widget

### 3.2 Event Loop
- [ ] Crossterm event handling (keyboard, resize)
- [ ] Agent event handling (streaming, tool calls)
- [ ] Render loop with proper refresh rate

### 3.3 Scrolling & Navigation
- [ ] Page up/down through message history
- [ ] Auto-scroll to bottom on new content
- [ ] Scroll position indicator

### 3.4 Input Handling
- [ ] Multi-line input with Shift+Enter
- [ ] Input history (up/down arrows)
- [ ] Basic line editing (home, end, delete word)

## Phase 4: Session Management

### 4.1 Persistence
- [ ] Save conversation to ~/.local/share/tau/sessions/
- [ ] JSON format for messages and metadata
- [ ] Auto-save after each turn

### 4.2 Session Commands
- [ ] /save [name] - save current session
- [ ] /load [name] - load a session
- [ ] /list - list saved sessions
- [ ] /clear - clear current conversation

## Phase 5: Additional Providers

### 5.1 OpenAI Provider
- [ ] Implement streaming for chat completions API
- [ ] Map tool calls to OpenAI format
- [ ] Handle function_call responses

### 5.2 Google Provider
- [ ] Implement streaming for Gemini API
- [ ] Map tool calls to Google format
- [ ] Handle different content structure

## Phase 6: Configuration

### 6.1 Config File
- [ ] Create ~/.config/tau/config.toml
- [ ] Default model and provider
- [ ] API key storage (or env var reference)
- [ ] Custom system prompt path

### 6.2 Environment
- [ ] Support ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
- [ ] TAU_MODEL override
- [ ] TAU_CONFIG_PATH override

---

## Priority Order

1. **Phase 1** - Tools are essential for a useful coding agent
2. **Phase 2.1** - Token tracking gives visibility into usage
3. **Phase 3** - TUI makes it pleasant to use
4. **Phase 2.2-2.3** - Robustness improvements
5. **Phase 4** - Session management for longer work
6. **Phase 5** - Additional providers expand compatibility
7. **Phase 6** - Configuration is convenience

## Current Status

- [x] tau-ai: Core types, Anthropic streaming
- [x] tau-agent: Agent loop, tool execution, events
- [x] tau-tui: Widget definitions
- [x] tau-cli: Full CLI with all tools
- [x] Release build working

### Phase 1: Complete ✓
- [x] Glob tool
- [x] Grep tool
- [x] List tool

### Phase 2: Complete ✓
- [x] Token tracking with cost display
- [x] Error handling & retry (exponential backoff for 429, 5xx, network errors)
- [x] Cancellation (Ctrl+C / Esc in TUI aborts current operation)

### Phase 3: Complete ✓
- [x] TUI mode wired up (`-t` flag)
- [x] Message list with scrolling
- [x] Input box
- [x] Status bar with token/cost info

### Phase 4: Complete ✓
- [x] Session persistence (JSONL format)
- [x] `--sessions` to list saved sessions
- [x] `--resume <id>` to resume a session

### Phase 6: Complete ✓
- [x] Config file support (`~/.config/tau/config.toml`)
- [x] `--init-config` to create default config
- [x] API keys can be in config or environment

### Phase 5: Complete ✓
- [x] OpenAI provider (Chat Completions API with streaming)
- [x] Google provider (Gemini API with streaming)
- [x] Session auto-save during conversations
- [x] Load messages into agent when resuming session

## All Phases Complete! 🎉

The tau CLI is now feature-complete with:
- All 7 tools (bash, read, write, edit, glob, grep, list)
- Three providers (Anthropic, OpenAI, Google)
- Full TUI mode
- Session persistence and resume
- Configuration file support
- Token tracking and cost display
- Retry with exponential backoff
- Cancellation support
