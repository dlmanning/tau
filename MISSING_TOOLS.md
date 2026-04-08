# Tools Claude Code has that tau does not

## High impact

- **LSP** — Language Server Protocol: go-to-definition, find-references, hover, call hierarchy
- **Agent** — Spawn subagents for parallel work (Explore, Plan, Verify)
- **WebSearch** — Search the web, return results with citations
- **WebFetch** — Fetch and analyze content from URLs
- **NotebookEdit** — Edit Jupyter notebook cells

## Medium impact

- **TaskCreate / TaskGet / TaskList / TaskUpdate / TaskOutput** — Structured task tracking with dependencies
- **EnterPlanMode / ExitPlanMode** — Explicit design-before-implement workflow
- **AskUserQuestion** — Multiple-choice questions with previews
- **SendMessage** — Cross-agent messaging
- **EnterWorktree / ExitWorktree** — Isolated git worktree management

## Lower impact

- **MCPTool / McpAuthTool** — Execute tools from MCP servers
- **ToolSearchTool / ListMcpResources / ReadMcpResource** — Discover and read MCP resources
- **SkillTool** — Slash commands (/commit, /review-pr, etc.)
- **CronCreate / CronDelete / CronList / RemoteTrigger** — Scheduled remote agents
- **PowerShell** — Windows command execution
