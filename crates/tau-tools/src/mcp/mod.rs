//! MCP (Model Context Protocol) client support: tools from external
//! MCP servers exposed to the agent as regular [`Tool`]s.
//!
//! Servers are declared in the host's config and connected at startup
//! by [`McpManager::connect_all`]; each remote tool becomes an
//! [`McpTool`] named `mcp__<server>__<tool>`. Built on the official
//! `rmcp` SDK (stdio child processes and streamable-HTTP endpoints).
//!
//! ## v1 limitations (deliberate)
//!
//! - **Tools only** — resources, prompts, and sampling are not
//!   surfaced.
//! - **Startup snapshot** — `notifications/tools/list_changed` is
//!   ignored; the tool set is frozen when the agent spawns.
//! - **No MCP-side cancellation** — when a call times out or the
//!   prompt is aborted, the request future is dropped without sending
//!   `notifications/cancelled`; the server may keep working.
//! - **Top-level schema normalization only** — see [`schema`].
//!
//! [`Tool`]: tau_agent::Tool

mod config;
mod content;
mod manager;
mod naming;
mod risk;
mod schema;
mod tool;

pub use config::{McpServerSpec, McpTransportSpec, McpTrust};
pub use manager::McpManager;
