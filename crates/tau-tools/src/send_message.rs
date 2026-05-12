//! SendMessage tool — send a message to a running or idle agent.
use crate::cached_schema;

use std::sync::{Arc, Weak};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::{AgentManager, AgentStatus};
use tau_agent::{ExecutionContext, Tool, ToolResult};

#[derive(Deserialize, JsonSchema)]
struct SendMessageArgs {
    /// Agent name (matched against description) or agent ID
    to: String,
    /// Message to send to the agent
    message: String,
}

/// Tool for sending messages between agents. Holds a `Weak<AgentManager>`
/// for the same reason as `AgentTool`: a strong `Arc<AgentManager>` here
/// would form a `manager → agent_specs → AgentSpec → tools →
/// SendMessageTool → manager` cycle that prevents the manager from
/// dropping when the host releases its reference.
pub struct SendMessageTool {
    manager: Weak<AgentManager>,
}

impl SendMessageTool {
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self {
            manager: Arc::downgrade(&manager),
        }
    }
}

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> &str {
        "Send a message to another agent by name or ID. If the agent is currently \
         running, the message is injected immediately (the agent sees it after its \
         current tool completes). If the agent is idle, it is resumed with the message \
         and this tool blocks until it finishes."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(SendMessageArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let manager = match self.manager.upgrade() {
            Some(m) => m,
            None => {
                return ToolResult::error("SendMessageTool: parent AgentManager has been dropped");
            }
        };

        let args: SendMessageArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let to = &args.to;
        let message = &args.message;

        let located = match manager.find_agent(to) {
            Some(found) => found,
            None => {
                return ToolResult::error(format!(
                    "No agent found matching '{}'. It may have been evicted or never spawned.",
                    to
                ));
            }
        };
        let agent_id = located.agent_id;
        let description = located.description;
        let status = located.status;

        // Wrap message with sender attribution so the receiving agent
        // knows this came from another agent, not the user.
        let wrapped = format!("[Message from parent agent]: {}", message);

        match status {
            AgentStatus::Running => {
                manager
                    .send_to_running(&agent_id, tau_ai::Message::user(&wrapped))
                    .await;
                ToolResult::text(format!(
                    "Message delivered to running agent '{}' ({})",
                    description, agent_id
                ))
            }
            AgentStatus::Idle | AgentStatus::Adopted => {
                match manager.send(&agent_id, &wrapped, ctx.cancel).await {
                    Ok(result) => ToolResult::text(format!(
                        "{}\n[Agent {} resumed | {} in + {} out tokens | {} tool calls | {}ms]",
                        result.text,
                        result.agent_id,
                        result.input_tokens,
                        result.output_tokens,
                        result.tool_use_count,
                        result.duration_ms,
                    )),
                    Err(e) => ToolResult::error(format!("Failed to resume agent: {}", e)),
                }
            }
        }
    }
}
