//! Agent state management and execution

use parking_lot::Mutex;
use std::sync::Arc;
use tau_ai::{Content, Message, Model, ReasoningLevel, Usage};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::{
    events::AgentEvent,
    tool::{BoxedTool, ToolResult, to_api_tool},
    transport::{AgentRunConfig, Transport},
};

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// System prompt
    pub system_prompt: Option<String>,
    /// Model to use
    pub model: Model,
    /// Reasoning/thinking level
    pub reasoning: ReasoningLevel,
    /// Maximum tokens per response
    pub max_tokens: Option<u32>,
}

/// Agent state
#[derive(Default)]
pub struct AgentState {
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Whether currently streaming
    pub is_streaming: bool,
    /// Current streaming message (partial)
    pub stream_message: Option<Message>,
    /// Total usage across all turns
    pub total_usage: Usage,
    /// Last error
    pub error: Option<String>,
}

/// The main agent that orchestrates conversations
pub struct Agent {
    config: AgentConfig,
    state: AgentState,
    tools: Vec<BoxedTool>,
    transport: Arc<dyn Transport>,
    event_tx: broadcast::Sender<AgentEvent>,
    /// Cancellation token, wrapped in Arc<Mutex> for external access during prompt
    cancel: Arc<Mutex<CancellationToken>>,
}

impl Agent {
    /// Create a new agent
    pub fn new(config: AgentConfig, transport: Arc<dyn Transport>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        Self {
            config,
            state: AgentState::default(),
            tools: vec![],
            transport,
            event_tx,
            cancel: Arc::new(Mutex::new(CancellationToken::new())),
        }
    }

    /// Subscribe to agent events
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Get the current state
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Set the system prompt
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.config.system_prompt = Some(prompt.into());
    }

    /// Set the model
    pub fn set_model(&mut self, model: Model) {
        self.config.model = model;
    }

    /// Set the reasoning level
    pub fn set_reasoning(&mut self, level: ReasoningLevel) {
        self.config.reasoning = level;
    }

    /// Add a tool
    pub fn add_tool(&mut self, tool: BoxedTool) {
        self.tools.push(tool);
    }

    /// Set tools (replaces existing)
    pub fn set_tools(&mut self, tools: Vec<BoxedTool>) {
        self.tools = tools;
    }

    /// Get tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Clear all messages
    pub fn clear_messages(&mut self) {
        self.state.messages.clear();
        self.state.total_usage = Usage::default();
        self.state.error = None;
    }

    /// Set messages (for loading from session)
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.state.messages = messages;
    }

    /// Get all messages
    pub fn messages(&self) -> &[Message] {
        &self.state.messages
    }

    /// Abort the current operation
    pub fn abort(&self) {
        self.cancel.lock().cancel();
    }

    /// Get a handle to cancel the current operation from outside
    /// Returns an Arc that can be used to cancel even while a prompt is running
    pub fn cancel_handle(&self) -> Arc<Mutex<CancellationToken>> {
        Arc::clone(&self.cancel)
    }

    /// Send a message and run the agent loop
    pub async fn prompt(&mut self, input: &str) -> Result<(), String> {
        self.prompt_with_content(vec![Content::text(input)]).await
    }

    /// Send a message with multiple content blocks
    pub async fn prompt_with_content(&mut self, content: Vec<Content>) -> Result<(), String> {
        use futures::StreamExt;

        // Reset cancellation token
        *self.cancel.lock() = CancellationToken::new();

        // Create user message
        let user_message = Message::User {
            content,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        // Build run config
        let run_config = AgentRunConfig {
            system_prompt: self.config.system_prompt.clone(),
            tools: self.tools.iter().map(|t| to_api_tool(t.as_ref())).collect(),
            model: self.config.model.clone(),
            reasoning: Some(self.config.reasoning),
            max_tokens: self.config.max_tokens,
            temperature: None,
        };

        self.state.is_streaming = true;
        self.state.error = None;

        // Emit agent start
        let _ = self.event_tx.send(AgentEvent::AgentStart);

        let mut turn = 0u32;
        let mut messages_to_add = vec![user_message.clone()];

        loop {
            turn += 1;

            // Get all messages for context
            let context_messages: Vec<Message> = self
                .state
                .messages
                .iter()
                .cloned()
                .chain(messages_to_add.iter().cloned())
                .collect();

            // For subsequent turns, use an empty user message
            let current_user_msg = if turn == 1 {
                user_message.clone()
            } else {
                // After tool execution, we continue with tool results already in context
                Message::User {
                    content: vec![],
                    timestamp: chrono::Utc::now().timestamp_millis(),
                }
            };

            // Run the transport
            let cancel_token = self.cancel.lock().clone();
            let mut event_stream = match self
                .transport
                .run(
                    context_messages,
                    current_user_msg,
                    &run_config,
                    cancel_token,
                )
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    self.state.error = Some(e.to_string());
                    self.state.is_streaming = false;
                    let _ = self.event_tx.send(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    return Err(e.to_string());
                }
            };

            let mut assistant_message: Option<Message> = None;
            let mut turn_usage = Usage::default();

            // Process events
            while let Some(event) = event_stream.next().await {
                // Forward event to subscribers
                let _ = self.event_tx.send(event.clone());

                match event {
                    AgentEvent::MessageUpdate { message } => {
                        self.state.stream_message = Some(message);
                    }
                    AgentEvent::MessageEnd { message } => {
                        self.state.stream_message = None;
                        assistant_message = Some(message);
                    }
                    AgentEvent::TurnEnd { usage, .. } => {
                        turn_usage = usage;
                    }
                    AgentEvent::Error { message } => {
                        self.state.error = Some(message.clone());
                        self.state.is_streaming = false;
                        return Err(message);
                    }
                    _ => {}
                }
            }

            // Update total usage
            self.state.total_usage.input += turn_usage.input;
            self.state.total_usage.output += turn_usage.output;
            self.state.total_usage.cache_read += turn_usage.cache_read;
            self.state.total_usage.cache_write += turn_usage.cache_write;
            self.state.total_usage.thinking += turn_usage.thinking;

            // Process assistant message
            if let Some(msg) = assistant_message {
                // Add messages to state
                for m in messages_to_add.drain(..) {
                    self.state.messages.push(m);
                }
                self.state.messages.push(msg.clone());

                // Check for tool calls
                let tool_calls = msg.tool_calls();
                if tool_calls.is_empty() {
                    // No tool calls, we're done
                    break;
                }

                // Execute tools
                let mut tool_results = vec![];
                for (id, name, args) in tool_calls {
                    // Find the tool
                    let tool = self.tools.iter().find(|t| t.name() == name);

                    let _ = self.event_tx.send(AgentEvent::ToolExecutionStart {
                        tool_call_id: id.to_string(),
                        tool_name: name.to_string(),
                        arguments: args.clone(),
                    });

                    let result = if let Some(tool) = tool {
                        let cancel = self.cancel.lock().clone();
                        tool.execute(id, args.clone(), cancel).await
                    } else {
                        ToolResult::error(format!("Tool not found: {}", name))
                    };

                    let _ = self.event_tx.send(AgentEvent::ToolExecutionEnd {
                        tool_call_id: id.to_string(),
                        tool_name: name.to_string(),
                        result: result.text_content(),
                        is_error: result.is_error,
                    });

                    // Create tool result message
                    let tool_result_msg =
                        Message::tool_result(id, name, result.content, result.is_error);
                    tool_results.push(tool_result_msg);
                }

                // Add tool results for next turn
                messages_to_add = tool_results;
            } else {
                // No message, something went wrong
                break;
            }
        }

        self.state.is_streaming = false;

        // Emit agent end
        let _ = self.event_tx.send(AgentEvent::AgentEnd {
            total_turns: turn,
            total_usage: self.state.total_usage.clone(),
        });

        Ok(())
    }
}
