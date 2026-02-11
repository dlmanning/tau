//! Streaming event types and utilities

use crate::types::{Content, Message, StopReason, Usage};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use tokio_stream::Stream;

/// Events emitted during message streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageEvent {
    /// Initial message structure
    Start { message: Message },
    /// Text content started
    TextStart { content_index: usize },
    /// Text content delta
    TextDelta { content_index: usize, delta: String },
    /// Text content completed
    TextEnd { content_index: usize, text: String },
    /// Thinking content started
    ThinkingStart { content_index: usize },
    /// Thinking content delta
    ThinkingDelta { content_index: usize, delta: String },
    /// Thinking content completed
    ThinkingEnd {
        content_index: usize,
        thinking: String,
    },
    /// Tool call started
    ToolCallStart {
        content_index: usize,
        id: String,
        name: String,
    },
    /// Tool call arguments delta (partial JSON)
    ToolCallDelta { content_index: usize, delta: String },
    /// Tool call completed
    ToolCallEnd {
        content_index: usize,
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// Message completed successfully
    Done {
        message: Message,
        stop_reason: StopReason,
        usage: Usage,
    },
    /// Error occurred
    Error { message: String },
}

impl MessageEvent {
    /// Check if this is a terminal event (Done or Error)
    pub fn is_terminal(&self) -> bool {
        matches!(self, MessageEvent::Done { .. } | MessageEvent::Error { .. })
    }

    /// Get the final message if this is a Done event
    pub fn into_message(self) -> Option<Message> {
        match self {
            MessageEvent::Done { message, .. } => Some(message),
            _ => None,
        }
    }
}

/// A stream of message events
pub type MessageEventStream = Pin<Box<dyn Stream<Item = MessageEvent> + Send>>;

/// Builder for constructing an assistant message from streaming events
#[derive(Debug, Default)]
pub struct MessageBuilder {
    #[allow(dead_code)]
    content: Vec<Content>,
    content_buffers: Vec<ContentBuffer>,
    usage: Usage,
    stop_reason: Option<StopReason>,
}

#[derive(Debug)]
enum ContentBuffer {
    Text(String),
    Thinking(String),
    ToolCall {
        id: String,
        name: String,
        arguments_json: String,
    },
}

impl MessageBuilder {
    /// Create a new message builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a streaming event and update the message state
    pub fn process_event(&mut self, event: &MessageEvent) {
        match event {
            MessageEvent::TextStart { content_index } => {
                self.ensure_buffer(*content_index, ContentBuffer::Text(String::new()));
            }
            MessageEvent::TextDelta {
                content_index,
                delta,
            } => {
                if let Some(ContentBuffer::Text(text)) =
                    self.content_buffers.get_mut(*content_index)
                {
                    text.push_str(delta);
                }
            }
            MessageEvent::TextEnd {
                content_index,
                text,
            } => {
                if *content_index < self.content_buffers.len() {
                    self.content_buffers[*content_index] = ContentBuffer::Text(text.clone());
                }
            }
            MessageEvent::ThinkingStart { content_index } => {
                self.ensure_buffer(*content_index, ContentBuffer::Thinking(String::new()));
            }
            MessageEvent::ThinkingDelta {
                content_index,
                delta,
            } => {
                if let Some(ContentBuffer::Thinking(thinking)) =
                    self.content_buffers.get_mut(*content_index)
                {
                    thinking.push_str(delta);
                }
            }
            MessageEvent::ThinkingEnd {
                content_index,
                thinking,
            } => {
                if *content_index < self.content_buffers.len() {
                    self.content_buffers[*content_index] =
                        ContentBuffer::Thinking(thinking.clone());
                }
            }
            MessageEvent::ToolCallStart {
                content_index,
                id,
                name,
            } => {
                self.ensure_buffer(
                    *content_index,
                    ContentBuffer::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments_json: String::new(),
                    },
                );
            }
            MessageEvent::ToolCallDelta {
                content_index,
                delta,
            } => {
                if let Some(ContentBuffer::ToolCall { arguments_json, .. }) =
                    self.content_buffers.get_mut(*content_index)
                {
                    arguments_json.push_str(delta);
                }
            }
            MessageEvent::ToolCallEnd {
                content_index,
                id,
                name,
                arguments,
            } => {
                if *content_index < self.content_buffers.len() {
                    self.content_buffers[*content_index] = ContentBuffer::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments_json: arguments.to_string(),
                    };
                }
            }
            MessageEvent::Done {
                stop_reason, usage, ..
            } => {
                self.stop_reason = Some(*stop_reason);
                self.usage = usage.clone();
            }
            _ => {}
        }
    }

    /// Build the final message
    pub fn build(self) -> Message {
        let content: Vec<Content> = self
            .content_buffers
            .into_iter()
            .map(|buf| match buf {
                ContentBuffer::Text(text) => Content::Text { text },
                ContentBuffer::Thinking(thinking) => Content::Thinking { thinking },
                ContentBuffer::ToolCall {
                    id,
                    name,
                    arguments_json,
                } => {
                    let arguments =
                        serde_json::from_str(&arguments_json).unwrap_or(serde_json::Value::Null);
                    Content::ToolCall {
                        id,
                        name,
                        arguments,
                    }
                }
            })
            .collect();

        Message::Assistant {
            content,
            metadata: crate::types::AssistantMetadata {
                usage: self.usage,
                stop_reason: self.stop_reason,
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            },
        }
    }

    /// Get the current partial message state
    pub fn current_content(&self) -> Vec<Content> {
        self.content_buffers
            .iter()
            .map(|buf| match buf {
                ContentBuffer::Text(text) => Content::Text { text: text.clone() },
                ContentBuffer::Thinking(thinking) => Content::Thinking {
                    thinking: thinking.clone(),
                },
                ContentBuffer::ToolCall {
                    id,
                    name,
                    arguments_json,
                } => {
                    let arguments = serde_json::from_str(arguments_json).unwrap_or_default();
                    Content::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments,
                    }
                }
            })
            .collect()
    }

    fn ensure_buffer(&mut self, index: usize, default: ContentBuffer) {
        while self.content_buffers.len() <= index {
            self.content_buffers
                .push(ContentBuffer::Text(String::new()));
        }
        self.content_buffers[index] = default;
    }
}
