//! Streaming event types and utilities

use std::pin::Pin;

use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use crate::types::{Api, AssistantMetadata, Content, Message, Provider, StopReason, Usage};

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
        signature: Option<String>,
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
    Thinking {
        text: String,
        signature: Option<String>,
    },
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
                // Auto-create buffer if TextStart was not emitted (e.g. OpenAI, Google)
                if self.content_buffers.len() <= *content_index {
                    self.ensure_buffer(*content_index, ContentBuffer::Text(String::new()));
                }
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
                self.ensure_buffer(
                    *content_index,
                    ContentBuffer::Thinking {
                        text: String::new(),
                        signature: None,
                    },
                );
            }
            MessageEvent::ThinkingDelta {
                content_index,
                delta,
            } => {
                if let Some(ContentBuffer::Thinking { text: thinking, .. }) =
                    self.content_buffers.get_mut(*content_index)
                {
                    thinking.push_str(delta);
                }
            }
            MessageEvent::ThinkingEnd {
                content_index,
                thinking,
                signature,
            } => {
                if *content_index < self.content_buffers.len() {
                    // Use signature from event; fall back to any accumulated via deltas
                    let sig = signature.clone().or_else(|| {
                        if let ContentBuffer::Thinking { signature: s, .. } =
                            &self.content_buffers[*content_index]
                        {
                            s.clone()
                        } else {
                            None
                        }
                    });
                    self.content_buffers[*content_index] = ContentBuffer::Thinking {
                        text: thinking.clone(),
                        signature: sig,
                    };
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
                ContentBuffer::Thinking { text, signature } => Content::Thinking {
                    thinking: text,
                    signature,
                },
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
                ContentBuffer::Thinking { text, signature } => Content::Thinking {
                    thinking: text.clone(),
                    signature: signature.clone(),
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

/// Tracks streaming content blocks and emits [`MessageEvent`]s.
///
/// Providers parse their SSE format and feed deltas into the accumulator,
/// which handles start/end event lifecycle and final [`Message`] construction.
/// This eliminates duplicated state-tracking and message-building logic
/// across providers.
pub struct StreamAccumulator {
    blocks: Vec<AccBlock>,
    usage: Usage,
    stop_reason: Option<StopReason>,
    error: Option<String>,
    api: Api,
    provider: Provider,
    model_id: String,
}

#[derive(Debug, Default)]
enum AccBlock {
    #[default]
    Empty,
    Text {
        text: String,
        started: bool,
        ended: bool,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
        started: bool,
        ended: bool,
    },
    ToolCall {
        id: String,
        name: String,
        args_json: String,
        started: bool,
        ended: bool,
    },
    RedactedThinking {
        data: String,
    },
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

impl StreamAccumulator {
    /// Create a new accumulator and the initial [`MessageEvent::Start`] event.
    pub fn new(api: Api, provider: Provider, model_id: String) -> (Self, MessageEvent) {
        let acc = Self {
            blocks: Vec::new(),
            usage: Usage::default(),
            stop_reason: None,
            error: None,
            api,
            provider,
            model_id: model_id.clone(),
        };
        let start = MessageEvent::Start {
            message: Message::Assistant {
                content: vec![],
                metadata: AssistantMetadata {
                    model: Some(model_id),
                    ..Default::default()
                },
            },
        };
        (acc, start)
    }

    /// Append a text delta. Auto-creates and starts the block if needed.
    pub fn text_delta(&mut self, index: usize, delta: &str) -> Vec<MessageEvent> {
        self.ensure_block(index);
        let mut events = Vec::new();

        if matches!(self.blocks[index], AccBlock::Empty) {
            self.blocks[index] = AccBlock::Text {
                text: String::new(),
                started: false,
                ended: false,
            };
        }

        if let AccBlock::Text {
            ref mut text,
            ref mut started,
            ..
        } = self.blocks[index]
        {
            if !*started {
                *started = true;
                events.push(MessageEvent::TextStart {
                    content_index: index,
                });
            }
            text.push_str(delta);
            events.push(MessageEvent::TextDelta {
                content_index: index,
                delta: delta.to_string(),
            });
        }
        events
    }

    /// Explicitly start a text block (Anthropic content_block_start).
    pub fn text_start(&mut self, index: usize) -> Vec<MessageEvent> {
        self.ensure_block(index);
        self.blocks[index] = AccBlock::Text {
            text: String::new(),
            started: true,
            ended: false,
        };
        vec![MessageEvent::TextStart {
            content_index: index,
        }]
    }

    /// Explicitly end a text block.
    pub fn text_end(&mut self, index: usize) -> Vec<MessageEvent> {
        if let Some(AccBlock::Text {
            text, ended, ..
        }) = self.blocks.get_mut(index)
        {
            *ended = true;
            vec![MessageEvent::TextEnd {
                content_index: index,
                text: text.clone(),
            }]
        } else {
            vec![]
        }
    }

    /// Start a thinking block.
    pub fn thinking_start(&mut self, index: usize) -> Vec<MessageEvent> {
        self.ensure_block(index);
        self.blocks[index] = AccBlock::Thinking {
            thinking: String::new(),
            signature: None,
            started: true,
            ended: false,
        };
        vec![MessageEvent::ThinkingStart {
            content_index: index,
        }]
    }

    /// Append a thinking delta.
    pub fn thinking_delta(&mut self, index: usize, delta: &str) -> Vec<MessageEvent> {
        if let Some(AccBlock::Thinking {
            thinking, ..
        }) = self.blocks.get_mut(index)
        {
            thinking.push_str(delta);
            vec![MessageEvent::ThinkingDelta {
                content_index: index,
                delta: delta.to_string(),
            }]
        } else {
            vec![]
        }
    }

    /// Accumulate a signature delta (no events emitted).
    pub fn thinking_signature_delta(&mut self, index: usize, sig_delta: &str) {
        if let Some(AccBlock::Thinking {
            signature, ..
        }) = self.blocks.get_mut(index)
        {
            match signature {
                Some(s) => s.push_str(sig_delta),
                None => *signature = Some(sig_delta.to_string()),
            }
        }
    }

    /// End a thinking block. `override_signature` replaces any accumulated signature.
    pub fn thinking_end(
        &mut self,
        index: usize,
        override_signature: Option<String>,
    ) -> Vec<MessageEvent> {
        if let Some(AccBlock::Thinking {
            thinking,
            signature,
            ended,
            ..
        }) = self.blocks.get_mut(index)
        {
            if let Some(sig) = override_signature {
                *signature = Some(sig);
            }
            *ended = true;
            vec![MessageEvent::ThinkingEnd {
                content_index: index,
                thinking: thinking.clone(),
                signature: signature.clone(),
            }]
        } else {
            vec![]
        }
    }

    /// Start a tool call block.
    pub fn tool_call_start(
        &mut self,
        index: usize,
        id: impl Into<String>,
        name: impl Into<String>,
    ) -> Vec<MessageEvent> {
        self.ensure_block(index);
        let id = id.into();
        let name = name.into();
        self.blocks[index] = AccBlock::ToolCall {
            id: id.clone(),
            name: name.clone(),
            args_json: String::new(),
            started: true,
            ended: false,
        };
        vec![MessageEvent::ToolCallStart {
            content_index: index,
            id,
            name,
        }]
    }

    /// Append a tool call arguments delta.
    pub fn tool_call_delta(&mut self, index: usize, delta: &str) -> Vec<MessageEvent> {
        if let Some(AccBlock::ToolCall {
            args_json, ..
        }) = self.blocks.get_mut(index)
        {
            args_json.push_str(delta);
            vec![MessageEvent::ToolCallDelta {
                content_index: index,
                delta: delta.to_string(),
            }]
        } else {
            vec![]
        }
    }

    /// Explicitly end a tool call block.
    pub fn tool_call_end(&mut self, index: usize) -> Vec<MessageEvent> {
        if let Some(AccBlock::ToolCall {
            id,
            name,
            args_json,
            ended,
            ..
        }) = self.blocks.get_mut(index)
        {
            *ended = true;
            let arguments =
                serde_json::from_str(args_json).unwrap_or(serde_json::Value::Null);
            vec![MessageEvent::ToolCallEnd {
                content_index: index,
                id: id.clone(),
                name: name.clone(),
                arguments,
            }]
        } else {
            vec![]
        }
    }

    /// Record a redacted thinking block (no events emitted).
    pub fn add_redacted_thinking(&mut self, index: usize, data: String) {
        self.ensure_block(index);
        self.blocks[index] = AccBlock::RedactedThinking { data };
    }

    /// Record a server tool use block (no events emitted).
    pub fn add_server_tool_use(
        &mut self,
        index: usize,
        id: String,
        name: String,
        input: serde_json::Value,
    ) {
        self.ensure_block(index);
        self.blocks[index] = AccBlock::ServerToolUse { id, name, input };
    }

    /// Mutable reference to usage for incremental updates.
    pub fn usage_mut(&mut self) -> &mut Usage {
        &mut self.usage
    }

    /// Set the stop reason.
    pub fn set_stop_reason(&mut self, reason: StopReason) {
        self.stop_reason = Some(reason);
    }

    /// Record an error. When [`finish`](Self::finish) is called, it will emit
    /// [`MessageEvent::Error`] instead of [`MessageEvent::Done`].
    pub fn set_error(&mut self, msg: impl Into<String>) {
        self.error = Some(msg.into());
        self.stop_reason = Some(StopReason::Error);
    }

    /// End the block at `index`, dispatching to the appropriate end method.
    /// `override_signature` is applied only to thinking blocks.
    pub fn end_block(
        &mut self,
        index: usize,
        override_signature: Option<String>,
    ) -> Vec<MessageEvent> {
        if index >= self.blocks.len() {
            return vec![];
        }
        let is_text = matches!(self.blocks[index], AccBlock::Text { .. });
        let is_thinking = matches!(self.blocks[index], AccBlock::Thinking { .. });
        let is_tool = matches!(self.blocks[index], AccBlock::ToolCall { .. });

        if is_thinking {
            self.thinking_end(index, override_signature)
        } else if is_text {
            self.text_end(index)
        } else if is_tool {
            self.tool_call_end(index)
        } else {
            vec![]
        }
    }

    /// Create an error event for immediate yield-and-return (without calling finish).
    pub fn error_event(msg: impl Into<String>) -> MessageEvent {
        MessageEvent::Error {
            message: msg.into(),
        }
    }

    /// End all open blocks, build the final message, and return terminal events.
    pub fn finish(self) -> Vec<MessageEvent> {
        let Self {
            blocks,
            usage,
            stop_reason,
            error,
            api,
            provider,
            model_id,
        } = self;

        let mut events = Vec::new();

        if let Some(error_msg) = error {
            events.push(MessageEvent::Error {
                message: error_msg,
            });
            return events;
        }

        // Single pass: emit End events for open blocks and collect content
        let mut content = Vec::new();
        for (index, block) in blocks.into_iter().enumerate() {
            match block {
                AccBlock::Empty => {}
                AccBlock::Text {
                    text,
                    started,
                    ended,
                } => {
                    if started && !ended {
                        events.push(MessageEvent::TextEnd {
                            content_index: index,
                            text: text.clone(),
                        });
                    }
                    if !text.is_empty() {
                        content.push(Content::Text { text });
                    }
                }
                AccBlock::Thinking {
                    thinking,
                    signature,
                    started,
                    ended,
                } => {
                    if started && !ended {
                        events.push(MessageEvent::ThinkingEnd {
                            content_index: index,
                            thinking: thinking.clone(),
                            signature: signature.clone(),
                        });
                    }
                    content.push(Content::Thinking {
                        thinking,
                        signature,
                    });
                }
                AccBlock::ToolCall {
                    id,
                    name,
                    args_json,
                    started,
                    ended,
                } => {
                    let arguments = serde_json::from_str(&args_json)
                        .unwrap_or(serde_json::Value::Null);
                    if started && !ended {
                        events.push(MessageEvent::ToolCallEnd {
                            content_index: index,
                            id: id.clone(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                        });
                    }
                    content.push(Content::ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                AccBlock::RedactedThinking { data } => {
                    content.push(Content::RedactedThinking { data });
                }
                AccBlock::ServerToolUse { id, name, input } => {
                    content.push(Content::ServerToolUse { id, name, input });
                }
            }
        }

        let stop_reason = stop_reason.unwrap_or(StopReason::Stop);
        let final_message = Message::Assistant {
            content,
            metadata: AssistantMetadata {
                api: Some(api),
                provider: Some(provider),
                model: Some(model_id),
                usage: usage.clone(),
                stop_reason: Some(stop_reason),
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            },
        };

        events.push(MessageEvent::Done {
            message: final_message,
            stop_reason,
            usage,
        });

        events
    }

    fn ensure_block(&mut self, index: usize) {
        while self.blocks.len() <= index {
            self.blocks.push(AccBlock::Empty);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_delta_without_text_start() {
        // Simulates OpenAI/Google behavior: TextDelta without prior TextStart
        let mut builder = MessageBuilder::new();
        builder.process_event(&MessageEvent::Start {
            message: Message::Assistant {
                content: vec![],
                metadata: Default::default(),
            },
        });
        // No TextStart — go straight to TextDelta
        builder.process_event(&MessageEvent::TextDelta {
            content_index: 0,
            delta: "Hello ".to_string(),
        });
        builder.process_event(&MessageEvent::TextDelta {
            content_index: 0,
            delta: "world".to_string(),
        });

        let msg = builder.build();
        let text = msg.text();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_text_with_start_and_end() {
        // Simulates Anthropic behavior: TextStart → TextDelta → TextEnd
        let mut builder = MessageBuilder::new();
        builder.process_event(&MessageEvent::Start {
            message: Message::Assistant {
                content: vec![],
                metadata: Default::default(),
            },
        });
        builder.process_event(&MessageEvent::TextStart { content_index: 0 });
        builder.process_event(&MessageEvent::TextDelta {
            content_index: 0,
            delta: "Hello".to_string(),
        });
        builder.process_event(&MessageEvent::TextEnd {
            content_index: 0,
            text: "Hello".to_string(),
        });

        let msg = builder.build();
        assert_eq!(msg.text(), "Hello");
    }

    #[test]
    fn test_tool_call_building() {
        let mut builder = MessageBuilder::new();
        builder.process_event(&MessageEvent::Start {
            message: Message::Assistant {
                content: vec![],
                metadata: Default::default(),
            },
        });
        builder.process_event(&MessageEvent::ToolCallStart {
            content_index: 0,
            id: "call_1".to_string(),
            name: "bash".to_string(),
        });
        builder.process_event(&MessageEvent::ToolCallEnd {
            content_index: 0,
            id: "call_1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        });

        let msg = builder.build();
        match &msg {
            Message::Assistant { content, .. } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    Content::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        assert_eq!(id, "call_1");
                        assert_eq!(name, "bash");
                        assert_eq!(arguments["command"], "ls");
                    }
                    other => panic!("expected ToolCall, got {:?}", other),
                }
            }
            other => panic!("expected Assistant, got {:?}", other),
        }
    }
}
