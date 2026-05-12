//! Reduce a stream of `AgentEvent`s into a turn's structured outcome.
//!
//! Testable without a transport — feed events via `observe()`.

use tau_ai::{Message, Usage};

use crate::types::events::AgentEvent;

#[derive(Default)]
pub struct StreamReducer {
    partial: Option<Message>,
    assistant: Option<Message>,
    usage: Usage,
    error: Option<String>,
}

pub struct StreamOutcome {
    /// Final assistant message (set on `MessageEnd`).
    pub assistant_message: Option<Message>,
    /// Token usage reported in `TurnEnd`.
    pub usage: Usage,
    /// Error message from a stream-level `Error` event.
    pub error: Option<String>,
    /// Last `MessageUpdate` payload before an interruption — useful
    /// for preserving partial text on overflow / cancellation.
    pub partial_message: Option<Message>,
}

impl StreamReducer {
    pub fn observe(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::MessageUpdate { message } => self.partial = Some(message.clone()),
            AgentEvent::MessageEnd { message } => {
                self.partial = None;
                self.assistant = Some(message.clone());
            }
            AgentEvent::TurnEnd { usage, .. } => self.usage = usage.clone(),
            AgentEvent::Error { message } => self.error = Some(message.clone()),
            _ => {}
        }
    }

    pub fn finalize(self) -> StreamOutcome {
        StreamOutcome {
            assistant_message: self.assistant,
            usage: self.usage,
            error: self.error,
            partial_message: self.partial,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_ai::{AssistantMetadata, Content};

    fn assistant(text: &str) -> Message {
        Message::Assistant {
            content: vec![Content::text(text)],
            metadata: AssistantMetadata::default(),
        }
    }

    #[test]
    fn normal_stream_yields_final_message_and_usage() {
        let mut r = StreamReducer::default();
        r.observe(&AgentEvent::MessageUpdate {
            message: assistant("partial"),
        });
        r.observe(&AgentEvent::MessageEnd {
            message: assistant("done"),
        });
        r.observe(&AgentEvent::TurnEnd {
            turn_number: 1,
            message: assistant("done"),
            usage: Usage {
                input: 100,
                output: 50,
                ..Default::default()
            },
        });
        let o = r.finalize();
        assert_eq!(o.assistant_message.unwrap().text(), "done");
        assert_eq!(o.usage.input, 100);
        assert!(o.error.is_none());
        assert!(o.partial_message.is_none());
    }

    #[test]
    fn error_mid_stream_preserves_partial() {
        let mut r = StreamReducer::default();
        r.observe(&AgentEvent::MessageUpdate {
            message: assistant("partial content"),
        });
        r.observe(&AgentEvent::Error {
            message: "context overflow".into(),
        });
        let o = r.finalize();
        assert!(o.assistant_message.is_none());
        assert_eq!(o.error.as_deref(), Some("context overflow"));
        assert_eq!(o.partial_message.unwrap().text(), "partial content");
    }
}
