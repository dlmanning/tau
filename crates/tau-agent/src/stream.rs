//! Stream reduction: collapse a stream of `AgentEvent`s into a turn outcome.

use tau_ai::{Message, Usage};

use crate::events::AgentEvent;

/// Accumulates events from a single model turn into a structured outcome.
///
/// Testable without a transport — just feed events via `observe()`.
#[derive(Default)]
pub struct StreamReducer {
    partial: Option<Message>,
    assistant: Option<Message>,
    usage: Usage,
    error: Option<String>,
}

/// The result of reducing one model turn's event stream.
pub struct StreamOutcome {
    /// The final assistant message (if the stream completed normally).
    pub assistant_message: Option<Message>,
    /// Token usage for this turn.
    pub usage: Usage,
    /// Error message (if the stream produced an error event).
    pub error: Option<String>,
    /// Partial message from an interrupted stream (useful for error recovery).
    pub partial_message: Option<Message>,
}

impl StreamReducer {
    /// Feed an event into the reducer.
    pub fn observe(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::MessageUpdate { message } => {
                self.partial = Some(message.clone());
            }
            AgentEvent::MessageEnd { message } => {
                self.partial = None;
                self.assistant = Some(message.clone());
            }
            AgentEvent::TurnEnd { usage, .. } => {
                self.usage = usage.clone();
            }
            AgentEvent::Error { message } => {
                self.error = Some(message.clone());
            }
            _ => {}
        }
    }

    /// Consume the reducer and produce the turn outcome.
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

    fn text_message(text: &str) -> Message {
        Message::Assistant {
            content: vec![Content::text(text)],
            metadata: AssistantMetadata::default(),
        }
    }

    #[test]
    fn test_normal_stream() {
        let mut reducer = StreamReducer::default();

        reducer.observe(&AgentEvent::MessageUpdate {
            message: text_message("partial"),
        });
        reducer.observe(&AgentEvent::MessageEnd {
            message: text_message("complete"),
        });
        reducer.observe(&AgentEvent::TurnEnd {
            turn_number: 1,
            message: text_message("complete"),
            usage: Usage {
                input: 100,
                output: 50,
                ..Default::default()
            },
        });

        let outcome = reducer.finalize();
        assert!(outcome.assistant_message.is_some());
        assert_eq!(outcome.assistant_message.unwrap().text(), "complete");
        assert_eq!(outcome.usage.input, 100);
        assert_eq!(outcome.usage.output, 50);
        assert!(outcome.error.is_none());
        assert!(outcome.partial_message.is_none());
    }

    #[test]
    fn test_error_mid_stream() {
        let mut reducer = StreamReducer::default();

        reducer.observe(&AgentEvent::MessageUpdate {
            message: text_message("partial content"),
        });
        reducer.observe(&AgentEvent::Error {
            message: "context overflow".to_string(),
        });

        let outcome = reducer.finalize();
        assert!(outcome.assistant_message.is_none());
        assert_eq!(outcome.error.as_deref(), Some("context overflow"));
        assert!(outcome.partial_message.is_some());
        assert_eq!(outcome.partial_message.unwrap().text(), "partial content");
    }

    #[test]
    fn test_empty_stream() {
        let reducer = StreamReducer::default();
        let outcome = reducer.finalize();
        assert!(outcome.assistant_message.is_none());
        assert!(outcome.error.is_none());
        assert!(outcome.partial_message.is_none());
        assert_eq!(outcome.usage.input, 0);
    }

    #[test]
    fn test_message_end_clears_partial() {
        let mut reducer = StreamReducer::default();

        reducer.observe(&AgentEvent::MessageUpdate {
            message: text_message("draft"),
        });
        reducer.observe(&AgentEvent::MessageEnd {
            message: text_message("final"),
        });

        let outcome = reducer.finalize();
        // partial should be None because MessageEnd cleared it
        assert!(outcome.partial_message.is_none());
        assert_eq!(outcome.assistant_message.unwrap().text(), "final");
    }

    #[test]
    fn test_error_after_message_end() {
        let mut reducer = StreamReducer::default();

        reducer.observe(&AgentEvent::MessageEnd {
            message: text_message("complete"),
        });
        reducer.observe(&AgentEvent::Error {
            message: "late error".to_string(),
        });

        let outcome = reducer.finalize();
        // Both are set — caller checks error first
        assert!(outcome.assistant_message.is_some());
        assert_eq!(outcome.error.as_deref(), Some("late error"));
        // partial should be None (MessageEnd cleared it)
        assert!(outcome.partial_message.is_none());
    }

    #[test]
    fn test_error_without_partial() {
        let mut reducer = StreamReducer::default();

        reducer.observe(&AgentEvent::Error {
            message: "network error".to_string(),
        });

        let outcome = reducer.finalize();
        assert!(outcome.partial_message.is_none());
        assert_eq!(outcome.error.as_deref(), Some("network error"));
    }
}
