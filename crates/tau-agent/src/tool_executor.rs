//! Tool execution helpers: run a single tool, validate arguments, check content.

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::broadcast;

use crate::events::AgentEvent;
use crate::tool::{BoxedTool, ExecutionContext, ToolResult, send_event};

/// Execute a single tool: emit events, check guard/validation, run, emit end event.
/// Standalone function so it can be called both inline (sequential) and from a
/// spawned task (parallel) without borrowing the actor state.
pub(crate) async fn run_single_tool(
    tool: Option<BoxedTool>,
    id: String,
    name: String,
    args: Value,
    validator: Option<Arc<jsonschema::Validator>>,
    event_tx: broadcast::Sender<AgentEvent>,
    ctx: ExecutionContext,
) -> ToolResult {
    let activity = tool
        .as_ref()
        .map(|t| t.activity_description(&args))
        .unwrap_or_else(|| format!("Running {}", name));
    send_event(
        &event_tx,
        AgentEvent::ToolExecutionStart {
            tool_call_id: id.clone(),
            tool_name: name.clone(),
            arguments: args.clone(),
            activity,
        },
    );

    let result = if let Some(tool) = tool {
        let validation_error = validator.and_then(|v| validate_with_validator(&args, &v));
        if let Some(err) = validation_error {
            ToolResult::error(err)
        } else {
            tool.execute(args, ctx).await
        }
    } else {
        ToolResult::error(format!("Tool not found: {}", name))
    };

    send_event(
        &event_tx,
        AgentEvent::ToolExecutionEnd {
            tool_call_id: id,
            tool_name: name,
            result: result.text_content(),
            is_error: result.is_error,
        },
    );

    result
}

/// Check if a message has meaningful content worth preserving.
pub(crate) fn has_meaningful_content(message: &tau_ai::Message) -> bool {
    use tau_ai::{Content, Message};

    let content = match message {
        Message::Assistant { content, .. }
        | Message::User { content, .. }
        | Message::ToolResult { content, .. }
        | Message::SystemInjection { content, .. } => content,
    };

    content.iter().any(|c| match c {
        Content::Text { text } => !text.trim().is_empty(),
        Content::Thinking { thinking, .. } => !thinking.trim().is_empty(),
        Content::ToolCall { name, .. } => !name.is_empty(),
        Content::Image { .. } => true,
        Content::RedactedThinking { .. } => true,
        Content::ServerToolUse { .. } => true,
        Content::ServerToolResult { .. } => true,
    })
}

/// Validate tool arguments using a pre-compiled validator.
/// Returns `Some(error_message)` if validation fails, `None` if valid.
pub(crate) fn validate_with_validator(
    args: &Value,
    validator: &jsonschema::Validator,
) -> Option<String> {
    let errors: Vec<String> = validator
        .iter_errors(args)
        .map(|e| {
            let path = e.instance_path().to_string();
            if path.is_empty() {
                e.to_string()
            } else {
                format!("{}: {}", path, e)
            }
        })
        .collect();

    if errors.is_empty() {
        None
    } else {
        Some(format!(
            "Tool argument validation failed:\n{}",
            errors.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_ai::{AssistantMetadata, Content, Message};

    fn validate_tool_args(args: &serde_json::Value, schema: &serde_json::Value) -> Option<String> {
        let validator = match jsonschema::validator_for(schema) {
            Ok(v) => v,
            Err(_) => return None,
        };
        validate_with_validator(args, &validator)
    }

    fn simple_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "count": { "type": "integer" }
            },
            "required": ["path"]
        })
    }

    #[test]
    fn test_validate_args_valid() {
        let args = serde_json::json!({"path": "/foo.rs", "count": 10});
        assert!(validate_tool_args(&args, &simple_schema()).is_none());
    }

    #[test]
    fn test_validate_args_missing_required() {
        let args = serde_json::json!({"count": 5});
        let err = validate_tool_args(&args, &simple_schema());
        assert!(err.is_some());
        let msg = err.unwrap();
        assert!(msg.contains("validation failed"), "got: {}", msg);
    }

    #[test]
    fn test_meaningful_content_text() {
        let msg = Message::Assistant {
            content: vec![Content::text("hello")],
            metadata: AssistantMetadata::default(),
        };
        assert!(has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_whitespace_only() {
        let msg = Message::Assistant {
            content: vec![Content::text("   \n\t  ")],
            metadata: AssistantMetadata::default(),
        };
        assert!(!has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_empty() {
        let msg = Message::Assistant {
            content: vec![],
            metadata: AssistantMetadata::default(),
        };
        assert!(!has_meaningful_content(&msg));
    }

    #[test]
    fn test_meaningful_content_tool_call() {
        let msg = Message::Assistant {
            content: vec![Content::tool_call("id1", "read", serde_json::json!({}))],
            metadata: AssistantMetadata::default(),
        };
        assert!(has_meaningful_content(&msg));
    }
}
