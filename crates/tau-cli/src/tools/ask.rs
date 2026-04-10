//! AskUserQuestion tool — ask the user a multiple-choice question.

use async_trait::async_trait;
use serde_json::json;
use tau_agent::interaction::{InteractionKind, InteractionRequest, InteractionResponse, QuestionOption};
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};

pub struct AskTool;

impl AskTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for AskTool {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn description(&self) -> &str {
        "Ask the user a question with a set of options. Use when you need \
         clarification, want the user to choose between approaches, or need \
         a decision before proceeding. Returns the label of the selected option."
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "Short label for this option"
                            },
                            "description": {
                                "type": "string",
                                "description": "Longer explanation of what this option means"
                            }
                        },
                        "required": ["label", "description"]
                    },
                    "minItems": 2,
                    "maxItems": 5,
                    "description": "Options for the user to choose from"
                }
            },
            "required": ["question", "options"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let question = match arguments.get("question").and_then(|v| v.as_str()) {
            Some(q) => q.to_string(),
            None => return ToolResult::error("Missing 'question'"),
        };

        let options_val = match arguments.get("options").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return ToolResult::error("Missing 'options' array"),
        };

        let mut options = Vec::new();
        for opt in options_val {
            let label = opt.get("label").and_then(|v| v.as_str()).unwrap_or("");
            let description = opt.get("description").and_then(|v| v.as_str()).unwrap_or("");
            options.push(QuestionOption {
                label: label.to_string(),
                description: description.to_string(),
            });
        }

        if options.len() < 2 {
            return ToolResult::error("At least 2 options are required");
        }

        let interaction_tx = match ctx.interaction {
            Some(ref tx) => tx.clone(),
            None => return ToolResult::error("No interactive session available"),
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        let request = InteractionRequest {
            kind: InteractionKind::AskQuestion { question, options },
            response_tx,
        };

        if interaction_tx.send(request).await.is_err() {
            return ToolResult::error("Interaction channel closed");
        }

        match response_rx.await {
            Ok(InteractionResponse::Answer(answer)) => {
                ToolResult::text(format!("User selected: {}", answer))
            }
            Ok(InteractionResponse::Cancelled) => {
                ToolResult::error("User cancelled the question")
            }
            Err(_) => ToolResult::error("Interaction channel closed"),
        }
    }
}
