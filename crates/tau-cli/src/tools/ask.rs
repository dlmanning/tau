//! AskUserQuestion tool — ask the user a multiple-choice question.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::interaction::{
    InteractionKind, InteractionRequest, InteractionResponse, QuestionOption,
};
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};

#[derive(Deserialize, JsonSchema)]
struct AskOption {
    /// Short label for this option
    label: String,
    /// Longer explanation of what this option means
    description: String,
}

#[derive(Deserialize, JsonSchema)]
struct AskArgs {
    /// The question to ask the user
    question: String,
    /// Options for the user to choose from (2-5 items)
    #[schemars(extend("minItems" = 2, "maxItems" = 5))]
    options: Vec<AskOption>,
}

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
        cached_schema!(AskArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: AskArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        if args.options.len() < 2 {
            return ToolResult::error("At least 2 options are required");
        }

        let options: Vec<QuestionOption> = args
            .options
            .into_iter()
            .map(|o| QuestionOption {
                label: o.label,
                description: o.description,
            })
            .collect();

        let interaction_tx = match ctx.interaction {
            Some(ref tx) => tx.clone(),
            None => return ToolResult::error("No interactive session available"),
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        let request = InteractionRequest {
            kind: InteractionKind::AskQuestion {
                question: args.question,
                options,
            },
            response_tx,
        };

        if interaction_tx.send(request).await.is_err() {
            return ToolResult::error("Interaction channel closed");
        }

        match response_rx.await {
            Ok(InteractionResponse::Answer(answer)) => {
                ToolResult::text(format!("User selected: {}", answer))
            }
            Ok(InteractionResponse::Cancelled) => ToolResult::error("User cancelled the question"),
            Err(_) => ToolResult::error("Interaction channel closed"),
        }
    }
}
