//! /model command - list and switch models

use super::CommandResult;
use tau_ai::Model;

pub struct ModelCommand;

impl ModelCommand {
    /// Execute /model command - opens selector if no args, or switches to matching model
    pub fn execute(
        args: &str,
        _current_model: &Model,
        available_models: &[Model],
    ) -> CommandResult {
        if args.is_empty() {
            // Open model selector (TUI) or list models (CLI)
            CommandResult::OpenModelSelector
        } else {
            // Try to find and switch to a model
            match find_model(args, available_models) {
                Some(model) => CommandResult::ChangeModel(model),
                None => CommandResult::Message(format!(
                    "No model found matching '{}'\nUse /model to list available models",
                    args
                )),
            }
        }
    }

    /// List models as text (for CLI mode)
    pub fn list_models_text(current_model: &Model, available_models: &[Model]) -> String {
        list_models(current_model, available_models)
    }
}

fn list_models(current: &Model, models: &[Model]) -> String {
    if models.is_empty() {
        return "No models available".to_string();
    }

    let mut output = String::from("Available models:\n");

    // Group by provider
    let mut by_provider: std::collections::HashMap<String, Vec<&Model>> =
        std::collections::HashMap::new();

    for model in models {
        by_provider
            .entry(model.provider.name().to_string())
            .or_default()
            .push(model);
    }

    for (provider, models) in by_provider.iter() {
        output.push_str(&format!("\n{}:\n", provider));
        for model in models {
            let marker = if model.id == current.id { " *" } else { "" };
            output.push_str(&format!("  {}{}\n", model.id, marker));
        }
    }

    output.push_str("\nSwitch with: /model <name>");
    output
}

fn find_model(query: &str, models: &[Model]) -> Option<Model> {
    let query_lower = query.to_lowercase();

    // Exact match first
    if let Some(model) = models.iter().find(|m| m.id.to_lowercase() == query_lower) {
        return Some(model.clone());
    }

    // Partial match
    if let Some(model) = models
        .iter()
        .find(|m| m.id.to_lowercase().contains(&query_lower))
    {
        return Some(model.clone());
    }

    // Match by name
    models
        .iter()
        .find(|m| m.name.to_lowercase().contains(&query_lower))
        .cloned()
}
