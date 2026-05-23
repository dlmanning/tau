//! /model command - list and switch models

use async_trait::async_trait;
use tau_ai::Model;

use super::Command;
use crate::driver::{Frontend, Session};

pub struct ModelCommand;

#[async_trait]
impl Command for ModelCommand {
    fn name(&self) -> &str {
        "model"
    }
    fn aliases(&self) -> &[&str] {
        &["m"]
    }
    fn description(&self) -> &str {
        "List models or switch model (/model <name>)"
    }
    async fn execute(&self, args: &str, session: &mut Session, frontend: &mut dyn Frontend) {
        let Some(config) = session.current_config().await else {
            frontend.show_error("Agent shut down.").await;
            return;
        };
        if args.is_empty() {
            let text = list_models(config.model(), session.available_models());
            frontend.show_system(&text).await;
            return;
        }
        match find_model(args, session.available_models()) {
            Some(model) => session.change_model(model, frontend).await,
            None => {
                frontend
                    .show_system(&format!(
                        "No model found matching '{}'\nUse /model to list available models",
                        args
                    ))
                    .await;
            }
        }
    }
}

fn list_models(current: &Model, models: &[Model]) -> String {
    if models.is_empty() {
        return "No models available".to_string();
    }

    let mut output = String::from("Available models:\n");

    let mut by_provider: std::collections::BTreeMap<String, Vec<&Model>> =
        std::collections::BTreeMap::new();

    for model in models {
        by_provider
            .entry(model.provider.name().to_string())
            .or_default()
            .push(model);
    }

    for (provider, models) in &by_provider {
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

    if let Some(model) = models.iter().find(|m| m.id.to_lowercase() == query_lower) {
        return Some(model.clone());
    }

    if let Some(model) = models
        .iter()
        .find(|m| m.id.to_lowercase().contains(&query_lower))
    {
        return Some(model.clone());
    }

    models
        .iter()
        .find(|m| m.name.to_lowercase().contains(&query_lower))
        .cloned()
}
