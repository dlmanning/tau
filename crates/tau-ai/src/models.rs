//! Model registry — auto-generated model data with public lookup API.

use crate::models_generated::{ModelEntry, MODEL_ENTRIES};
use crate::{Api, CostInfo, InputType, Model, Provider};

impl ModelEntry {
    fn to_model(&self) -> Model {
        Model {
            id: self.id.to_string(),
            name: self.name.to_string(),
            api: match self.api {
                "AnthropicMessages" => Api::AnthropicMessages,
                "OpenAICompletions" => Api::OpenAICompletions,
                "OpenAIResponses" => Api::OpenAIResponses,
                "GoogleGenerativeAI" => Api::GoogleGenerativeAI,
                _ => unreachable!("unknown api: {}", self.api),
            },
            provider: parse_provider(self.provider).unwrap_or(Provider::Custom),
            base_url: self.base_url.to_string(),
            reasoning: self.reasoning,
            input_types: {
                let mut types = Vec::new();
                if self.input_text {
                    types.push(InputType::Text);
                }
                if self.input_image {
                    types.push(InputType::Image);
                }
                types
            },
            cost: CostInfo {
                input: self.cost_input,
                output: self.cost_output,
                cache_read: self.cost_cache_read,
                cache_write: self.cost_cache_write,
                thinking: self.cost_thinking,
            },
            context_window: self.context_window,
            max_tokens: self.max_tokens,
            headers: Default::default(),
        }
    }
}

/// Look up a model by provider and ID.
pub fn get_model(provider: Provider, id: &str) -> Option<Model> {
    MODEL_ENTRIES
        .iter()
        .find(|e| e.id == id && e.provider == provider.name())
        .map(|e| e.to_model())
}

/// Look up a model by ID only (first match across all providers).
pub fn get_model_by_id(id: &str) -> Option<Model> {
    MODEL_ENTRIES
        .iter()
        .find(|e| e.id == id)
        .map(|e| e.to_model())
}

/// Get all models for a specific provider.
pub fn get_models(provider: Provider) -> Vec<Model> {
    MODEL_ENTRIES
        .iter()
        .filter(|e| e.provider == provider.name())
        .map(|e| e.to_model())
        .collect()
}

/// Get all registered models.
pub fn get_all_models() -> Vec<Model> {
    MODEL_ENTRIES.iter().map(|e| e.to_model()).collect()
}

/// Get all providers that have at least one registered model.
pub fn get_providers() -> Vec<Provider> {
    let mut providers = Vec::new();
    for entry in MODEL_ENTRIES {
        if let Some(p) = parse_provider(entry.provider) {
            if !providers.contains(&p) {
                providers.push(p);
            }
        }
    }
    providers
}

fn parse_provider(name: &str) -> Option<Provider> {
    match name {
        "Anthropic" => Some(Provider::Anthropic),
        "OpenAI" => Some(Provider::OpenAI),
        "Google" => Some(Provider::Google),
        "Groq" => Some(Provider::Groq),
        "Cerebras" => Some(Provider::Cerebras),
        "xAI" => Some(Provider::XAI),
        _ => None,
    }
}
