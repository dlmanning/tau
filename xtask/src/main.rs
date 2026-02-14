use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

const API_URL: &str = "https://models.dev/api.json";

const PROVIDER_KEYS: &[&str] = &["anthropic", "openai", "google", "groq", "cerebras", "xai"];

/// Returns (provider_name, api, base_url) for a models.dev provider key.
fn provider_config(key: &str) -> Option<(&'static str, &'static str, &'static str)> {
    match key {
        "anthropic" => Some((
            "Anthropic",
            "AnthropicMessages",
            "https://api.anthropic.com",
        )),
        "openai" => Some((
            "OpenAI",
            "OpenAIResponses",
            "https://api.openai.com/v1",
        )),
        "google" => Some((
            "Google",
            "GoogleGenerativeAI",
            "https://generativelanguage.googleapis.com/v1beta",
        )),
        "groq" => Some((
            "Groq",
            "OpenAICompletions",
            "https://api.groq.com/openai/v1",
        )),
        "cerebras" => Some((
            "Cerebras",
            "OpenAICompletions",
            "https://api.cerebras.ai/v1",
        )),
        "xai" => Some(("xAI", "OpenAICompletions", "https://api.x.ai/v1")),
        _ => None,
    }
}

// ── Internal model entry (used during generation) ───────────────────────────

struct ModelEntryData {
    id: String,
    name: String,
    provider: String,
    api: String,
    base_url: String,
    reasoning: bool,
    input_text: bool,
    input_image: bool,
    cost_input: f64,
    cost_output: f64,
    cost_cache_read: f64,
    cost_cache_write: f64,
    cost_thinking: f64,
    context_window: u32,
    max_tokens: u32,
}

// ── Overrides JSON types ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Overrides {
    #[serde(default)]
    additions: Vec<OverrideAddition>,
    #[serde(default)]
    patches: BTreeMap<String, OverridePatch>,
}

#[derive(Deserialize)]
struct OverrideAddition {
    id: String,
    name: String,
    provider: String,
    api: String,
    base_url: String,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    input_types: Vec<String>,
    #[serde(default)]
    cost: OverrideCost,
    #[serde(default)]
    context_window: u32,
    #[serde(default)]
    max_tokens: u32,
}

#[derive(Deserialize, Default)]
struct OverrideCost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}

#[derive(Deserialize)]
struct OverridePatch {
    cost: Option<OverrideCost>,
    reasoning: Option<bool>,
    context_window: Option<u32>,
    max_tokens: Option<u32>,
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("generate-models") => generate_models(),
        Some(cmd) => anyhow::bail!("Unknown command: {cmd}"),
        None => {
            eprintln!("Usage: cargo xtask <command>");
            eprintln!("Commands:");
            eprintln!("  generate-models    Fetch models from models.dev and generate models_generated.rs");
            Ok(())
        }
    }
}

fn generate_models() -> Result<()> {
    eprintln!("Fetching models from {API_URL}...");

    let response: serde_json::Value = reqwest::blocking::get(API_URL)
        .context("Failed to fetch models.dev API")?
        .json()
        .context("Failed to parse API response")?;

    let mut entries = Vec::new();

    for &key in PROVIDER_KEYS {
        let (provider_name, api, base_url) =
            provider_config(key).expect("All provider keys should be in config");

        let models = match response.get(key).and_then(|p| p.get("models")) {
            Some(m) => m,
            None => {
                eprintln!("  Warning: No models found for provider '{key}'");
                continue;
            }
        };

        let models_obj = models
            .as_object()
            .context(format!("Expected models to be an object for '{key}'"))?;

        for (_model_key, model_data) in models_obj {
            // Filter: only models with tool_call support
            if !model_data
                .get("tool_call")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                continue;
            }

            // Skip deprecated models
            if model_data.get("status").and_then(|v| v.as_str()) == Some("deprecated") {
                continue;
            }

            let full_id = model_data
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            // Strip provider prefix (e.g. "anthropic/claude-opus-4.5" -> "claude-opus-4.5")
            let id = full_id
                .strip_prefix(&format!("{key}/"))
                .unwrap_or(full_id);

            let name = model_data
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(id);

            let reasoning = model_data
                .get("reasoning")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let input_modalities = model_data
                .get("modalities")
                .and_then(|m| m.get("input"))
                .and_then(|v| v.as_array());

            let input_text = input_modalities
                .map(|arr| arr.iter().any(|v| v.as_str() == Some("text")))
                .unwrap_or(true);
            let input_image = input_modalities
                .map(|arr| arr.iter().any(|v| v.as_str() == Some("image")))
                .unwrap_or(false);

            let cost = model_data.get("cost");
            let cost_input = cost
                .and_then(|c| c.get("input"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let cost_output = cost
                .and_then(|c| c.get("output"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let cost_cache_read = cost
                .and_then(|c| c.get("cache_read"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let cost_cache_write = cost
                .and_then(|c| c.get("cache_write"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let cost_thinking = if reasoning { cost_output } else { 0.0 };

            let limit = model_data.get("limit");
            let context_window = limit
                .and_then(|l| l.get("context"))
                .and_then(|v| v.as_u64())
                .unwrap_or(128000) as u32;
            let max_tokens = limit
                .and_then(|l| l.get("output"))
                .and_then(|v| v.as_u64())
                .unwrap_or(8192) as u32;

            entries.push(ModelEntryData {
                id: id.to_string(),
                name: name.to_string(),
                provider: provider_name.to_string(),
                api: api.to_string(),
                base_url: base_url.to_string(),
                reasoning,
                input_text,
                input_image,
                cost_input,
                cost_output,
                cost_cache_read,
                cost_cache_write,
                cost_thinking,
                context_window,
                max_tokens,
            });
        }

        eprintln!(
            "  {}: {} models",
            provider_name,
            entries.iter().filter(|e| e.provider == provider_name).count()
        );
    }

    // Load and apply overrides
    let overrides_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("overrides.json");
    if overrides_path.exists() {
        let overrides_data =
            std::fs::read_to_string(&overrides_path).context("Failed to read overrides.json")?;
        let overrides: Overrides =
            serde_json::from_str(&overrides_data).context("Failed to parse overrides.json")?;

        // Apply additions (only if model not already present)
        for addition in overrides.additions {
            if !entries
                .iter()
                .any(|e| e.id == addition.id && e.provider == addition.provider)
            {
                let cost_output = addition.cost.output.unwrap_or(0.0);
                let cost_thinking = if addition.reasoning {
                    cost_output
                } else {
                    0.0
                };
                eprintln!("  + Adding override: {}", addition.id);
                entries.push(ModelEntryData {
                    id: addition.id,
                    name: addition.name,
                    provider: addition.provider,
                    api: addition.api,
                    base_url: addition.base_url,
                    reasoning: addition.reasoning,
                    input_text: addition.input_types.iter().any(|t| t == "Text"),
                    input_image: addition.input_types.iter().any(|t| t == "Image"),
                    cost_input: addition.cost.input.unwrap_or(0.0),
                    cost_output,
                    cost_cache_read: addition.cost.cache_read.unwrap_or(0.0),
                    cost_cache_write: addition.cost.cache_write.unwrap_or(0.0),
                    cost_thinking,
                    context_window: addition.context_window,
                    max_tokens: addition.max_tokens,
                });
            }
        }

        // Apply patches
        for (patch_key, patch) in overrides.patches {
            // patch_key is like "anthropic/claude-opus-4-5"
            let Some((provider_key, model_id)) = patch_key.split_once('/') else {
                eprintln!("  Warning: invalid patch key '{patch_key}' (expected provider/model)");
                continue;
            };

            let provider_name = provider_config(provider_key)
                .map(|(name, _, _)| name)
                .unwrap_or(provider_key);

            if let Some(entry) = entries
                .iter_mut()
                .find(|e| e.id == model_id && e.provider == provider_name)
            {
                eprintln!("  ~ Patching: {patch_key}");
                if let Some(cost) = &patch.cost {
                    if let Some(v) = cost.input {
                        entry.cost_input = v;
                    }
                    if let Some(v) = cost.output {
                        entry.cost_output = v;
                    }
                    if let Some(v) = cost.cache_read {
                        entry.cost_cache_read = v;
                    }
                    if let Some(v) = cost.cache_write {
                        entry.cost_cache_write = v;
                    }
                }
                if let Some(v) = patch.reasoning {
                    entry.reasoning = v;
                }
                if let Some(v) = patch.context_window {
                    entry.context_window = v;
                }
                if let Some(v) = patch.max_tokens {
                    entry.max_tokens = v;
                }
            } else {
                eprintln!("  Warning: patch target '{patch_key}' not found in model list");
            }
        }
    }

    // Sort by provider then model ID for deterministic output
    entries.sort_by(|a, b| a.provider.cmp(&b.provider).then(a.id.cmp(&b.id)));

    // Generate the Rust source
    let output = generate_source(&entries);

    // Write to models_generated.rs
    let output_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("crates/tau-ai/src/models_generated.rs");

    std::fs::write(&output_path, &output).context("Failed to write models_generated.rs")?;

    eprintln!(
        "Generated {} model entries -> {}",
        entries.len(),
        output_path.display()
    );

    Ok(())
}

fn generate_source(entries: &[ModelEntryData]) -> String {
    let mut out = String::new();

    writeln!(
        out,
        "// This file is auto-generated by `cargo xtask generate-models`."
    )
    .unwrap();
    writeln!(out, "// Do not edit manually.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "#[derive(Debug, Clone, Copy)]").unwrap();
    writeln!(out, "pub(crate) struct ModelEntry {{").unwrap();
    for (name, ty) in [
        ("id", "&'static str"),
        ("name", "&'static str"),
        ("provider", "&'static str"),
        ("api", "&'static str"),
        ("base_url", "&'static str"),
        ("reasoning", "bool"),
        ("input_text", "bool"),
        ("input_image", "bool"),
        ("cost_input", "f64"),
        ("cost_output", "f64"),
        ("cost_cache_read", "f64"),
        ("cost_cache_write", "f64"),
        ("cost_thinking", "f64"),
        ("context_window", "u32"),
        ("max_tokens", "u32"),
    ] {
        writeln!(out, "    pub {name}: {ty},").unwrap();
    }
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "pub(crate) const MODEL_ENTRIES: &[ModelEntry] = &[").unwrap();

    for entry in entries {
        writeln!(out, "    ModelEntry {{").unwrap();
        writeln!(out, "        id: {:?},", entry.id).unwrap();
        writeln!(out, "        name: {:?},", entry.name).unwrap();
        writeln!(out, "        provider: {:?},", entry.provider).unwrap();
        writeln!(out, "        api: {:?},", entry.api).unwrap();
        writeln!(out, "        base_url: {:?},", entry.base_url).unwrap();
        writeln!(out, "        reasoning: {},", entry.reasoning).unwrap();
        writeln!(out, "        input_text: {},", entry.input_text).unwrap();
        writeln!(out, "        input_image: {},", entry.input_image).unwrap();
        writeln!(
            out,
            "        cost_input: {},",
            format_f64(entry.cost_input)
        )
        .unwrap();
        writeln!(
            out,
            "        cost_output: {},",
            format_f64(entry.cost_output)
        )
        .unwrap();
        writeln!(
            out,
            "        cost_cache_read: {},",
            format_f64(entry.cost_cache_read)
        )
        .unwrap();
        writeln!(
            out,
            "        cost_cache_write: {},",
            format_f64(entry.cost_cache_write)
        )
        .unwrap();
        writeln!(
            out,
            "        cost_thinking: {},",
            format_f64(entry.cost_thinking)
        )
        .unwrap();
        writeln!(out, "        context_window: {},", entry.context_window).unwrap();
        writeln!(out, "        max_tokens: {},", entry.max_tokens).unwrap();
        writeln!(out, "    }},").unwrap();
    }

    writeln!(out, "];").unwrap();
    out
}

fn format_f64(v: f64) -> String {
    if v == 0.0 {
        "0.0".to_string()
    } else if v == v.floor() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}
