//! Built-in tools for the coding agent

/// Generate a JSON Schema for a tool args struct, stripped of metadata
/// that LLM APIs don't need (`$schema`, `title`). Inlines any `$ref`/`$defs`
/// so the schema is self-contained without JSON Schema references.
pub(crate) fn tool_schema<T: schemars::JsonSchema>() -> serde_json::Value {
    let schema = schemars::schema_for!(T);
    let mut value = serde_json::to_value(schema).unwrap();
    if let Some(obj) = value.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
    }
    inline_refs(&mut value);
    value
}

/// Cached schema: computes once via `LazyLock`, returns a clone.
macro_rules! cached_schema {
    ($T:ty) => {{
        static SCHEMA: std::sync::LazyLock<serde_json::Value> =
            std::sync::LazyLock::new(|| super::tool_schema::<$T>());
        SCHEMA.clone()
    }};
}

pub mod agent;
mod ask;
mod bash;
mod edit;
mod glob;
mod grep;
mod list;
pub mod lsp;
mod read;
mod send_message;
mod web_fetch;
mod write;

/// Recursively resolve `$ref` pointers against `$defs` and inline them.
fn inline_refs(value: &mut serde_json::Value) {
    let defs = value
        .as_object()
        .and_then(|obj| obj.get("$defs"))
        .cloned();

    if let Some(defs) = &defs {
        resolve_refs(value, defs);
        if let Some(obj) = value.as_object_mut() {
            obj.remove("$defs");
        }
    }
}

fn resolve_refs(value: &mut serde_json::Value, defs: &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(ref_val) = map.get("$ref").and_then(|v| v.as_str()).map(String::from) {
                if let Some(type_name) = ref_val.strip_prefix("#/$defs/") {
                    if let Some(def) = defs.get(type_name) {
                        let mut resolved = def.clone();
                        resolve_refs(&mut resolved, defs);
                        *value = resolved;
                        return;
                    }
                }
            }
            for v in map.values_mut() {
                resolve_refs(v, defs);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                resolve_refs(v, defs);
            }
        }
        _ => {}
    }
}

pub use agent::AgentTool;
pub use ask::AskTool;
pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use list::ListTool;
pub use lsp::LspTool;
pub use read::ReadTool;
pub use send_message::SendMessageTool;
pub use web_fetch::WebFetchTool;
pub use write::WriteTool;
