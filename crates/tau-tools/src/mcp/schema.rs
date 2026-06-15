//! Input-schema normalization.
//!
//! tau's actor validates tool arguments against `parameters_schema()`
//! with the `jsonschema` crate before execution, and the Anthropic API
//! requires `input_schema.type == "object"`. Server-supplied schemas
//! vary in quality, so we apply a minimal **top-level-only** cleanup —
//! deep rewriting risks breaking valid schemas the providers would
//! have accepted.

use serde_json::{Map, Value, json};

/// Normalize a remote tool's input schema:
/// 1. ensure `"type": "object"`,
/// 2. ensure a `"properties"` object exists,
/// 3. strip top-level `"$schema"` (avoids draft-resolution surprises),
/// 4. verify the result compiles as a JSON Schema — if not, fall back
///    to a permissive `{"type":"object"}` so the tool stays usable
///    (the server still validates its own inputs).
pub(crate) fn normalize(raw: &Map<String, Value>, tool_label: &str) -> Value {
    let mut schema = raw.clone();
    schema
        .entry("type")
        .or_insert_with(|| Value::String("object".into()));
    schema
        .entry("properties")
        .or_insert_with(|| Value::Object(Map::new()));
    schema.remove("$schema");

    let value = Value::Object(schema);
    match jsonschema::validator_for(&value) {
        Ok(_) => value,
        Err(e) => {
            tracing::warn!(
                tool = tool_label,
                error = %e,
                "MCP tool schema does not compile; using permissive fallback"
            );
            json!({ "type": "object", "additionalProperties": true })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn fills_missing_type_and_properties() {
        let out = normalize(&obj(json!({})), "t");
        assert_eq!(out["type"], "object");
        assert!(out["properties"].is_object());
    }

    #[test]
    fn strips_top_level_schema_key_and_keeps_rest() {
        let out = normalize(
            &obj(json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": { "a": { "type": "string" } },
                "required": ["a"],
            })),
            "t",
        );
        assert!(out.get("$schema").is_none());
        assert_eq!(out["required"], json!(["a"]));
        assert_eq!(out["properties"]["a"]["type"], "string");
    }

    #[test]
    fn uncompilable_schema_falls_back_to_permissive() {
        // `type: 42` is not a valid schema.
        let out = normalize(&obj(json!({ "type": 42 })), "t");
        assert_eq!(out, json!({ "type": "object", "additionalProperties": true }));
    }
}
