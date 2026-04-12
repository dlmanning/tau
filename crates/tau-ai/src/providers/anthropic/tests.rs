use super::CacheScope;
use super::convert::{CacheControl, split_system_prompt};
use super::request::ThinkingConfig;

#[test]
fn test_cache_scope_serialization() {
    assert_eq!(
        serde_json::to_value(CacheScope::Global).unwrap(),
        serde_json::json!("global")
    );
    assert_eq!(
        serde_json::to_value(CacheScope::Org).unwrap(),
        serde_json::json!("org")
    );
}

#[test]
fn test_cache_control_serialization_minimal() {
    let cc = CacheControl {
        control_type: "ephemeral".to_string(),
        scope: None,
        ttl: None,
    };
    let json = serde_json::to_value(&cc).unwrap();
    assert_eq!(json, serde_json::json!({"type": "ephemeral"}));
    assert!(json.get("scope").is_none());
    assert!(json.get("ttl").is_none());
}

#[test]
fn test_cache_control_serialization_full() {
    let cc = CacheControl {
        control_type: "ephemeral".to_string(),
        scope: Some(CacheScope::Global),
        ttl: Some("1h".to_string()),
    };
    let json = serde_json::to_value(&cc).unwrap();
    assert_eq!(
        json,
        serde_json::json!({"type": "ephemeral", "scope": "global", "ttl": "1h"})
    );
}

#[test]
fn test_thinking_config_adaptive_serialization() {
    let config = ThinkingConfig::Adaptive {
        thinking_type: "adaptive".to_string(),
        display: None,
    };
    let json = serde_json::to_value(&config).unwrap();
    assert_eq!(json, serde_json::json!({"type": "adaptive"}));
    assert!(json.get("budget_tokens").is_none());
}

#[test]
fn test_thinking_config_enabled_serialization() {
    let config = ThinkingConfig::Enabled {
        thinking_type: "enabled".to_string(),
        budget_tokens: 4096,
        display: None,
    };
    let json = serde_json::to_value(&config).unwrap();
    assert_eq!(
        json,
        serde_json::json!({"type": "enabled", "budget_tokens": 4096})
    );
}

#[test]
fn test_split_system_prompt_no_boundary() {
    let blocks = split_system_prompt("Hello world", None, &None, &None);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "Hello world");
    assert!(blocks[0].cache_control.is_some());
}

#[test]
fn test_split_system_prompt_boundary_not_found() {
    let blocks = split_system_prompt(
        "Hello world",
        Some("<!-- BOUNDARY -->"),
        &Some(CacheScope::Org),
        &None,
    );
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "Hello world");
}

#[test]
fn test_split_system_prompt_boundary_splits() {
    let prompt = "Static part<!-- BOUNDARY -->Dynamic part";
    let blocks = split_system_prompt(
        prompt,
        Some("<!-- BOUNDARY -->"),
        &Some(CacheScope::Global),
        &Some("1h".to_string()),
    );
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].text, "Static part");
    assert!(blocks[0].cache_control.is_some());
    let cc = blocks[0].cache_control.as_ref().unwrap();
    assert!(matches!(cc.scope, Some(CacheScope::Global)));
    assert_eq!(cc.ttl.as_deref(), Some("1h"));
    assert_eq!(blocks[1].text, "Dynamic part");
    assert!(blocks[1].cache_control.is_none());
}

#[test]
fn test_split_system_prompt_respects_caller_scope() {
    let prompt = "Static<!-- B -->Dynamic";
    let blocks = split_system_prompt(prompt, Some("<!-- B -->"), &Some(CacheScope::Org), &None);
    assert_eq!(blocks.len(), 2);
    let cc = blocks[0].cache_control.as_ref().unwrap();
    assert!(matches!(cc.scope, Some(CacheScope::Org)));
}

#[test]
fn test_split_system_prompt_boundary_at_edges() {
    let prompt = "<!-- B -->Dynamic only";
    let blocks = split_system_prompt(prompt, Some("<!-- B -->"), &Some(CacheScope::Global), &None);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "Dynamic only");
    assert!(blocks[0].cache_control.is_none());

    let prompt = "Static only<!-- B -->";
    let blocks = split_system_prompt(prompt, Some("<!-- B -->"), &Some(CacheScope::Global), &None);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "Static only");
    assert!(blocks[0].cache_control.is_some());
}

#[test]
fn test_split_system_prompt_boundary_is_entire_prompt() {
    let prompt = "<!-- B -->";
    let blocks = split_system_prompt(prompt, Some("<!-- B -->"), &Some(CacheScope::Global), &None);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].text, "<!-- B -->");
    assert!(blocks[0].cache_control.is_some());
}
