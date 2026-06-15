//! Map MCP `CallToolResult` content into tau's `ToolResult`.

use rmcp::model::{CallToolResult, RawContent, ResourceContents};
use tau_agent::ToolResult;
use tau_ai::Content;

pub(crate) fn map_call_tool_result(result: CallToolResult) -> ToolResult {
    let mut content: Vec<Content> = result
        .content
        .iter()
        .map(|c| match &c.raw {
            RawContent::Text(t) => Content::text(t.text.clone()),
            RawContent::Image(i) => Content::Image {
                data: i.data.clone(),
                mime_type: i.mime_type.clone(),
            },
            RawContent::Resource(r) => match &r.resource {
                ResourceContents::TextResourceContents { uri, text, .. } => {
                    Content::text(format!("[resource {uri}]\n{text}"))
                }
                ResourceContents::BlobResourceContents { uri, blob, .. } => Content::text(
                    format!("[binary resource: {uri}, {} base64 bytes — not supported]", blob.len()),
                ),
            },
            RawContent::Audio(a) => {
                Content::text(format!("[audio content ({}) — not supported]", a.mime_type))
            }
            RawContent::ResourceLink(r) => {
                Content::text(format!("[resource link: {} ({})]", r.uri, r.name))
            }
        })
        .collect();

    // A tool_result block must never be empty for the provider: fall
    // back to the structured payload, then to an explicit marker.
    if content.is_empty() {
        let text = match &result.structured_content {
            Some(v) => serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()),
            None => "(no content)".to_string(),
        };
        content.push(Content::text(text));
    }

    ToolResult {
        content,
        is_error: result.is_error.unwrap_or(false),
        details: result.structured_content.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{Annotated, RawTextContent};
    use serde_json::json;

    fn text_block(s: &str) -> rmcp::model::Content {
        Annotated::new(
            RawContent::Text(RawTextContent {
                text: s.into(),
                meta: None,
            }),
            None,
        )
    }

    fn text_of(c: &Content) -> &str {
        match c {
            Content::Text { text } => text,
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn text_and_error_map_through() {
        let result = CallToolResult::error(vec![text_block("boom")]);
        let mapped = map_call_tool_result(result);
        assert!(mapped.is_error);
        assert_eq!(text_of(&mapped.content[0]), "boom");
    }

    #[test]
    fn image_passes_through_as_image_content() {
        let result = CallToolResult::success(vec![Annotated::new(
            RawContent::Image(rmcp::model::RawImageContent {
                data: "aGk=".into(),
                mime_type: "image/png".into(),
                meta: None,
            }),
            None,
        )]);
        let mapped = map_call_tool_result(result);
        match &mapped.content[0] {
            Content::Image { data, mime_type } => {
                assert_eq!(data, "aGk=");
                assert_eq!(mime_type, "image/png");
            }
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    fn structured_only_result_becomes_pretty_json() {
        let mut result = CallToolResult::success(vec![]);
        result.structured_content = Some(json!({"answer": 42}));
        let mapped = map_call_tool_result(result);
        assert!(text_of(&mapped.content[0]).contains("\"answer\": 42"));
        assert_eq!(mapped.details, Some(json!({"answer": 42})));
    }

    #[test]
    fn fully_empty_result_gets_marker() {
        let result = CallToolResult::success(vec![]);
        let mapped = map_call_tool_result(result);
        assert_eq!(text_of(&mapped.content[0]), "(no content)");
        assert!(!mapped.is_error);
    }
}
