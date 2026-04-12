//! Context overflow detection across LLM providers.
//!
//! Regex-based classification of error strings that indicate the input
//! exceeded the model's context window. Covers Anthropic, OpenAI, Google,
//! Bedrock, xAI, Groq, OpenRouter, llama.cpp, LM Studio, MiniMax, Kimi.

use std::sync::LazyLock;

use regex::Regex;

/// Compiled regex patterns for detecting context overflow errors across providers.
static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        // Generic / multi-provider patterns
        r"(?i)context.?length.?exceed",
        r"(?i)maximum.?context.?length",
        r"(?i)context.?window.?(exceed|full|limit)",
        r"(?i)too.?many.?tokens",
        r"(?i)prompt.?is.?too.?long",
        r"(?i)input.?too.?long",
        r"(?i)token.?limit.?(exceed|reach)",
        r"(?i)content.?too.?large",
        // Anthropic / Bedrock
        r"(?i)prompt.?too.?long",
        r"(?i)request.?too.?large",
        r"(?i)messages?.?too.?long",
        // OpenAI
        r"(?i)maximum.?number.?of.?tokens",
        r"(?i)reduce.?the.?length",
        r"(?i)context_length_exceeded",
        // "max_tokens" only when followed by overflow language (not config errors)
        r"(?i)max_tokens.*(exceed|limit|too|overflow)",
        // Google (Gemini)
        r"(?i)exceeds?.+token.?limit",
        r"(?i)input.?token.?limit",
        // xAI / Groq / OpenRouter
        r"(?i)context.?overflow",
        r"(?i)sequence.?too.?long",
        // llama.cpp / LM Studio
        r"(?i)context.?size.?exceed",
        r"(?i)n_ctx",
        r"(?i)slot.?context.?overflow",
        // MiniMax / Kimi
        r"(?i)total.?tokens?.?exceed",
        r"(?i)max_prompt_tokens",
        // HTTP status-based (embedded in error strings)
        r"\b413\b",
    ]
    .iter()
    .filter_map(|p| Regex::new(p).ok())
    .collect()
});

/// Regex for HTTP 400 status codes in error strings (e.g. "400 Bad Request", "HTTP 400", "status: 400").
/// Requires word boundary to avoid matching port numbers or IDs containing "400".
static HTTP_400_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:status|http|error)[:\s]*400\b|\b400\s+bad\s+request")
        .expect("valid regex literal")
});

/// Check if an error indicates a context overflow / too many tokens.
pub fn is_context_overflow(error: &str) -> bool {
    // HTTP 400 with token-related keywords — requires structured 400 reference
    if HTTP_400_PATTERN.is_match(error) {
        let lower = error.to_lowercase();
        if lower.contains("token") || lower.contains("context") || lower.contains("length") {
            return true;
        }
    }

    OVERFLOW_PATTERNS.iter().any(|re| re.is_match(error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overflow_anthropic_prompt_too_long() {
        assert!(is_context_overflow("prompt is too long"));
        assert!(is_context_overflow("Prompt is too long for this model"));
    }

    #[test]
    fn test_overflow_anthropic_request_too_large() {
        assert!(is_context_overflow("request too large"));
    }

    #[test]
    fn test_overflow_anthropic_messages_too_long() {
        assert!(is_context_overflow("messages too long"));
    }

    #[test]
    fn test_overflow_openai_context_length_exceeded() {
        assert!(is_context_overflow(
            "This model's maximum context length is 128000 tokens. context_length_exceeded"
        ));
    }

    #[test]
    fn test_overflow_openai_reduce_length() {
        assert!(is_context_overflow(
            "Please reduce the length of the messages"
        ));
    }

    #[test]
    fn test_overflow_openai_max_tokens_with_exceeds() {
        assert!(is_context_overflow("max_tokens exceeds the model limit"));
    }

    #[test]
    fn test_no_overflow_max_tokens_config_error() {
        // "max_tokens" alone (e.g. config validation) should NOT match
        assert!(!is_context_overflow(
            "max_tokens parameter must be positive"
        ));
        assert!(!is_context_overflow("invalid value for max_tokens"));
    }

    #[test]
    fn test_overflow_google_token_limit() {
        assert!(is_context_overflow(
            "Request exceeds the token limit for this model"
        ));
    }

    #[test]
    fn test_overflow_google_input_token_limit() {
        assert!(is_context_overflow("Input token limit exceeded"));
    }

    #[test]
    fn test_overflow_generic_too_many_tokens() {
        assert!(is_context_overflow("too many tokens in the request"));
    }

    #[test]
    fn test_overflow_generic_context_window() {
        assert!(is_context_overflow("context window exceeded"));
        assert!(is_context_overflow("context window full"));
        assert!(is_context_overflow("exceeds context window limit"));
    }

    #[test]
    fn test_overflow_generic_token_limit() {
        assert!(is_context_overflow("token limit exceeded"));
        assert!(is_context_overflow("token limit reached"));
    }

    #[test]
    fn test_overflow_llama_cpp_n_ctx() {
        assert!(is_context_overflow("n_ctx exceeded, cannot process"));
    }

    #[test]
    fn test_overflow_llama_cpp_slot_overflow() {
        assert!(is_context_overflow("slot context overflow"));
    }

    #[test]
    fn test_overflow_lm_studio_context_size() {
        assert!(is_context_overflow("context size exceeded"));
    }

    #[test]
    fn test_overflow_groq_sequence_too_long() {
        assert!(is_context_overflow("sequence too long for model"));
    }

    #[test]
    fn test_overflow_minimax_total_tokens() {
        assert!(is_context_overflow("total tokens exceed the limit"));
    }

    #[test]
    fn test_overflow_http_413() {
        assert!(is_context_overflow("HTTP 413 Payload Too Large"));
    }

    #[test]
    fn test_overflow_http_400_with_token() {
        assert!(is_context_overflow("HTTP 400: token count exceeds limit"));
        assert!(is_context_overflow(
            "status: 400 - too many tokens in context"
        ));
        assert!(is_context_overflow("error 400: context length exceeded"));
    }

    #[test]
    fn test_overflow_http_400_bad_request_with_context() {
        assert!(is_context_overflow("400 Bad Request: context too large"));
    }

    #[test]
    fn test_no_overflow_normal_errors() {
        assert!(!is_context_overflow("401 Unauthorized"));
        assert!(!is_context_overflow("rate limit exceeded"));
        assert!(!is_context_overflow("internal server error 500"));
        assert!(!is_context_overflow("connection timeout"));
        assert!(!is_context_overflow("invalid API key"));
    }

    #[test]
    fn test_no_overflow_http_400_without_keywords() {
        // 400 alone without token/context/length keywords should NOT match
        assert!(!is_context_overflow("400 Bad Request: invalid field"));
    }

    #[test]
    fn test_no_overflow_400_in_unrelated_context() {
        // "400" appearing as port or ID should NOT match
        assert!(!is_context_overflow(
            "connected to port 14001 with token auth"
        ));
        assert!(!is_context_overflow(
            "processed 400 items in context manager"
        ));
    }
}
