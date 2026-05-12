//! Context-overflow detection across LLM providers.
//!
//! Regex-based classification of error strings that indicate the input
//! exceeded the model's context window.

use std::sync::LazyLock;

use regex::Regex;

static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        // Generic / multi-provider
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
        // HTTP status
        r"\b413\b",
    ]
    .iter()
    .filter_map(|p| Regex::new(p).ok())
    .collect()
});

static HTTP_400_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:status|http|error)[:\s]*400\b|\b400\s+bad\s+request")
        .expect("valid regex literal")
});

pub fn is_context_overflow(error: &str) -> bool {
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
    fn anthropic_prompt_too_long() {
        assert!(is_context_overflow("prompt is too long"));
    }

    #[test]
    fn openai_context_length_exceeded() {
        assert!(is_context_overflow("context_length_exceeded"));
    }

    #[test]
    fn http_400_with_token_qualifier_is_overflow() {
        assert!(is_context_overflow(
            "status 400: prompt has too many tokens"
        ));
    }

    #[test]
    fn unrelated_errors_pass_through() {
        assert!(!is_context_overflow("401 Unauthorized"));
        assert!(!is_context_overflow("rate limit exceeded"));
        assert!(!is_context_overflow("internal server error 500"));
    }
}
