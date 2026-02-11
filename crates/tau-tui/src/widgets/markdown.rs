//! Markdown rendering for terminal UI

use crate::theme::Theme;
use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

/// Convert markdown text to styled ratatui Lines
pub fn render_markdown<'a>(text: &str, theme: &Theme, width: usize) -> Vec<Line<'a>> {
    let mut lines: Vec<Line<'a>> = Vec::new();
    let mut current_line: Vec<Span<'a>> = Vec::new();
    let mut current_style = theme.base_style();
    let mut in_code_block = false;
    let mut code_block_content = String::new();
    let mut list_depth: usize = 0;

    let parser = Parser::new(text);

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    // Flush current line
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                    current_style = match level {
                        pulldown_cmark::HeadingLevel::H1 => theme
                            .accent_style()
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                        pulldown_cmark::HeadingLevel::H2 => {
                            theme.accent_style().add_modifier(Modifier::BOLD)
                        }
                        _ => theme.accent_style(),
                    };
                }
                Tag::Paragraph => {
                    // Start new paragraph
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                }
                Tag::CodeBlock(_) => {
                    in_code_block = true;
                    code_block_content.clear();
                    // Flush current line and add blank line before code
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                }
                Tag::List(_) => {
                    list_depth += 1;
                }
                Tag::Item => {
                    // Start list item with bullet
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                    let indent = "  ".repeat(list_depth.saturating_sub(1));
                    current_line.push(Span::styled(format!("{}• ", indent), theme.dim_style()));
                }
                Tag::Emphasis => {
                    current_style = current_style.add_modifier(Modifier::ITALIC);
                }
                Tag::Strong => {
                    current_style = current_style.add_modifier(Modifier::BOLD);
                }
                Tag::Strikethrough => {
                    current_style = current_style.add_modifier(Modifier::CROSSED_OUT);
                }
                Tag::Link { .. } => {
                    current_style = Style::default().fg(theme.link);
                }
                _ => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Heading(_) => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                    current_style = theme.base_style();
                }
                TagEnd::Paragraph => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                    lines.push(Line::from("")); // Blank line after paragraph
                }
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    // Render code block with background styling
                    let code_style = Style::default().fg(theme.code).add_modifier(Modifier::DIM);

                    for code_line in code_block_content.lines() {
                        let display_line = if code_line.len() > width.saturating_sub(4) {
                            format!("  {}…", &code_line[..width.saturating_sub(5)])
                        } else {
                            format!("  {}", code_line)
                        };
                        lines.push(Line::from(Span::styled(display_line, code_style)));
                    }
                    lines.push(Line::from("")); // Blank line after code block
                }
                TagEnd::List(_) => {
                    list_depth = list_depth.saturating_sub(1);
                    if list_depth == 0 {
                        lines.push(Line::from("")); // Blank line after list
                    }
                }
                TagEnd::Item => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                }
                TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                    current_style = theme.base_style();
                }
                TagEnd::Link => {
                    current_style = theme.base_style();
                }
                _ => {}
            },
            Event::Text(text) => {
                if in_code_block {
                    code_block_content.push_str(&text);
                } else {
                    // Wrap text if needed
                    let text_str = text.to_string();
                    current_line.push(Span::styled(text_str, current_style));
                }
            }
            Event::Code(code) => {
                // Inline code
                let code_style = Style::default().fg(theme.code).add_modifier(Modifier::BOLD);
                current_line.push(Span::styled(format!("`{}`", code), code_style));
            }
            Event::SoftBreak => {
                current_line.push(Span::raw(" "));
            }
            Event::HardBreak => {
                if !current_line.is_empty() {
                    lines.push(Line::from(std::mem::take(&mut current_line)));
                }
            }
            _ => {}
        }
    }

    // Flush remaining content
    if !current_line.is_empty() {
        lines.push(Line::from(current_line));
    }

    // Remove trailing empty lines
    while lines.last().is_some_and(|l| {
        l.spans.is_empty() || (l.spans.len() == 1 && l.spans[0].content.is_empty())
    }) {
        lines.pop();
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_text() {
        let theme = Theme::dark();
        let lines = render_markdown("Hello, world!", &theme, 80);
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_code_block() {
        let theme = Theme::dark();
        let md = "```rust\nfn main() {}\n```";
        let lines = render_markdown(md, &theme, 80);
        assert!(!lines.is_empty());
    }
}
