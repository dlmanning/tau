//! Markdown rendering for terminal UI

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};
use unicode_width::UnicodeWidthStr;

use super::super::theme::Theme;

/// Convert markdown text to styled ratatui Lines
pub fn render_markdown<'a>(text: &str, theme: &Theme, width: usize) -> Vec<Line<'a>> {
    let mut lines: Vec<Line<'a>> = Vec::new();
    let mut current_line: Vec<Span<'a>> = Vec::new();
    let mut style_stack: Vec<Style> = vec![theme.base_style()];
    let mut current_style = theme.base_style();
    let mut in_code_block = false;
    let mut code_block_content = String::new();
    let mut list_depth: usize = 0;

    // Table state
    let mut in_table = false;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut current_cell = String::new();

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    let parser = Parser::new_ext(text, options);

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
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
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                }
                Tag::CodeBlock(_) => {
                    in_code_block = true;
                    code_block_content.clear();
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                }
                Tag::List(_) => {
                    list_depth += 1;
                }
                Tag::Item => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                    let indent = "  ".repeat(list_depth.saturating_sub(1));
                    current_line.push(Span::styled(format!("{}• ", indent), theme.dim_style()));
                }
                Tag::Emphasis => {
                    style_stack.push(current_style);
                    current_style = current_style.add_modifier(Modifier::ITALIC);
                }
                Tag::Strong => {
                    style_stack.push(current_style);
                    current_style = current_style.add_modifier(Modifier::BOLD);
                }
                Tag::Strikethrough => {
                    style_stack.push(current_style);
                    current_style = current_style.add_modifier(Modifier::CROSSED_OUT);
                }
                Tag::Link { .. } => {
                    current_style = Style::default().fg(theme.link);
                }
                Tag::Table(_) => {
                    if !current_line.is_empty() {
                        lines.push(Line::from(std::mem::take(&mut current_line)));
                    }
                    in_table = true;
                    table_rows.clear();
                }
                Tag::TableHead => {
                    current_row.clear();
                }
                Tag::TableRow => {
                    current_row.clear();
                }
                Tag::TableCell => {
                    current_cell.clear();
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
                    let code_style = Style::default().fg(theme.code).add_modifier(Modifier::DIM);

                    for code_line in code_block_content.lines() {
                        let max_chars = width.saturating_sub(4);
                        let char_count = code_line.chars().count();
                        let display_line = if char_count > max_chars {
                            let truncated: String = code_line
                                .chars()
                                .take(max_chars.saturating_sub(1))
                                .collect();
                            format!("  {}…", truncated)
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
                    current_style = style_stack.pop().unwrap_or_else(|| theme.base_style());
                }
                TagEnd::Link => {
                    current_style = theme.base_style();
                }
                TagEnd::TableCell => {
                    current_row.push(std::mem::take(&mut current_cell));
                }
                TagEnd::TableHead => {
                    table_rows.push(std::mem::take(&mut current_row));
                }
                TagEnd::TableRow => {
                    table_rows.push(std::mem::take(&mut current_row));
                }
                TagEnd::Table => {
                    // Calculate column widths
                    let num_cols = table_rows.iter().map(|r| r.len()).max().unwrap_or(0);
                    let mut col_widths = vec![0usize; num_cols];
                    for row in &table_rows {
                        for (i, cell) in row.iter().enumerate() {
                            if i < num_cols {
                                // Display width, not byte length — CJK
                                // and emoji cells are wider than their
                                // char count and much narrower than
                                // their byte count.
                                col_widths[i] = col_widths[i].max(cell.width());
                            }
                        }
                    }
                    // Clamp total width
                    let total: usize =
                        col_widths.iter().sum::<usize>() + num_cols.saturating_sub(1) * 3;
                    if total > width {
                        let scale = width as f64 / total as f64;
                        for w in &mut col_widths {
                            *w = (*w as f64 * scale).max(3.0) as usize;
                        }
                    }

                    let head_style = theme.accent_style().add_modifier(Modifier::BOLD);
                    let cell_style = theme.base_style();
                    let sep_style = theme.dim_style();

                    for (row_idx, row) in table_rows.iter().enumerate() {
                        let mut spans: Vec<Span<'a>> = Vec::new();
                        for (i, cell) in row.iter().enumerate() {
                            if i > 0 {
                                spans.push(Span::styled(" │ ", sep_style));
                            }
                            let w = col_widths.get(i).copied().unwrap_or(cell.width());
                            // Pad by display width (format!'s width
                            // counts chars, mis-padding wide glyphs).
                            let pad = w.saturating_sub(cell.width());
                            let padded = format!("{}{}", cell, " ".repeat(pad));
                            let style = if row_idx == 0 { head_style } else { cell_style };
                            spans.push(Span::styled(padded, style));
                        }
                        lines.push(Line::from(spans));
                        // Separator after header row
                        if row_idx == 0 {
                            let sep: String = col_widths
                                .iter()
                                .map(|w| "─".repeat(*w))
                                .collect::<Vec<_>>()
                                .join("─┼─");
                            lines.push(Line::from(Span::styled(sep, sep_style)));
                        }
                    }
                    lines.push(Line::from(""));
                    in_table = false;
                    table_rows.clear();
                }
                _ => {}
            },
            Event::Text(text) => {
                if in_code_block {
                    code_block_content.push_str(&text);
                } else if in_table {
                    current_cell.push_str(&text);
                } else {
                    let text_str = text.to_string();
                    current_line.push(Span::styled(text_str, current_style));
                }
            }
            Event::Code(code) => {
                if in_table {
                    current_cell.push('`');
                    current_cell.push_str(&code);
                    current_cell.push('`');
                } else {
                    let code_style = Style::default().fg(theme.code).add_modifier(Modifier::BOLD);
                    current_line.push(Span::styled(format!("`{}`", code), code_style));
                }
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

    if !current_line.is_empty() {
        lines.push(Line::from(current_line));
    }

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

    /// Columns must align by display width: a CJK cell (2 columns per
    /// glyph) and an ASCII cell of the same display width should pad
    /// their rows to identical rendered widths.
    #[test]
    fn table_columns_align_by_display_width() {
        let theme = Theme::dark();
        let md = "| h1 | h2 |\n|----|----|\n| 你好 | x |\n| abcd | y |";
        let lines = render_markdown(md, &theme, 80);
        let rendered: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        // "你好" renders 4 columns wide, same as "abcd" — both data
        // rows must have equal total display width.
        let data_rows: Vec<&String> = rendered
            .iter()
            .filter(|r| r.contains('你') || r.contains("abcd"))
            .collect();
        assert_eq!(data_rows.len(), 2);
        assert_eq!(
            data_rows[0].width(),
            data_rows[1].width(),
            "rows mis-aligned: {:?} vs {:?}",
            data_rows[0],
            data_rows[1]
        );
    }
}
