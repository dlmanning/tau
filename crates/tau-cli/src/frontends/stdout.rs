//! Stdout / stdin frontend covering both one-shot (`tau "..."`) and
//! REPL (`tau --no-tui`) modes.

use std::io::{self, IsTerminal, Write};

use async_trait::async_trait;
use tau_agent::{
    AgentEvent, InteractionKind, InteractionRequest, InteractionResponse,
};
use tau_ai::{Model, Usage};

use crate::driver::{Frontend, SessionStart, UserInput};

enum Mode {
    /// One prompt, then quit. Suppresses the banner; no prompt prefix.
    OneShot { prompt: Option<String> },
    /// Interactive REPL: banner, prompt prefix, persists until EOF.
    Repl,
}

pub struct StdoutFrontend {
    mode: Mode,
    /// Tracks how many characters of the current assistant message we
    /// have already printed, so deltas land correctly without
    /// re-printing the prefix.
    rendered_chars: usize,
}

impl StdoutFrontend {
    pub fn one_shot(prompt: String) -> Self {
        Self {
            mode: Mode::OneShot {
                prompt: Some(prompt),
            },
            rendered_chars: 0,
        }
    }

    pub fn repl() -> Self {
        Self {
            mode: Mode::Repl,
            rendered_chars: 0,
        }
    }

    fn is_repl(&self) -> bool {
        matches!(self.mode, Mode::Repl)
    }
}

#[async_trait]
impl Frontend for StdoutFrontend {
    async fn on_session_start(&mut self, info: SessionStart<'_>) {
        if !self.is_repl() {
            // One-shot mode: print the user's prompt as the "tau> ..." line.
            if let Mode::OneShot { prompt: Some(p) } = &self.mode {
                println!("tau> {}", p);
                println!();
            }
            return;
        }
        if std::io::stderr().is_terminal() {
            let model_id = &info.model.id;
            let short = model_id.split('/').next_back().unwrap_or(model_id);
            if let Some(sid) = info.session_id {
                eprintln!("tau ({}) session: {}", short, sid);
            } else {
                eprintln!("tau ({})", short);
            }
            eprintln!();
        }
    }

    async fn next_input(&mut self) -> Option<UserInput> {
        match &mut self.mode {
            Mode::OneShot { prompt } => prompt.take().map(UserInput::Prompt),
            Mode::Repl => {
                print!("> ");
                io::stdout().flush().ok();
                let line = tokio::task::spawn_blocking(|| {
                    let mut buf = String::new();
                    match io::stdin().read_line(&mut buf) {
                        Ok(0) => None,
                        Ok(_) => Some(buf),
                        Err(_) => None,
                    }
                })
                .await
                .ok()
                .flatten()?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    // Empty input — re-prompt rather than quitting.
                    return Box::pin(self.next_input()).await;
                }
                if trimmed.starts_with('/') {
                    Some(UserInput::Command(trimmed.to_string()))
                } else {
                    Some(UserInput::Prompt(trimmed.to_string()))
                }
            }
        }
    }

    async fn render_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::MessageUpdate { message } => {
                let text = message.text();
                let chars: Vec<char> = text.chars().collect();
                if chars.len() > self.rendered_chars {
                    let delta: String = chars[self.rendered_chars..].iter().collect();
                    print!("{}", delta);
                    io::stdout().flush().ok();
                    self.rendered_chars = chars.len();
                }
            }
            AgentEvent::MessageEnd { .. } => {
                println!();
                self.rendered_chars = 0;
            }
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                if self.is_repl() {
                    print!("\n[{}...", tool_name);
                    io::stdout().flush().ok();
                } else {
                    println!("\n[Running {}...]", tool_name);
                }
            }
            AgentEvent::ToolExecutionUpdate {
                tool_name, lines, ..
            } => {
                if self.is_repl() {
                    for line in lines {
                        print!(" {}", line.content);
                    }
                    io::stdout().flush().ok();
                } else {
                    for line in lines {
                        println!("[{}: {}]", tool_name, line.content);
                    }
                }
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                result,
                is_error,
                ..
            } => {
                if self.is_repl() {
                    if is_error {
                        println!(" error]");
                        let preview = crate::utils::truncate_chars(&result, 80);
                        println!("  {}", preview.replace('\n', " "));
                    } else {
                        let char_count = result.chars().count();
                        let first_line = result.lines().next().unwrap_or("");
                        if char_count <= 60 && !result.contains('\n') {
                            println!(" {}]", result);
                        } else if first_line.chars().count() <= 50 {
                            println!(" {}...]", first_line);
                        } else {
                            let preview: String = first_line.chars().take(40).collect();
                            println!(" {}...]", preview);
                        }
                    }
                } else if is_error {
                    println!("[{} failed: {}]", tool_name, result);
                } else {
                    let preview = crate::utils::truncate_chars(&result, 200);
                    println!("[{}: {}]", tool_name, preview);
                }
            }
            AgentEvent::CompactionStart { reason } => {
                println!(
                    "[Compacting context ({})]",
                    crate::utils::compaction_reason_str(reason)
                );
            }
            AgentEvent::CompactionEnd {
                tokens_before,
                tokens_after,
            } => {
                println!("[Compacted: ~{} -> ~{} tokens]", tokens_before, tokens_after);
            }
            AgentEvent::Error { message } => {
                eprintln!("\nError: {}", message);
            }
            _ => {}
        }
    }

    async fn render_turn_end(&mut self, total_usage: &Usage, model: &Model) {
        let cost = total_usage.calculate_cost(model);
        match self.mode {
            Mode::OneShot { .. } => {
                println!(
                    "\n[Tokens: {} in, {} out | Cost: ${:.4}]",
                    total_usage.input, total_usage.output, cost.total
                );
            }
            Mode::Repl => {
                if io::stdout().is_terminal() {
                    println!(
                        "[{} in, {} out | ${:.4}]",
                        total_usage.input, total_usage.output, cost.total
                    );
                }
                println!();
            }
        }
    }

    async fn show_system(&mut self, text: &str) {
        println!("{}", text);
    }

    async fn show_error(&mut self, text: &str) {
        eprintln!("{}", text);
    }

    async fn handle_interaction(&mut self, req: InteractionRequest) {
        match req.kind {
            InteractionKind::AskQuestion { question, options } => {
                println!("\n{}", question);
                for (i, opt) in options.iter().enumerate() {
                    println!("  {}) {} — {}", i + 1, opt.label, opt.description);
                }
                print!("Enter choice (1-{}): ", options.len());
                io::stdout().flush().ok();
                let num_options = options.len();
                let line = read_line_blocking().await;
                let response = match line.trim().parse::<usize>() {
                    Ok(n) if n >= 1 && n <= num_options => {
                        InteractionResponse::Answer(options[n - 1].label.clone())
                    }
                    _ => InteractionResponse::Cancelled,
                };
                let _ = req.response_tx.send(response);
            }
            InteractionKind::Typed { schema_id, payload } => match schema_id.as_str() {
                "plan.submit" => {
                    let plan: tau_tools::Plan = match serde_json::from_value(payload) {
                        Ok(p) => p,
                        Err(e) => {
                            let _ = req.response_tx.send(InteractionResponse::Rejected {
                                reason: format!("Invalid plan payload: {e}"),
                            });
                            return;
                        }
                    };
                    println!("\nPlan submitted ({} step(s)):", plan.items.len());
                    for step in &plan.items {
                        println!("  - {}: {}", step.id, step.title);
                    }
                    print!("Approve plan? [y/N]: ");
                    io::stdout().flush().ok();
                    let line = read_line_blocking().await;
                    let response = match line.trim().to_ascii_lowercase().as_str() {
                        "y" | "yes" => InteractionResponse::Approved { payload: None },
                        _ => InteractionResponse::Rejected {
                            reason: "User declined plan".into(),
                        },
                    };
                    let _ = req.response_tx.send(response);
                }
                "tool.confirm" => {
                    let tool_name = payload
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let activity = payload
                        .get("activity")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    println!("\nApproval required for {tool_name}: {activity}");
                    print!("Approve? [y/N]: ");
                    io::stdout().flush().ok();
                    let line = read_line_blocking().await;
                    let response = match line.trim().to_ascii_lowercase().as_str() {
                        "y" | "yes" => InteractionResponse::Approved { payload: None },
                        _ => InteractionResponse::Rejected {
                            reason: "User declined".into(),
                        },
                    };
                    let _ = req.response_tx.send(response);
                }
                other => {
                    let _ = req.response_tx.send(InteractionResponse::Rejected {
                        reason: format!("Unknown typed interaction schema: {other}"),
                    });
                }
            },
        }
    }

    fn can_render_approval(&self) -> bool {
        // Stdout reads y/N from stdin synchronously — yes, we can.
        true
    }
}

async fn read_line_blocking() -> String {
    tokio::task::spawn_blocking(|| {
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).ok();
        buf
    })
    .await
    .unwrap_or_default()
}
