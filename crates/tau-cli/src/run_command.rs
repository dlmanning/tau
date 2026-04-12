//! Non-interactive single-command execution mode

use tau_agent::{Agent, AgentEvent};
use tau_ai::Model;

pub(crate) async fn run_command(
    agent: &mut Agent,
    command: &str,
    model: &Model,
    interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) -> anyhow::Result<()> {
    println!("tau> {}", command);
    println!();

    let mut receiver = agent.subscribe();
    let model_for_cost = model.clone();

    let handle = tokio::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            match event {
                AgentEvent::MessageUpdate { message } => {
                    let text = message.text();
                    if !text.is_empty() {
                        print!("\r{}", text);
                    }
                }
                AgentEvent::MessageEnd { message } => {
                    println!("\r{}", message.text());
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    println!("\n[Running {}...]", tool_name);
                }
                AgentEvent::ToolExecutionUpdate {
                    tool_name, content, ..
                } => {
                    println!("[{}: {}]", tool_name, content);
                }
                AgentEvent::ToolExecutionEnd {
                    tool_name,
                    result,
                    is_error,
                    ..
                } => {
                    if is_error {
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
                    println!(
                        "[Compacted: ~{} -> ~{} tokens]",
                        tokens_before, tokens_after
                    );
                }
                AgentEvent::Error { message } => {
                    eprintln!("Error: {}", message);
                }
                AgentEvent::AgentEnd { total_usage, .. } => {
                    let cost = total_usage.calculate_cost(&model_for_cost);
                    println!(
                        "\n[Tokens: {} in, {} out | Cost: ${:.4}]",
                        total_usage.input, total_usage.output, cost.total
                    );
                    break;
                }
                _ => {}
            }
        }
    });

    let interaction_handle = tokio::spawn(handle_interaction_stdin(interaction_rx));

    agent.prompt(command).await?;

    // Wait for event handler to finish (it breaks on AgentEnd)
    match tokio::time::timeout(std::time::Duration::from_secs(2), handle).await {
        Ok(_) => {}
        Err(_) => tracing::debug!("Event handler did not finish in time"),
    }
    interaction_handle.abort();

    Ok(())
}

/// Handle interaction requests by printing to stdout and reading from stdin.
/// Used in non-TUI modes (command mode and simple interactive mode).
pub(crate) async fn handle_interaction_stdin(
    mut rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) {
    use tau_agent::interaction::{InteractionKind, InteractionResponse};

    while let Some(request) = rx.recv().await {
        match request.kind {
            InteractionKind::AskQuestion { question, options } => {
                println!("\n{}", question);
                for (i, opt) in options.iter().enumerate() {
                    println!("  {}) {} — {}", i + 1, opt.label, opt.description);
                }
                print!("Enter choice (1-{}): ", options.len());
                std::io::Write::flush(&mut std::io::stdout()).ok();

                let num_options = options.len();
                let line = tokio::task::spawn_blocking(|| {
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input).ok();
                    input
                })
                .await
                .unwrap_or_default();

                let response = match line.trim().parse::<usize>() {
                    Ok(n) if n >= 1 && n <= num_options => {
                        InteractionResponse::Answer(options[n - 1].label.clone())
                    }
                    _ => InteractionResponse::Cancelled,
                };

                let _ = request.response_tx.send(response);
            }
        }
    }
}
