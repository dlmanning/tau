//! `tau mcp list` — connect to configured servers and show their tools.

use tau_agent::ToolRisk;

use crate::config::Config;

/// Connects to every enabled server (same path as a real session),
/// prints status + namespaced tools, and shuts back down. Exits
/// non-zero if any server failed so scripts can health-check configs.
pub(crate) async fn list(cfg: &Config) -> anyhow::Result<()> {
    let specs = cfg.mcp_specs();
    if specs.is_empty() {
        println!("No MCP servers configured.");
        println!("Add [mcp_servers.<name>] sections to {}", Config::config_path().display());
        return Ok(());
    }

    let manager = tau_tools::mcp::McpManager::connect_all(specs).await;

    for (name, count) in manager.servers().await {
        println!("{name}: connected ({count} tools)");
    }
    let tools = manager.tools().await;
    for tool in &tools {
        let risk = match tool.risk(&serde_json::Value::Null) {
            ToolRisk::Safe => "safe",
            ToolRisk::Local => "local",
            ToolRisk::Elevated => "elevated",
        };
        let desc = tool.description().lines().next().unwrap_or("");
        println!("  {:<44} [{risk:<8}] {desc}", tool.name());
    }

    let failed = !manager.failures().is_empty();
    for (name, err) in manager.failures() {
        eprintln!("{name}: FAILED — {err}");
    }
    manager.shutdown_all().await;

    if failed {
        std::process::exit(1);
    }
    Ok(())
}
