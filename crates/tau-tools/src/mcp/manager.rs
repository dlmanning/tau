//! Connection lifecycle for configured MCP servers.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use rmcp::model::{ClientCapabilities, ClientInfo, Implementation};
use rmcp::service::RunningService;
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use tau_agent::BoxedTool;

use super::config::{McpServerSpec, McpTransportSpec};
use super::tool::McpTool;
use super::{naming, risk, schema};

/// How long a server gets to spawn/connect + initialize + list tools.
/// Per-call timeouts are configured separately (`McpServerSpec`).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

struct McpServerConnection {
    name: String,
    service: RunningService<RoleClient, ClientInfo>,
    /// Snapshot from `tools/list` at connect time. v1 ignores
    /// `tools/list_changed` — see module docs.
    tools: Vec<rmcp::model::Tool>,
    spec: McpServerSpec,
}

/// Owns the live MCP connections. Mirrors the LSP manager's lifecycle:
/// constructed at startup, shared via `Arc`, `shutdown_all` on exit.
pub struct McpManager {
    connections: tokio::sync::Mutex<Vec<McpServerConnection>>,
    failures: Vec<(String, String)>,
}

impl McpManager {
    /// Connect to every spec concurrently. A server that fails to
    /// spawn, initialize, or list tools (or exceeds the 15s connect
    /// timeout) is recorded in [`failures`](Self::failures) and
    /// skipped — never fatal to startup.
    pub async fn connect_all(specs: Vec<McpServerSpec>) -> Self {
        let attempts = specs.into_iter().map(|spec| async {
            let name = spec.name.clone();
            match tokio::time::timeout(CONNECT_TIMEOUT, connect_one(spec)).await {
                Ok(Ok(conn)) => Ok(conn),
                Ok(Err(e)) => Err((name, e.to_string())),
                Err(_) => Err((
                    name,
                    format!("connect timed out after {}s", CONNECT_TIMEOUT.as_secs()),
                )),
            }
        });
        let mut connections = Vec::new();
        let mut failures = Vec::new();
        for outcome in futures::future::join_all(attempts).await {
            match outcome {
                Ok(conn) => connections.push(conn),
                Err((name, err)) => {
                    tracing::warn!(server = %name, error = %err, "MCP server failed");
                    failures.push((name, err));
                }
            }
        }
        // Deterministic server order → deterministic tool list →
        // stable provider prompt cache.
        connections.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            connections: tokio::sync::Mutex::new(connections),
            failures,
        }
    }

    /// Servers that failed to connect, with reasons.
    pub fn failures(&self) -> &[(String, String)] {
        &self.failures
    }

    /// Test-only: build a manager around already-running client
    /// services (e.g. served over an in-process duplex transport),
    /// performing the same `tools/list` snapshot as `connect_one`.
    #[doc(hidden)]
    pub async fn from_running_services(
        services: Vec<(McpServerSpec, RunningService<RoleClient, ClientInfo>)>,
    ) -> anyhow::Result<Self> {
        let mut connections = Vec::new();
        for (spec, service) in services {
            let tools = service.list_all_tools().await?;
            connections.push(McpServerConnection {
                name: spec.name.clone(),
                service,
                tools,
                spec,
            });
        }
        connections.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self {
            connections: tokio::sync::Mutex::new(connections),
            failures: Vec::new(),
        })
    }

    /// `(server, tool_count)` per live connection.
    pub async fn servers(&self) -> Vec<(String, usize)> {
        self.connections
            .lock()
            .await
            .iter()
            .map(|c| (c.name.clone(), c.tools.len()))
            .collect()
    }

    /// Build one [`McpTool`] per exposed remote tool, applying the
    /// spec's include/exclude lists and the provider-safe naming pass.
    pub async fn tools(&self) -> Vec<BoxedTool> {
        let connections = self.connections.lock().await;
        let mut taken = HashSet::new();
        let mut out: Vec<BoxedTool> = Vec::new();
        for conn in connections.iter() {
            for remote in &conn.tools {
                if !conn.spec.allows_tool(&remote.name) {
                    continue;
                }
                let Some(name) = naming::tool_name(&conn.name, &remote.name, &mut taken) else {
                    tracing::warn!(
                        server = %conn.name,
                        tool = %remote.name,
                        "skipping MCP tool: name collision"
                    );
                    continue;
                };
                let (tool_risk, category) =
                    risk::classify(conn.spec.trust, remote.annotations.as_ref());
                let description = remote
                    .description
                    .as_deref()
                    .or(remote.title.as_deref())
                    .unwrap_or("");
                out.push(Arc::new(McpTool {
                    name,
                    label: format!("{}:{}", conn.name, remote.name),
                    description: format!(
                        "MCP tool from server '{}'. {}",
                        conn.name, description
                    ),
                    schema: schema::normalize(&remote.input_schema, &remote.name),
                    remote_name: remote.name.to_string(),
                    server_name: conn.name.clone(),
                    peer: conn.service.peer().clone(),
                    timeout: conn.spec.call_timeout,
                    risk: tool_risk,
                    category,
                }) as BoxedTool);
            }
        }
        out
    }

    /// Cancel every connection (terminating stdio children). Called
    /// once at host shutdown, like `LspManager::shutdown_all`.
    pub async fn shutdown_all(&self) {
        let connections: Vec<McpServerConnection> =
            self.connections.lock().await.drain(..).collect();
        for conn in connections {
            if let Err(e) = conn.service.cancel().await {
                tracing::warn!(server = %conn.name, error = %e, "MCP shutdown error");
            }
        }
    }
}

async fn connect_one(spec: McpServerSpec) -> anyhow::Result<McpServerConnection> {
    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("tau", env!("CARGO_PKG_VERSION")),
    );
    let service = match &spec.transport {
        McpTransportSpec::Stdio { command, args, env } => {
            let command = expand(command)?;
            let mut expanded_env = Vec::new();
            for (k, v) in env {
                expanded_env.push((k.clone(), expand(v)?));
            }
            let args: Vec<String> = args.iter().map(|a| expand(a)).collect::<Result<_, _>>()?;
            let transport =
                TokioChildProcess::new(tokio::process::Command::new(command).configure(|c| {
                    c.args(&args);
                    c.envs(expanded_env);
                }))?;
            client_info.serve(transport).await?
        }
        McpTransportSpec::Http { url, auth_header } => {
            let mut config = StreamableHttpClientTransportConfig::with_uri(expand(url)?);
            if let Some(header) = auth_header {
                config.auth_header = Some(expand(header)?);
            }
            let transport = rmcp::transport::StreamableHttpClientTransport::with_client(
                reqwest::Client::default(),
                config,
            );
            client_info.serve(transport).await?
        }
    };
    let tools = service.list_all_tools().await?;
    Ok(McpServerConnection {
        name: spec.name.clone(),
        service,
        tools,
        spec,
    })
}

/// Expand `${VAR}` references from the process environment. A
/// referenced-but-unset variable is an error so the server fails
/// loudly instead of connecting with an empty credential.
fn expand(input: &str) -> anyhow::Result<String> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find('}') else {
            anyhow::bail!("unterminated ${{...}} in config value");
        };
        let var = &after[..end];
        match std::env::var(var) {
            Ok(v) => out.push_str(&v),
            Err(_) => anyhow::bail!("environment variable '{var}' is not set"),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_replaces_vars_and_errors_on_missing() {
        // SAFETY: test-only env mutation; tests in this module run in
        // one process and no other test reads this variable.
        unsafe { std::env::set_var("TAU_MCP_TEST_VAR", "hunter2") };
        assert_eq!(
            expand("Bearer ${TAU_MCP_TEST_VAR}").unwrap(),
            "Bearer hunter2"
        );
        assert_eq!(expand("no vars").unwrap(), "no vars");
        assert!(expand("${TAU_MCP_TEST_VAR_UNSET}").is_err());
        assert!(expand("${unterminated").is_err());
    }
}
