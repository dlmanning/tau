//! JSON-RPC 2.0 transport over stdio for LSP servers

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, oneshot};

/// A JSON-RPC 2.0 client that communicates with an LSP server over stdio.
pub struct LspClient {
    stdin: Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>>,
    next_id: AtomicI64,
    _child: Child,
}

#[derive(Serialize)]
struct JsonRpcRequest<P: Serialize> {
    jsonrpc: &'static str,
    id: i64,
    method: String,
    params: P,
}

#[derive(Serialize)]
struct JsonRpcNotification<P: Serialize> {
    jsonrpc: &'static str,
    method: String,
    params: P,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    id: Option<i64>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcError {
    code: i64,
    message: String,
}

impl LspClient {
    /// Spawn an LSP server process and return a client connected to it.
    pub async fn spawn(
        command: &str,
        args: &[String],
        cwd: &Path,
    ) -> anyhow::Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .current_dir(cwd)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("no stdout"))?;

        let pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Spawn background reader task
        let pending_clone = pending.clone();
        tokio::spawn(async move {
            if let Err(e) = read_loop(stdout, pending_clone).await {
                tracing::debug!("LSP read loop ended: {}", e);
            }
        });

        Ok(Self {
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicI64::new(1),
            _child: child,
        })
    }

    /// Send a request and wait for the response.
    pub async fn request<P: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        params: P,
    ) -> anyhow::Result<R> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();

        self.pending.lock().await.insert(id, tx);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };
        self.send_message(&serde_json::to_string(&request)?).await?;

        let response = tokio::time::timeout(std::time::Duration::from_secs(30), rx)
            .await
            .map_err(|_| anyhow::anyhow!("LSP request timed out: {}", method))?
            .map_err(|_| anyhow::anyhow!("LSP response channel closed"))?;

        serde_json::from_value(response).map_err(Into::into)
    }

    /// Send a notification (no response expected).
    pub async fn notify<P: Serialize>(&self, method: &str, params: P) -> anyhow::Result<()> {
        let notification = JsonRpcNotification {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
        };
        self.send_message(&serde_json::to_string(&notification)?).await
    }

    /// Send an LSP shutdown request and exit notification.
    #[allow(dead_code)]
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        // shutdown is a request
        let _: serde_json::Value = self.request("shutdown", serde_json::Value::Null).await?;
        // exit is a notification
        self.notify("exit", serde_json::Value::Null).await?;
        Ok(())
    }

    async fn send_message(&self, json: &str) -> anyhow::Result<()> {
        let msg = format!("Content-Length: {}\r\n\r\n{}", json.len(), json);
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(msg.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }
}

/// Background task: read Content-Length framed messages from stdout, dispatch responses.
async fn read_loop(
    stdout: ChildStdout,
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>>,
) -> anyhow::Result<()> {
    let mut reader = BufReader::new(stdout);
    let mut header_buf = String::new();

    loop {
        // Read headers until empty line
        let mut content_length: Option<usize> = None;
        loop {
            header_buf.clear();
            let n = reader.read_line(&mut header_buf).await?;
            if n == 0 {
                return Ok(()); // EOF
            }
            let line = header_buf.trim();
            if line.is_empty() {
                break;
            }
            if let Some(val) = line.strip_prefix("Content-Length: ") {
                content_length = val.parse().ok();
            }
        }

        let len = match content_length {
            Some(len) => len,
            None => continue,
        };

        // Read body
        let mut body = vec![0u8; len];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut body).await?;
        let body_str = String::from_utf8_lossy(&body);

        // Try to parse as response
        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&body_str) {
            if let Some(id) = response.id {
                let mut pending = pending.lock().await;
                if let Some(tx) = pending.remove(&id) {
                    let value = if let Some(error) = response.error {
                        // Return error as a JSON value so the caller can handle it
                        serde_json::json!({
                            "error": { "code": error.code, "message": error.message }
                        })
                    } else {
                        response.result.unwrap_or(serde_json::Value::Null)
                    };
                    let _ = tx.send(value);
                }
            }
            // Notifications from server (id is None) — just ignore for now
        }
    }
}
