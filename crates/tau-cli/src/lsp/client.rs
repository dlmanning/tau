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
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<RpcResult>>>>,
    next_id: AtomicI64,
    _child: Mutex<Child>,
}

/// Result from a JSON-RPC response — either a value or a server error.
enum RpcResult {
    Ok(serde_json::Value),
    Err { code: i64, message: String },
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
            .stderr(std::process::Stdio::piped()) // capture for debugging, don't null
            .kill_on_drop(true)
            .spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("no stdout"))?;

        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    tracing::debug!(target: "lsp_stderr", "{}", line.trim_end());
                    line.clear();
                }
            });
        }

        let pending: Arc<Mutex<HashMap<i64, oneshot::Sender<RpcResult>>>> =
            Arc::new(Mutex::new(HashMap::new()));

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
            _child: Mutex::new(child),
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

        let rpc_result = tokio::time::timeout(std::time::Duration::from_secs(60), rx)
            .await
            .map_err(|_| anyhow::anyhow!("LSP request timed out after 60s: {}", method))?
            .map_err(|_| anyhow::anyhow!("LSP server connection lost"))?;

        match rpc_result {
            RpcResult::Ok(value) => serde_json::from_value(value).map_err(Into::into),
            RpcResult::Err { code, message } => {
                Err(anyhow::anyhow!("LSP server error ({}): {}", code, message))
            }
        }
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
        let _: serde_json::Value = self.request("shutdown", serde_json::Value::Null).await?;
        self.notify("exit", serde_json::Value::Null).await?;
        Ok(())
    }

    /// Check if the server process is still alive.
    pub async fn is_alive(&self) -> bool {
        let mut child = self._child.lock().await;
        matches!(child.try_wait(), Ok(None))
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
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<RpcResult>>>>,
) -> anyhow::Result<()> {
    let mut reader = BufReader::new(stdout);
    let mut header_buf = String::new();

    loop {
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

        let mut body = vec![0u8; len];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut body).await?;
        let body_str = String::from_utf8_lossy(&body);

        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&body_str) {
            if let Some(id) = response.id {
                let mut pending = pending.lock().await;
                if let Some(tx) = pending.remove(&id) {
                    let result = if let Some(error) = response.error {
                        RpcResult::Err {
                            code: error.code,
                            message: error.message,
                        }
                    } else {
                        RpcResult::Ok(response.result.unwrap_or(serde_json::Value::Null))
                    };
                    let _ = tx.send(result);
                }
            }
            // Notifications from server (id is None) — logged via stderr
        }
    }
}
