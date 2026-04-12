//! WebFetch tool - fetches and converts web content

use std::net::IpAddr;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

/// Maximum content size to return (characters)
const MAX_CONTENT_CHARS: usize = 100_000;
/// HTTP request timeout
const FETCH_TIMEOUT_SECS: u64 = 60;
/// Maximum HTTP response body size (10 MB)
const MAX_CONTENT_BYTES: usize = 10 * 1024 * 1024;

#[derive(Deserialize, JsonSchema)]
struct WebFetchArgs {
    /// The URL to fetch content from
    url: String,
    /// The prompt describing what information to extract from the page
    prompt: Option<String>,
}

/// Tool for fetching web content and converting HTML to markdown
pub struct WebFetchTool;

impl WebFetchTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn label(&self) -> &str {
        "Fetch"
    }

    fn description(&self) -> &str {
        "Fetches content from a specified URL and converts HTML to markdown.\n\
         - Takes a URL and a prompt as input\n\
         - Fetches the URL content, converts HTML to markdown\n\
         - Returns the page content for you to analyze with the given prompt\n\
         - Use this tool when you need to retrieve and analyze web content\n\n\
         Usage notes:\n\
         - The URL must be a fully-formed valid URL\n\
         - HTTP URLs will be automatically upgraded to HTTPS\n\
         - The prompt should describe what information you want to extract from the page\n\
         - This tool is read-only and does not modify any files\n\
         - Results may be truncated if the content is very large\n\
         - When a URL redirects to a different host, the tool will inform you and \
         provide the redirect URL. You should then make a new request with that URL.\n\
         - For GitHub URLs, prefer using the gh CLI via bash instead \
         (e.g., gh pr view, gh issue view, gh api)"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(WebFetchArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: WebFetchArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let url_str = &args.url;
        let prompt = args.prompt.as_deref().unwrap_or("");

        // Validate and normalize URL
        let url = match validate_url(url_str) {
            Ok(u) => u,
            Err(e) => return ToolResult::error(e),
        };

        // Block requests to private/internal IPs (SSRF protection)
        if let Err(e) = check_host_safety(&url).await {
            return ToolResult::error(e);
        }

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("Cancelled");
        }

        ctx.progress.send(format!("Fetching {}", url));

        // Fetch the content
        let start = std::time::Instant::now();
        let response = match fetch_url(&url).await {
            Ok(r) => r,
            Err(e) => return ToolResult::error(format!("Failed to fetch URL: {}", e)),
        };

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("Cancelled");
        }

        let status = response.status;
        let content_type = response.content_type;
        let body = response.body;
        let final_url = response.url;
        let duration_ms = start.elapsed().as_millis();

        if status >= 400 {
            return ToolResult::error(format!(
                "HTTP {} fetching {}\nDuration: {}ms",
                status, final_url, duration_ms
            ));
        }

        // Convert to markdown if HTML
        let markdown = if content_type.contains("html") {
            html_to_markdown(&body)
        } else {
            body
        };

        // Truncate if too large
        let (result, truncated) = if markdown.chars().count() > MAX_CONTENT_CHARS {
            let truncated_content: String = markdown.chars().take(MAX_CONTENT_CHARS).collect();
            (truncated_content, true)
        } else {
            (markdown, false)
        };

        let mut output = format!(
            "Prompt: {}\nURL: {}\nStatus: {}\nContent-Type: {}\nDuration: {}ms\n",
            prompt, final_url, status, content_type, duration_ms
        );

        if truncated {
            output.push_str(&format!(
                "Note: Content truncated to {} characters\n",
                MAX_CONTENT_CHARS
            ));
        }

        output.push_str("\n---\n\n");
        output.push_str(&result);

        ToolResult::text(output)
    }
}

struct FetchResponse {
    status: u16,
    content_type: String,
    body: String,
    url: String,
}

fn validate_url(url_str: &str) -> Result<String, String> {
    let url = url::Url::parse(url_str).map_err(|e| format!("Invalid URL: {}", e))?;

    // Must be http or https
    match url.scheme() {
        "http" | "https" => {}
        scheme => return Err(format!("Unsupported URL scheme: {}", scheme)),
    }

    // Must have a host
    if url.host().is_none() {
        return Err("URL must have a host".to_string());
    }

    // Must not contain credentials
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URL must not contain credentials".to_string());
    }

    // URL length limit
    if url_str.len() > 2000 {
        return Err("URL too long (max 2000 characters)".to_string());
    }

    // Upgrade http to https
    let mut url = url;
    if url.scheme() == "http" {
        url.set_scheme("https").ok();
    }

    Ok(url.to_string())
}

/// Check if an IP address is private, loopback, or link-local.
fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()             // 127.0.0.0/8
            || v4.is_private()           // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            || v4.is_link_local()        // 169.254.0.0/16 (includes cloud metadata)
            || v4.is_unspecified()       // 0.0.0.0
            || v4.is_broadcast()         // 255.255.255.255
            || (o[0] == 100 && (o[1] & 0xC0) == 64) // 100.64.0.0/10 shared/CGN
        }
        IpAddr::V6(v6) => {
            // Check IPv4-mapped addresses (::ffff:x.x.x.x) against IPv4 rules
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(mapped));
            }
            v6.is_loopback()             // ::1
            || v6.is_unspecified()       // ::
            || (v6.segments()[0] & 0xffc0) == 0xfe80  // fe80::/10 link-local
            || (v6.segments()[0] & 0xfe00) == 0xfc00 // fc00::/7 unique-local
        }
    }
}

/// Resolve a URL's host and reject private/internal IP addresses.
///
/// Note: DNS is resolved here then again by reqwest, so a TOCTOU gap
/// exists (DNS rebinding). Low risk for a local CLI tool.
async fn check_host_safety(url_str: &str) -> Result<(), String> {
    let url = url::Url::parse(url_str).map_err(|e| format!("Invalid URL: {}", e))?;

    let host = url.host_str().ok_or("URL has no host")?;

    // If the host is already an IP literal, check it directly
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_blocked_ip(ip) {
            return Err(format!(
                "Blocked request to private/internal address: {}",
                ip
            ));
        }
        return Ok(());
    }

    // Resolve hostname and check all returned IPs
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs = tokio::net::lookup_host(format!("{}:{}", host, port))
        .await
        .map_err(|e| format!("DNS resolution failed for {}: {}", host, e))?;

    let addrs: Vec<_> = addrs.collect();
    if addrs.is_empty() {
        return Err(format!("DNS resolution returned no addresses for {}", host));
    }

    for addr in &addrs {
        if is_blocked_ip(addr.ip()) {
            return Err(format!(
                "Blocked request: {} resolves to private/internal address {}",
                host,
                addr.ip()
            ));
        }
    }

    Ok(())
}

/// Check if two URLs have the same domain (ignoring www. prefix)
fn is_same_domain(a: &str, b: &str) -> bool {
    fn strip_www(host: &str) -> &str {
        host.strip_prefix("www.").unwrap_or(host)
    }
    let host_a = url::Url::parse(a)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_string()));
    let host_b = url::Url::parse(b)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_string()));
    match (host_a, host_b) {
        (Some(a), Some(b)) => strip_www(&a) == strip_www(&b),
        _ => false,
    }
}

async fn fetch_url(url: &str) -> Result<FetchResponse, String> {
    // Don't auto-follow redirects — we need to detect cross-domain redirects
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent("Claude-User (tau; +https://github.com/anthropics)")
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let mut current_url = url.to_string();
    let mut redirects = 0;
    const MAX_REDIRECTS: u32 = 10;

    loop {
        let response = client
            .get(&current_url)
            .header("Accept", "text/markdown, text/html, */*")
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    "Request timed out after 60 seconds".to_string()
                } else if e.is_connect() {
                    format!("Connection failed: {}", e)
                } else {
                    e.to_string()
                }
            })?;

        let status = response.status().as_u16();

        // Handle redirects
        if (300..400).contains(&status) {
            if redirects >= MAX_REDIRECTS {
                return Err(format!("Too many redirects (max {})", MAX_REDIRECTS));
            }
            redirects += 1;

            let location = response
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or("Redirect without Location header")?;

            // Resolve relative redirects
            let redirect_url = url::Url::parse(location)
                .or_else(|_| url::Url::parse(&current_url).and_then(|base| base.join(location)))
                .map_err(|e| format!("Invalid redirect URL: {}", e))?
                .to_string();

            // Block redirects to private/internal IPs
            check_host_safety(&redirect_url).await?;

            // Block cross-domain redirects — report to model so it can re-request
            if !is_same_domain(&current_url, &redirect_url) {
                return Ok(FetchResponse {
                    status,
                    content_type: "text/plain".to_string(),
                    body: format!(
                        "This URL redirects to a different domain: {}\n\
                         Make a new web_fetch request with that URL to follow the redirect.",
                        redirect_url
                    ),
                    url: current_url,
                });
            }

            current_url = redirect_url;
            continue;
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/plain")
            .to_string();

        // Check content length before downloading
        if let Some(len) = response.content_length() {
            if len as usize > MAX_CONTENT_BYTES {
                return Err(format!(
                    "Response too large: {} bytes (max {} bytes)",
                    len, MAX_CONTENT_BYTES
                ));
            }
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        if bytes.len() > MAX_CONTENT_BYTES {
            return Err(format!(
                "Response too large: {} bytes (max {} bytes)",
                bytes.len(),
                MAX_CONTENT_BYTES
            ));
        }

        let body = String::from_utf8_lossy(&bytes).to_string();

        return Ok(FetchResponse {
            status,
            content_type,
            body,
            url: current_url,
        });
    }
}

fn html_to_markdown(html: &str) -> String {
    htmd::convert(html).unwrap_or_else(|_| html.to_string())
}
