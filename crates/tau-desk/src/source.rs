use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::broadcast;

use tau_agent::tool::BoxedTool;

use crate::Result;

pub type SourceId = String;

/// A platform integration. Sources are pure data providers: they expose
/// tools (read + write), and optionally signal that something changed.
/// They do not produce cards, do not know about piles, and do not own
/// card identity. All editorial decisions are made by agent prompts.
#[async_trait]
pub trait Source: Send + Sync {
    fn id(&self) -> &str;

    /// Tools this source contributes. Both reads (`jira_get_issue`,
    /// `gh_list_review_requests`) and writes (`jira_post_comment`,
    /// `gh_post_pr_comment`). Write tools are gated by the runtime's
    /// `ApprovalPolicy` for immediate-write flows; deferred writes go
    /// through `enqueue_draft` and dispatch via the registry on approval.
    fn tools(&self) -> Vec<BoxedTool>;

    /// Optional push channel for sources with native events (Slack RTM,
    /// GitHub event API). HTTP-poll-only sources return `None` and rely
    /// on scheduled scans.
    fn watch(&self) -> Option<broadcast::Receiver<ChangeNotice>> {
        None
    }
}

/// Notice that something on a source has changed. Routed to a focused
/// agent prompt or a registered mechanical handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeNotice {
    pub source: SourceId,
    pub summary: String,
    pub context: Value,
}

/// Holds the registered sources + their contributed tools.
pub struct SourceRegistry {
    sources: HashMap<SourceId, Arc<dyn Source>>,
    watch_tx: broadcast::Sender<ChangeNotice>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        let (watch_tx, _) = broadcast::channel(64);
        Self {
            sources: HashMap::new(),
            watch_tx,
        }
    }

    /// Errors if a source with the same id is already registered, or if
    /// any of its tool names collide with existing tools in the registry.
    pub fn register(&mut self, source: Arc<dyn Source>) -> Result<()> {
        let id = source.id().to_string();
        if self.sources.contains_key(&id) {
            return Err(crate::error::Error::Other(anyhow::anyhow!(
                "source `{id}` already registered"
            )));
        }

        // Tool-name collision check across registered sources.
        let existing: std::collections::HashSet<String> = self
            .sources
            .values()
            .flat_map(|s| s.tools().into_iter().map(|t| t.name().to_string()))
            .collect();
        for tool in source.tools() {
            if existing.contains(tool.name()) {
                return Err(crate::error::Error::Other(anyhow::anyhow!(
                    "tool name collision: `{}` already provided by another source",
                    tool.name()
                )));
            }
        }

        // Forward this source's watch channel into the merged stream, if any.
        if let Some(mut rx) = source.watch() {
            let tx = self.watch_tx.clone();
            tokio::spawn(async move {
                while let Ok(notice) = rx.recv().await {
                    if tx.send(notice).is_err() {
                        break;
                    }
                }
            });
        }

        self.sources.insert(id, source);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&Arc<dyn Source>> {
        self.sources.get(id)
    }

    /// All tools contributed by all registered sources. Hosts merge this
    /// with `tau-desk`'s desk-state tools when constructing agent builders.
    pub fn all_tools(&self) -> Vec<BoxedTool> {
        self.sources.values().flat_map(|s| s.tools()).collect()
    }

    /// Fan-in of every source's `watch()` channel. Sources without
    /// native push channels never fire here; rely on scheduled scans.
    pub fn merged_watch(&self) -> broadcast::Receiver<ChangeNotice> {
        self.watch_tx.subscribe()
    }

    /// Publish a notice into the merged stream. Used by host-side
    /// webhook ingestion ([`DeskAgent::ingest_signal`](crate::DeskAgent::ingest_signal))
    /// so externally-delivered signals look the same as ones produced
    /// by a registered source's `watch()` channel.
    pub fn publish(&self, notice: ChangeNotice) {
        let _ = self.watch_tx.send(notice);
    }
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
