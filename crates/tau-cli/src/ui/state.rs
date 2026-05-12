use std::collections::HashMap;

use ratatui::layout::Rect;
use tau_agent::AgentConfig;
use tau_ai::Model;

use super::{
    theme::Theme,
    widgets::{InputBox, SelectorState, message_list::ChatMessage},
};
use tokio::sync::mpsc;

use super::types::{GitBranchState, PendingInteraction, PendingPlan, UiMessage};

/// Token and cost accounting for the session.
#[derive(Default)]
pub(super) struct UsageStats {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub cost: f64,
}

impl UsageStats {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn accumulate(&mut self, usage: &tau_ai::Usage, model: &Model) {
        self.input_tokens += usage.input;
        self.output_tokens += usage.output;
        self.cache_read += usage.cache_read;
        self.cache_write += usage.cache_write;
        let cost = usage.calculate_cost(model);
        self.cost += cost.total;
    }
}

/// Per-agent progress tracking for richer subagent display.
pub(super) struct AgentProgress {
    pub description: String,
    pub tool_count: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub activity: String,
    pub finished: bool,
}

impl AgentProgress {
    pub fn new(description: String) -> Self {
        Self {
            description,
            tool_count: 0,
            input_tokens: 0,
            output_tokens: 0,
            activity: "starting...".to_string(),
            finished: false,
        }
    }
}

/// TUI application state.
pub(super) struct TuiState {
    /// Chat messages
    pub messages: Vec<ChatMessage>,
    /// Per-agent progress state keyed by agent_id.
    pub agent_progress: HashMap<String, AgentProgress>,
    /// Insertion order of agent IDs for tree rendering.
    pub agent_order: Vec<String>,
    /// Input box
    pub input: InputBox,
    /// Current scroll position
    pub scroll: usize,
    /// Whether to auto-follow new content at the bottom
    pub follow_bottom: bool,
    /// Whether agent is currently processing
    pub is_processing: bool,
    /// Git branch name with background refresh.
    pub git_branch: GitBranchState,
    /// Current status message
    pub status: String,
    /// Theme
    pub theme: Theme,
    /// Token and cost stats
    pub usage: UsageStats,
    /// Model for cost calculation
    pub model: Model,
    /// Current reasoning level
    pub reasoning: tau_ai::ReasoningLevel,
    /// Whether adaptive thinking is enabled (fixed per session; reasoning level
    /// can change at runtime but adaptive mode cannot be toggled).
    pub thinking_adaptive: bool,
    /// Available models for selection
    pub available_models: Vec<Model>,
    /// Channel to send messages to agent handler
    pub ui_tx: mpsc::Sender<UiMessage>,
    /// Model selector state
    pub model_selector: SelectorState,
    /// Branch selector state
    pub branch_selector: SelectorState,
    /// Pending interaction request (question waiting for user to pick an option)
    pub pending_interaction: Option<PendingInteraction>,
    /// Pending plan submission awaiting Approve / Execute now / Reject.
    pub pending_plan: Option<PendingPlan>,
}

impl TuiState {
    pub fn new(
        config: &AgentConfig,
        available_models: Vec<Model>,
        ui_tx: mpsc::Sender<UiMessage>,
    ) -> Self {
        let mut input = InputBox::new().with_placeholder("Type a message...");
        input.set_focused(true);

        let current_index = available_models
            .iter()
            .position(|m| m.id == config.model.id)
            .unwrap_or(0);

        let model_selector = SelectorState {
            selected: current_index,
            ..Default::default()
        };

        Self {
            messages: vec![],
            agent_progress: HashMap::new(),
            agent_order: Vec::new(),
            input,
            scroll: 0,
            follow_bottom: true,
            is_processing: false,
            git_branch: GitBranchState::new(),
            status: "Ready".to_string(),
            theme: Theme::dark(),
            usage: UsageStats::default(),
            model: config.model.clone(),
            reasoning: config.reasoning,
            thinking_adaptive: config.thinking_adaptive,
            available_models,
            ui_tx,
            model_selector,
            branch_selector: SelectorState::default(),
            pending_interaction: None,
            pending_plan: None,
        }
    }

    /// Open the branch selector popup. Invoked by the TUI frontend's
    /// `Frontend::open_branch_selector` impl when the user runs bare
    /// `/branch`. Selection navigates with Up/Down and emits
    /// `UiMessage::Branch(Some(idx))` on Enter.
    pub fn open_branch_selector(&mut self) {
        if !self.messages.is_empty() {
            self.branch_selector.selected = self.messages.len().saturating_sub(1);
            self.branch_selector.show();
        }
    }

    /// Send a UI message, logging a warning if the channel is closed.
    pub async fn send_ui(&self, msg: UiMessage) {
        if self.ui_tx.send(msg).await.is_err() {
            tracing::warn!("UI message channel closed");
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        self.follow_bottom = true;
    }

    /// Show a system message.
    pub fn show_system_message(&mut self, content: &str) {
        self.messages.push(ChatMessage::system(content));
        self.scroll_to_bottom();
    }

    /// Sync model/reasoning from agent config (call before rendering).
    /// Sync mutable agent config (model, reasoning level, thinking
    /// mode) into the display. Called by the TUI frontend's
    /// `Frontend::on_config_change` impl after `/model` / `/thinking`.
    pub fn sync_from_config(&mut self, config: &AgentConfig) {
        self.model = config.model.clone();
        self.reasoning = config.reasoning;
    }

    /// Reset token/cost counters and agent progress.
    pub fn reset_stats(&mut self) {
        self.usage.reset();
        self.agent_progress.clear();
        self.agent_order.clear();
    }

    /// Recalculate scroll position based on content height.
    /// Call before rendering the conversation area.
    pub fn clamp_scroll(&mut self, conversation_inner: Rect) {
        let content_height = super::widgets::message_list::calculate_message_height(
            &self.messages,
            conversation_inner.width as usize,
            &self.theme,
        );
        let max_scroll = content_height.saturating_sub(conversation_inner.height as usize);
        if self.follow_bottom {
            self.scroll = max_scroll;
        } else {
            self.scroll = self.scroll.min(max_scroll);
            if self.scroll >= max_scroll {
                self.follow_bottom = true;
            }
        }
    }
}
