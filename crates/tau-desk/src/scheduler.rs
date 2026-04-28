use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::source::SourceId;

pub type TaskId = String;
pub type TaskName = String;

/// What fires a task. `OnSignal` consumes the matched source's
/// `watch()` channel; bursts are coalesced per `Concurrency::Coalesce`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Trigger {
    /// Standard cron expression. Timezone is host-configured.
    Cron(String),
    Interval(Duration),
    /// Only fires via `desk.trigger_scan(name)`. Useful for ad-hoc
    /// rescans and for tests.
    Manual,
    OnSignal(SourceId),
}

/// Concurrency policy when a fire arrives while a previous run of the
/// same task is still in flight.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum Concurrency {
    /// Drop the new fire with a warning. Default for `Cron`.
    Skip,
    /// Collapse multiple fires within `window` into one run. Default
    /// for `OnSignal`.
    Coalesce { window: Duration },
    /// Run independently. Caller must ensure mutations don't conflict.
    Parallel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub id: TaskId,
    pub name: TaskName,
    pub trigger: Trigger,
    pub concurrency: Concurrency,
    pub prompt: PromptSpec,
    pub enabled: bool,
}

/// Prompt for a per-task agent. `Hydrated` is the common case: a
/// template plus a `HydrationSpec` describing which slices of current
/// desk state to interpolate at fire time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptSpec {
    Plain(String),
    Hydrated {
        template: String,
        include: HydrationSpec,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HydrationSpec {
    pub cards_in: Vec<crate::card::CardPile>,
    pub drafts: bool,
    pub activity_recent: usize,
    pub notes: bool,
    pub brief: bool,
    /// Surfaces `last_modified_by` + recency on each card so the agent
    /// can see "user moved this 2h ago" and respect it (soft conflict
    /// resolution).
    pub show_provenance: bool,
}
