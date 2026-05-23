//! The [`History`] trait — a conversation's worth of state, from
//! the agent's point of view.
//!
//! Implementations are typically [`Branch`](crate::Branch) instances
//! over a [`Repository`](crate::Repository), but the trait says
//! nothing about that — it speaks in conversation terms: messages,
//! system prompt, tools, summaries. Hosts can write their own
//! backends (SQLite, network) against the same surface.
//!
//! What the agent runtime reads to build the next API request comes
//! from three methods on this trait:
//! - [`messages`](Self::messages) → API `messages` field
//! - [`system_prompt`](Self::system_prompt) → API `system` field
//! - [`tools`](Self::tools) → API `tools` field
//!
//! The three together describe the full prompt at the current tip.

use async_trait::async_trait;
use tau_ai::{Content, Message};

use crate::objects::ToolDef;

/// Error type for [`History`] operations.
///
/// A thin newtype around `Box<dyn std::error::Error + Send + Sync>`.
/// Host-side backend implementations can surface their own error
/// types via the blanket `From<E>` impl below, without depending
/// on this crate's internal error enum. Runtime code that consumes
/// a `History` typically maps this into its own error type at the
/// boundary.
///
/// The newtype (vs. a bare type alias) buys us ergonomic
/// conversions: any `E: std::error::Error + Send + Sync + 'static`
/// converts via `?` or `.into()`, eliminating the
/// `Box::new(...) as HistoryError` ceremony at call sites.
pub struct HistoryError(Box<dyn std::error::Error + Send + Sync>);

impl HistoryError {
    /// Construct from a `Display`-able message — useful for ad-hoc
    /// errors that don't have a backing `Error` type.
    pub fn msg(s: impl Into<String>) -> Self {
        HistoryError(s.into().into())
    }

    /// Reach into the underlying boxed error if you need to
    /// downcast, chain, etc.
    pub fn as_inner(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
        &*self.0
    }

    /// Unwrap to the boxed inner error.
    pub fn into_inner(self) -> Box<dyn std::error::Error + Send + Sync> {
        self.0
    }
}

impl std::fmt::Debug for HistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for HistoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

// Note: deliberately no `impl std::error::Error for HistoryError`.
// Adding it would conflict with the reflexive `impl<T> From<T> for T`
// once we have the blanket `From<E: Error> for HistoryError` below
// — `E = HistoryError` would satisfy both. Callers that need an
// `&dyn Error` reach for `as_inner()`; callers crossing into a
// boxed-error boundary use the `From<HistoryError> for Box<dyn Error
// + Send + Sync>` impl below or `into_inner()`.

impl<E> From<E> for HistoryError
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(err: E) -> Self {
        HistoryError(Box::new(err))
    }
}

/// Allow `?` to widen a `HistoryError` into a generic boxed error
/// at API boundaries that want `Box<dyn Error + Send + Sync>`.
impl From<HistoryError> for Box<dyn std::error::Error + Send + Sync> {
    fn from(err: HistoryError) -> Self {
        err.0
    }
}

/// A conversation history — the read/write surface an agent uses to
/// see what it has said, what it can do, and to record what just
/// happened.
///
/// Six methods, all stated in conversation terms:
///
/// - **Read** the model-visible messages ([`messages`](Self::messages)).
/// - **Read** the system prompt in effect ([`system_prompt`](Self::system_prompt)).
/// - **Read** the tool surface in effect ([`tools`](Self::tools)).
/// - **Read** the metadata produced by the most recent compaction
///   ([`previous_summary`](Self::previous_summary)).
/// - **Append** a batch of messages from one turn ([`append`](Self::append)).
/// - **Compact** an old prefix into a single summary message
///   ([`compact_prefix`](Self::compact_prefix)).
///
/// Notably absent: forking, merging, tip-by-hash lookup,
/// system-prompt / tools mutation. Those are graph-shaped operations
/// the agent runtime doesn't need. They live on the concrete backend
/// type — see [`Branch`](crate::Branch) for the git-flavored API.
/// `previous_summary` is read-only on the trait — the only writer is
/// `compact_prefix`.
///
/// # Concurrency
///
/// Implementations are [`Send`] + [`Sync`] so they can cross await
/// points and be shared via `Arc<dyn History>`. Each instance is
/// owned by at most one writer (typically one actor).
// async_trait is required for dyn-Trait usage (Arc<dyn History>);
// AFIT doesn't yet support `dyn Trait` natively. Don't "modernize" this.
#[async_trait]
pub trait History: Send + Sync {
    /// All messages the model will see, oldest first. Goes into the
    /// API request's `messages` field, chained with any per-turn
    /// pending messages by the runtime.
    async fn messages(&self) -> Result<Vec<Message>, HistoryError>;

    /// The system prompt content in effect at the current tip.
    /// Goes into the API request's `system` field. `None` if the
    /// branch has no system prompt set.
    async fn system_prompt(&self) -> Result<Option<Vec<Content>>, HistoryError>;

    /// The tool surface in effect at the current tip. Goes into the
    /// API request's `tools` field. Empty `Vec` if the branch has
    /// no tools set. By design, "unset" and "explicitly empty" tool
    /// sets are indistinguishable here — both yield an empty `Vec`,
    /// which is exactly what the API request needs.
    async fn tools(&self) -> Result<Vec<ToolDef>, HistoryError>;

    /// The summary text recorded by the most recent
    /// [`compact_prefix`](Self::compact_prefix), if any. Threaded
    /// into the next compaction's update template — never shown to
    /// the model directly in a regular turn.
    async fn previous_summary(&self) -> Result<Option<String>, HistoryError>;

    /// Append a batch of messages — typically one turn's worth of
    /// output: an assistant message and any tool results. The batch
    /// becomes a single commit on the branch.
    async fn append(&self, messages: Vec<Message>) -> Result<(), HistoryError>;

    /// Replace the first `end` messages with a single
    /// `summary_message`, and record `summary_text` as the new
    /// `previous_summary`. Both updates land in a single commit.
    /// This method is the sole writer of `previous_summary` — the
    /// trait exposes no separate setter.
    async fn compact_prefix(
        &self,
        end: usize,
        summary_message: Message,
        summary_text: String,
    ) -> Result<(), HistoryError>;
}
