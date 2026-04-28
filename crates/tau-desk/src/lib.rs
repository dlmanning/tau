//! Ambient agent layer for `tau-agent`.
//!
//! A "desk" is a long-lived process that surfaces work items ("cards") to
//! an engineer, queues actions ("drafts") for them to approve, and hands
//! off to coding sessions when needed. The user mutates state alongside
//! the agent; provenance is recorded on every change.
//!
//! See `plans/TAU_DESK.md` for the design rationale. Skeleton-only at
//! present: type and trait surface, with implementation bodies stubbed.

pub mod activity;
pub mod brief;
pub mod card;
pub mod desk;
pub mod draft;
pub mod error;
pub mod event;
pub mod handler;
pub mod mute;
pub mod now;
pub(crate) mod ops;
pub mod provenance;
pub mod scheduler;
pub mod source;
pub mod storage;
pub mod token;
pub mod tombstone;
pub mod tools;

pub use activity::{ActivityEntry, ActivityFeed, ActivityId, ActivityKind, SessionSeed};
pub use brief::{Brief, BriefStat};
pub use card::{AgentTake, Attachment, CardBody, CardData, CardId, CardPile};
pub use desk::{DeskAgent, DeskConfig};
pub use draft::{ActionOutcome, Draft, DraftId, DraftStatus};
pub use error::{Error, Result};
pub use event::DeskEvent;
pub use handler::{HandlerContext, MechanicalHandler};
pub use mute::SuggestionMutes;
pub use now::{NowZone, PickUpView, SuggestionView};
pub use provenance::{CardEvent, CardEventKind, Provenance};
pub use scheduler::{HydrationSpec, PromptSpec, ScheduledTask, TaskId, TaskName, Trigger};
pub use source::{ChangeNotice, Source, SourceId, SourceRegistry};
pub use storage::{
    CardFilter, DeskStorage, MemDeskStorage, StorageChange, TaskRunState, UpsertOutcome,
};
pub use token::{KeychainTokenStore, MemTokenStore, SecretString, TokenStore};
pub use tombstone::DismissalRecord;
