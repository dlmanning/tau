use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use chrono::Utc;
use parking_lot::RwLock;
use tokio::sync::broadcast;

use crate::Result;
use crate::activity::{ActivityEntry, ActivityId};
use crate::brief::Brief;
use crate::card::{CardData, CardId, CardPile};
use crate::draft::{Draft, DraftId, DraftStatus};
use crate::error::Error;
use crate::tombstone::DismissalRecord;

/// Pluggable persistence for the desk. Everything inside `tau-desk`
/// (manager, draft queue, activity feed, scheduler state) talks through
/// this trait; the engine is implementation detail behind the boundary.
///
/// The two impls that ship with the crate:
/// - [`MemDeskStorage`] — in-memory, for tests and ephemeral hosts.
/// - `TursoDeskStorage` — production default (separate crate / future).
///
/// Hosts wanting alternative engines implement the trait.
#[async_trait]
pub trait DeskStorage: Send + Sync {
    // ----- Cards -----
    async fn upsert_card(&self, card: &CardData) -> Result<UpsertOutcome>;
    async fn read_card(&self, id: &CardId) -> Result<Option<CardData>>;
    async fn read_card_by_ref(&self, external_ref: &str) -> Result<Option<CardData>>;
    async fn list_cards(&self, filter: CardFilter) -> Result<Vec<CardData>>;
    async fn delete_card(&self, id: &CardId) -> Result<()>;

    // ----- Tombstones -----
    async fn add_tombstone(&self, ref_: &str, reason: Option<String>) -> Result<()>;
    async fn remove_tombstone(&self, ref_: &str) -> Result<bool>;
    async fn list_tombstones(&self) -> Result<Vec<DismissalRecord>>;
    async fn read_tombstone(&self, ref_: &str) -> Result<Option<DismissalRecord>>;

    // ----- Drafts -----
    async fn write_draft(&self, draft: &Draft) -> Result<()>;
    async fn read_draft(&self, id: &DraftId) -> Result<Option<Draft>>;
    async fn list_drafts(&self, status: Option<DraftStatus>) -> Result<Vec<Draft>>;

    // ----- Activity -----
    /// Appends `entry` and returns the assigned `seq` (storage owns
    /// monotonic seq).
    async fn append_activity(&self, entry: &ActivityEntry) -> Result<u64>;
    /// Most recent N entries, newest-first.
    async fn list_activity(&self, limit: usize) -> Result<Vec<ActivityEntry>>;
    /// Entries with seq strictly greater than `seq`, oldest-first.
    async fn activity_since(&self, seq: u64) -> Result<Vec<ActivityEntry>>;
    async fn read_activity(&self, id: &ActivityId) -> Result<Option<ActivityEntry>>;

    // ----- Brief (singleton) -----
    async fn read_brief(&self) -> Result<Option<Brief>>;
    async fn write_brief(&self, brief: &Brief) -> Result<()>;

    // ----- Suggestion mutes -----
    async fn add_mute(&self, seed_from: &str) -> Result<()>;
    async fn remove_mute(&self, seed_from: &str) -> Result<bool>;
    async fn list_mutes(&self) -> Result<Vec<String>>;

    // ----- Source state (cursors / last-synced timestamps) -----
    async fn read_source_state(&self, source_id: &str) -> Result<Option<serde_json::Value>>;
    async fn write_source_state(&self, source_id: &str, state: &serde_json::Value) -> Result<()>;

    // ----- Scheduler state -----
    async fn read_task_state(&self, task_id: &str) -> Result<Option<TaskRunState>>;
    async fn write_task_state(&self, task_id: &str, state: &TaskRunState) -> Result<()>;

    /// Optional native change subscription. Engines without native CDC
    /// return `None` and the manager emits events manually.
    fn subscribe(&self) -> Option<broadcast::Receiver<StorageChange>> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsertOutcome {
    Inserted,
    Updated,
    /// The upsert's `external_ref` matched an existing card with a
    /// different `id`. Storage kept the existing canonical `id` and
    /// updated the rest of the card's fields.
    Merged,
}

#[derive(Debug, Clone, Default)]
pub struct CardFilter {
    pub pile: Option<CardPile>,
    pub external_ref: Option<String>,
    pub pinned_only: bool,
    pub limit: Option<usize>,
}

/// Per-task scheduler state, persisted across restarts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskRunState {
    pub last_run_at: Option<chrono::DateTime<chrono::Utc>>,
    pub last_status: Option<String>,
}

/// CDC-style row change. Engines with native CDC translate their
/// internal stream into this; the manager translates these into
/// `DeskEvent`s.
#[derive(Debug, Clone)]
pub enum StorageChange {
    Card(CardId),
    Tombstone(String),
    Draft(DraftId),
    Activity(ActivityId),
    Brief,
    Mute(String),
}

// ============================================================================
// MemDeskStorage
// ============================================================================

/// In-memory storage, for tests and ephemeral hosts. `subscribe()`
/// returns `None`; the manager emits change events manually.
///
/// Concurrency: a single `RwLock` over the whole state. Reads can be
/// concurrent; writes are exclusive. Acceptable for testing scale; not
/// designed for production load.
pub struct MemDeskStorage {
    state: RwLock<MemState>,
}

struct MemState {
    cards: HashMap<CardId, CardData>,
    /// `external_ref` → canonical `CardId`.
    by_ref: HashMap<String, CardId>,
    tombstones: HashMap<String, DismissalRecord>,
    drafts: HashMap<DraftId, Draft>,
    /// Append-only, ordered by insertion (== seq order).
    activity: Vec<ActivityEntry>,
    activity_index: HashMap<ActivityId, usize>,
    activity_seq: u64,
    brief: Option<Brief>,
    mutes: HashSet<String>,
    source_state: HashMap<String, serde_json::Value>,
    task_state: HashMap<String, TaskRunState>,
}

impl MemDeskStorage {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(MemState {
                cards: HashMap::new(),
                by_ref: HashMap::new(),
                tombstones: HashMap::new(),
                drafts: HashMap::new(),
                activity: Vec::new(),
                activity_index: HashMap::new(),
                activity_seq: 0,
                brief: None,
                mutes: HashSet::new(),
                source_state: HashMap::new(),
                task_state: HashMap::new(),
            }),
        }
    }
}

impl Default for MemDeskStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DeskStorage for MemDeskStorage {
    // ----- Cards -----

    async fn upsert_card(&self, card: &CardData) -> Result<UpsertOutcome> {
        let mut state = self.state.write();

        // Tombstone check: external_ref dismissed → reject before mutating.
        if let Some(r) = &card.external_ref {
            if let Some(record) = state.tombstones.get(r) {
                return Err(Error::Tombstoned {
                    external_ref: record.external_ref.clone(),
                    dismissed_at: record.dismissed_at,
                    reason: record.reason.clone(),
                });
            }
        }

        // Resolve canonical id and outcome class.
        let (canonical_id, outcome) = match &card.external_ref {
            Some(r) => match state.by_ref.get(r) {
                Some(existing_id) if existing_id != &card.id => {
                    (existing_id.clone(), UpsertOutcome::Merged)
                }
                _ => {
                    if state.cards.contains_key(&card.id) {
                        (card.id.clone(), UpsertOutcome::Updated)
                    } else {
                        (card.id.clone(), UpsertOutcome::Inserted)
                    }
                }
            },
            None => {
                if state.cards.contains_key(&card.id) {
                    (card.id.clone(), UpsertOutcome::Updated)
                } else {
                    (card.id.clone(), UpsertOutcome::Inserted)
                }
            }
        };

        // Build the stored card. Use canonical id; preserve created_at
        // on Updated/Merged so the caller can't accidentally reset it.
        let mut stored = card.clone();
        stored.id = canonical_id.clone();
        if outcome != UpsertOutcome::Inserted {
            let (existing_created, existing_ref) = match state.cards.get(&canonical_id) {
                Some(e) => (Some(e.created_at), e.external_ref.clone()),
                None => (None, None),
            };
            if let Some(ca) = existing_created {
                stored.created_at = ca;
            }
            // If external_ref changed (or was removed), drop the stale
            // by_ref entry.
            if let Some(old_ref) = existing_ref {
                if Some(&old_ref) != stored.external_ref.as_ref() {
                    state.by_ref.remove(&old_ref);
                }
            }
        }

        if let Some(r) = &stored.external_ref {
            state.by_ref.insert(r.clone(), canonical_id.clone());
        }
        state.cards.insert(canonical_id, stored);

        Ok(outcome)
    }

    async fn read_card(&self, id: &CardId) -> Result<Option<CardData>> {
        Ok(self.state.read().cards.get(id).cloned())
    }

    async fn read_card_by_ref(&self, external_ref: &str) -> Result<Option<CardData>> {
        let state = self.state.read();
        Ok(state
            .by_ref
            .get(external_ref)
            .and_then(|id| state.cards.get(id))
            .cloned())
    }

    async fn list_cards(&self, filter: CardFilter) -> Result<Vec<CardData>> {
        let state = self.state.read();
        let mut results: Vec<CardData> = state
            .cards
            .values()
            .filter(|c| {
                filter.pile.is_none_or(|p| c.pile == p)
                    && filter
                        .external_ref
                        .as_ref()
                        .is_none_or(|r| c.external_ref.as_ref() == Some(r))
                    && (!filter.pinned_only || c.pinned)
            })
            .cloned()
            .collect();

        // Default sort: newest-first by last_modified.
        results.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn delete_card(&self, id: &CardId) -> Result<()> {
        let mut state = self.state.write();
        if let Some(card) = state.cards.remove(id) {
            if let Some(r) = &card.external_ref {
                state.by_ref.remove(r);
            }
        }
        Ok(())
    }

    // ----- Tombstones -----

    async fn add_tombstone(&self, ref_: &str, reason: Option<String>) -> Result<()> {
        let mut state = self.state.write();
        state.tombstones.insert(
            ref_.to_string(),
            DismissalRecord {
                external_ref: ref_.to_string(),
                dismissed_at: Utc::now(),
                reason,
            },
        );
        Ok(())
    }

    async fn remove_tombstone(&self, ref_: &str) -> Result<bool> {
        Ok(self.state.write().tombstones.remove(ref_).is_some())
    }

    async fn list_tombstones(&self) -> Result<Vec<DismissalRecord>> {
        let mut records: Vec<DismissalRecord> =
            self.state.read().tombstones.values().cloned().collect();
        records.sort_by(|a, b| b.dismissed_at.cmp(&a.dismissed_at));
        Ok(records)
    }

    async fn read_tombstone(&self, ref_: &str) -> Result<Option<DismissalRecord>> {
        Ok(self.state.read().tombstones.get(ref_).cloned())
    }

    // ----- Drafts -----

    async fn write_draft(&self, draft: &Draft) -> Result<()> {
        self.state
            .write()
            .drafts
            .insert(draft.id.clone(), draft.clone());
        Ok(())
    }

    async fn read_draft(&self, id: &DraftId) -> Result<Option<Draft>> {
        Ok(self.state.read().drafts.get(id).cloned())
    }

    async fn list_drafts(&self, status: Option<DraftStatus>) -> Result<Vec<Draft>> {
        let state = self.state.read();
        let mut drafts: Vec<Draft> = state
            .drafts
            .values()
            .filter(|d| status.is_none_or(|s| d.status == s))
            .cloned()
            .collect();
        drafts.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(drafts)
    }

    // ----- Activity -----

    async fn append_activity(&self, entry: &ActivityEntry) -> Result<u64> {
        let mut state = self.state.write();
        state.activity_seq += 1;
        let seq = state.activity_seq;
        let mut e = entry.clone();
        e.seq = seq;
        let idx = state.activity.len();
        state.activity_index.insert(e.id.clone(), idx);
        state.activity.push(e);
        Ok(seq)
    }

    async fn list_activity(&self, limit: usize) -> Result<Vec<ActivityEntry>> {
        let state = self.state.read();
        let len = state.activity.len();
        let start = len.saturating_sub(limit);
        Ok(state.activity[start..].iter().rev().cloned().collect())
    }

    async fn activity_since(&self, seq: u64) -> Result<Vec<ActivityEntry>> {
        let state = self.state.read();
        Ok(state
            .activity
            .iter()
            .filter(|e| e.seq > seq)
            .cloned()
            .collect())
    }

    async fn read_activity(&self, id: &ActivityId) -> Result<Option<ActivityEntry>> {
        let state = self.state.read();
        Ok(state
            .activity_index
            .get(id)
            .and_then(|&idx| state.activity.get(idx).cloned()))
    }

    // ----- Brief -----

    async fn read_brief(&self) -> Result<Option<Brief>> {
        Ok(self.state.read().brief.clone())
    }

    async fn write_brief(&self, brief: &Brief) -> Result<()> {
        self.state.write().brief = Some(brief.clone());
        Ok(())
    }

    // ----- Suggestion mutes -----

    async fn add_mute(&self, seed_from: &str) -> Result<()> {
        self.state.write().mutes.insert(seed_from.to_string());
        Ok(())
    }

    async fn remove_mute(&self, seed_from: &str) -> Result<bool> {
        Ok(self.state.write().mutes.remove(seed_from))
    }

    async fn list_mutes(&self) -> Result<Vec<String>> {
        let mut mutes: Vec<String> = self.state.read().mutes.iter().cloned().collect();
        mutes.sort();
        Ok(mutes)
    }

    // ----- Source state -----

    async fn read_source_state(&self, source_id: &str) -> Result<Option<serde_json::Value>> {
        Ok(self.state.read().source_state.get(source_id).cloned())
    }

    async fn write_source_state(
        &self,
        source_id: &str,
        state_value: &serde_json::Value,
    ) -> Result<()> {
        self.state
            .write()
            .source_state
            .insert(source_id.to_string(), state_value.clone());
        Ok(())
    }

    // ----- Scheduler state -----

    async fn read_task_state(&self, task_id: &str) -> Result<Option<TaskRunState>> {
        Ok(self.state.read().task_state.get(task_id).cloned())
    }

    async fn write_task_state(&self, task_id: &str, run_state: &TaskRunState) -> Result<()> {
        self.state
            .write()
            .task_state
            .insert(task_id.to_string(), run_state.clone());
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    use crate::activity::{ActivityEntry, ActivityKind};
    use crate::card::{CardBody, CardData, CardPile};
    use crate::draft::{Draft, DraftStatus};
    use crate::provenance::Provenance;

    fn pr_card(id: &str, ext: Option<&str>) -> CardData {
        let now = Utc::now();
        CardData {
            id: id.to_string(),
            pile: CardPile::NeedsYou,
            external_ref: ext.map(String::from),
            body: CardBody::Pr {
                url: "https://github.com/x/y/pull/1".into(),
                title: "Test PR".into(),
                repo: "x/y".into(),
                author: "alice".into(),
                ci: None,
            },
            agent_take: None,
            attachments: vec![],
            metadata: serde_json::json!({}),
            pinned: false,
            created_at: now,
            last_modified: now,
            last_modified_by: Provenance::Agent {
                agent_id: Some("morning_scan".into()),
            },
            last_modified_reason: None,
            history: VecDeque::new(),
        }
    }

    fn activity(id: &str, text: &str) -> ActivityEntry {
        ActivityEntry {
            id: id.to_string(),
            seq: 0,
            at: Utc::now(),
            text: text.into(),
            kind: Some(ActivityKind::AgentMessage),
            suggest_session: None,
        }
    }

    #[tokio::test]
    async fn upsert_inserts_then_updates() {
        let s = MemDeskStorage::new();
        let card = pr_card("a", None);

        assert_eq!(s.upsert_card(&card).await.unwrap(), UpsertOutcome::Inserted);
        assert_eq!(s.upsert_card(&card).await.unwrap(), UpsertOutcome::Updated);
        assert_eq!(s.list_cards(CardFilter::default()).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn merge_by_external_ref_keeps_canonical_id() {
        let s = MemDeskStorage::new();
        let original = pr_card("orig-id", Some("github:pr/1"));
        s.upsert_card(&original).await.unwrap();

        // Same external_ref, different agent-picked id.
        let dup = pr_card("dup-id", Some("github:pr/1"));
        let outcome = s.upsert_card(&dup).await.unwrap();
        assert_eq!(outcome, UpsertOutcome::Merged);

        // Stored under the original id.
        assert!(s.read_card(&"orig-id".into()).await.unwrap().is_some());
        assert!(s.read_card(&"dup-id".into()).await.unwrap().is_none());

        // Lookup by ref still resolves.
        let by_ref = s.read_card_by_ref("github:pr/1").await.unwrap();
        assert_eq!(by_ref.unwrap().id, "orig-id");

        // Only one card total.
        assert_eq!(s.list_cards(CardFilter::default()).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn upsert_blocked_by_tombstone() {
        let s = MemDeskStorage::new();
        s.add_tombstone("github:pr/1", Some("nope".into()))
            .await
            .unwrap();

        let card = pr_card("a", Some("github:pr/1"));
        match s.upsert_card(&card).await {
            Err(Error::Tombstoned { external_ref, .. }) => {
                assert_eq!(external_ref, "github:pr/1");
            }
            other => panic!("expected Tombstoned, got {:?}", other),
        }
        assert_eq!(s.list_cards(CardFilter::default()).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn upsert_preserves_created_at() {
        let s = MemDeskStorage::new();
        let mut card = pr_card("a", None);
        let original_created = card.created_at;
        s.upsert_card(&card).await.unwrap();

        // Caller submits with a different created_at; storage ignores it.
        card.created_at = original_created + chrono::Duration::hours(1);
        s.upsert_card(&card).await.unwrap();

        let stored = s.read_card(&"a".into()).await.unwrap().unwrap();
        assert_eq!(stored.created_at, original_created);
    }

    #[tokio::test]
    async fn changing_external_ref_cleans_old_index() {
        let s = MemDeskStorage::new();
        let mut card = pr_card("a", Some("ref-1"));
        s.upsert_card(&card).await.unwrap();

        card.external_ref = Some("ref-2".into());
        s.upsert_card(&card).await.unwrap();

        assert!(s.read_card_by_ref("ref-1").await.unwrap().is_none());
        assert!(s.read_card_by_ref("ref-2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn delete_card_drops_by_ref_index() {
        let s = MemDeskStorage::new();
        s.upsert_card(&pr_card("a", Some("ref-1"))).await.unwrap();
        s.delete_card(&"a".into()).await.unwrap();

        assert!(s.read_card(&"a".into()).await.unwrap().is_none());
        assert!(s.read_card_by_ref("ref-1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_cards_filters_and_sorts() {
        let s = MemDeskStorage::new();
        let mut a = pr_card("a", None);
        a.last_modified = Utc::now();
        a.pile = CardPile::NeedsYou;

        let mut b = pr_card("b", None);
        b.last_modified = a.last_modified - chrono::Duration::seconds(1);
        b.pile = CardPile::Watching;

        let mut c = pr_card("c", None);
        c.last_modified = a.last_modified + chrono::Duration::seconds(1);
        c.pile = CardPile::NeedsYou;

        s.upsert_card(&a).await.unwrap();
        s.upsert_card(&b).await.unwrap();
        s.upsert_card(&c).await.unwrap();

        // Filter by pile, sorted newest-first.
        let needs = s
            .list_cards(CardFilter {
                pile: Some(CardPile::NeedsYou),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(
            needs.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            vec!["c", "a"]
        );

        // Limit
        let limited = s
            .list_cards(CardFilter {
                limit: Some(2),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn tombstone_lifecycle() {
        let s = MemDeskStorage::new();
        s.add_tombstone("ref-1", Some("dismissed".into()))
            .await
            .unwrap();

        let record = s.read_tombstone("ref-1").await.unwrap().unwrap();
        assert_eq!(record.reason.as_deref(), Some("dismissed"));

        assert!(s.remove_tombstone("ref-1").await.unwrap());
        assert!(s.read_tombstone("ref-1").await.unwrap().is_none());

        // Removing again returns false.
        assert!(!s.remove_tombstone("ref-1").await.unwrap());
    }

    #[tokio::test]
    async fn activity_assigns_monotonic_seq() {
        let s = MemDeskStorage::new();
        let s1 = s.append_activity(&activity("a", "first")).await.unwrap();
        let s2 = s.append_activity(&activity("b", "second")).await.unwrap();
        let s3 = s.append_activity(&activity("c", "third")).await.unwrap();

        assert!(s1 < s2 && s2 < s3);

        // list_activity returns newest-first.
        let recent = s.list_activity(10).await.unwrap();
        assert_eq!(
            recent.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["c", "b", "a"]
        );

        // activity_since returns oldest-first, exclusive.
        let since = s.activity_since(s1).await.unwrap();
        assert_eq!(
            since.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
    }

    #[tokio::test]
    async fn activity_list_respects_limit() {
        let s = MemDeskStorage::new();
        for i in 0..5 {
            s.append_activity(&activity(&format!("a{i}"), "x"))
                .await
                .unwrap();
        }
        let recent = s.list_activity(3).await.unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].id, "a4");
    }

    #[tokio::test]
    async fn drafts_filter_by_status() {
        let s = MemDeskStorage::new();
        let d = |id: &str, status: DraftStatus| Draft {
            id: id.into(),
            source_id: None,
            tool_name: "noop".into(),
            arguments: serde_json::json!({}),
            rationale: None,
            status,
            created_at: Utc::now(),
            resolved_at: None,
            outcome: None,
        };
        s.write_draft(&d("p1", DraftStatus::Pending)).await.unwrap();
        s.write_draft(&d("p2", DraftStatus::Pending)).await.unwrap();
        s.write_draft(&d("a1", DraftStatus::Approved))
            .await
            .unwrap();

        let pending = s.list_drafts(Some(DraftStatus::Pending)).await.unwrap();
        assert_eq!(pending.len(), 2);

        let all = s.list_drafts(None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn mutes_round_trip() {
        let s = MemDeskStorage::new();
        s.add_mute("jira:PLT-312").await.unwrap();
        s.add_mute("jira:PLT-313").await.unwrap();
        s.add_mute("jira:PLT-312").await.unwrap(); // idempotent

        let mutes = s.list_mutes().await.unwrap();
        assert_eq!(mutes, vec!["jira:PLT-312", "jira:PLT-313"]);

        assert!(s.remove_mute("jira:PLT-312").await.unwrap());
        assert!(!s.remove_mute("jira:PLT-312").await.unwrap());
        assert_eq!(s.list_mutes().await.unwrap(), vec!["jira:PLT-313"]);
    }
}
