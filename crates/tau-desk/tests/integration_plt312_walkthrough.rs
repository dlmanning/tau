//! Canonical PLT-312 walkthrough from `plans/TAU_DESK.md` ("Walking
//! PLT-312 through the model"). Validates the full lifecycle:
//!
//! 1. **Birth** — agent's first scan upserts the Jira card.
//! 2. **Cross-source synthesis** — Slack thread attached to the same card.
//! 3. **User edits** — user moves it to `Watching`.
//! 4. **Agent respects edit** — next scan sees status unchanged, revises
//!    its take rather than re-piling.
//! 5. **Source-driven move** — Jira ticket transitions; agent moves it
//!    back to `NeedsYou`.
//! 6. **Retire** — ticket falls out of assigned set; agent retires it.
//! 7. **Re-emerge after retire** — ticket re-assigned three weeks later;
//!    upsert succeeds (retire is soft) with attachment history intact.
//! 8. **Re-emerge after dismiss** — same scenario but the user dismissed
//!    instead of letting the agent retire; upsert returns
//!    `Err(Tombstoned)`; agent falls back to `add_activity` with
//!    `ActivityKind::TombstoneHit`.
//!
//! Each milestone is asserted before the next scan runs, so a regression
//! pinpoints which leg of the lifecycle broke.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tau_agent::test_utils::{MockTransport, make_test_config};
use tau_agent::tool::{BoxedTool, ExecutionContext, Tool, ToolResult};
use tau_agent::{ApprovalPolicy, DefaultApprovalPolicy};
use tau_desk::{
    ActivityKind, CardFilter, CardPile, DeskAgent, DeskConfig, DeskStorage, MemDeskStorage,
    Provenance, Source,
};
use tau_session::{FsStorage, SessionManager};

const PLT_REF: &str = "https://jira.example/browse/PLT-312";
const PLT_ID: &str = "jira:PLT-312";

// ---------------------------------------------------------------------------
// Mutable mock sources — data evolves between scans.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct JiraData {
    issues: Arc<Mutex<Vec<Value>>>,
}

impl JiraData {
    fn new(initial: Vec<Value>) -> Self {
        Self {
            issues: Arc::new(Mutex::new(initial)),
        }
    }
    fn set(&self, issues: Vec<Value>) {
        *self.issues.lock().unwrap() = issues;
    }
}

struct MockJiraSource {
    data: JiraData,
}

#[async_trait]
impl Source for MockJiraSource {
    fn id(&self) -> &str {
        "jira"
    }
    fn tools(&self) -> Vec<BoxedTool> {
        vec![Arc::new(JiraListAssignedTool {
            data: self.data.clone(),
        })]
    }
}

struct JiraListAssignedTool {
    data: JiraData,
}

#[async_trait]
impl Tool for JiraListAssignedTool {
    fn name(&self) -> &str {
        "jira_list_assigned"
    }
    fn description(&self) -> &str {
        "List Jira issues currently assigned to you."
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _: Value, _: ExecutionContext) -> ToolResult {
        let issues = self.data.issues.lock().unwrap().clone();
        ToolResult::text(serde_json::to_string(&issues).unwrap())
    }
}

#[derive(Clone)]
struct SlackData {
    mentions: Arc<Mutex<Vec<Value>>>,
}

impl SlackData {
    fn new(initial: Vec<Value>) -> Self {
        Self {
            mentions: Arc::new(Mutex::new(initial)),
        }
    }
    fn set(&self, mentions: Vec<Value>) {
        *self.mentions.lock().unwrap() = mentions;
    }
}

struct MockSlackSource {
    data: SlackData,
}

#[async_trait]
impl Source for MockSlackSource {
    fn id(&self) -> &str {
        "slack"
    }
    fn tools(&self) -> Vec<BoxedTool> {
        vec![Arc::new(SlackListMentionsTool {
            data: self.data.clone(),
        })]
    }
}

struct SlackListMentionsTool {
    data: SlackData,
}

#[async_trait]
impl Tool for SlackListMentionsTool {
    fn name(&self) -> &str {
        "slack_list_mentions"
    }
    fn description(&self) -> &str {
        "List recent Slack mentions in your subscribed channels."
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _: Value, _: ExecutionContext) -> ToolResult {
        let mentions = self.data.mentions.lock().unwrap().clone();
        ToolResult::text(serde_json::to_string(&mentions).unwrap())
    }
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn plt_in_state(status: &str) -> Value {
    json!({
        "url":     PLT_REF,
        "key":     "PLT-312",
        "title":   "Decide rate limiter — token bucket vs leaky bucket",
        "project": "PLT",
        "status":  status,
    })
}

fn plt_card_body(status: &str) -> Value {
    json!({
        "kind":    "jira",
        "url":     PLT_REF,
        "title":   "Decide rate limiter — token bucket vs leaky bucket",
        "project": "PLT",
        "status":  status,
    })
}

fn upsert_args(pile: &str, status: &str) -> Value {
    json!({
        "id":           PLT_ID,
        "external_ref": PLT_REF,
        "pile":         pile,
        "body":         plt_card_body(status),
    })
}

// ---------------------------------------------------------------------------
// The walkthrough
// ---------------------------------------------------------------------------

#[tokio::test]
async fn plt312_full_lifecycle() {
    // ---- Mock platform data, mutated between scans -----------------------
    let jira = JiraData::new(vec![plt_in_state("Needs decision")]);
    let slack = SlackData::new(vec![json!({
        "thread_url": "https://slack.example/C123/p1700000000",
        "channel":    "platform-eng",
        "from":       "Maya",
        "excerpt":    "Re PLT-312: I lean token bucket; David's down to implement either.",
        "linked":     PLT_ID
    })]);

    // ---- Scripted agent transcript across 6 scans -------------------------
    //
    // The model isn't really reasoning here; we just dictate the actions
    // it would take given each scan's hydrated state. Mock data + state
    // assertions between scans give us pinpoint coverage.
    let transport = Arc::new(
        MockTransport::new()
            // ===== Scan 1: Birth + cross-source synthesis =====
            .with_tool_call_response("jira_list_assigned", "s1.t1", json!({}))
            .with_tool_call_response("slack_list_mentions", "s1.t2", json!({}))
            .with_tool_call_response("desk_upsert_card", "s1.t3", upsert_args("needs_you", "Needs decision"))
            .with_tool_call_response(
                "desk_attach_to_card",
                "s1.t4",
                json!({
                    "card_id": PLT_ID,
                    "kind":    "slack-thread",
                    "url":     "https://slack.example/C123/p1700000000",
                    "summary": "Maya leans token bucket; David ready to implement either."
                }),
            )
            .with_text_response("PLT-312 surfaced.")
            // ===== Scan 2: Agent respects user's recent move =====
            .with_tool_call_response("jira_list_assigned", "s2.t1", json!({}))
            .with_tool_call_response(
                "desk_update_take",
                "s2.t2",
                json!({
                    "card_id": PLT_ID,
                    "ask":     "still waiting on David — I'll watch this",
                    "note":    "ticket status unchanged since you flagged you're waiting on him."
                }),
            )
            .with_text_response("respected user's move; revised take only.")
            // ===== Scan 3: Source-driven move back to NeedsYou =====
            .with_tool_call_response("jira_list_assigned", "s3.t1", json!({}))
            .with_tool_call_response(
                "desk_move_card",
                "s3.t2",
                json!({
                    "card_id": PLT_ID,
                    "to":      "needs_you",
                    "reason":  "ticket transitioned to In Review; you're tagged decider"
                }),
            )
            .with_text_response("PLT-312 needs you again.")
            // ===== Scan 4: Retire (no longer assigned) =====
            .with_tool_call_response("jira_list_assigned", "s4.t1", json!({}))
            .with_tool_call_response(
                "desk_retire_card",
                "s4.t2",
                json!({
                    "card_id": PLT_ID,
                    "reason":  "no longer assigned to me"
                }),
            )
            .with_text_response("retired.")
            // ===== Scan 5: Re-emerge after retire (soft, restores) =====
            .with_tool_call_response("jira_list_assigned", "s5.t1", json!({}))
            .with_tool_call_response(
                "desk_upsert_card",
                "s5.t2",
                upsert_args("needs_you", "Needs decision"),
            )
            .with_text_response("restored.")
            // ===== Scan 6: Re-emerge after dismiss (blocked → fallback) =====
            .with_tool_call_response("jira_list_assigned", "s6.t1", json!({}))
            .with_tool_call_response(
                "desk_upsert_card",
                "s6.t2",
                upsert_args("needs_you", "Needs decision"),
            )
            .with_tool_call_response(
                "desk_add_activity",
                "s6.t3",
                json!({
                    "text": "PLT-312 has been reassigned to you, but you dismissed it earlier.",
                    "kind": {
                        "type": "tombstone_hit",
                        "external_ref": PLT_REF,
                        "original_summary": "Decide rate limiter — token vs leaky"
                    }
                }),
            )
            .with_text_response("noted in activity."),
    );

    // ---- Build the desk ----------------------------------------------------
    let tmp = tempfile::tempdir().unwrap();
    let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
    let sessions = Arc::new(SessionManager::new(session_storage));
    let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultApprovalPolicy);
    let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

    let mut cfg = DeskConfig::new(
        transport.clone(),
        storage.clone(),
        sessions,
        approval,
        make_test_config(),
        tmp.path().to_path_buf(),
    );
    cfg.sources
        .register(Arc::new(MockJiraSource { data: jira.clone() }))
        .unwrap();
    cfg.sources
        .register(Arc::new(MockSlackSource { data: slack.clone() }))
        .unwrap();

    let desk = DeskAgent::new(cfg).await.unwrap();

    // ============== Scan 1: Birth + cross-source synthesis ==============
    desk.run_task_once(&"morning_scan".into(), "Run".into())
        .await
        .unwrap();

    let card = storage
        .read_card(&PLT_ID.to_string())
        .await
        .unwrap()
        .expect("birth: card exists");
    assert_eq!(card.pile, CardPile::NeedsYou);
    assert_eq!(card.external_ref.as_deref(), Some(PLT_REF));
    assert_eq!(
        card.last_modified_by,
        Provenance::Agent {
            agent_id: Some("morning_scan".into())
        }
    );
    assert_eq!(card.attachments.len(), 1, "synthesis: slack attached");
    assert_eq!(card.attachments[0].kind, "slack-thread");

    // ============== User edits: move to Watching ==============
    slack.set(vec![]); // Slack mention already attached; subsequent scans see no new mention.
    desk.user_move_card(&PLT_ID.to_string(), CardPile::Watching)
        .await
        .unwrap();

    let card = storage.read_card(&PLT_ID.to_string()).await.unwrap().unwrap();
    assert_eq!(card.pile, CardPile::Watching);
    assert_eq!(card.last_modified_by, Provenance::User);

    // ============== Scan 2: Agent respects user's edit ==============
    desk.run_task_once(&"morning_scan".into(), "Run".into())
        .await
        .unwrap();

    let card = storage.read_card(&PLT_ID.to_string()).await.unwrap().unwrap();
    assert_eq!(card.pile, CardPile::Watching, "agent did not re-pile");
    let take = card.agent_take.as_ref().expect("scan 2: take present");
    assert!(take.note.as_deref().unwrap().contains("unchanged"));
    // Attachments preserved across the take update.
    assert_eq!(card.attachments.len(), 1);

    // ============== Scan 3: Source-driven move (status changed) ==============
    jira.set(vec![plt_in_state("In Review")]);
    desk.run_task_once(&"morning_scan".into(), "Run".into())
        .await
        .unwrap();

    let card = storage.read_card(&PLT_ID.to_string()).await.unwrap().unwrap();
    assert_eq!(card.pile, CardPile::NeedsYou, "agent moved it back");
    assert_eq!(
        card.last_modified_by,
        Provenance::Agent {
            agent_id: Some("morning_scan".into())
        },
        "agent move wins over stale user edit"
    );
    assert_eq!(card.attachments.len(), 1, "attachments still intact");

    // ============== Scan 4: Retire (no longer assigned) ==============
    jira.set(vec![]);
    desk.run_task_once(&"morning_scan".into(), "Run".into())
        .await
        .unwrap();

    let card = storage.read_card(&PLT_ID.to_string()).await.unwrap().unwrap();
    assert_eq!(card.pile, CardPile::Done);
    assert_eq!(
        card.last_modified_reason.as_deref(),
        Some("no longer assigned to me")
    );

    // ============== Scan 5: Re-emerge after retire (soft) ==============
    jira.set(vec![plt_in_state("Needs decision")]);
    desk.run_task_once(&"morning_scan".into(), "Run".into())
        .await
        .unwrap();

    let card = storage.read_card(&PLT_ID.to_string()).await.unwrap().unwrap();
    assert_eq!(card.pile, CardPile::NeedsYou, "retire is soft; restored");
    assert_eq!(
        card.attachments.len(),
        1,
        "attachment history preserved across retire-and-restore"
    );

    // ============== User dismisses ==============
    desk.user_dismiss_card(&PLT_ID.to_string(), Some("not now".into()))
        .await
        .unwrap();

    assert!(
        storage.read_card(&PLT_ID.to_string()).await.unwrap().is_none(),
        "dismissal removes card"
    );
    assert!(
        storage.read_tombstone(PLT_REF).await.unwrap().is_some(),
        "tombstone in place"
    );

    // ============== Scan 6: Re-emerge blocked → fallback to activity ==============
    desk.run_task_once(&"morning_scan".into(), "Run".into())
        .await
        .unwrap();

    // Card stays absent — tombstone blocked the upsert.
    assert!(storage.read_card(&PLT_ID.to_string()).await.unwrap().is_none());

    // The agent's fallback `add_activity` produced a TombstoneHit entry.
    let activity = storage.list_activity(20).await.unwrap();
    let hit = activity
        .iter()
        .find(|a| matches!(&a.kind, Some(ActivityKind::TombstoneHit { external_ref, .. }) if external_ref == PLT_REF))
        .expect("tombstone-hit activity entry");
    assert!(hit.text.contains("dismissed"));

    // Sanity: at no point did we leak a duplicate card under a different id.
    let all = storage.list_cards(CardFilter::default()).await.unwrap();
    let plt_cards: Vec<_> = all
        .iter()
        .filter(|c| c.external_ref.as_deref() == Some(PLT_REF))
        .collect();
    assert!(
        plt_cards.is_empty(),
        "no PLT-312 card should exist after dismissal+blocked re-emergence"
    );
}

// ---------------------------------------------------------------------------
// Smaller companion: draft rejection lifecycle
// ---------------------------------------------------------------------------

/// Agent enqueues a draft → user rejects with reason → status flips to
/// Rejected, `DeskEvent::DraftRejected` fires. Closes the loop on the
/// draft lifecycle that the morning-scan integration test only opens.
#[tokio::test]
async fn draft_rejection_flow() {
    let transport = Arc::new(
        MockTransport::new()
            .with_tool_call_response(
                "desk_enqueue_draft",
                "t1",
                json!({
                    "tool_name": "gh_post_pr_comment",
                    "arguments": { "pr": 1, "body": "lgtm" },
                    "rationale": "pattern matches your prior approvals",
                    "summary":   "Comment on PR #1"
                }),
            )
            .with_text_response("queued."),
    );

    let tmp = tempfile::tempdir().unwrap();
    let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
    let sessions = Arc::new(SessionManager::new(session_storage));
    let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultApprovalPolicy);
    let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

    let cfg = DeskConfig::new(
        transport,
        storage.clone(),
        sessions,
        approval,
        make_test_config(),
        tmp.path().to_path_buf(),
    );
    let desk = DeskAgent::new(cfg).await.unwrap();
    let mut events = desk.subscribe();

    desk.run_task_once(&"webhook:gh".into(), "Draft a comment".into())
        .await
        .unwrap();

    let drafts = storage
        .list_drafts(Some(tau_desk::DraftStatus::Pending))
        .await
        .unwrap();
    assert_eq!(drafts.len(), 1);
    let draft_id = drafts[0].id.clone();

    desk.reject_draft(&draft_id, Some("not yet".into()))
        .await
        .unwrap();

    let updated = storage.read_draft(&draft_id).await.unwrap().unwrap();
    assert_eq!(updated.status, tau_desk::DraftStatus::Rejected);
    assert!(updated.resolved_at.is_some());

    // The DraftRejected event fires through the desk event stream.
    let mut saw = false;
    for _ in 0..20 {
        match tokio::time::timeout(std::time::Duration::from_millis(50), events.recv()).await {
            Ok(Ok(tau_desk::DeskEvent::DraftRejected { draft_id: id, reason })) => {
                assert_eq!(id, draft_id);
                assert_eq!(reason.as_deref(), Some("not yet"));
                saw = true;
                break;
            }
            Ok(Ok(_)) => continue,
            Ok(Err(_)) | Err(_) => break,
        }
    }
    assert!(saw, "DraftRejected event should fire");
}
