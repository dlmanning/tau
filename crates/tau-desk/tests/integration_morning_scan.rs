//! End-to-end integration test: simulates a full morning scan against a
//! mock GitHub source and a mock Slack source.
//!
//! The scan agent is the real `tau-agent` actor wired with the real
//! desk-state tools and the real `MemDeskStorage`. Only the LLM response
//! stream and the source platforms themselves are mocked — every other
//! component is the production code path.
//!
//! Validates:
//! - Source registration + tool contribution into the agent's tool set.
//! - Multi-turn agent loop: source-read → desk-state mutations →
//!   terminating text response.
//! - Card upsert with body + take, cross-source attachment, draft
//!   enqueue, activity append (with session seed), brief update.
//! - Now-zone projection picks up the suggestion produced by the scan.
//! - Provenance stamping on all agent-driven mutations.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tau_agent::test_utils::{MockTransport, make_test_config};
use tau_agent::tool::{BoxedTool, ExecutionContext, Tool, ToolResult};
use tau_agent::{ApprovalPolicy, DefaultApprovalPolicy};
use tau_desk::{
    CardBody, CardFilter, CardPile, DeskAgent, DeskConfig, DeskStorage, DraftStatus,
    MemDeskStorage, Provenance, Source,
};
use tau_session::{FsStorage, SessionManager};

// ---------------------------------------------------------------------------
// Mock sources
// ---------------------------------------------------------------------------

struct MockGithubSource {
    prs: Vec<Value>,
}

#[async_trait]
impl Source for MockGithubSource {
    fn id(&self) -> &str {
        "gh"
    }

    fn tools(&self) -> Vec<BoxedTool> {
        vec![Arc::new(GhListReviewRequestsTool {
            prs: self.prs.clone(),
        })]
    }
}

struct GhListReviewRequestsTool {
    prs: Vec<Value>,
}

#[async_trait]
impl Tool for GhListReviewRequestsTool {
    fn name(&self) -> &str {
        "gh_list_review_requests"
    }

    fn description(&self) -> &str {
        "List PRs awaiting your review."
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, _arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        ToolResult::text(serde_json::to_string(&self.prs).unwrap())
    }
}

struct MockSlackSource {
    mentions: Vec<Value>,
}

#[async_trait]
impl Source for MockSlackSource {
    fn id(&self) -> &str {
        "slack"
    }

    fn tools(&self) -> Vec<BoxedTool> {
        vec![Arc::new(SlackListMentionsTool {
            mentions: self.mentions.clone(),
        })]
    }
}

struct SlackListMentionsTool {
    mentions: Vec<Value>,
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

    async fn execute(&self, _arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        ToolResult::text(serde_json::to_string(&self.mentions).unwrap())
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn morning_scan_end_to_end() {
    // ---- Source fixtures ---------------------------------------------------
    let prs = vec![json!({
        "number": 4821,
        "url":    "https://github.com/restaurants-api/restaurants-api/pull/4821",
        "title":  "Switch order service to the new payments SDK",
        "repo":   "restaurants-api",
        "author": "priya.s",
        "ci":     "passing",
        "age":    "opened 2d"
    })];
    let mentions = vec![json!({
        "thread_url": "https://slack.example/archives/C123/p1700000000",
        "channel":    "platform-eng",
        "from":       "Maya",
        "excerpt":    "Re PLT-312: I lean token bucket; David's down to implement either.",
        "linked":     "jira:PLT-312"
    })];

    // ---- Scripted agent transcript ----------------------------------------
    //
    // 9 tool calls + a terminating text response. Each turn assumes the
    // model has made the appropriate inference from the previous tool
    // result; we don't simulate the model's reasoning, just its actions.
    let transport = MockTransport::new()
        // 1. Read PRs from GitHub.
        .with_tool_call_response("gh_list_review_requests", "t1", json!({}))
        // 2. Read Slack mentions.
        .with_tool_call_response("slack_list_mentions", "t2", json!({}))
        // 3. Upsert PR card.
        .with_tool_call_response(
            "desk_upsert_card",
            "t3",
            json!({
                "id":           "github:pr/4821",
                "external_ref": "github:pr/4821",
                "pile":         "needs_you",
                "body": {
                    "kind":   "pr",
                    "url":    "https://github.com/restaurants-api/restaurants-api/pull/4821",
                    "title":  "Switch order service to the new payments SDK",
                    "repo":   "restaurants-api",
                    "author": "priya.s",
                    "ci":     "passing"
                }
            }),
        )
        // 4. Add the agent's take on the PR.
        .with_tool_call_response(
            "desk_update_take",
            "t4",
            json!({
                "card_id": "github:pr/4821",
                "ask":     "review — Priya is blocked on a deploy window",
                "note":    "I skimmed it; one risky change in refund_test.rs worth flagging."
            }),
        )
        // 5. Upsert the Jira card.
        .with_tool_call_response(
            "desk_upsert_card",
            "t5",
            json!({
                "id":           "jira:PLT-312",
                "external_ref": "https://jira.example/browse/PLT-312",
                "pile":         "needs_you",
                "body": {
                    "kind":    "jira",
                    "url":     "https://jira.example/browse/PLT-312",
                    "title":   "Decide rate limiter — token bucket vs leaky bucket",
                    "project": "PLT",
                    "status":  "Needs decision"
                }
            }),
        )
        // 6. Cross-source synthesis: attach the Slack thread to the Jira card.
        .with_tool_call_response(
            "desk_attach_to_card",
            "t6",
            json!({
                "card_id": "jira:PLT-312",
                "kind":    "slack-thread",
                "url":     "https://slack.example/archives/C123/p1700000000",
                "summary": "Maya leans token bucket; David ready to implement either."
            }),
        )
        // 7. Enqueue a draft PR comment.
        .with_tool_call_response(
            "desk_enqueue_draft",
            "t7",
            json!({
                "tool_name": "gh_post_pr_comment",
                "arguments": {
                    "pr": 4821,
                    "body": "The retry loop in process_refund at line 84 doesn't bound attempts — worth capping."
                },
                "rationale": "You asked me to flag retry loops without bounds back in August.",
                "summary":   "Comment on PR #4821"
            }),
        )
        // 8. Activity entry with a session-handoff suggestion.
        .with_tool_call_response(
            "desk_add_activity",
            "t8",
            json!({
                "text": "I can start drafting the rate-limiter spike for PLT-312 — you'd just review.",
                "suggest_session": {
                    "title":     "Spike: rate limiter — token vs leaky bucket",
                    "project":   "restaurants-api",
                    "branch":    "spike/rate-limiter-plt-312",
                    "kickoff":   "Read PLT-312, the #platform-eng thread, and existing RateLimiter wrapper. Sketch a 30-line comparison + a recommendation. No code changes yet.",
                    "seed_from": "jira:PLT-312"
                }
            }),
        )
        // 9. Wrap up with the brief.
        .with_tool_call_response(
            "desk_update_brief",
            "t9",
            json!({
                "greeting": "Good morning, Alex.",
                "summary":  "Quiet overnight. One PR needs your eyes; PLT-312 is ready for scoping.",
                "stats": [
                    { "label": "Review queue",    "value": "1", "delta": "+1 overnight" },
                    { "label": "Assigned tickets", "value": "1", "delta": null }
                ]
            }),
        )
        // 10. Terminating text — agent loop exits.
        .with_text_response("Morning scan complete.");

    // ---- Build the desk ----------------------------------------------------
    let tmp = tempfile::tempdir().unwrap();
    let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
    let sessions = Arc::new(SessionManager::new(session_storage));
    let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultApprovalPolicy);
    let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

    let mut cfg = DeskConfig::new(
        Arc::new(transport),
        storage.clone(),
        sessions,
        approval,
        make_test_config(),
        tmp.path().to_path_buf(),
    );

    cfg.sources
        .register(Arc::new(MockGithubSource { prs }))
        .expect("register gh");
    cfg.sources
        .register(Arc::new(MockSlackSource { mentions }))
        .expect("register slack");

    let desk = DeskAgent::new(cfg).await.expect("desk");

    // ---- Run the scan ------------------------------------------------------
    desk.run_task_once(&"morning_scan".to_string(), "Run the morning scan.".into())
        .await
        .expect("scan should complete");

    // ---- Verify final state -----------------------------------------------

    // Two cards landed in NeedsYou.
    let needs = storage
        .list_cards(CardFilter {
            pile: Some(CardPile::NeedsYou),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(needs.len(), 2, "expected 2 NeedsYou cards");

    let pr = storage
        .read_card(&"github:pr/4821".to_string())
        .await
        .unwrap()
        .expect("pr card");
    assert!(matches!(pr.body, CardBody::Pr { .. }));
    assert_eq!(
        pr.last_modified_by,
        Provenance::Agent {
            agent_id: Some("morning_scan".into())
        }
    );
    let take = pr.agent_take.expect("pr should have a take");
    assert!(take.ask.unwrap().contains("Priya"));
    assert!(take.note.unwrap().contains("refund_test"));

    // Jira card has the Slack thread attachment.
    let jira = storage
        .read_card(&"jira:PLT-312".to_string())
        .await
        .unwrap()
        .expect("jira card");
    assert!(matches!(jira.body, CardBody::Jira { .. }));
    assert_eq!(jira.attachments.len(), 1, "jira should have 1 attachment");
    let att = &jira.attachments[0];
    assert_eq!(att.kind, "slack-thread");
    assert!(att.summary.contains("token bucket"));

    // One draft, both as a draft row and as a card in the Drafts pile.
    let drafts = storage.list_drafts(Some(DraftStatus::Pending)).await.unwrap();
    assert_eq!(drafts.len(), 1);
    let d = &drafts[0];
    assert_eq!(d.tool_name, "gh_post_pr_comment");
    assert_eq!(d.source_id.as_deref(), Some("gh"));

    let drafts_pile = storage
        .list_cards(CardFilter {
            pile: Some(CardPile::Drafts),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(drafts_pile.len(), 1);
    assert!(matches!(drafts_pile[0].body, CardBody::Draft { .. }));

    // Activity feed has at least the suggest-session entry.
    let activity = storage.list_activity(20).await.unwrap();
    let with_seed: Vec<_> = activity
        .iter()
        .filter(|a| a.suggest_session.is_some())
        .collect();
    assert_eq!(with_seed.len(), 1);
    let seed = with_seed[0].suggest_session.as_ref().unwrap();
    assert_eq!(seed.title, "Spike: rate limiter — token vs leaky bucket");
    assert_eq!(seed.seed_from.as_deref(), Some("jira:PLT-312"));

    // Brief was written.
    let brief = storage.read_brief().await.unwrap().expect("brief");
    assert_eq!(brief.greeting, "Good morning, Alex.");
    assert_eq!(brief.stats.len(), 2);

    // Now-zone projection picks up the suggestion.
    let zone = desk.now_zone().await.unwrap();
    assert!(zone.pickup.is_none(), "no hibernated session yet");
    assert_eq!(zone.suggestions.len(), 1);
    assert_eq!(
        zone.suggestions[0].seed.seed_from.as_deref(),
        Some("jira:PLT-312")
    );

    // After the user mutes the suggestion, it disappears from Now zone.
    desk.user_mute_suggestion("jira:PLT-312").await.unwrap();
    let zone = desk.now_zone().await.unwrap();
    assert_eq!(zone.suggestions.len(), 0);
}
