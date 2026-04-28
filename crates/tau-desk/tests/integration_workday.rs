//! Cross-subsystem integration test: a small "workday" vignette that
//! exercises pieces in concert.
//!
//! 1. **Webhook → mechanical handler.** A `pr_merged` notice arrives
//!    via `ingest_signal`. A registered handler retires the PR card
//!    with `Provenance::Source`.
//! 2. **User → chat.** The user asks the chat agent to draft a
//!    thank-you comment. The chat agent (mocked transport) calls
//!    `desk_enqueue_draft` with `gh_post_pr_comment` as the target
//!    tool.
//! 3. **User approves draft.** `desk.approve_draft` looks up
//!    `gh_post_pr_comment` in the source registry, dispatches with
//!    the stored arguments through a synthetic `ExecutionContext`,
//!    and captures the outcome.
//! 4. **Final state assertions.** Source-tool was invoked with the
//!    right args; draft is `Approved` with `outcome.success = true`;
//!    associated draft card moved Drafts → Done; mechanical handler
//!    ran exactly once.
//!
//! Validates: mechanical handlers + chat-as-`tau-session` + draft
//! enqueue/approve + source-tool dispatch + provenance routing
//! (Agent for chat-driven mutations, Source for the webhook handler).

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tau_agent::test_utils::{MockTransport, make_test_config};
use tau_agent::tool::{BoxedTool, ExecutionContext, Tool, ToolResult};
use tau_agent::{ApprovalPolicy, DefaultApprovalPolicy};
use tau_desk::{
    CardBody, CardData, CardEvent, CardEventKind, CardPile, DeskAgent, DeskConfig, DeskStorage,
    DraftStatus, HandlerContext, MechanicalHandler, MemDeskStorage, Provenance, Source,
};
use tau_session::{FsStorage, SessionManager};

// ---------------------------------------------------------------------------
// Mock source: a single write tool we'll target via draft.
// ---------------------------------------------------------------------------

struct MockGhSource {
    post_calls: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl Source for MockGhSource {
    fn id(&self) -> &str {
        "gh"
    }
    fn tools(&self) -> Vec<BoxedTool> {
        vec![Arc::new(GhPostPrCommentTool {
            calls: self.post_calls.clone(),
        })]
    }
}

struct GhPostPrCommentTool {
    calls: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl Tool for GhPostPrCommentTool {
    fn name(&self) -> &str {
        "gh_post_pr_comment"
    }
    fn description(&self) -> &str {
        "Post a comment on a GitHub pull request."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pr":   { "type": "integer" },
                "body": { "type": "string" }
            },
            "required": ["pr", "body"]
        })
    }
    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        self.calls.lock().push(arguments);
        ToolResult::text("comment posted: gh.example/c/9001")
    }
}

// ---------------------------------------------------------------------------
// Mechanical handler: retires PR cards on `pr_merged` notice.
// ---------------------------------------------------------------------------

struct PrMergedHandler {
    fired: Arc<Mutex<u32>>,
}

#[async_trait]
impl MechanicalHandler for PrMergedHandler {
    fn id(&self) -> &str {
        "pr_merged"
    }

    fn handles(&self, notice: &tau_desk::ChangeNotice) -> bool {
        notice.source == "gh"
            && notice.context.get("event").and_then(|v| v.as_str()) == Some("pr_merged")
    }

    async fn apply(
        &self,
        notice: tau_desk::ChangeNotice,
        ctx: &HandlerContext,
    ) -> tau_desk::Result<()> {
        *self.fired.lock() += 1;

        let card_id = notice
            .context
            .get("card_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| tau_desk::Error::Other(anyhow::anyhow!("missing card_id")))?
            .to_string();

        ctx.mutate_card(
            &card_id,
            CardEventKind::Retired {
                reason: Some("PR merged".into()),
            },
            Some("PR merged".into()),
            |c| c.pile = CardPile::Done,
        )
        .await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn workday_vignette() {
    // ---- Mocks --------------------------------------------------------
    let post_calls = Arc::new(Mutex::new(Vec::new()));
    let handler_fired = Arc::new(Mutex::new(0u32));

    // Chat transcript: chat agent calls desk_enqueue_draft with a
    // gh_post_pr_comment payload, then a terminating text response.
    let transport = Arc::new(
        MockTransport::new()
            .with_tool_call_response(
                "desk_enqueue_draft",
                "chat-1",
                json!({
                    "tool_name": "gh_post_pr_comment",
                    "arguments": {
                        "pr":   4821,
                        "body": "Thanks for shipping this — clean SDK swap."
                    },
                    "rationale": "User asked for a thank-you comment.",
                    "summary":   "Thank-you on PR #4821"
                }),
            )
            .with_text_response("Drafted it for your review."),
    );

    // ---- Build desk ---------------------------------------------------
    let tmp = tempfile::tempdir().unwrap();
    let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
    let sessions = Arc::new(SessionManager::new(session_storage));
    let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultApprovalPolicy);
    let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

    let mut cfg = DeskConfig::new(
        transport,
        storage.clone(),
        sessions,
        approval,
        make_test_config(),
        tmp.path().to_path_buf(),
    );
    cfg.sources
        .register(Arc::new(MockGhSource {
            post_calls: post_calls.clone(),
        }))
        .unwrap();
    let desk = DeskAgent::new(cfg).await.unwrap();
    desk.register_handler(Arc::new(PrMergedHandler {
        fired: handler_fired.clone(),
    }));

    // ---- Pre-seed: a PR card sitting in NeedsYou ----------------------
    let now = chrono::Utc::now();
    let agent_prov = Provenance::Agent {
        agent_id: Some("morning_scan".into()),
    };
    let pr_card = CardData {
        id: "github:pr/4821".into(),
        pile: CardPile::NeedsYou,
        external_ref: Some("github:pr/4821".into()),
        body: CardBody::Pr {
            url: "https://github.com/x/y/pull/4821".into(),
            title: "Switch payments SDK".into(),
            repo: "x/y".into(),
            author: "priya.s".into(),
            ci: Some("passing".into()),
        },
        agent_take: None,
        attachments: vec![],
        metadata: json!({}),
        pinned: false,
        created_at: now,
        last_modified: now,
        last_modified_by: agent_prov.clone(),
        last_modified_reason: None,
        history: VecDeque::from([CardEvent {
            at: now,
            by: agent_prov,
            kind: CardEventKind::Created,
        }]),
    };
    storage.upsert_card(&pr_card).await.unwrap();

    // ---- Step 1: webhook arrives, handler retires the card ------------
    desk.ingest_signal(tau_desk::ChangeNotice {
        source: "gh".into(),
        summary: "PR #4821 merged".into(),
        context: json!({ "event": "pr_merged", "card_id": "github:pr/4821" }),
    })
    .await
    .unwrap();

    assert_eq!(*handler_fired.lock(), 1);
    let retired = storage
        .read_card(&"github:pr/4821".to_string())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retired.pile, CardPile::Done);
    assert_eq!(
        retired.last_modified_by,
        Provenance::Source {
            source_id: "gh".into()
        },
        "handler stamped Source provenance"
    );

    // ---- Step 2: user asks chat → chat enqueues a draft ---------------
    desk.ask("Please draft a thank-you comment on PR #4821.".into())
        .await
        .unwrap()
        .await
        .unwrap();

    let drafts = storage.list_drafts(Some(DraftStatus::Pending)).await.unwrap();
    assert_eq!(drafts.len(), 1, "chat should have queued one draft");
    let draft = &drafts[0];
    assert_eq!(draft.tool_name, "gh_post_pr_comment");
    assert_eq!(draft.source_id.as_deref(), Some("gh"));
    let draft_id = draft.id.clone();

    // The Drafts pile shows the corresponding card.
    let drafts_pile = storage
        .list_cards(tau_desk::CardFilter {
            pile: Some(CardPile::Drafts),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(drafts_pile.len(), 1);
    assert!(matches!(drafts_pile[0].body, CardBody::Draft { .. }));
    // Chat-driven mutation stamps `agent_id: "chat"`.
    assert_eq!(
        drafts_pile[0].last_modified_by,
        Provenance::Agent {
            agent_id: Some("chat".into())
        }
    );

    // No real source-tool dispatch yet.
    assert!(post_calls.lock().is_empty());

    // ---- Step 3: user approves the draft → tool dispatches ------------
    let outcome = desk.approve_draft(&draft_id).await.unwrap();
    assert!(outcome.success);
    assert!(outcome.summary.contains("comment posted"));

    // The fake source-tool received the agent's payload.
    let calls = post_calls.lock().clone();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["pr"], 4821);
    assert!(
        calls[0]["body"]
            .as_str()
            .unwrap()
            .contains("clean SDK swap")
    );

    // Draft row resolved.
    let stored_draft = storage.read_draft(&draft_id).await.unwrap().unwrap();
    assert_eq!(stored_draft.status, DraftStatus::Approved);
    assert!(stored_draft.outcome.unwrap().success);

    // Draft card moved Drafts → Done.
    let draft_card = storage
        .read_card(&format!("card-{draft_id}"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(draft_card.pile, CardPile::Done);

    // ---- Step 4: shutdown hibernates the chat session -----------------
    let chat_id = desk.chat_session_id().expect("chat session was created");
    desk.shutdown().await.unwrap();
    let info = desk
        .sessions()
        .list()
        .await
        .unwrap()
        .into_iter()
        .find(|s| s.id == chat_id)
        .unwrap();
    assert_eq!(info.status, tau_session::SessionStatus::Hibernated);
}
