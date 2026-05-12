//! Plan-submission flow tests are *not* ported to v2.
//!
//! The v1 `plan.rs` exercises `tau_tools::SubmitPlanTool` and its
//! `Plan` / `PlanFile` / `PlanStep` data model. Those types are
//! host-side conventions, not v2 runtime concerns:
//!
//! - v2's `InteractionRequest` defines the `Typed { schema_id, payload }`
//!   shape and the runtime treats the payload as opaque JSON.
//! - The `plan.submit` schema, the `Plan` struct, and the
//!   `SubmitPlanTool` implementation all live in (or would live in)
//!   a v2-host crate analogous to v1's `tau-tools`.
//!
//! Since v2 ships no host tools, there is nothing in this crate to
//! test against here. The flow tested by v1's `plan.rs` is
//! "host tool emits a Typed interaction request → host UI replies →
//! tool returns" — that round-trip is exercised in `approval.rs`
//! against the runtime's own `tool.confirm` schema.

#[test]
fn placeholder() {
    // Empty test so this file compiles into a binary and shows up in
    // `cargo test` output as a 0-test crate.
}
