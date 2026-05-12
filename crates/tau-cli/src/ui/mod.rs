//! TUI implementation for tau.
//!
//! `TuiFrontend` is the consumer-facing surface — it implements
//! [`crate::driver::Frontend`] over the existing widgets, renderer,
//! state, and event handlers. The rest of this module is internal
//! infrastructure shared by the frontend and tests.
//!
//! ## Screen layout
//!
//! The interface is divided into four horizontal strips, top to bottom:
//!
//! | Area | Height | Renderer | Contents |
//! |------|--------|----------|----------|
//! | **Header** | 1 | `render_header` | τ glyph (rainbow when processing, green when idle), cwd in `{ }` brackets, clock (MM/DD/YYYY HH:MM:SS AM) |
//! | **Conversation** | flex | `render_conversation` | Message thread — user (▶), assistant (◀), tools (⚙), agents (◇), system (●), steer (▷). Bottom border shows status (Ready/Thinking/Cancelling). |
//! | **Status line** | 1 | `render_status_line` | Model name, thinking level, token counts, cache stats, cost |
//! | **Input** | 3 | `InputBox` widget | Text entry with placeholder |
//!
//! The header style is inspired by the HP 48GX calculator status area.

mod agents;
mod constants;
mod events;
mod frontend;
mod input;
mod render;
mod state;
mod theme;
mod types;
mod widgets;

pub use frontend::TuiFrontend;
