//! Drain sub-machine: checking the steering / follow-up queues and
//! blocking on background follow-ups.

use std::sync::atomic::Ordering;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::core::command::Command;
use crate::core::state::State;
use crate::core::transitions as t;

use super::{DrainPhase, Phase, Turn, TurnSub, handle_busy_command};

pub(super) async fn step_drain(
    dp: DrainPhase,
    state: &mut State,
    urgent_rx: &mut mpsc::Receiver<Command>,
    normal_rx: &mut mpsc::Receiver<Command>,
    prompt_cancel: &CancellationToken,
) -> Phase {
    match dp {
        DrainPhase::CheckQueues => {
            let drained = t::apply_drain_queues(&state.frame, &mut state.conv);
            match drained.source {
                t::DrainedFrom::Steering => Phase::Turn(Turn {
                    first_user_message: None,
                    sub: TurnSub::Prepare {
                        pending: drained.messages,
                    },
                }),
                t::DrainedFrom::FollowUps => {
                    // Decrement the bg-pending counter once per *background
                    // subagent completion* drained — not per message. Host
                    // `follow_up()` messages share this queue but are not
                    // paired with an `expect_follow_up()` increment, so
                    // counting them would zero the counter early and end the
                    // prompt while bg agents are still running.
                    let completed = drained
                        .messages
                        .iter()
                        .filter(|m| t::is_subagent_completion(m))
                        .count() as u32;
                    if completed > 0 {
                        let _ = state.shared.pending_follow_ups.fetch_update(
                            Ordering::Release,
                            Ordering::Acquire,
                            |n| Some(n.saturating_sub(completed)),
                        );
                    }
                    Phase::Turn(Turn {
                        first_user_message: None,
                        sub: TurnSub::Prepare {
                            pending: drained.messages,
                        },
                    })
                }
                t::DrainedFrom::Nothing => {
                    if state.shared.pending_follow_ups.load(Ordering::Acquire) > 0 {
                        Phase::Turn(Turn {
                            first_user_message: None,
                            sub: TurnSub::Drain(DrainPhase::WaitingForBackground),
                        })
                    } else {
                        Phase::Done(Ok(()))
                    }
                }
            }
        }
        DrainPhase::WaitingForBackground => loop {
            tokio::select! {
                biased;
                _ = prompt_cancel.cancelled() => break Phase::Done(Ok(())),
                Some(cmd) = urgent_rx.recv() => match cmd {
                    Command::FollowUp(msg) => {
                        t::apply_enqueue_follow_up(&mut state.conv, msg);
                        break Phase::Turn(Turn {
                            first_user_message: None,
                            sub: TurnSub::Drain(DrainPhase::CheckQueues),
                        });
                    }
                    other => handle_busy_command(state, other),
                },
                Some(cmd) = normal_rx.recv() => handle_busy_command(state, cmd),
            }
        },
    }
}
