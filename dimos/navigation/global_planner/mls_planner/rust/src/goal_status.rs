// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Goal ids, blocked goals, and the nav_status reports for each goal.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::voxel::Xyz;
use dimos_module::{error_throttled, Output};
use lcm_msgs::actionlib_msgs::{GoalID, GoalStatus};
use lcm_msgs::geometry_msgs::Point;
use lcm_msgs::std_msgs::Time;

/// The last status reported. Held across each publish, so a heartbeat can
/// never send an older goal's status after a newer one.
pub(crate) type LatestStatus = Arc<tokio::sync::Mutex<Option<Report>>>;

/// The text of a PREEMPTED report, which says why the goal was preempted.
/// Matches dimos/navigation/spec.py.
pub(crate) const PREEMPT_REPLACED: &str = "replaced";
pub(crate) const PREEMPT_CANCELED: &str = "canceled";

/// Why a goal that ran out its blocked timeout was aborted.
pub(crate) const NO_SAFE_PATH: &str = "no safe path";
pub(crate) const NO_POSE: &str = "no robot pose";

/// How often the latest goal status repeats between transitions.
const STATUS_HEARTBEAT: Duration = Duration::from_secs(1);

/// A goal that cannot be planned, when that began, and why.
#[derive(Clone, Copy)]
pub(crate) struct Blocked {
    pub(crate) goal: Goal,
    since: Instant,
    pub(crate) reason: &'static str,
}

impl Blocked {
    /// The block this goal is in, carried over when it was blocked already.
    pub(crate) fn of(
        goal: Goal,
        reason: &'static str,
        earlier: Option<Blocked>,
        now: Instant,
    ) -> Self {
        let since = earlier
            .filter(|earlier| earlier.goal == goal)
            .map_or(now, |earlier| earlier.since);
        Self {
            goal,
            since,
            reason,
        }
    }

    /// Time left before the goal is aborted. Zero once it has run out.
    pub(crate) fn remaining(&self, timeout: Duration, now: Instant) -> Duration {
        timeout.saturating_sub(now.duration_since(self.since))
    }
}

/// Names a goal in its status reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GoalId {
    /// Stamp of the message that set the goal. Senders match reports on it.
    pub(crate) sec: i32,
    pub(crate) nsec: i32,
    /// Arrival order, which tells a displaced goal from the one after it.
    pub(crate) arrival: u64,
}

impl GoalId {
    pub(crate) fn same_stamp(&self, other: &GoalId) -> bool {
        (self.sec, self.nsec) == (other.sec, other.nsec)
    }
}

/// The active goal and the id its status reports carry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Goal {
    pub(crate) id: GoalId,
    pub(crate) position: Xyz,
}

/// One goal's state as nav_status carries it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Report {
    id: GoalId,
    status: i8,
    reason: &'static str,
    remaining_m: Option<f32>,
}

impl Report {
    pub(crate) fn new(id: GoalId, status: i8, reason: &'static str) -> Self {
        Self {
            id,
            status,
            reason,
            remaining_m: None,
        }
    }

    pub(crate) fn with_remaining(mut self, remaining_m: f32) -> Self {
        self.remaining_m = Some(remaining_m);
        self
    }

    /// Whether this differs from the last report by more than the distance.
    fn is_transition_from(&self, last: Option<&Report>) -> bool {
        last.is_none_or(|last| {
            (last.id, last.status, last.reason) != (self.id, self.status, self.reason)
        })
    }

    fn is_terminal(&self) -> bool {
        ![GoalStatus::PENDING, GoalStatus::ACTIVE].contains(&self.status)
    }

    fn message(&self) -> GoalStatus {
        let text = match self.remaining_m {
            Some(remaining_m) => format!("{}, {remaining_m:.2} m from the goal", self.reason),
            None => self.reason.to_string(),
        };
        GoalStatus {
            goal_id: GoalID {
                stamp: Time {
                    sec: self.id.sec,
                    nsec: self.id.nsec,
                },
                id: format!("{}.{:09}", self.id.sec, self.id.nsec),
            },
            status: self.status as u8,
            text,
        }
    }
}

/// Record a report as the latest. True when it is a transition to publish
/// now. A report for a goal older than the latest is dropped, and so is one
/// for a goal that is already over.
fn record(latest: &mut Option<Report>, report: &Report) -> bool {
    let stale = latest.as_ref().is_some_and(|last| {
        last.id.arrival > report.id.arrival
            || (last.id.arrival == report.id.arrival && last.is_terminal())
    });
    if stale {
        return false;
    }
    let transition = report.is_transition_from(latest.as_ref());
    *latest = Some(report.clone());
    transition
}

/// Publishes goal status on every transition and on the heartbeat.
#[derive(Clone)]
pub(crate) struct StatusReporter {
    pub(crate) out: Output<GoalStatus>,
    pub(crate) latest: LatestStatus,
}

impl StatusReporter {
    pub(crate) async fn report(&self, report: Report) {
        let mut latest = self.latest.lock().await;
        if record(&mut latest, &report) {
            self.publish(&report).await;
        }
    }

    pub(crate) async fn heartbeat(self) {
        let mut tick = tokio::time::interval(STATUS_HEARTBEAT);
        loop {
            tick.tick().await;
            let latest = self.latest.lock().await;
            if let Some(report) = latest.as_ref() {
                self.publish(report).await;
            }
        }
    }

    async fn publish(&self, report: &Report) {
        if let Err(e) = self.out.publish(&report.message()).await {
            error_throttled!(
                Duration::from_secs(1),
                error = %e,
                topic = %self.out.topic,
                "Goal status failed to publish",
            );
        }
    }
}

/// What a goal message asks for.
#[derive(Debug, PartialEq)]
pub(crate) enum GoalRequest {
    Set(Xyz),
    Cancel,
    Invalid,
}

/// A finite point sets a goal and an all-NaN point cancels. Any other
/// non-finite point is invalid.
pub(crate) fn goal_request(p: &Point) -> GoalRequest {
    let goal = (p.x as f32, p.y as f32, p.z as f32);
    if goal.0.is_finite() && goal.1.is_finite() && goal.2.is_finite() {
        GoalRequest::Set(goal)
    } else if p.x.is_nan() && p.y.is_nan() && p.z.is_nan() {
        GoalRequest::Cancel
    } else {
        GoalRequest::Invalid
    }
}

/// Whether a worker pass stops the follower. A goal message that leaves no
/// goal is a cancel, and the follower is still on the last path.
pub(crate) fn stop_due(goal_changed: bool, goal: Option<Goal>) -> bool {
    goal_changed && goal.is_none()
}

/// Whether an arrival stops the follower. A goal reached before it got a path
/// leaves the follower on the path of the goal before it.
pub(crate) fn stop_on_arrival(followed: Option<GoalId>, goal: GoalId) -> bool {
    !followed.is_some_and(|followed| followed.same_stamp(&goal))
}

pub(crate) fn clear_if_held(slot: &Mutex<Option<Goal>>, goal: Goal) -> bool {
    let mut guard = slot.lock().expect("goal mutex");
    let held = *guard == Some(goal);
    if held {
        *guard = None;
    }
    held
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_stops_the_follower_only_when_a_goal_message_leaves_no_goal() {
        let goal = Goal {
            id: goal_id(1),
            position: (1.0, 0.0, 0.0),
        };
        assert!(stop_due(true, None));
        assert!(!stop_due(true, Some(goal)));
        // The worker clearing a reached goal is not a goal message.
        assert!(!stop_due(false, None));
    }

    #[test]
    fn an_arrival_stops_the_follower_unless_it_is_on_that_goals_path() {
        assert!(!stop_on_arrival(Some(goal_id(2)), goal_id(2)));
        // Reached before it got a path, so the follower is still on the last goal's.
        assert!(stop_on_arrival(Some(goal_id(1)), goal_id(2)));
        assert!(stop_on_arrival(None, goal_id(2)));
    }

    /// The id of the goal that arrived in this order, stamped at that second.
    fn goal_id(arrival: u64) -> GoalId {
        GoalId {
            sec: arrival as i32,
            nsec: 5,
            arrival,
        }
    }

    fn point(x: f64, y: f64, z: f64) -> Point {
        Point { x, y, z }
    }

    #[test]
    fn goal_request_sets_on_finite_cancels_on_all_nan_and_rejects_the_rest() {
        assert_eq!(
            goal_request(&point(1.0, 2.0, 3.0)),
            GoalRequest::Set((1.0, 2.0, 3.0))
        );
        assert_eq!(
            goal_request(&point(f64::NAN, f64::NAN, f64::NAN)),
            GoalRequest::Cancel
        );
        assert_eq!(
            goal_request(&point(f64::NAN, 0.0, 0.0)),
            GoalRequest::Invalid
        );
        assert_eq!(
            goal_request(&point(0.0, f64::INFINITY, 0.0)),
            GoalRequest::Invalid
        );
    }

    #[test]
    fn a_goal_is_cleared_only_while_it_is_still_the_one_held() {
        let goal = |arrival| Goal {
            id: goal_id(arrival),
            position: (1.0, 0.0, 0.0),
        };
        // The second goal replaced the first while the first was blocked.
        let slot = Mutex::new(Some(goal(2)));
        assert!(!clear_if_held(&slot, goal(1)));
        assert_eq!(*slot.lock().unwrap(), Some(goal(2)));
        assert!(clear_if_held(&slot, goal(2)));
        assert_eq!(*slot.lock().unwrap(), None);
    }

    #[test]
    fn a_block_keeps_its_start_for_the_same_goal_only() {
        let goal = |arrival| Goal {
            id: goal_id(arrival),
            position: (1.0, 0.0, 0.0),
        };
        let start = Instant::now();
        let later = start + Duration::from_secs(1);
        let first = Blocked::of(goal(1), NO_POSE, None, start);
        let still = Blocked::of(goal(1), NO_SAFE_PATH, Some(first), later);
        assert_eq!(still.since, start);
        assert_eq!(still.reason, NO_SAFE_PATH);
        assert_eq!(
            Blocked::of(goal(2), NO_SAFE_PATH, Some(first), later).since,
            later
        );
    }

    #[test]
    fn a_block_runs_out_once_the_timeout_has_passed() {
        let start = Instant::now();
        let blocked = Blocked {
            goal: Goal {
                id: goal_id(1),
                position: (1.0, 0.0, 0.0),
            },
            since: start,
            reason: NO_SAFE_PATH,
        };
        let timeout = Duration::from_secs(2);
        assert_eq!(
            blocked.remaining(timeout, start + Duration::from_millis(500)),
            Duration::from_millis(1500)
        );
        assert!(blocked.remaining(timeout, start + timeout).is_zero());
        assert!(blocked
            .remaining(timeout, start + Duration::from_secs(5))
            .is_zero());
    }

    #[test]
    fn a_report_publishes_on_a_transition_and_not_on_a_distance_change() {
        let mut latest = None;
        let active = Report::new(goal_id(1), GoalStatus::ACTIVE, "following the path");
        assert!(record(&mut latest, &active.clone().with_remaining(4.0)));
        assert!(!record(&mut latest, &active.clone().with_remaining(3.0)));
        assert_eq!(latest.as_ref().and_then(|r| r.remaining_m), Some(3.0));
        assert!(record(
            &mut latest,
            &Report::new(goal_id(1), GoalStatus::SUCCEEDED, "reached")
        ));
    }

    #[test]
    fn a_report_for_an_older_goal_is_dropped() {
        let mut latest = None;
        let pending = Report::new(goal_id(2), GoalStatus::PENDING, "planning");
        assert!(record(&mut latest, &pending));
        assert!(!record(
            &mut latest,
            &Report::new(goal_id(1), GoalStatus::ACTIVE, "following the path")
        ));
        assert_eq!(latest, Some(pending));
    }

    #[test]
    fn a_report_for_a_goal_that_is_over_is_dropped() {
        let mut latest = None;
        let preempted = Report::new(goal_id(1), GoalStatus::PREEMPTED, PREEMPT_CANCELED);
        assert!(record(&mut latest, &preempted));
        assert!(!record(
            &mut latest,
            &Report::new(goal_id(1), GoalStatus::ACTIVE, "following the path")
        ));
        assert_eq!(latest, Some(preempted));
    }

    #[test]
    fn an_aborted_goal_is_over() {
        let mut latest = None;
        assert!(record(
            &mut latest,
            &Report::new(goal_id(1), GoalStatus::ABORTED, "no safe path")
        ));
        assert!(!record(
            &mut latest,
            &Report::new(goal_id(1), GoalStatus::ACTIVE, "following the path")
        ));
    }

    #[test]
    fn a_report_message_carries_the_id_status_reason_and_distance() {
        let msg = Report::new(goal_id(7), GoalStatus::ABORTED, "no safe path")
            .with_remaining(1.5)
            .message();
        assert_eq!(msg.goal_id.id, "7.000000005");
        assert_eq!((msg.goal_id.stamp.sec, msg.goal_id.stamp.nsec), (7, 5));
        assert_eq!(msg.status, GoalStatus::ABORTED as u8);
        assert_eq!(msg.text, "no safe path, 1.50 m from the goal");
    }
}
