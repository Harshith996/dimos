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

//! The Go2's clock on the host's. A stream's own stamps (a data-channel message's) are moved onto the host clock by the smallest
//! arrival - stamp seen in a sliding window: the message that crossed fastest, so neither a wrong
//! robot clock nor queueing moves it. Same estimator as the Go2 DDS module's `HostClock`.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

const NANOS_PER_SEC: i64 = 1_000_000_000;
const WINDOW_SECS: i64 = 30;
/// An offset this far from the estimate is a clock reset (robot reboot, host step), not jitter.
const RESET_NS: i64 = NANOS_PER_SEC;

pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Default)]
pub struct HostClock {
    /// (host second, smallest arrival - device stamp in it)
    minima: VecDeque<(i64, i64)>,
    offset_ns: Option<i64>,
}

impl HostClock {
    /// Fold in one message's device stamp and host arrival time.
    pub fn observe(&mut self, device_ns: i64, arrival_ns: u64) {
        let arrival_ns = arrival_ns as i64;
        let offset = arrival_ns - device_ns;
        if self
            .offset_ns
            .is_some_and(|estimate| (offset - estimate).abs() >= RESET_NS)
        {
            tracing::warn!(
                estimate_ns = self.offset_ns,
                offset_ns = offset,
                "Go2 clock offset jumped, restarting its estimate"
            );
            self.minima.clear();
        }
        let second = arrival_ns.div_euclid(NANOS_PER_SEC);
        match self.minima.back_mut() {
            Some(last) if last.0 == second => last.1 = last.1.min(offset),
            _ => {
                self.minima.push_back((second, offset));
                while self
                    .minima
                    .front()
                    .is_some_and(|f| f.0 < second - WINDOW_SECS)
                {
                    self.minima.pop_front();
                }
            }
        }
        self.offset_ns = self.minima.iter().map(|m| m.1).min();
    }

    /// The device stamp on the host clock; None before the first observation.
    pub fn host_ns(&self, device_ns: i64) -> Option<u64> {
        Some(device_ns.saturating_add(self.offset_ns?).max(0) as u64)
    }

    pub fn offset_ns(&self) -> Option<i64> {
        self.offset_ns
    }
}

/// Observe and place in one: a stamped message's host time, never later than its arrival.
pub fn place(clock: &mut HostClock, device_ns: Option<i64>, arrival_ns: u64) -> u64 {
    let Some(device_ns) = device_ns.filter(|ns| *ns > 0) else {
        return arrival_ns;
    };
    clock.observe(device_ns, arrival_ns);
    clock
        .host_ns(device_ns)
        .map_or(arrival_ns, |at| at.min(arrival_ns))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    #[test]
    fn the_fastest_arrival_sets_the_offset() {
        let mut clock = HostClock::default();
        assert_eq!(clock.host_ns(5), None);
        // robot 1000 s, host 1.7e9 s; the first arrives undelayed, later ones late
        let device = 1_000_000 * MS;
        let host = 1_700_000_000_000 * MS;
        clock.observe(device, host as u64);
        clock.observe(device + 5 * MS, (host + 9 * MS) as u64);
        clock.observe(device + 10 * MS, (host + 30 * MS) as u64);
        assert_eq!(clock.host_ns(device + 7), Some((host + 7) as u64));
    }

    #[test]
    fn a_robot_reboot_restarts_the_estimate() {
        let mut clock = HostClock::default();
        let host = 1_700_000_000_000 * MS;
        clock.observe(1_000_000 * MS, host as u64);
        clock.observe(2_000 * MS, (host + 100_000 * MS) as u64);
        assert_eq!(
            clock.host_ns(2_000 * MS),
            Some((host + 100_000 * MS) as u64)
        );
    }

    #[test]
    fn place_falls_back_to_arrival_and_never_lands_after_it() {
        let mut clock = HostClock::default();
        assert_eq!(place(&mut clock, None, 42), 42);
        assert_eq!(place(&mut clock, Some(0), 43), 43);
        assert_eq!(
            place(&mut clock, Some(1_000 * MS), (2_000 * MS) as u64),
            (2_000 * MS) as u64
        );
        // later and slower: placed by the earlier, faster one
        assert_eq!(
            place(&mut clock, Some(1_100 * MS), (2_150 * MS) as u64),
            (2_100 * MS) as u64
        );
    }
}
