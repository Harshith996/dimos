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

//! The Go2's clocks on the host's. A stream's own stamps (a data-channel message's, or a video
//! frame's RTP time through the RTCP sender reports) are moved onto the host clock by the smallest
//! arrival - stamp seen in a sliding window: the message that crossed fastest, so neither a wrong
//! robot clock nor queueing moves it. Same estimator as the Go2 DDS module's `HostClock`.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

const NANOS_PER_SEC: i64 = 1_000_000_000;
const WINDOW_SECS: i64 = 30;
/// An offset this far from the estimate is a clock reset (robot reboot, host step), not jitter.
const RESET_NS: i64 = NANOS_PER_SEC;
/// Seconds between 1900 (NTP) and 1970 (Unix).
const NTP_UNIX_OFFSET_SECS: u64 = 2_208_988_800;
/// Sender reports this far apart (robot time) before their RTP rate replaces the codec's nominal one.
const RATE_SPAN_NS: i64 = 2 * NANOS_PER_SEC;

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

/// An RTCP NTP timestamp (seconds since 1900 in the high word, fraction in the low) as Unix ns.
pub fn ntp_to_unix_ns(ntp: u64) -> i64 {
    let secs = (ntp >> 32).saturating_sub(NTP_UNIX_OFFSET_SECS) as i64;
    let frac = ((ntp & 0xFFFF_FFFF) * NANOS_PER_SEC as u64) >> 32;
    secs * NANOS_PER_SEC + frac as i64
}

/// RTP timestamps on the robot's clock, through its RTCP sender reports (each pairs an RTP time with
/// the robot's NTP time at that instant). The tick rate comes from two reports far enough apart,
/// the codec's nominal rate until then.
#[derive(Debug)]
pub struct RtpTimeline {
    nominal_hz: f64,
    last_rtp: Option<u32>,
    extended: i64,
    /// (robot ns, extended RTP) of the first report since a reset and of the latest
    first: Option<(i64, i64)>,
    latest: Option<(i64, i64)>,
}

impl RtpTimeline {
    pub fn new(nominal_hz: u32) -> Self {
        Self {
            nominal_hz: nominal_hz.max(1) as f64,
            last_rtp: None,
            extended: 0,
            first: None,
            latest: None,
        }
    }

    /// The RTP time with wraps counted, relative to the stream's first.
    fn extend(&mut self, rtp: u32) -> i64 {
        match self.last_rtp {
            Some(last) => self.extended += rtp.wrapping_sub(last) as i32 as i64,
            None => self.extended = 0,
        }
        self.last_rtp = Some(rtp);
        self.extended
    }

    pub fn sender_report(&mut self, ntp: u64, rtp: u32) {
        let at = (ntp_to_unix_ns(ntp), self.extend(rtp));
        if self.latest.is_some_and(|(ns, _)| at.0 <= ns) {
            // the robot's NTP time stepped back: start over from this report
            self.first = None;
        }
        self.first.get_or_insert(at);
        self.latest = Some(at);
    }

    pub fn hz(&self) -> f64 {
        match (self.first, self.latest) {
            (Some((n0, r0)), Some((n1, r1))) if n1 - n0 >= RATE_SPAN_NS && r1 > r0 => {
                (r1 - r0) as f64 * NANOS_PER_SEC as f64 / (n1 - n0) as f64
            }
            _ => self.nominal_hz,
        }
    }

    /// A frame's RTP time on the robot's clock (ns); None before the first sender report.
    pub fn robot_ns(&mut self, rtp: u32) -> Option<i64> {
        let ticks = self.extend(rtp);
        let (ns, at_ticks) = self.latest?;
        Some(ns + ((ticks - at_ticks) as f64 * NANOS_PER_SEC as f64 / self.hz()) as i64)
    }
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

    #[test]
    fn ntp_converts_to_unix() {
        let ntp = ((NTP_UNIX_OFFSET_SECS + 1_700_000_000) << 32) | (1u64 << 31);
        assert_eq!(ntp_to_unix_ns(ntp), 1_700_000_000_500 * MS);
    }

    #[test]
    fn frames_land_between_sender_reports_at_the_measured_rate() {
        let ntp = |unix_ms: u64| {
            ((NTP_UNIX_OFFSET_SECS + unix_ms / 1000) << 32) | (((unix_ms % 1000) << 32) / 1000)
        };
        // a 1 kHz RTP clock that the codec calls 90 kHz, wrapping on the way
        let mut timeline = RtpTimeline::new(90_000);
        let rtp0 = u32::MAX - 500;
        assert_eq!(timeline.robot_ns(rtp0), None);
        timeline.sender_report(ntp(1_000_000), rtp0);
        timeline.sender_report(ntp(1_005_000), rtp0.wrapping_add(5_000));
        assert!((timeline.hz() - 1_000.0).abs() < 1e-6);
        assert_eq!(
            timeline.robot_ns(rtp0.wrapping_add(5_250)),
            Some(1_005_250 * MS)
        );
    }
}
