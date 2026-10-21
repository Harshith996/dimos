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

//! Every wait, timeout and buffer size the module uses, from GO2Connection's config (seconds unless named otherwise).

use std::time::Duration;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    /// how long a request waits for the Go2's answer
    pub answer_timeout: f64,
    /// silence on cmd_vel this long stops the base
    pub cmd_vel_timeout: f64,
    /// a timed move resends its velocity this often
    pub move_period: f64,
    /// after StandUp, before the Go2 accepts BalanceStand
    pub stand_up_settle: f64,
    /// after BalanceStand, before the Rage Mode toggle
    pub rage_balance_settle: f64,
    /// after the Rage Mode toggle, before SwitchJoystick
    pub rage_switch_settle: f64,
    /// after selecting a motion mode
    pub motion_mode_settle: f64,
    /// between reconnect attempts (auto_reconnect)
    pub reconnect_pause: f64,
    /// each signaling HTTP request
    pub signaling_timeout: f64,
    /// ICE candidate gathering before the offer goes out
    pub ice_gather_timeout: f64,
    /// the Go2 validating the data channel
    pub validation_timeout: f64,
    /// data channel heartbeat period
    pub heartbeat_period: f64,
    /// ICE keepalive period
    pub ice_keepalive_period: f64,
    /// silence this long marks the link disconnected (logged only; it may come back)
    pub link_quiet_timeout: f64,
    /// silence this long means the link is gone: reconnect, or exit without auto_reconnect
    pub link_lost_timeout: f64,
    /// while the video waits for a keyframe, ask the Go2 for one this often
    pub keyframe_retry_period: f64,
    /// window the Go2-to-host clock offset is the minimum over
    pub clock_window: f64,
    /// an offset this far from the estimate is a clock reset, not jitter
    pub clock_reset: f64,
    /// RTP packets the decoder may fall behind by before they're dropped and it resyncs on a keyframe
    pub video_queue_packets: usize,
    /// frames in the decoder the capture clock remembers
    pub capture_max_waiting: usize,
}

pub fn secs(seconds: f64) -> Duration {
    Duration::from_secs_f64(seconds.max(0.0))
}

#[cfg(test)]
impl Timing {
    /// Any valid timing, for tests that need one (the real values come from GO2Connection's config).
    pub fn for_tests() -> Self {
        Self {
            answer_timeout: 5.0,
            cmd_vel_timeout: 0.2,
            move_period: 0.01,
            stand_up_settle: 3.0,
            rage_balance_settle: 0.3,
            rage_switch_settle: 2.0,
            motion_mode_settle: 5.0,
            reconnect_pause: 2.0,
            signaling_timeout: 4.0,
            ice_gather_timeout: 4.0,
            validation_timeout: 8.0,
            heartbeat_period: 2.0,
            ice_keepalive_period: 2.0,
            link_quiet_timeout: 5.0,
            link_lost_timeout: 30.0,
            keyframe_retry_period: 1.0,
            clock_window: 30.0,
            clock_reset: 1.0,
            video_queue_packets: 2048,
            capture_max_waiting: 16,
        }
    }
}
