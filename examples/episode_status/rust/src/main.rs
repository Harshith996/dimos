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

use dimos_generated_messages::{codec::Message, dimos_msgs::msg::EpisodeStatus};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().collect();
    if arguments.len() != 4 {
        return Err(
            "usage: episode-status-consumer <seed|seed-be|echo|echo-be> <input> <output>".into(),
        );
    }
    let status = match arguments[1].as_str() {
        "seed" | "seed-be" => EpisodeStatus {
            ts: 17.25,
            state: "recording".into(),
            episodes_saved: 2,
            episodes_discarded: 1,
            last_event: "start".into(),
            task_label: vec!["".into()],
        },
        "echo" | "echo-be" => EpisodeStatus::decode(&fs::read(&arguments[2])?)?,
        _ => return Err("Expected seed, seed-be, echo or echo-be".into()),
    };
    let little = arguments[1] != "seed-be" && arguments[1] != "echo-be";
    fs::write(&arguments[3], status.encode_endian(little)?)?;
    println!(
        "Rust EpisodeStatus: {}, labels={}",
        status.state,
        status.task_label.len()
    );
    Ok(())
}
