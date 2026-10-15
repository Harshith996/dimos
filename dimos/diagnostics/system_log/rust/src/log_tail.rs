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

//! Follows every *.jsonl in a directory from its start, across the RotatingFileHandler's renames
//! (main.jsonl → main.jsonl.1 and a fresh main.jsonl).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;

struct Followed {
    file: File,
    inode: u64,
    partial: Vec<u8>,
}

pub struct JsonlTail {
    directory: PathBuf,
    followed: BTreeMap<String, Followed>,
}

impl JsonlTail {
    pub fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            followed: BTreeMap::new(),
        }
    }

    /// The complete lines written since the last call, as (file name, line).
    pub fn read(&mut self) -> Vec<(String, String)> {
        let mut lines = Vec::new();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&self.directory)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        paths.sort();
        for path in paths {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let Ok(inode) = std::fs::metadata(&path).map(|m| m.ino()) else {
                continue;
            };
            if let Some(mut old) = self.followed.remove(&name) {
                if old.inode == inode {
                    self.followed.insert(name.clone(), old);
                } else {
                    // rotated: the handle still reads the renamed file, so finish it before starting the new one
                    drain(&name, &mut old, &mut lines);
                }
            }
            if !self.followed.contains_key(&name) {
                let Ok(file) = File::open(&path) else {
                    continue;
                };
                self.followed.insert(
                    name.clone(),
                    Followed {
                        file,
                        inode,
                        partial: Vec::new(),
                    },
                );
            }
            drain(
                &name,
                self.followed.get_mut(&name).expect("inserted above"),
                &mut lines,
            );
        }
        lines
    }
}

fn drain(name: &str, followed: &mut Followed, lines: &mut Vec<(String, String)>) {
    if followed.file.read_to_end(&mut followed.partial).is_err() {
        return;
    }
    let Some(last_newline) = followed.partial.iter().rposition(|&b| b == b'\n') else {
        return;
    };
    let rest = followed.partial.split_off(last_newline + 1);
    for line in String::from_utf8_lossy(&followed.partial).lines() {
        if !line.trim().is_empty() {
            lines.push((name.to_string(), line.to_string()));
        }
    }
    followed.partial = rest;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn follows_appends_partial_lines_and_rotation() {
        let dir = std::env::temp_dir().join(format!("system_log_tail_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let main = dir.join("main.jsonl");
        fs::write(&main, "{\"event\":\"a\"}\n{\"event\":\"b").unwrap();
        fs::write(dir.join("notes.txt"), "not a log\n").unwrap();
        let mut tail = JsonlTail::new(dir.clone());
        let got = |pairs: Vec<(String, String)>| {
            pairs.into_iter().map(|(_, line)| line).collect::<Vec<_>>()
        };
        assert_eq!(got(tail.read()), vec!["{\"event\":\"a\"}"]);

        fs::OpenOptions::new()
            .append(true)
            .open(&main)
            .unwrap()
            .write_all(b"\"}\n")
            .unwrap();
        assert_eq!(got(tail.read()), vec!["{\"event\":\"b\"}"]);

        // what RotatingFileHandler does: a last line to the old file, rename it, start a new one
        fs::OpenOptions::new()
            .append(true)
            .open(&main)
            .unwrap()
            .write_all(b"{\"event\":\"c\"}\n")
            .unwrap();
        fs::rename(&main, dir.join("main.jsonl.1")).unwrap();
        fs::write(&main, "{\"event\":\"d\"}\n").unwrap();
        assert_eq!(
            got(tail.read()),
            vec!["{\"event\":\"c\"}", "{\"event\":\"d\"}"]
        );
        assert_eq!(got(tail.read()), Vec::<String>::new());
        fs::remove_dir_all(dir).unwrap();
    }
}
