# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process boundary to the repository's pinned LeRobot environment."""

from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
from typing import Any

from dimos.experimental.isolated_python.module import isolated_python_run_command
from dimos.imitation.loop.recording import digest

PROJECT = Path(__file__).resolve().parents[3] / "native" / "python" / "lerobot"


def command(*args: str) -> list[str]:
    python = PROJECT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(
            f"Install the isolated trainer first: uv sync --locked --project {PROJECT}"
        )
    return isolated_python_run_command(
        PROJECT, "--project", str(PROJECT), "python", "-m", "dimos_lerobot.act_loop", *args
    )


def train(
    snapshot: Path,
    output: Path,
    *,
    steps: int = 2000,
    device: str = "cpu",
    smoke: bool = False,
    initial: Path | None = None,
) -> None:
    args = [
        "train",
        str(snapshot.resolve()),
        str(output.resolve()),
        "--steps",
        str(steps),
        "--device",
        device,
    ]
    if smoke:
        args.append("--smoke")
    if initial is not None:
        args += ["--initial", str(initial.resolve())]
    subprocess.run(command(*args), check=True)


def read_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Verify weights, processors and configuration before loading either runtime."""
    manifest: dict[str, Any] = json.loads((checkpoint / "loop.json").read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise ValueError("checkpoint has no immutable runtime file manifest")
    for name, expected in files.items():
        if Path(name).name != name or digest(checkpoint / name) != expected:
            raise ValueError(f"checkpoint file changed: {name}")
    if files["model.safetensors"] != manifest["model_sha256"]:
        raise ValueError("checkpoint model identity differs from its file manifest")
    return manifest


class ActWorker:
    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.manifest = read_checkpoint(checkpoint)
        self.identity: str = self.manifest["model_sha256"]
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("ACT worker is already started")
        self.process = subprocess.Popen(
            command("serve", str(self.checkpoint), "--device", self.device),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            result = self._receive(120)
            if result != {"ready": True}:
                raise RuntimeError(f"ACT worker did not initialize: {result}")
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError("ACT worker timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"ACT worker exited ({self.process.poll()})")
        result: dict[str, Any] = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def predict(self, observation: dict[str, Any]) -> list[list[float]]:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(json.dumps(observation, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return list(self._receive(30)["actions"])

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
