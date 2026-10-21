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

from typing import Any

from fastapi.testclient import TestClient
import pytest

from dimos.core.coordination.module_coordinator import ModuleDescriptor
from experimental.gateway import rpcs
from experimental.gateway.state import State

SPEAK = {
    "name": "speak",
    "doc": "Say something.",
    "params": [],
    "return_type": "str",
    "skill": True,
}


class FakeBus:
    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.calls: list[tuple[str, Any]] = []

    def call(self, name: str, args: list[Any], kwargs: dict[str, Any], timeout: float) -> Any:
        self.calls.append((name, kwargs))
        if not self.running:
            raise TimeoutError()
        if name == "Coordinator/list_modules":
            return [
                ModuleDescriptor(
                    "Speaker", "x.Speaker", ["start", "speak", "volume"], "robot0/Speaker"
                )
            ]
        if kwargs.get("text") == "explode":
            raise RuntimeError("speaker on fire")
        return {"said": kwargs.get("text")}


@pytest.fixture
def bus(state: State, monkeypatch: pytest.MonkeyPatch) -> FakeBus:
    fake = FakeBus()
    state.rpcs = rpcs.Rpcs(fake)
    monkeypatch.setattr(rpcs, "live_run", lambda: rpcs.Run("run-1", "demo"))
    monkeypatch.setattr(
        state.scanner,
        "cache",
        lambda: {"modules": {"speaker": {"class": "x.Speaker", "rpcs": [SPEAK]}}},
    )
    return fake


def test_rpcs_are_listed_with_what_the_scan_found_and_called(
    client: TestClient, bus: FakeBus
) -> None:
    listed = client.get("/runs/rpcs").json()
    assert listed["run"] == {"runId": "run-1", "blueprint": "demo"}
    # every Module's own RPCs (start) are left out; volume isn't in the scan, so it has no doc or skill flag
    assert [(r["rpc"], r["skill"], r["doc"]) for r in listed["rpcs"]] == [
        ("robot0/Speaker/speak", True, "Say something."),
        ("robot0/Speaker/volume", None, ""),
    ]
    called = client.post(
        "/runs", json={"rpc": {"module": "Speaker", "name": "speak", "args": {"text": "hi"}}}
    ).json()
    assert (
        called["ok"]
        and called["result"] == {"said": "hi"}
        and called["rpc"] == "robot0/Speaker/speak"
    )
    failed = client.post(
        "/runs",
        json={"rpc": {"module": "robot0/Speaker", "name": "speak", "args": {"text": "explode"}}},
    ).json()
    assert not failed["ok"] and "speaker on fire" in failed["error"]


def test_unknown_modules_and_rpcs_never_reach_the_robot(client: TestClient, bus: FakeBus) -> None:
    assert (
        client.post("/runs", json={"rpc": {"module": "Nope", "name": "speak"}}).status_code == 404
    )
    assert (
        client.post("/runs", json={"rpc": {"module": "Speaker", "name": "fly"}}).status_code == 404
    )
    assert [name for name, _ in bus.calls if name != "Coordinator/list_modules"] == []
    both = client.post(
        "/runs", json={"blueprint": "demo", "rpc": {"module": "Speaker", "name": "speak"}}
    )
    assert both.status_code == 400


def test_nothing_running_means_no_rpcs(client: TestClient, bus: FakeBus) -> None:
    bus.running = False
    assert client.get("/runs/rpcs").json() == {"rpcs": [], "run": None}
    assert (
        client.post("/runs", json={"rpc": {"module": "Speaker", "name": "speak"}}).status_code
        == 409
    )
