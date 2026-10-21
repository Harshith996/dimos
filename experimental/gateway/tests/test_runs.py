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

from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi.testclient import TestClient
import psutil
import pytest

from dimos.core.run_registry import RunEntry
from experimental.gateway import launches, overrides as ov, store


def until(check: Callable[[], bool], seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def output_has(client: TestClient, text: str) -> Callable[[], bool]:
    return lambda: text in (client.get("/runs").json()["launch"] or {}).get("output", "")


@pytest.fixture
def launched(client: TestClient) -> Any:
    yield
    launch = launches.current_launch()
    if launch and launch["phase"] in launches.ACTIVE:
        client.post("/runs/stop", json={"runId": launch["runId"]})


def test_launch_runs_dimos_with_saved_and_one_off_config(
    client: TestClient, launched: None
) -> None:
    client.put("/config/global", json={"overrides": {"n_workers": 3}})
    store.set_blueprint_config(
        "demo", {"n_workers": 5, "viewer": "none"}, {"agent": {"api_key": "k1"}}
    )
    started = client.post(
        "/runs", json={"blueprint": "demo", "config": {"global": {"n_workers": 4, "replay": True}}}
    )
    assert started.status_code == 200, started.text
    assert started.json()["phase"] == "starting"
    until(output_has(client, "secret k1"))
    until(output_has(client, "dtop 1"))
    launch = client.get("/runs").json()["launch"]
    assert "--n-workers=4" in launch["output"] and "--replay" in launch["output"]
    assert "--viewer=none" in launch["output"]
    assert f"run --run-id {launch['runId']} demo" in launch["output"].splitlines()[0]
    assert "k1" not in launch["output"].splitlines()[0]
    assert launch["modules"]["agent"]["api_key"] == ov.HIDDEN
    assert launch["oneOff"]["global"] == {"n_workers": 4, "replay": True}
    again = client.post("/runs", json={"blueprint": "demo"})
    assert again.status_code == 400
    assert client.post("/runs/stop").status_code == 400
    assert client.post("/runs/stop", json={"pid": launch["pid"]}).status_code == 400
    assert client.post("/runs/stop", json={"runId": "nope"}).status_code == 400
    stopped = client.post("/runs/stop", json={"runId": launch["runId"]})
    assert stopped.status_code == 200, stopped.text
    assert client.get("/runs").json()["launch"]["phase"] == "stopped"


def test_a_module_launches_like_a_blueprint_and_exec_runs_args_as_typed(
    client: TestClient, launched: None
) -> None:
    assert client.post("/runs", json={}).status_code == 400
    assert client.post("/runs", json={"blueprint": "demo", "module": "cam"}).status_code == 400
    assert client.post("/runs/exec", json={"args": ["--n-workers=7"]}).status_code == 400
    client.put("/config/global", json={"overrides": {"n_workers": 3}})
    started = client.post("/runs", json={"module": "cam"})
    assert started.status_code == 200, started.text
    until(output_has(client, "args"))
    launch = client.get("/runs").json()["launch"]
    assert (
        f"run --run-id {launch['runId']} cam" in launch["output"]
        and "--n-workers=3" in launch["output"]
    )
    client.post("/runs/stop", json={"runId": launch["runId"]})
    until(lambda: client.get("/runs").json()["launch"]["phase"] == "stopped")
    typed = client.post("/runs/exec", json={"args": ["demo", "--n-workers=7"]})
    assert typed.status_code == 200, typed.text
    until(output_has(client, "args"))
    first = client.get("/runs").json()["launch"]["output"].splitlines()[0]
    assert first.endswith(f"run --run-id {typed.json()['runId']} demo --n-workers=7")
    restarted = client.post("/runs/restart", json={"runId": typed.json()["runId"]})
    assert restarted.status_code == 200, restarted.text
    until(output_has(client, "args"))
    again = client.get("/runs").json()["launch"]["output"].splitlines()[0]
    assert again.endswith("demo --n-workers=7") and "--n-workers=3" not in again


def test_a_failed_launch_reports_its_last_output_line(client: TestClient) -> None:
    client.post("/runs", json={"blueprint": "fail"})
    until(lambda: client.get("/runs").json()["launch"]["phase"] == "failed")
    assert client.get("/runs").json()["launch"]["error"] == "ValueError: no robot answered"


def test_restart_launches_the_last_blueprint_again(client: TestClient, launched: None) -> None:
    assert client.post("/runs/restart", json={"runId": "nope"}).status_code == 400
    client.post("/runs", json={"blueprint": "demo", "config": {"n_workers": 2}})
    until(output_has(client, "args"))
    first = client.get("/runs").json()["launch"]["runId"]
    assert client.post("/runs/restart").status_code == 400
    restarted = client.post("/runs/restart", json={"runId": first})
    assert restarted.status_code == 200, restarted.text
    assert restarted.json()["runId"] != first
    assert restarted.json()["oneOff"]["global"] == {"n_workers": 2}
    second = restarted.json()["runId"]
    assert client.post("/runs/stop", json={"runId": second}).status_code == 200
    until(lambda: client.get("/runs").json()["launch"]["phase"] == "stopped")
    again = client.post("/runs/restart", json={"runId": second})
    assert again.status_code == 200, again.text
    assert again.json()["runId"] not in (first, second)


def test_restart_copies_a_run_the_gateway_did_not_launch(
    client: TestClient, launched: None, tmp_path: Path
) -> None:
    command = [
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        "run",
        "--run-id",
        "x",
        "outside",
    ]
    env = {**os.environ, "OUTSIDE_RUN": "yes"}
    outside = subprocess.Popen(command, cwd=tmp_path, env=env, start_new_session=True)
    threading.Thread(target=outside.wait, daemon=True).start()
    RunEntry(
        run_id="20260101-120000-outside",
        pid=outside.pid,
        blueprint="outside",
        started_at="2026-01-01T12:00:00+00:00",
        log_dir=str(tmp_path),
    ).save()
    assert client.post("/runs/restart", json={"runId": "nope"}).status_code == 400
    restarted = client.post("/runs/restart", json={"runId": "20260101-120000-outside"})
    assert restarted.status_code == 200, restarted.text
    assert outside.poll() is not None
    again = psutil.Process(restarted.json()["pid"])
    run_id = restarted.json()["runId"]
    assert again.cmdline()[-4:] == ["run", "--run-id", run_id, "outside"]
    assert again.cwd() == str(tmp_path.resolve())
    assert again.environ()["OUTSIDE_RUN"] == "yes"
    assert restarted.json()["blueprint"] == "outside"
    assert client.post("/runs/stop", json={"runId": run_id}).status_code == 200


def test_runs_lists_the_registry(client: TestClient) -> None:
    entry = RunEntry(
        run_id="20260101-120000-demo",
        pid=os.getpid(),
        blueprint="demo",
        started_at="2026-01-01T12:00:00+00:00",
        log_dir="/tmp/x",
    )
    entry.save()
    runs = client.get("/runs").json()["runs"]
    assert runs == [
        {
            "run_id": "20260101-120000-demo",
            "pid": os.getpid(),
            "blueprint": "demo",
            "started_at": "2026-01-01T12:00:00+00:00",
            "log_dir": "/tmp/x",
        }
    ]
    refused = client.post("/runs", json={"blueprint": "demo"})
    assert refused.status_code == 400
    assert "stop it first" in refused.json()["error"]


def test_run_logs_now_and_past(client: TestClient, home: Path) -> None:
    assert client.get("/runs/logs").status_code == 404
    assert client.get("/runs/log_hist").json() == {"runs": []}
    for run_id in ("20260101-110000-old", "20260101-120000-demo"):
        (home / "logs" / run_id).mkdir(parents=True)
    run = home / "logs" / "20260101-120000-demo"
    lines = [
        {"level": "info", "event": "hello", "logger": "a"},
        {"level": "error", "event": "boom", "logger": "b"},
    ]
    (run / "main.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (home / "logs" / "20260101-110000-old" / "main.jsonl").write_text("")
    page = client.get("/runs/logs").json()
    assert page["runId"] == "20260101-120000-demo"
    assert [r["event"] for r in page["records"]] == ["hello", "boom"]
    assert page["loggers"] == ["a", "b"]
    past = client.get("/runs/log_hist").json()["runs"]
    assert [r["runId"] for r in past] == ["20260101-120000-demo", "20260101-110000-old"]
    assert not past[0]["running"]
    errors = client.get("/runs/log_hist/20260101-120000-demo?level=error").json()
    assert [r["event"] for r in errors["records"]] == ["boom"]
    assert client.get("/runs/log_hist/20990101-000000-nope").status_code == 404
    with (run / "main.jsonl").open("a") as handle:
        handle.write(json.dumps({"level": "info", "event": "later"}) + "\n")
    tail = client.get(f"/runs/logs?after={page['offset']}").json()
    assert [r["event"] for r in tail["records"]] == ["later"]
