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

import json
import os
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest
import yaml

from dimos.core.global_config import GlobalConfig
from dimos.core.run_registry import RunEntry
from dimos.robot.all_blueprints import all_blueprints
from experimental.gateway import overrides as ov, robots, store
from experimental.gateway.state import API_VERSION, ApiError, State

ARG = {
    "default": None,
    "description": None,
    "required": False,
    "secret": False,
    "base": False,
    "json_compatible": True,
}
CONFIG = {
    "name": "demo",
    "global": {"n_workers": {**ARG, "type": "int", "default": 2}},
    "modules": {
        "camera": {
            "class": "x.Camera",
            "args": {
                "fps": {**ARG, "type": "int", "default": 30},
                "api_key": {**ARG, "type": "str", "secret": True},
            },
        }
    },
}
REMOVED = [
    ("get", "/blueprints"),
    ("get", "/blueprints/demo"),
    ("post", "/blueprints/demo/args"),
    ("put", "/blueprints/demo/config"),
    ("get", "/robots"),
    ("get", "/global-config"),
    ("get", "/skills"),
    ("post", "/skills/call"),
    ("get", "/runs/latest/log"),
]


@pytest.fixture
def introspected(state: State, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    state.listed = [{"name": "demo", "kind": "builtin"}]

    async def fake(args: list[str], stdin: Any = None) -> Any:
        calls.append((args, stdin))
        if args[0] == "config":
            return CONFIG
        if args[0] == "blueprint":
            return {"name": args[1], "file": "x.py", "line": 3, "modules": []}
        bad = [a for a in stdin or [] if "fps=fast" in a]
        return {
            "args": [{"tokens": [a], "target": "camera.fps", "error": "not an int"} for a in bad]
        }

    monkeypatch.setattr(state, "introspected", fake)
    return calls


def test_server_routes(client: TestClient, checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.get("/healthz").text == "ok"
    monkeypatch.delenv("DESKTOP_DIMOS_RANGE", raising=False)
    info = client.get("/info").json()
    assert {
        k: info[k]
        for k in (
            "dir",
            "found",
            "installed",
            "version",
            "range",
            "inRange",
            "api",
            "commit",
            "tag",
        )
    } == {
        "dir": str(checkout),
        "found": True,
        "installed": True,
        "version": "0.0.14",
        "range": "",
        "inRange": True,
        "api": API_VERSION,
        "commit": None,
        "tag": None,
    }
    monkeypatch.setenv("DESKTOP_DIMOS_RANGE", ">=0.0.14b1 <0.1")
    assert client.get("/info").json()["inRange"] is True
    monkeypatch.setenv("DESKTOP_DIMOS_RANGE", ">=0.1 <0.2")
    narrow = client.get("/info").json()
    assert {
        k: narrow[k]
        for k in ("found", "installed", "version", "range", "inRange", "api", "commit", "tag")
    } == {
        "found": True,
        "installed": True,
        "version": "0.0.14",
        "range": ">=0.1 <0.2",
        "inRange": False,
        "api": API_VERSION,
        "commit": None,
        "tag": None,
    }
    assert info["python"]
    assert info["paths"]["dimosDir"] == str(checkout)
    assert info["server"]["zenohNamespace"] == "ns"


def test_openapi_lists_the_routes(client: TestClient) -> None:
    routes = client.get("/openapi.json").json()["paths"]
    assert {"/runs", "/runs/skills", "/catalog", "/config/global"} <= set(routes)
    slow = {path for path, methods in routes.items() if methods.get("get", {}).get("x-slow")}
    assert slow == {
        "/catalog/blueprints/{name}",
        "/catalog/blueprints/{name}/config",
        "/catalog/modules/{name}",
        "/catalog/modules/{name}/config",
    }
    assert not any(
        path.startswith(("/blueprints", "/skills", "/robots", "/global-config")) for path in routes
    )


@pytest.mark.parametrize(("method", "path"), REMOVED)
def test_the_4x_routes_are_gone(client: TestClient, method: str, path: str) -> None:
    assert getattr(client, method)(path).status_code in (404, 405)


def test_global_config_is_saved_in_the_gateways_own_state(client: TestClient, home: Path) -> None:
    answer = client.put("/config/global", json={"overrides": {"robot_ip": "10.0.0.2"}})
    assert answer.status_code == 200
    assert answer.json()["overrides"] == {"robot_ip": "10.0.0.2"}
    saved = json.loads((home / "state" / "gateway" / "settings.json").read_text())
    assert saved["global_config"] == {"robot_ip": "10.0.0.2"}
    shown = client.get("/config/global").json()
    assert "robot_ip" in shown["schema"]["properties"]
    assert shown["defaults"]["rerun_open"] == "none"


def test_global_config_refuses_what_dimos_would(client: TestClient) -> None:
    unknown = client.put("/config/global", json={"overrides": {"robot_iq": "x"}})
    assert unknown.status_code == 400
    assert "robot_iq" in unknown.json()["error"]
    wrong = client.put("/config/global", json={"overrides": {"n_workers": "many"}})
    assert wrong.status_code == 400


def test_a_secret_is_hidden_and_kept_when_sent_back(client: TestClient) -> None:
    client.put("/config/global", json={"overrides": {"typesafe_api_key": "s3cret"}})
    assert client.get("/config/global").json()["overrides"] == {"typesafe_api_key": ov.HIDDEN}
    client.put(
        "/config/global", json={"overrides": {"typesafe_api_key": ov.HIDDEN, "n_workers": 2}}
    )
    assert store.global_config_overrides() == {"typesafe_api_key": "s3cret", "n_workers": 2}


def test_blueprint_config_round_trip(client: TestClient, introspected: list[Any]) -> None:
    sent = {"global": {"n_workers": 3}, "modules": {"camera": {"fps": 5, "api_key": "k"}}}
    put = client.put("/config/blueprints/demo", json=sent)
    assert put.status_code == 200, put.text
    assert store.blueprint_config("demo") == sent
    shown = client.get("/config/blueprints/demo").json()
    assert shown == {
        "name": "demo",
        "global": {"n_workers": 3},
        "modules": {"camera": {"api_key": ov.HIDDEN, "fps": 5}},
    }
    kept = client.put(
        "/config/blueprints/demo", json={"modules": {"camera": {"api_key": ov.HIDDEN}}}
    )
    assert kept.status_code == 200
    assert store.blueprint_config("demo")["modules"] == {"camera": {"api_key": "k"}}


def test_blueprint_config_lists_every_refused_key(
    client: TestClient, introspected: list[Any]
) -> None:
    refused = client.put(
        "/config/blueprints/demo",
        json={"global": {"n_workers": "many"}, "modules": {"camera": {"fps": "fast"}}},
    )
    assert refused.status_code == 400
    keys = {p["key"] for p in refused.json()["problems"]}
    assert keys == {"global", "modules.camera.fps"}
    assert store.blueprint_config("demo") == {"global": {}, "modules": {}}
    assert client.get("/config/blueprints/nope").status_code == 404


def test_a_blueprints_config_schema_is_global_and_module_fields(
    client: TestClient, introspected: list[Any]
) -> None:
    schema = client.get("/catalog/blueprints/demo/config").json()
    assert schema["global"]["n_workers"]["default"] == 2
    assert schema["modules"]["camera"]["args"]["api_key"]["secret"]


def test_a_blueprint_running_from_another_checkout_is_read_from_that_one(
    client: TestClient, state: State, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    branch = home / "dimos_branch"
    (branch / ".venv" / "bin").mkdir(parents=True)
    (branch / "pyproject.toml").write_text('[project]\nname = "dimos"\n')
    state.listed = [{"name": "demo", "kind": "builtin"}]
    read_from: list[Any] = []

    async def fake(args: list[str], stdin: Any = None, source: Any = None) -> Any:
        read_from.append(source and source.checkout)
        if source is not None and "broken" in str(source.run_id):
            raise ApiError(500, "ImportError: no get_by_name")
        if args[0] == "config":
            return {**CONFIG, "name": args[1], "global": {} if source else CONFIG["global"]}
        stream = "lidar_with_robot" if source else "lidar"
        streams = [
            {"name": "lidar", "type": "x.PointCloud2", "direction": "in", "wire": stream}
            | {"topic": None}
        ]
        modules: list[dict[str, Any]] = [
            {"name": "mapper", "class": "x.Mapper", "doc": "", "summary": "", "streams": streams}
            | {"rpcs": [], "skills": [], "file": None, "line": None}
        ]
        return {"name": args[1], "file": None, "line": None, "modules": modules}

    monkeypatch.setattr(state, "introspected", fake)

    def wire_shown(name: str = "demo") -> tuple[str, Any, Any]:
        found = client.get(f"/catalog/blueprints/{name}?fields=modules,checkout,checkoutError")
        body = found.json()
        return body["modules"][0]["streams"][0]["wire"], body["checkout"], body["checkoutError"]

    def started_from(program: Path, run_id: str, blueprint: str = "demo") -> RunEntry:
        entry = RunEntry(
            run_id=run_id,
            pid=os.getpid(),
            blueprint=blueprint,
            started_at="2026-01-01T12:00:00+00:00",
            log_dir="/tmp/x",
            original_argv=[str(program), "run", blueprint],
        )
        entry.save()
        return entry

    assert wire_shown() == ("lidar", None, None)
    # a blueprint only a branch has is unknown until it runs from that branch
    assert client.get("/catalog/blueprints/branch-only").status_code == 404
    assert client.get("/config/blueprints/branch-only").status_code == 404

    from_gateway = started_from(state.dimos_dir / ".venv" / "bin" / "dimos", "20260101-120000-demo")
    assert wire_shown() == ("lidar", None, None)
    from_gateway.remove()
    started_from(branch / ".venv" / "bin" / "dimos", "20260101-120001-demo")
    assert wire_shown() == ("lidar_with_robot", str(branch), None)
    assert read_from[-1] == branch
    assert client.get("/catalog/blueprints/demo/config").json()["global"] == {}

    only = started_from(branch / ".venv" / "bin" / "dimos", "20260101-120003-only", "branch-only")
    assert wire_shown("branch-only") == ("lidar_with_robot", str(branch), None)
    assert client.get("/catalog/blueprints/branch-only/config").json()["name"] == "branch-only"
    assert client.get("/config/blueprints/branch-only").status_code == 200
    only.remove()
    assert client.get("/catalog/blueprints/branch-only").status_code == 404

    started_from(branch / ".venv" / "bin" / "dimos", "20260101-120002-broken")
    wire, checkout, error = wire_shown()
    assert (wire, checkout) == ("lidar", None) and "ImportError" in error


def test_a_bad_blueprint_name_is_refused(client: TestClient) -> None:
    assert client.get("/catalog/blueprints/-x").status_code == 400
    assert client.get("/config/blueprints/-x").status_code == 400


def test_robots_list_their_blueprints_which_carry_the_annotations(client: TestClient) -> None:
    go2 = client.get("/catalog/robots").json()["robots"]["go2"]
    assert "unitree-go2" in go2["blueprints"]
    found = client.get("/catalog/blueprints?fields=blueprints.name,blueprints.recommended_config")
    bp = next(b for b in found.json()["blueprints"] if b["name"] == "unitree-go2")
    assert set(bp) == {"name", "recommended_config"}
    assert any(s["kind"] == "pick" for s in bp["recommended_config"])


def test_annotations_name_registered_blueprints_and_real_fields() -> None:
    source = robots.load()
    for robot_id, robot in source["robots"].items():
        for name, blueprint in robot["blueprints"].items():
            assert name in all_blueprints, f"{robot_id} lists {name}, which isn't registered"
            settings = [
                *blueprint.get("recommended_config", []),
                *robot.get("defaults", {}).get("recommended_config", []),
            ]
            for setting in settings:
                if "global" in setting:
                    assert setting["global"] in GlobalConfig.model_fields


def test_dimos_yaml_declares_this_api_and_its_endpoints(client: TestClient) -> None:
    manifest = yaml.safe_load((Path(__file__).parents[3] / "dimos.yaml").read_text())
    assert manifest["api"]["version"] == API_VERSION
    routes = client.get("/openapi.json").json()["paths"]
    served = {
        f"{method.upper()} {path.lstrip('/')}" for path, ops in routes.items() for method in ops
    }
    declared = {f"{e['method']} {e['path']}" for e in manifest["provides"]["endpoints"]}
    assert declared == served
