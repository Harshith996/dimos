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

"""Run-bound clients for existing planners and advertised skills.

The app owns request progress, never a robot controller or a second planner.
Long scans and planning calls cannot block the worker's Hold request queue.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.core.introspection.module.info import ModuleInfo
from dimos.core.run_registry import RunEntry, is_pid_alive
from dimos.manipulation.manipulation_spec import UNCONFIRMED_STOP
from dimos.porcelain.dimos import Dimos

PLANNER_ACTIONS = frozenset(
    {
        "evaluate",
        "plan_to_joints",
        "plan_to_poses",
        "plan_cartesian",
        "execute",
        "clear_planned_path",
    }
)


@dataclass
class PlannerClient:
    module: Any
    session: str
    description: dict[str, Any]
    plan: dict[str, Any] | None = None


class OperatorClient:
    def __init__(self, run: RunEntry) -> None:
        self.run = run
        self.session = uuid4().hex
        self.planners: dict[str, PlannerClient] = {}
        self.skills: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        self.perception: dict[str, Any] = {}
        self.teleop: list[Any] = []
        self._lock = RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[Any] | None = None
        self._generation = 0
        self._job: dict[str, Any] | None = None

    def start(self, app: Dimos, modules: list[ModuleInfo]) -> None:
        for info in modules:
            name = info.instance_name or info.name
            methods = {method.name for method in info.rpcs}
            module: Any = app.get_module(name)
            if {"describe_operator", "operator_request"} <= methods:
                identity = module.describe_operator()
                if identity["run_id"] != self.run.run_id:
                    raise RuntimeError("Planner belongs to a different run")
                session = identity["session"]
                model = module.operator_request(session, "model")
                groups = module.operator_request(session, "list_planning_groups")
                self.planners[name] = PlannerClient(
                    module,
                    session,
                    {
                        "module": name,
                        "groups": groups,
                        "model_id": hashlib.sha256(
                            json.dumps(model, sort_keys=True).encode()
                        ).hexdigest(),
                    },
                )
            if "operator_action" in methods and "get_web_views" in methods:
                self.teleop.append(module)
            if "get_detected_objects" in methods:
                self.perception[name] = module
            if "get_skills" in methods:
                for skill in module.get_skills():
                    reason = None
                    if CAP_MOVEMENT in skill.uses or skill.lifecycle != "instant":
                        reason = (
                            "Use the planner controls; this skill has no app cancellation contract."
                        )
                    self.skills[name, skill.func_name] = (
                        module,
                        {
                            "module": name,
                            "name": skill.func_name,
                            "schema": json.loads(skill.args_schema),
                            "uses": list(skill.uses),
                            "available": reason is None,
                            "unavailable_reason": reason,
                        },
                    )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="desktop-operator")

    def require(self, params: dict[str, Any]) -> None:
        if params.get("run_id") != self.run.run_id or params.get("session") != self.session:
            raise RuntimeError("Operator session changed; inspect the current workspace")
        if not is_pid_alive(self.run.pid) or not self.run.registry_path.exists():
            raise RuntimeError("This blueprint stopped")

    def snapshot(self) -> dict[str, Any]:
        planners = []
        for planner in self.planners.values():
            state = planner.module.operator_request(planner.session, "get_state")
            with self._lock:
                plan = planner.plan if state["has_pending_plan"] else None
            planners.append({**planner.description, "state": state, "plan": plan})
        with self._lock:
            job = dict(self._job) if self._job is not None else None
        return {
            "run_id": self.run.run_id,
            "session": self.session,
            "planners": planners,
            "skills": [description for _, description in self.skills.values()],
            "objects": [
                {"module": name, "objects": module.get_detected_objects()}
                for name, module in self.perception.items()
            ],
            "job": job,
        }

    def _hold_teleop(self) -> None:
        for module in self.teleop:
            # The advertised view carries the receiver-issued session identity.
            view = module.get_web_views()[0]
            session = parse_qs(urlsplit(view["path"]).query).get("operator_session", [None])[0]
            if session is None:
                raise RuntimeError("Teleop does not advertise session-bound Hold")
            module.operator_action("hold", session, 0)  # Hold deliberately accepts any epoch.

    def cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        self.require(params)
        with self._lock:
            self._generation += 1
            cancelled = self._future is not None and self._future.cancel()
            if self._job is not None and self._job["status"] == "running":
                self._job = {**self._job, "status": "cancelled" if cancelled else "cancelling"}
        self._hold_teleop()
        results = {}
        for name, planner in self.planners.items():
            results[name] = planner.module.operator_request(planner.session, "cancel")
            with self._lock:
                planner.plan = None
        failures = [
            f"{name}: {result.get('message') or result['status']['name']}"
            for name, result in results.items()
            if result["status"]["name"] in {status.name for status in UNCONFIRMED_STOP}
        ]
        if failures:
            raise RuntimeError("Motion stop was not confirmed: " + "; ".join(failures))
        return {"results": results}

    def submit(self, params: dict[str, Any]) -> dict[str, Any]:
        self.require(params)
        kind, name, method = params.get("kind"), params.get("module"), params.get("method")
        if not isinstance(name, str) or not isinstance(method, str):
            raise ValueError("Choose an advertised module and operation")
        payload = params.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("Action arguments must be an object")
        if kind == "planner":
            if name not in self.planners or method not in PLANNER_ACTIONS:
                raise ValueError("Unknown planner operation")
        elif kind == "skill":
            if (name, method) not in self.skills:
                raise ValueError("This skill is not advertised by the selected run")
            description = self.skills[name, method][1]
            if not description["available"]:
                raise ValueError(description["unavailable_reason"])
            schema = description["schema"]
            if set(payload) - set(schema.get("properties", {})) or set(
                schema.get("required", [])
            ) - set(payload):
                raise ValueError("Skill arguments do not match its advertised schema")
        else:
            raise ValueError("Choose a planner operation or an advertised skill")
        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("An operator action is still running")
            generation = self._generation
            job = {
                "id": uuid4().hex,
                "kind": kind,
                "module": name,
                "method": method,
                "payload": payload,
                "status": "running",
                "started_at": time.time(),
                "result": None,
                "error": None,
            }
            self._job = job
            assert self._executor is not None
            self._future = self._executor.submit(self._invoke, generation, job, params)
            return dict(job)

    def _invoke(self, generation: int, job: dict[str, Any], params: dict[str, Any]) -> None:
        try:
            self.require(params)
            name, method, payload = job["module"], job["method"], job["payload"]
            with self._lock:
                if generation != self._generation:
                    return
            if job["kind"] == "planner":
                planner = self.planners[name]
                if method == "execute":
                    with self._lock:
                        if generation != self._generation:
                            return
                        self._hold_teleop()
                        result = planner.module.operator_request(planner.session, method, payload)
                else:
                    result = planner.module.operator_request(planner.session, method, payload)
                with self._lock:
                    if generation == self._generation and method.startswith("plan_"):
                        planner.plan = result.get("plan")
                    if method == "clear_planned_path":
                        planner.plan = None
            else:
                module = self.skills[name, method][0]
                value = getattr(module, method)(**payload)
                result = value.agent_encode() if hasattr(value, "agent_encode") else value
            with self._lock:
                if generation == self._generation:
                    self._job = {**job, "status": "completed", "result": result}
        except Exception as exc:
            with self._lock:
                if generation == self._generation:
                    self._job = {**job, "status": "failed", "error": str(exc)}
        finally:
            with self._lock:
                if self._job and self._job["id"] == job["id"] and generation != self._generation:
                    self._job = {**job, "status": "cancelled"}

    def close(self) -> None:
        with self._lock:
            self._generation += 1
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
