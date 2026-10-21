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

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Protocol

from fastapi.encoders import jsonable_encoder

from dimos.protocol.rpc.spec import DEFAULT_RPC_TIMEOUT
from experimental.gateway.errors import ApiError

PING_TIMEOUT_S = 1.0
CALL_TIMEOUT_S = DEFAULT_RPC_TIMEOUT  # what a module's own RPC call waits


class Bus(Protocol):
    def call(self, name: str, args: list[Any], kwargs: dict[str, Any], timeout: float) -> Any: ...


class CoordinatorBus:
    def __init__(self) -> None:
        self._coordinator: Any = None
        self._lock = threading.Lock()

    def call(self, name: str, args: list[Any], kwargs: dict[str, Any], timeout: float) -> Any:
        from dimos.core.coordination.coordinator_rpc import CoordinatorRPC
        from dimos.core.transport_factory import rpc_backend

        with self._lock:
            if self._coordinator is None:
                rpc = rpc_backend()()
                rpc.start()
                self._coordinator = CoordinatorRPC(rpc)
        result, _ = self._coordinator.rpc.call_sync(name, (args, kwargs), rpc_timeout=timeout)
        return result


@dataclass
class Run:
    run_id: str | None
    blueprint: str | None


def live_run() -> Run:
    from dimos.core.run_registry import list_runs

    entries = sorted(list_runs(alive_only=True), key=lambda e: str(e.run_id), reverse=True)
    if not entries:
        return Run(None, None)
    return Run(str(entries[0].run_id), str(entries[0].blueprint))


class Rpcs:
    """The running blueprint's RPCs, called directly (no MCP, so no capability holds): this API is for buttons,
    agents use the python API."""

    def __init__(self, bus: Bus | None = None) -> None:
        self.bus = bus or CoordinatorBus()

    def modules(self) -> list[Any] | None:
        try:
            return list(self.bus.call("Coordinator/list_modules", [], {}, PING_TIMEOUT_S))
        except TimeoutError:
            return None

    def listed(self, scanned: dict[str, Any] | None) -> dict[str, Any]:
        """Each running module's RPCs, with the doc, params and skill flag the last scan found (None if it didn't)."""
        from experimental.gateway.introspect import base_rpcs

        descriptors, run = self.modules(), live_run()
        if descriptors is None:
            return {"rpcs": [], "run": None}
        by_class = {m["class"]: m for m in (scanned or {}).get("modules", {}).values()}
        found = []
        for d in descriptors:
            module = d.rpc_name or d.class_name
            known = {r["name"]: r for r in by_class.get(d.qualified_path, {}).get("rpcs", [])}
            for name in sorted(d.rpc_names):
                if name in known or name not in base_rpcs():
                    seen = known.get(
                        name, {"doc": "", "params": None, "return_type": None, "skill": None}
                    )
                    found.append(
                        {**seen, "name": name, "module": module, "rpc": f"{module}/{name}"}
                    )
        return {
            "rpcs": sorted(found, key=lambda r: r["rpc"]),
            "run": {"runId": run.run_id, "blueprint": run.blueprint},
        }

    def call(
        self, module: str, name: str, args: dict[str, Any], run_id: str | None = None
    ) -> dict[str, Any]:
        descriptors, run = self.modules(), live_run()
        if descriptors is None:
            raise ApiError(409, "no blueprint is running, so there are no RPCs to call")
        if run_id is not None and run_id != run.run_id:
            raise ApiError(404, f"run {run_id} isn't the running one ({run.run_id})")
        target = next((d for d in descriptors if module in (d.rpc_name, d.class_name)), None)
        if target is None:
            running = ", ".join(sorted(d.rpc_name or d.class_name for d in descriptors)) or "none"
            raise ApiError(
                404, f"the running blueprint has no module {module} (its modules: {running})"
            )
        if name not in target.rpc_names:
            raise ApiError(
                404, f"{module} has no RPC {name} (its RPCs: {', '.join(sorted(target.rpc_names))})"
            )
        rpc = f"{target.rpc_name or target.class_name}/{name}"
        answer = {"rpc": rpc, "runId": run.run_id, "blueprint": run.blueprint}
        try:
            result = self.bus.call(rpc, [], args, CALL_TIMEOUT_S)
        except TimeoutError:
            raise ApiError(500, f"{rpc} didn't answer within {int(CALL_TIMEOUT_S)} s")
        except Exception as error:
            return {
                **answer,
                "ok": False,
                "result": None,
                "error": f"{type(error).__name__}: {error}",
            }
        try:
            encoded = jsonable_encoder(result)
        except Exception:
            encoded = str(result)
        return {**answer, "ok": True, "result": encoded, "error": None}
