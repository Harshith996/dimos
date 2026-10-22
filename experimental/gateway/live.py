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

from collections.abc import Callable
import importlib
import threading
from typing import Any

from fastapi.encoders import jsonable_encoder

from experimental.gateway.errors import ApiError


class NothingRunningError(ApiError):
    def __init__(self) -> None:
        super().__init__(409, "nothing running")


def instance_id(remote_name: str) -> str:
    """The coordinator's key for an instance, `/` written as `:` so it is one path segment: `robot0:go2connection`."""
    # a namespaced instance's RPC name is its key; an un-namespaced one's is its class name (key = lowercase class)
    # TODO(core): list_modules could publish the coordinator's key itself, instead of the gateway rebuilding it
    return remote_name.replace("/", ":") if "/" in remote_name else remote_name.lower()


def connect_dimos() -> Any:
    """`Dimos.connect()` on the transport the gateway's launch ran with (its `config.global.transport`), since
    connect finds the coordinator on the gateway's own GlobalConfig transport."""
    from dimos.core.global_config import GlobalConfig, global_config
    from dimos.porcelain.dimos import Dimos
    from experimental.gateway.launches import current_launch

    launch = current_launch() or {}
    transport = launch.get("config", {}).get("global", {}).get("transport")
    global_config.update(transport=transport or GlobalConfig().transport)
    return Dimos.connect(timeout=1.0)


def load_class(path: str) -> Any:
    """A module class from its class path (what Dimos.restart and Dimos.unload take)."""
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def live_run_id() -> str | None:
    from dimos.core.run_registry import get_most_recent

    entry = get_most_recent(alive_only=True)
    return None if entry is None else str(entry.run_id)


class Live:
    """The running blueprint through one kept-open `Dimos.connect()`, reconnected when the run changes. Its RPCs are
    called directly (no MCP, so no capability holds): this API is for buttons, agents use the python API."""

    def __init__(
        self,
        connect: Callable[[], Any] = connect_dimos,
        run_id: Callable[[], str | None] = live_run_id,
    ) -> None:
        self._connect, self._run_id = connect, run_id
        self._app: Any = None
        self._connected_to: str | None = None
        self._lock = threading.Lock()

    def app(self) -> Any:
        run = self._run_id()
        with self._lock:
            if self._app is not None and run != self._connected_to:
                self._app.stop()
                self._app = None
            if run is None:
                raise NothingRunningError()
            if self._app is None:
                try:
                    self._app = self._connect()
                except (RuntimeError, TimeoutError):
                    raise NothingRunningError() from None
                self._connected_to = run
            return self._app

    def modules(self, scanned: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Each instance with its RPCs (doc, params and skill flag as the last scan found them) and streams."""
        from experimental.gateway.introspect import base_rpcs

        app = self.app()
        by_class = {
            module["class"]: module for module in (scanned or {}).get("modules", {}).values()
        }
        streams = self.streams()
        found = []
        for info in app.list_modules():
            id_ = instance_id(info.instance_name)
            known = {
                rpc["name"]: rpc for rpc in by_class.get(info.qualified_path, {}).get("rpcs", [])
            }
            rpcs = [
                {
                    **known.get(rpc.name, {"params": None, "return_type": None, "skill": None}),
                    "name": rpc.name,
                    "module": id_,
                    "doc": known.get(rpc.name, {}).get("doc") or rpc.documentation or "",
                }
                for rpc in sorted(info.rpcs, key=lambda rpc: rpc.name)
                if rpc.name in known or rpc.name not in base_rpcs()
            ]
            found.append(
                {
                    "id": id_,
                    "class": info.class_name,
                    "catalogId": info.qualified_path,
                    "namespace": id_.rsplit(":", 1)[0] if ":" in id_ else None,
                    "doc": info.documentation or "",
                    "rpcs": rpcs,
                    "streams": [stream for stream in streams if stream["module"] == id_],
                }
            )
        return sorted(found, key=lambda module: module["id"])

    def module(self, scanned: dict[str, Any] | None, id_: str) -> dict[str, Any]:
        modules = self.modules(scanned)
        found = next((listed for listed in modules if listed["id"] == id_), None)
        if found is None:
            running = ", ".join(listed["id"] for listed in modules) or "none"
            raise ApiError(404, f"no running module {id_} (running: {running})")
        return found

    def streams(self) -> list[dict[str, Any]]:
        return [
            {
                "name": stream.name,
                "type": stream.type_name,
                "direction": stream.direction,
                "channel": stream.channel,
                "module": instance_id(str(stream.module_name or "")),
            }
            for stream in self.app().list_streams()
        ]

    def call(
        self,
        scanned: dict[str, Any] | None,
        id_: str,
        name: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        module = self.module(scanned, id_)
        if name not in {rpc["name"] for rpc in module["rpcs"]}:
            listed = ", ".join(rpc["name"] for rpc in module["rpcs"])
            raise ApiError(404, f"{id_} has no RPC {name} (its RPCs: {listed})")
        # porcelain takes a namespaced instance's key, or an un-namespaced one's class name
        handle = self.app().get_module(id_.replace(":", "/") if ":" in id_ else module["class"])
        try:
            result = getattr(handle, name)(*args, **kwargs)
        except TimeoutError:
            raise ApiError(500, f"{id_}/{name} didn't answer in time")
        except Exception as error:
            return {"ok": False, "result": None, "error": f"{type(error).__name__}: {error}"}
        try:
            encoded = jsonable_encoder(result)
        except Exception:
            encoded = str(result)
        return {"ok": True, "result": encoded, "error": None}

    # TODO(core): Dimos.restart and Dimos.unload take a module class, so with several namespaced copies of one class
    # (robot0:go2connection, robot1:go2connection) they refuse ("Multiple instances"). Doing this cleanly needs them
    # to take an instance name: the coordinator already has restart_module_by_name, but unloading by name has no RPC.
    def restart(self, id_: str) -> dict[str, Any]:
        return self._try(lambda app, cls: app.restart(cls), id_)

    def unload(self, id_: str) -> dict[str, Any]:
        return self._try(lambda app, cls: app.unload(cls), id_)

    def _try(self, action: Callable[[Any, Any], None], id_: str) -> dict[str, Any]:
        module = self.module(None, id_)
        try:
            action(self.app(), load_class(module["catalogId"]))
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}
        return {"ok": True, "error": None}

    def peek(self, name: str, timeout: float) -> dict[str, Any]:
        stream = next((listed for listed in self.streams() if listed["name"] == name), None)
        if stream is None:
            raise ApiError(404, f"no running stream {name}")
        try:
            value = self.app().peek_stream(name, timeout)
        except LookupError as error:
            raise ApiError(404, str(error))
        if value is None:
            raise ApiError(504, f"{name} sent nothing within {timeout} s")
        try:
            encoded = jsonable_encoder(value)
        except Exception:
            encoded = str(value)
        return {
            "name": name,
            "type": stream["type"],
            "value": encoded,
            "ts": getattr(value, "ts", None),
        }
