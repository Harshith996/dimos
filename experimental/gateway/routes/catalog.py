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

import asyncio
from typing import Any

from fastapi import APIRouter

from experimental.gateway import catalog, introspect, launches, responses
from experimental.gateway.state import ApiError, GatewayState, State

router = APIRouter(prefix="/catalog")
# imports dimos in a child process: seconds
SLOW = {"x-slow": True}
MAX_ERRORS = 200


def check_name(name: str) -> None:
    if not name or name.startswith("-") or any(c.isspace() for c in name):
        raise ApiError(400, f"bad name: {name}")


def picked(answer: dict[str, Any], fields: str | None) -> dict[str, Any]:
    try:
        return catalog.pick(answer, fields)
    except catalog.FieldError as error:
        raise ApiError(400, str(error))


async def listed(state: State) -> list[dict[str, Any]]:
    return state.listed or (await asyncio.to_thread(introspect.blueprint_list))["blueprints"]


async def blueprint_list(state: State) -> list[dict[str, Any]]:
    entries = await listed(state)
    notes = await asyncio.to_thread(
        catalog.notes_for, state.dimos_dir, [e["name"] for e in entries]
    )
    return catalog.blueprints(entries, state.scanner.cache(), notes)


async def find_blueprint(state: State, name: str) -> dict[str, Any]:
    """A blueprint the gateway's checkout has, or one running from another checkout."""
    check_name(name)
    found = next((b for b in await blueprint_list(state) if b["name"] == name), None)
    if found is None and await asyncio.to_thread(running_source, state, name) is not None:
        found = catalog.blueprints([{"name": name}], None, {"robots": {}})[0]
    if found is None:
        raise ApiError(404, f"no blueprint {name}")
    return found


async def has_locally(state: State, name: str) -> bool:
    return any(entry["name"] == name for entry in await listed(state))


def running_source(state: State, name: str) -> launches.RunSource | None:
    """Where the live run of this blueprint came from, when that isn't the gateway's checkout."""
    run = launches.run_of(name)
    source = launches.run_source(str(run["run_id"])) if run else None
    if source is None or source.checkout.resolve() == state.dimos_dir.resolve():
        return None
    return source


async def from_run_checkout(state: State, args: list[str], local: bool) -> dict[str, Any]:
    """Introspect from the checkout the live run came from (falling back to the gateway's when it has the blueprint)."""
    source = await asyncio.to_thread(running_source, state, args[1])
    if source is None:
        return {**await state.introspected(args), "checkout": None, "checkoutError": None}
    try:
        found = await state.introspected(args, source=source)
        return {**found, "checkout": str(source.checkout), "checkoutError": None}
    except ApiError as error:
        failed = f"reading it from {source.checkout} failed: {error}"
        if not local:
            raise ApiError(500, failed)
        return {**await state.introspected(args), "checkout": None, "checkoutError": failed}


async def scan(state: State) -> dict[str, Any]:
    return await asyncio.to_thread(state.scanner.snapshot)


@router.get("", response_model=responses.Catalog)
async def everything(
    state: GatewayState,
    module: str | None = None,
    blueprint: str | None = None,
    fields: str | None = None,
) -> dict[str, Any]:
    """Every blueprint, module, RPC and robot from the last scan, with how stale it is.

    ?module= (a name or class) keeps that module, its RPCs and the blueprints using it; ?blueprint= keeps that
    blueprint, its modules and their RPCs.
    """
    entries = await listed(state)
    notes = await asyncio.to_thread(
        catalog.notes_for, state.dimos_dir, [e["name"] for e in entries]
    )
    cache = state.scanner.cache()
    blueprints, modules = catalog.blueprints(entries, cache, notes), catalog.modules(cache)
    if blueprint is not None:
        blueprints = [b for b in blueprints if b["name"] == blueprint]
        names = {name for b in blueprints for name in b["modules"]}
        modules = [m for m in modules if m["name"] in names]
    if module is not None:
        modules = [m for m in modules if module in (m["name"], m["class"])]
        names = {m["name"] for m in modules}
        blueprints = [b for b in blueprints if names & set(b["modules"])]
    kept = {m["name"] for m in modules}
    answer = {
        "blueprints": blueprints,
        "modules": modules,
        "rpcs": [r for r in catalog.rpcs(cache) if r["module"] in kept],
        "robots": catalog.robot_list(notes),
        "types": notes.get("types", {}),
        "errors": catalog.errors(cache, MAX_ERRORS),
        "scan": await scan(state),
    }
    return picked(answer, fields)


@router.post("/scan", status_code=202, response_model=responses.ScanStarted)
async def scan_all(state: GatewayState) -> dict[str, Any]:
    """Scan every blueprint and module again (or join the running scan); the cache changes when it finishes. Follow it at /jobs/{job}."""
    return {"job": state.scanner.start("requested").id}


@router.post("/scan/blueprints/{name}", status_code=202, response_model=responses.ScanStarted)
async def scan_blueprint(state: GatewayState, name: str) -> dict[str, Any]:
    """Scan just this blueprint and its modules, and update their entries in the cache (or join the running full scan)."""
    await find_blueprint(state, name)
    return {"job": state.scanner.start_one(name).id}


@router.get("/blueprints", response_model=responses.BlueprintList)
async def blueprints(state: GatewayState, fields: str | None = None) -> dict[str, Any]:
    """Every blueprint, including external ones and ones that don't import, with what the last scan found."""
    return picked({"blueprints": await blueprint_list(state), "scan": await scan(state)}, fields)


@router.get("/blueprints/{name}", response_model=responses.BlueprintDetail, openapi_extra=SLOW)
async def blueprint(state: GatewayState, name: str, fields: str | None = None) -> dict[str, Any]:
    """A blueprint, plus its file and line and its modules' streams and RPCs.

    While it runs from another checkout (a branch started in a terminal), it's read from that one, so the modules,
    streams and remappings are the live ones (and a blueprint only that checkout has is found); `checkout` names it,
    and `checkoutError` says why it couldn't be read from there.
    """
    found = await find_blueprint(state, name)
    wanted = {f.strip().split(".")[0] for f in fields.split(",")} if fields else None
    if wanted is None or wanted & {"file", "line", "modules", "checkout", "checkoutError"}:
        found.update(
            await from_run_checkout(state, ["blueprint", name], await has_locally(state, name))
        )
        found.pop("config", None)
    return picked(found, fields)


@router.get(
    "/blueprints/{name}/config", response_model=responses.BlueprintSchema, openapi_extra=SLOW
)
async def blueprint_config(state: GatewayState, name: str) -> Any:
    """The blueprint's config schema: GlobalConfig's fields and each module's, with types, defaults and choices.

    Read from the checkout of its live run, like GET /catalog/blueprints/{name}.
    """
    await find_blueprint(state, name)
    found = await from_run_checkout(state, ["config", name], await has_locally(state, name))
    found.pop("checkout")
    found.pop("checkoutError")
    return found


async def find_module(state: State, name: str) -> dict[str, Any]:
    check_name(name)
    found = next(
        (m for m in catalog.modules(state.scanner.cache()) if name in (m["name"], m["class"])), None
    )
    if found is None:
        raise ApiError(404, f"no module {name} in the last scan")
    return found


async def importable_module(state: State, name: str) -> dict[str, Any]:
    found = await find_module(state, name)
    if "<locals>" in found["class"]:
        raise ApiError(
            404, f"{found['class']} is defined inside a function, so it can't be loaded on its own"
        )
    return found


@router.get("/modules", response_model=responses.ModuleList)
async def modules(state: GatewayState, fields: str | None = None) -> dict[str, Any]:
    return picked(
        {"modules": catalog.modules(state.scanner.cache()), "scan": await scan(state)}, fields
    )


@router.get("/modules/{name}", response_model=responses.ModuleDetail, openapi_extra=SLOW)
async def module(state: GatewayState, name: str) -> dict[str, Any]:
    """A module, the blueprints using it, and its RPCs (`name` is its registry name or class)."""
    found = await importable_module(state, name)
    detail = await state.introspected(["module", found["class"]])
    return {**found, **{k: detail[k] for k in ("rpcs", "file", "line")}}


@router.get("/modules/{name}/config", response_model=responses.ModuleSchema, openapi_extra=SLOW)
async def module_config(state: GatewayState, name: str) -> Any:
    """The module's config fields with the code's defaults; saved values are per blueprint (/config/blueprints/{name})."""
    found = await importable_module(state, name)
    return await state.introspected(["module_config", found["class"]])


@router.get("/rpcs", response_model=responses.CatalogRpcs)
async def rpcs(
    state: GatewayState, namespace: str | None = None, fields: str | None = None
) -> dict[str, Any]:
    """Every module's RPCs the last scan found (`skill` says which are also skills); ?namespace= names them as that namespace would. The running blueprint's are at /runs/rpcs."""
    found = catalog.rpcs(state.scanner.cache(), namespace)
    return picked({"rpcs": found, "scan": await scan(state)}, fields)


@router.get("/rpcs/{module}/{name}", response_model=responses.CatalogRpc)
async def rpc(
    state: GatewayState, module: str, name: str, namespace: str | None = None
) -> dict[str, Any]:
    wanted = f"{namespace}/{module}" if namespace else module
    for found in catalog.rpcs(state.scanner.cache(), namespace):
        if found["name"] == name and found["module"] in (module, wanted):
            return found
    raise ApiError(404, f"no RPC {module}/{name} in the last scan")


@router.get("/robots", response_model=responses.Robots)
async def robots(state: GatewayState, fields: str | None = None) -> dict[str, Any]:
    """Every robot dimos supports, by type, with its blueprints by name."""
    entries = await listed(state)
    notes = await asyncio.to_thread(
        catalog.notes_for, state.dimos_dir, [e["name"] for e in entries]
    )
    return picked({"types": notes.get("types", {}), "robots": catalog.robot_list(notes)}, fields)


@router.get("/msgs/{name}", response_model=responses.MessageType)
async def message_type(name: str) -> dict[str, Any]:
    """A message type's definition, by the type name topics use (e.g. sensor_msgs.Image). Not filled in yet."""
    check_name(name)
    return {"name": name, "definition": None}
