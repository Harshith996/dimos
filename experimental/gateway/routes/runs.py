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
from typing import Annotated, Any

from fastapi import APIRouter, Path as PathParam
from pydantic import BaseModel, JsonValue

from experimental.gateway import launches, logs, overrides as ov, responses, store
from experimental.gateway.launches import LaunchConfig
from experimental.gateway.skills import SkillError
from experimental.gateway.state import ApiError, GatewayState, State

router = APIRouter()


class SkillCall(BaseModel):
    module: str | None = None
    name: str
    args: dict[str, JsonValue] = {}
    namespace: str | None = None


class LaunchRequest(BaseModel):
    """Exactly one of: a blueprint or a module to launch (with one-off `config`), or a skill to call."""

    blueprint: str | None = None
    module: str | None = None
    skill: SkillCall | None = None
    config: JsonValue = None


class ExecRequest(BaseModel):
    args: list[str]


class RunRequest(BaseModel):
    """Which run (the gateway's own launches have their runId from the start)."""

    runId: str


async def is_own_launch(run_id: str) -> bool:
    launch = await asyncio.to_thread(launches.current_launch)
    return launch is not None and launch["runId"] == run_id


def with_saved(name: str, one_off: ov.LaunchOverrides, args: list[str]) -> LaunchConfig:
    """/config/global, then what's saved for `name`, then this launch's own config."""
    saved = store.blueprint_config(name)
    effective = ov.merge(
        ov.merge(store.global_config_overrides(), saved["global"]), one_off.global_
    )
    try:
        ov.check_global(effective)
    except ValueError as error:
        raise ApiError(400, str(error))
    modules = ov.merge_modules(saved["modules"], one_off.modules)
    return LaunchConfig(effective, modules, one_off, list(args))


async def launch_config(state: State, name: str, config: JsonValue) -> LaunchConfig:
    try:
        one_off = ov.parse(config)
        one_off.global_ = ov.without_hidden(one_off.global_)
        one_off.modules = {m: ov.without_hidden(f) for m, f in one_off.modules.items()}
        ov.check_global(one_off.global_)
    except ValueError as error:
        raise ApiError(400, str(error))
    return with_saved(name, one_off, [])


async def started(state: State, name: str, config: LaunchConfig) -> dict[str, Any]:
    try:
        launch = await asyncio.to_thread(launches.start, state.dimos_dir, name, config)
    except launches.StillRunningError as error:
        raise ApiError(400, str(error))
    except launches.RunError as error:
        raise ApiError(500, str(error))
    state.bus.launch(launch)
    return launch


@router.get("/runs", response_model=responses.Runs)
async def run_list() -> dict[str, Any]:
    return {
        "runs": await asyncio.to_thread(launches.registry_runs),
        "launch": await asyncio.to_thread(launches.current_launch),
    }


@router.post("/runs", response_model=responses.Launch | responses.SkillResult)
async def launch(state: GatewayState, request: LaunchRequest) -> dict[str, Any]:
    """Launch a blueprint or a module with the saved config plus this launch's `config`, or call a skill of the running blueprint."""
    given = [k for k in ("blueprint", "module", "skill") if getattr(request, k) is not None]
    if len(given) != 1:
        raise ApiError(400, "give exactly one of blueprint, module or skill")
    if request.skill is not None:
        call = request.skill
        module = (
            f"{call.namespace}/{call.module}" if call.namespace and call.module else call.module
        )
        try:
            return await asyncio.to_thread(state.skills.call, call.name, call.args, module)
        except SkillError as error:
            raise ApiError(error.status, str(error))
    name = str(request.blueprint or request.module)
    if not name or name.startswith("-"):
        raise ApiError(400, f"bad name: {name}")
    return await started(state, name, await launch_config(state, name, request.config))


@router.post("/runs/exec", response_model=responses.Launch)
async def exec_run(state: GatewayState, request: ExecRequest) -> dict[str, Any]:
    """Run exactly `dimos run <args>` (args[0] is the blueprint or module), without the saved config."""
    if not request.args or request.args[0].startswith("-"):
        raise ApiError(400, "args must start with a blueprint or module name")
    name, *rest = request.args
    return await started(state, name, LaunchConfig(args=rest, exact=True))


@router.get("/runs/skills", response_model=responses.Skills)
async def skill_list(state: GatewayState) -> dict[str, Any]:
    """The running blueprint's skills, live, with each one's JSON schema; call one with POST /runs {skill}."""
    return await asyncio.to_thread(state.skills.listed)


@router.post("/runs/stop", response_model=responses.Stopped)
async def stop(state: GatewayState, request: RunRequest) -> dict[str, Any]:
    """Stop the run `runId`."""
    try:
        return {"output": await launches.stop(request.runId)}
    except launches.NoSuchRunError as error:
        raise ApiError(400, str(error))
    except launches.RunError as error:
        raise ApiError(500, str(error))
    finally:
        state.bus.launch(await asyncio.to_thread(launches.current_launch))


@router.post("/runs/restart", response_model=responses.Launch)
async def restart(state: GatewayState, request: RunRequest) -> dict[str, Any]:
    """Stop the run `runId` and launch it again.

    The gateway's own launch starts again with the config saved now; a run it didn't launch starts again with the
    same command line, directory and environment.
    """
    run_id, ours = request.runId, await is_own_launch(request.runId)
    current = await asyncio.to_thread(launches.current_launch)
    active = ours and current is not None and current["phase"] in launches.ACTIVE
    # a run started outside the gateway (e.g. `dimos run` in a terminal) restarts as a copy of its process
    try:
        run = (
            None
            if ours and not active
            else await asyncio.to_thread(launches.run_to_relaunch, run_id)
        )
    except launches.RunError as error:
        raise ApiError(400, str(error))
    if run is not None:
        try:
            command, cwd, env = await asyncio.to_thread(launches.snapshot, int(run["pid"]))
            await launches.stop(run_id)
            started = await asyncio.to_thread(
                launches.relaunch, str(run["blueprint"]), command, cwd, env
            )
        except launches.RunError as error:
            raise ApiError(500, str(error))
        state.bus.launch(started)
        return started
    last = launches.last_launch_args()
    if last is None:
        raise ApiError(400, "the dimos gateway hasn't launched anything yet")
    blueprint, previous = last
    # a /runs/exec launch restarts exactly as typed
    config = (
        LaunchConfig(args=previous.args, exact=True)
        if previous.exact
        else with_saved(blueprint, previous.one_off, [])
    )
    try:
        if active:
            await launches.stop(run_id)
        started = await asyncio.to_thread(launches.start, state.dimos_dir, blueprint, config)
    except launches.RunError as error:
        raise ApiError(500, str(error))
    state.bus.launch(started)
    return started


def current_run_id() -> str:
    launch = launches.current_launch()
    if launch is not None and launch["phase"] in launches.ACTIVE:
        return str(launch["runId"])
    running = launches.registry_runs()
    return str(running[0]["run_id"]) if running else "latest"


async def read_log(
    state: State,
    run_id: str,
    after: int | None,
    level: str | None,
    q: str | None,
    limit: int | None,
) -> dict[str, Any]:
    file = await asyncio.to_thread(logs.find, state.dimos_dir, run_id)
    if file is None:
        raise ApiError(
            404, "no run has a log yet" if run_id == "latest" else f"no log for run {run_id}"
        )
    return await asyncio.to_thread(
        logs.read, file, run_id, after, limit or 1000, q or None, level or None
    )


@router.get("/runs/logs", response_model=responses.RunLog)
async def logs_now(
    state: GatewayState,
    after: int | None = None,
    level: str | None = None,
    q: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """The running run's log, or the latest run's when nothing runs."""
    run_id = await asyncio.to_thread(current_run_id)
    return await read_log(state, run_id, after, level, q, limit)


@router.get("/runs/log_hist", response_model=responses.LogHistory)
async def log_history(state: GatewayState) -> dict[str, Any]:
    """Every run with a log, newest first."""
    return {"runs": await asyncio.to_thread(logs.history, state.dimos_dir)}


@router.get("/runs/log_hist/{runId}", response_model=responses.RunLog)
async def past_log(
    state: GatewayState,
    run_id: Annotated[str, PathParam(alias="runId")],
    after: int | None = None,
    level: str | None = None,
    q: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    return await read_log(state, run_id, after, level, q, limit)


@router.get("/runs/stats", response_model=responses.ProcessStats)
async def process_stats(state: GatewayState) -> dict[str, Any]:
    """dtop for the running blueprint: cpu, memory, threads, fds and io per process, with a minute of cpu history."""
    return await asyncio.to_thread(state.process_stats.snapshot)
