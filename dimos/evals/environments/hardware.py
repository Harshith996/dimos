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

"""A real robot driven through ``dimos run``, with an operator in the loop.

The operator readies the scene before each case and confirms it; the code never
moves the robot to reset it. When a case ends the robot is halted in place, then
dimos stops the way ``dimos stop`` does, so each module's stop() runs (a Go2 sits).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field

from dimos.agents.mcp.mcp_adapter import McpAdapter
from dimos.evals.environments.live import LiveEnvironment, LiveEnvironmentConfig
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dimos.e2e_tests.dimos_cli_call import DimosCliCall
    from dimos.evals.types import RunningEnvironment
    from dimos.memory.store.base import Store
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

logger = setup_logger()


class HardwareEnvironmentConfig(LiveEnvironmentConfig):
    task: str
    """The case's instruction, shown to the operator before the robot starts."""
    stop_tool: str | None = None
    """MCP tool that halts motion before the robot disconnects, e.g. ``cancel``."""
    module_env: dict[str, str] = Field(default_factory=dict)
    at_rest_rad_s: float = 0.02
    launch_timeout_s: float = 300.0
    graceful_stop: bool = True


class HardwareEnvironment(LiveEnvironment):
    """A real robot; the operator resets the scene and confirms it is ready."""

    config: HardwareEnvironmentConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mcp_url: str | None = None

    def start(self, modules: Sequence[str]) -> RunningEnvironment:
        answer = input(
            f"\nNext eval task: {self.config.task}\n"
            "Is the scene reset and the robot ready, with the area clear? [y/N] "
        )
        if not answer.strip().lower().startswith("y"):
            raise RuntimeError("operator did not confirm the robot is ready")
        running = super().start(modules)
        self._mcp_url = running.mcp_url
        return running

    def configure_launch(self, proc: DimosCliCall) -> None:
        proc.extra_env.update(self.config.module_env)

    def latest_pose(self, recording: Store) -> PoseStamped:
        if "odom" not in recording.streams:
            raise LookupError("No odometry recorded")
        return cast("PoseStamped", recording.streams.odom.last().data)

    def settle(self, budget_s: float) -> None:
        """An arm is at rest once every joint is slower than ``at_rest_rad_s`` for ``at_rest_s``."""
        recording = self._recording
        if recording is None or "coordinator_joint_state" not in recording.streams:
            super().settle(budget_s)
            return
        rest_since: float | None = None
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            try:
                state = recording.streams.coordinator_joint_state.last().data
            except LookupError:
                return
            now = time.monotonic()
            if any(abs(v) > self.config.at_rest_rad_s for v in state.velocity):
                rest_since = None
            elif rest_since is None:
                rest_since = now
            elif now - rest_since >= self.config.at_rest_s:
                return
            time.sleep(self.config.settle_poll_s)

    def stop(self) -> None:
        try:
            if self.config.stop_tool is not None and self._mcp_url is not None:
                try:
                    McpAdapter(self._mcp_url).call_tool(self.config.stop_tool)
                except Exception as e:
                    logger.warning("could not halt the robot", tool=self.config.stop_tool, error=e)
        finally:
            self._mcp_url = None
            super().stop()
