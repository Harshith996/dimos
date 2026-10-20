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

import pytest

from dimos.agents.annotation import skill
from dimos.core.module import Module
from dimos.navigation import spec


def test_a_planner_without_the_navigation_skills_fails_to_define() -> None:
    with pytest.raises(TypeError, match="go_to, stop_navigation"):

        class _Planner(Module, spec.GlobalPlanner):
            async def go_to(self, x: float, y: float) -> str:
                return ""


def test_a_planner_with_the_navigation_skills_defines() -> None:
    class _Planner(Module, spec.GlobalPlanner):
        @skill
        async def go_to(self, x: float, y: float) -> str:
            """Go."""
            return ""

        @skill
        async def stop_navigation(self) -> str:
            """Stop."""
            return ""
