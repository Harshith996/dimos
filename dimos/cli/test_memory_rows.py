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

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from dimos.cli.dimos import main as app


class _Vec:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z

    def __sub__(self, other: _Vec) -> _Vec:
        return _Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def length(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5


class _OdomStream:
    def __init__(self, observations: list[SimpleNamespace]) -> None:
        self._observations = observations

    def range_time(self, start: float, stop: float) -> _OdomStream:
        t0 = self._observations[0].ts
        return _OdomStream([o for o in self._observations if t0 + start <= o.ts < t0 + stop])

    def to_list(self) -> list[SimpleNamespace]:
        return list(self._observations)

    def __iter__(self):
        return iter(self._observations)


def _patch_odom(mocker: MockerFixture) -> None:
    def obs(ts: float, x: float, y: float = 0.0) -> SimpleNamespace:
        return SimpleNamespace(ts=ts, data=SimpleNamespace(position=_Vec(x, y)))

    stream = _OdomStream([obs(1000.0, 0.0), obs(1010.0, 3.0), obs(1030.0, 3.0, 4.0)])
    store = mocker.Mock()
    store.streams.odom = stream
    mocker.patch("dimos.evals.suites.lib.generate.open_dataset", return_value=store)


@pytest.mark.parametrize("window", ["10:x", "10:5", "nope"])
def test_memory_rows_cli_rejects_bad_windows(
    window: str, tmp_path: Path, mocker: MockerFixture
) -> None:
    _patch_odom(mocker)
    result = CliRunner().invoke(
        app,
        ["evals", "memory-rows", "desk", "-o", str(tmp_path / "rows.json"), "--window", window],
    )
    assert result.exit_code != 0


def test_memory_rows_cli_writes_deterministic_json(tmp_path: Path, mocker: MockerFixture) -> None:
    _patch_odom(mocker)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    for path in (first, second):
        result = CliRunner().invoke(
            app,
            [
                "evals",
                "memory-rows",
                "desk",
                "--window",
                "0:20",
                "--window",
                "0:31",
                "-o",
                str(path),
            ],
        )
        assert result.exit_code == 0, result.stdout
    assert first.read_bytes() == second.read_bytes()
    assert b"desk_disp_0_20" in first.read_bytes()
    assert b"desk_path_0_31" in first.read_bytes()
