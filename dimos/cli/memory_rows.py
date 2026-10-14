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

"""CLI for emitting odom memory eval rows as JSON."""

from __future__ import annotations

from pathlib import Path

import typer

from dimos.evals.suites.lib.generate import memory_rows, write_rows


def _parse_window(item: str) -> tuple[float, float]:
    start_text, sep, stop_text = item.partition(":")
    if not sep:
        raise typer.BadParameter(f"window must be START:STOP, got {item!r}")
    try:
        start, stop = float(start_text), float(stop_text)
    except ValueError as exc:
        raise typer.BadParameter(f"window bounds must be numbers, got {item!r}") from exc
    if stop <= start:
        raise typer.BadParameter(f"window stop must be greater than start, got {item!r}")
    return start, stop


def memory_rows_cmd(
    dataset: str = typer.Argument(help="Recording name or path (e.g. go2_short)"),
    output: Path = typer.Option(..., "-o", "--output", help="Output JSON path"),
    window: list[str] = typer.Option(
        ...,
        "--window",
        help="Relative-second window as START:STOP; repeat for multiple windows",
    ),
) -> None:
    """Emit displacement and path-length memory eval rows for a recording."""
    windows = [_parse_window(item) for item in window]
    rows = memory_rows(dataset, windows)
    write_rows(output, rows)
    typer.echo(f"wrote {len(rows)} rows to {output}")
