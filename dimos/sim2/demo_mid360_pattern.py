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

"""Plot one scan in laser coordinates, without physics, a mount or a viewer."""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from dimos.sim2.sensors.lidar.models.mid360 import Mid360


def angular_points(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("xyz must be a nonempty (N, 3) sensor-frame array")
    if not np.isfinite(xyz).all() or np.any(np.linalg.norm(xyz, axis=1) == 0):
        raise ValueError("xyz must contain finite, nonzero directions or returns")
    result: NDArray[np.float64] = np.rad2deg(
        np.column_stack(
            (
                np.arctan2(xyz[:, 1], xyz[:, 0]),
                np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])),
            )
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", type=float, default=0.0, help="Simulation acquisition time, seconds"
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/mid360-100ms.png"))
    parser.add_argument("--comparison", type=Path, help="NPZ: xyz (N,3), offset_time_ns (N,)")
    parser.add_argument("--label", help="Comparison provenance, e.g. G1 raw returns, 2026-10-06")
    args = parser.parse_args()
    if not np.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if bool(args.comparison) != bool(args.label):
        parser.error("--comparison and --label must be supplied together")

    rays = Mid360().scan(args.start, 0.1)
    scans = [("sim2: continuous Mid360 firing directions", rays.directions, rays.offsets * 1e3)]
    if args.comparison:
        with np.load(args.comparison, allow_pickle=False) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float64)
            offsets = np.asarray(data["offset_time_ns"], dtype=np.float64) / 1e6
        if offsets.shape != (len(xyz),) or not np.isfinite(offsets).all():
            raise ValueError("offset_time_ns must have one finite timestamp per point")
        if np.any(offsets < 0) or np.any(offsets >= 100):
            raise ValueError("comparison must be one 100 ms scan, with offsets in [0, 100 ms)")
        scans.append((args.label, xyz, offsets))

    figure = Figure(figsize=(14, 2.3 + 4 * len(scans)), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(scans), 2, squeeze=False, width_ratios=(1.6, 1))
    figure.suptitle(
        "MID360 / 100 ms at 10 Hz\n"
        f"20,000 firings / 200,000 per second / simulation time {args.start:g} s",
        fontsize=17,
    )
    summaries = []
    for row, (label, xyz, offsets) in enumerate(scans):
        angles = angular_points(xyz)
        for column, (bounds, title) in enumerate(
            (((-180, 180), label), ((-45, 45), "Front sector, same points"))
        ):
            axis = axes[row, column]
            points = axis.scatter(
                *angles.T, c=offsets, cmap="viridis", vmin=0, vmax=100, s=1.5, linewidths=0
            )
            axis.set(title=title, xlim=bounds, ylim=(-12, 62))
            axis.set_xlabel("Azimuth (degrees)")
            axis.set_ylabel("Elevation above laser XY plane (degrees)")
            axis.grid(alpha=0.2)
            axis.set_axisbelow(True)
        summaries.append(
            {
                "label": label,
                "points": len(xyz),
                "elevation_degrees": [float(angles[:, 1].min()), float(angles[:, 1].max())],
                "offset_ms": [float(offsets.min()), float(offsets.max())],
            }
        )
    figure.colorbar(points, ax=axes.ravel().tolist(), label="Acquisition time within scan (ms)")
    figure.supxlabel(
        "Laser coordinates: +X forward, +Y left, +Z up. No robot mount rotation.\n"
        "Firing directions are not observed returns; real clouds omit misses and occluded rays.\n"
        "Sequence phases are not fitted. This plot alone does not establish hardware fidelity.",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(json.dumps({"output": str(args.output), "scans": summaries}, indent=2))


if __name__ == "__main__":
    main()
