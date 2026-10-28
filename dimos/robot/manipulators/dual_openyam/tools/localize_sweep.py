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

"""Find the localize thresholds for the overhead camera from a recording, offline.

Run: python -m dimos.robot.manipulators.dual_openyam.tools.localize_sweep
         "label" ["label" ...] --dataset <memory.db> [--from <s>] [--duration <s>]
         [--detector owlv2|gdino|omdet] [--floors 0.25,0.15,0.10,0.07] [--accepts 0.40,0.25,0.15,0.10]

The window is embedded once. A probe pass with both thresholds at the floor of the grid
reports the best score the detector gives each label at all. Then every (candidate_floor,
accept_score) pair of the grid localizes all labels in one pass, with min_views 1 for the
fixed camera, and the table shows which labels each pair found, with their scores, and how
many instances it made of each (more than one of a label that is alone on the table is a
false split). Pick the loosest pair that finds every label once and put it in
DUAL_OPENYAM_LOCALIZE_PARAMS (blueprints/grasp.py).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from dimos.memory.store.sqlite import SqliteStore
from dimos.perception.localize.dandetect import DanDetector
from dimos.perception.localize.localize import embed_index
from dimos.perception.localize.rig import Rig, window_tf
from dimos.robot.manipulators.dual_openyam.localize_registration import localize_params


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("labels", nargs="+", help="object labels, the names the agent will use")
    parser.add_argument("--dataset", type=Path, required=True, help="memory recording database")
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="window start (s)")
    parser.add_argument("--duration", type=float, default=None, help="window length (s)")
    parser.add_argument("--embedder", default="siglip")
    parser.add_argument("--detector", default="owlv2")
    parser.add_argument("--segmenter", default="edgetam")
    parser.add_argument("--floors", default="0.25,0.15,0.10,0.07", help="candidate_floor values")
    parser.add_argument("--accepts", default="0.40,0.25,0.15,0.10", help="accept_score values")
    args = parser.parse_args()

    store = SqliteStore(path=args.dataset)
    rig = Rig.from_store(store)
    lo, hi = rig.time_range()
    after = lo + args.start
    before = after + args.duration if args.duration is not None else hi
    rig.tf = window_tf(store, after, before)
    floors, accepts = _floats(args.floors), _floats(args.accepts)
    base = localize_params()

    with DanDetector(args.embedder, args.detector, args.segmenter) as dan:
        index = embed_index(rig, dan.embedder, after, before)
        print(f"window {args.start:.0f}s + {before - after:.0f}s, {index.count()} frames embedded")

        probe = replace(base, candidate_floor=min(floors), accept_score=min(accepts))
        print(f"\nprobe: candidate_floor {probe.candidate_floor} accept_score {probe.accept_score}")
        for label, hits in zip(
            args.labels, dan.localize(args.labels, index=index, rig=rig, params=probe), strict=True
        ):
            best = max((h.semantic_score for h in hits), default=None)
            print(
                f"  {label:22s} best score {best if best is None else f'{best:.2f}'}  instances {len(hits)}"
            )

        print("\ngrid: candidate_floor / accept_score -> labels found (score x instances)")
        for floor in floors:
            for accept in accepts:
                if accept < floor:
                    continue
                params = replace(base, candidate_floor=floor, accept_score=accept)
                results = dan.localize(args.labels, index=index, rig=rig, params=params)
                found = [
                    f"{label} ({max(h.semantic_score for h in hits):.2f} x{len(hits)})"
                    for label, hits in zip(args.labels, results, strict=True)
                    if hits
                ]
                missing = [
                    label for label, hits in zip(args.labels, results, strict=True) if not hits
                ]
                print(
                    f"  {floor:.2f} / {accept:.2f}: {len(found)}/{len(args.labels)}  "
                    f"found {', '.join(found) or 'none'}  missing {', '.join(missing) or 'none'}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
