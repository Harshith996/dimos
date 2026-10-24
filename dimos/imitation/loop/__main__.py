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

"""Offline preparation and training; operate the robot through DimOS Desktop."""

import argparse
from pathlib import Path

from dimos.imitation.loop.act import train
from dimos.imitation.loop.recording import export_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    snapshot = sub.add_parser("snapshot", help="Freeze accepted human training segments")
    snapshot.add_argument("root", type=Path)
    snapshot.add_argument("output", type=Path)
    training = sub.add_parser("train", help="Convert to LeRobot and train ACT in isolation")
    training.add_argument("snapshot", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, default=2000)
    training.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    training.add_argument("--initial", type=Path)
    training.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "snapshot":
        export_snapshot(args.root, args.output)
    else:
        train(
            args.snapshot,
            args.output,
            steps=args.steps,
            device=args.device,
            initial=args.initial,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
