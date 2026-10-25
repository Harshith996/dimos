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

"""Official LeRobot ACT on a frozen DimOS correction snapshot, in isolation.

Training reads a standard LeRobot dataset. Each accepted human segment is an
episode, so the dataset API pads action chunks at correction boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import time
from typing import Any

from dimos_lerobot.dataprep import write as write_dataset
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train as lerobot_train
from lerobot.utils.logging_utils import MetricsTracker
import torch

from dimos.imitation.dataprep.schema import OutputConfig
from dimos.imitation.loop.checkpoint import read_checkpoint
from dimos.imitation.loop.recording import (
    CONTRACT_KEYS,
    collection_profile,
    digest,
    snapshot_samples,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dataset(snapshot: Path, manifest: dict[str, Any]) -> Path:
    dataset = snapshot / "dataset"
    index_path = snapshot / "dataset.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["selection_sha256"] != digest(snapshot / "manifest.json"):
            raise ValueError("frozen dataset selection was modified")
        if set(index["files"]) != {
            str(path.relative_to(dataset)) for path in dataset.rglob("*") if path.is_file()
        }:
            raise ValueError("frozen dataset files were modified")
        for relative, expected in index["files"].items():
            if digest(dataset / relative) != expected:
                raise ValueError("frozen dataset was modified")
        return dataset
    if dataset.exists():
        raise FileExistsError("dataset conversion is incomplete; create a fresh snapshot")
    profile = collection_profile(manifest)
    features = {
        key: feature.model_dump()
        for key, feature in {**profile.observations, **profile.actions}.items()
    }
    features["complementary_info.is_filled"] = {
        "dtype": "bool",
        "shape": (1,),
        "names": ["is_filled"],
    }
    write_dataset(
        snapshot_samples(snapshot),
        OutputConfig(
            path=dataset,
            metadata={
                "repo_id": "local/openyam",
                "robot_type": manifest["robot_type"],
                "fps": manifest["fps"],
                "feature_schema": features,
            },
        ),
    )
    files = {
        str(path.relative_to(dataset)): digest(path)
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    }
    index_path.write_text(
        json.dumps(
            {"files": files, "selection_sha256": digest(snapshot / "manifest.json")}, indent=2
        )
        + "\n"
    )
    return dataset


class TrainingProgress(logging.Filter):
    """Mirror official trainer metrics into the Desktop job artifact."""

    def __init__(self, output: Path, steps: int) -> None:
        super().__init__()
        self.output, self.steps = output, steps
        self.metrics: list[dict[str, int | float]] = []

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.msg, MetricsTracker):
            return True
        metrics = record.msg.to_dict()
        if not math.isfinite(metrics["loss"]):
            raise RuntimeError("LeRobot reported a non-finite training loss")
        self.metrics.append(metrics)
        partial = self.output / "progress.json.partial"
        partial.write_text(json.dumps(metrics))
        partial.replace(self.output / "progress.json")
        print(f"ACT step {metrics['steps']}/{self.steps}: loss={metrics['loss']:.5f}", flush=True)
        return True


def train(
    snapshot: Path, output: Path, *, steps: int, device: str, smoke: bool, initial: Path | None
) -> None:
    if steps <= 0:
        raise ValueError("steps must be positive")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["schema_version"] != 2:
        raise ValueError("training requires the artifact-backed LeRobot contract")
    if output.exists():
        raise FileExistsError(f"checkpoint output already exists: {output}")
    output.mkdir(parents=True)
    job: dict[str, Any] = dict(
        status="running",
        started_at=time.time(),
        steps=steps,
        device=device,
        snapshot=str(snapshot),
        snapshot_sha256=sha256(snapshot / "manifest.json"),
        initial=str(initial) if initial else None,
        smoke=smoke,
        seed=0,
    )
    job_path = output / "job.json"
    job_path.write_text(json.dumps(job, indent=2))
    try:
        torch.set_num_threads(2)
        dataset_path = prepare_dataset(snapshot, manifest)
        job["dataset_index_sha256"] = digest(snapshot / "dataset.json")
        height, width, channels = manifest["image_shape"]
        dof = len(manifest["joint_names"])
        if initial is None:
            config = ACTConfig(
                input_features={
                    "observation.state": PolicyFeature(FeatureType.STATE, (dof,)),
                    "observation.images.wrist": PolicyFeature(
                        FeatureType.VISUAL, (channels, height, width)
                    ),
                },
                output_features={"action": PolicyFeature(FeatureType.ACTION, (dof,))},
                device=device,
                push_to_hub=False,
                chunk_size=8 if smoke else 50,
                n_action_steps=4 if smoke else 10,
                dim_model=64 if smoke else 512,
                n_heads=4 if smoke else 8,
                dim_feedforward=128 if smoke else 3200,
                n_encoder_layers=1 if smoke else 4,
                n_vae_encoder_layers=1 if smoke else 4,
                pretrained_backbone_weights=None if smoke else "ResNet18_Weights.IMAGENET1K_V1",
            )
        else:
            original = read_checkpoint(initial)
            smoke = bool(original["smoke"])
            job["smoke"] = smoke
            for key in CONTRACT_KEYS:
                if original[key] != manifest[key]:
                    raise ValueError(f"fine-tuning contract differs: {key}")
            loaded_config = PreTrainedConfig.from_pretrained(initial)
            if not isinstance(loaded_config, ACTConfig):
                raise ValueError("initial checkpoint must be ACT")
            config = loaded_config
            config.device = device
            config.pretrained_path = initial
            config.push_to_hub = False
            job["parent_model_sha256"] = original["model_sha256"]
        dataset_info = json.loads((dataset_path / "meta" / "info.json").read_text())
        job["dataset_frames"] = dataset_info["total_frames"]
        job["dataset_episodes"] = dataset_info["total_episodes"]
        job["trainer"] = "lerobot.scripts.lerobot_train.train"
        job["normalization"] = "official trainer: statistics from the selected training dataset"
        training = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id="local/openyam", root=str(dataset_path), video_backend="pyav"
            ),
            policy=config,
            output_dir=output / "training",
            job_name="openyam-act",
            seed=0,
            num_workers=0,
            batch_size=min(8, job["dataset_frames"]),
            steps=steps,
            env_eval_freq=0,
            log_freq=1 if steps < 100 else 100,
            save_freq=steps,
        )
        progress = TrainingProgress(output, steps)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addFilter(progress)
        try:
            lerobot_train(training)
        finally:
            logger.removeFilter(progress)
        if not progress.metrics:
            raise RuntimeError("LeRobot completed without reporting training metrics")
        saved = output / "training" / "checkpoints" / "last" / "pretrained_model"
        if not (saved / "model.safetensors").is_file():
            raise RuntimeError("LeRobot did not publish its final checkpoint")
        for path in saved.iterdir():
            if path.is_file():
                shutil.copyfile(path, output / path.name)
        (output / "snapshot.json").write_text(json.dumps(manifest, indent=2))
        (output / "metrics.json").write_text(json.dumps(progress.metrics, indent=2))
        job.update(
            status="complete", finished_at=time.time(), last_loss=progress.metrics[-1]["loss"]
        )
        contract = {key: manifest[key] for key in CONTRACT_KEYS}
        (output / "loop.json").write_text(
            json.dumps(
                {
                    **contract,
                    "model_sha256": sha256(output / "model.safetensors"),
                    "files": {
                        path.name: digest(path)
                        for path in sorted(output.iterdir())
                        if path.is_file() and path.name != "job.json"
                    },
                    "n_action_steps": config.n_action_steps,
                    "smoke": smoke,
                    "training_job": job,
                },
                indent=2,
            )
        )
    except BaseException as exc:
        job.update(status="failed", error=str(exc), finished_at=time.time())
        raise
    finally:
        job_path.write_text(json.dumps(job, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--initial", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        if args.output is None:
            parser.error("training needs an output directory")
        train(
            args.input,
            args.output,
            steps=args.steps,
            device=args.device,
            smoke=args.smoke,
            initial=args.initial,
        )


if __name__ == "__main__":
    main()
