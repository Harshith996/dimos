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
import base64
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
from dimos_lerobot.dataprep import write as write_dataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
import numpy as np
import torch
from torch.utils.data import DataLoader

from dimos.imitation.dataprep.schema import OutputConfig
from dimos.imitation.loop.act import read_checkpoint
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
        torch.manual_seed(0)
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
            policy = ACTPolicy(config)
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
            policy = ACTPolicy.from_pretrained(initial, config=config)
            job["parent_model_sha256"] = original["model_sha256"]
        dataset = LeRobotDataset(
            repo_id="local/openyam",
            root=dataset_path,
            delta_timestamps={
                "action": [index / manifest["fps"] for index in range(config.chunk_size)]
            },
            video_backend="pyav",
        )
        job["dataset_frames"] = len(dataset)
        job["dataset_episodes"] = dataset.num_episodes
        if initial is None:
            pre, post = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
        else:
            pre, post = make_pre_post_processors(
                config,
                pretrained_path=str(initial),
                preprocessor_overrides={"device_processor": {"device": device}},
            )
        policy.to(device).train()
        optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
        loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True, num_workers=0)
        batches = iter(loader)
        losses = []
        for step in range(steps):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            loss, _ = policy.forward(pre(batch))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite ACT loss")
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
                print(f"ACT step {step + 1}/{steps}: loss={value:.5f}", flush=True)
        policy.save_pretrained(output)
        pre.save_pretrained(output)
        post.save_pretrained(output)
        (output / "snapshot.json").write_text(json.dumps(manifest, indent=2))
        (output / "losses.json").write_text(json.dumps(losses))
        job.update(status="complete", finished_at=time.time(), last_loss=losses[-1])
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


def serve(checkpoint: Path, device: str) -> None:
    read_checkpoint(checkpoint)
    wire = sys.stdout
    with redirect_stdout(sys.stderr):
        torch.set_num_threads(2)
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, ACTConfig):
            raise ValueError("checkpoint must be ACT")
        config.device = device
        policy = ACTPolicy.from_pretrained(checkpoint, config=config).to(device).eval()
        pre, post = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    wire.write(json.dumps({"ready": True}) + "\n")
    wire.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            raw = cv2.imdecode(
                np.frombuffer(base64.b64decode(request["image_png"]), np.uint8), cv2.IMREAD_COLOR
            )
            if raw is None:
                raise ValueError("invalid RGB observation")
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            with torch.inference_mode(), redirect_stdout(sys.stderr):
                batch = {
                    "observation.state": torch.tensor(request["state"], dtype=torch.float32),
                    "observation.images.wrist": torch.tensor(
                        rgb.transpose(2, 0, 1).copy(), dtype=torch.float32
                    )
                    / 255,
                }
                chunk = post(policy.predict_action_chunk(pre(batch)))[0, : config.n_action_steps]
            actions = chunk.cpu().numpy()
            if not np.isfinite(actions).all():
                raise ValueError("ACT returned non-finite actions")
            response = {"actions": actions.tolist()}
        except Exception as exc:
            response = {"error": str(exc)}
        wire.write(json.dumps(response, allow_nan=False) + "\n")
        wire.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "serve"])
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
    else:
        serve(args.input, args.device)


if __name__ == "__main__":
    main()
