#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: R1PRO_RL_PYTHON=python bash scripts/r1pro_rl_prepare.sh NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${R1PRO_RL_PYTHON:-python}"
output_dir="$1"
mkdir -p "$(dirname "$output_dir")"
mkdir "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${R1PRO_RL_DEPS:+:$R1PRO_RL_DEPS}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

"$python_bin" -m pytest -c /dev/null \
  --confcutdir=dimos/robot/galaxea/r1pro/rl \
  -o 'markers=mujoco: isolated MuJoCo simulation checks' \
  -q dimos/robot/galaxea/r1pro/rl | tee "$output_dir/tests.log"
"$python_bin" -m dimos.robot.galaxea.r1pro.rl.static_audit \
  --output "$output_dir/static-evidence.json" | tee "$output_dir/static-audit.log"
for controller in hold differential-ik; do
  for case_id in 0 1 2 3; do
    "$python_bin" -m dimos.robot.galaxea.r1pro.rl.evaluate \
      --output "$output_dir/$controller-$case_id" --case "$case_id" \
      --controller "$controller" > "$output_dir/$controller-$case_id.log"
  done
done
"$python_bin" -m dimos.robot.galaxea.r1pro.rl.evaluate \
  --output "$output_dir/recorded-synthetic" \
  --target-stream configs/r1pro_rl_target_stream_example.json \
  --controller differential-ik > "$output_dir/recorded-synthetic.log"

"$python_bin" - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = {"training_updates": 0, "hardware_access": False, "cases": []}
for case_id in range(4):
    manifests = [json.loads((root / f"{controller}-{case_id}/manifest.json").read_text())
                 for controller in ("hold", "differential-ik")]
    for key in ("frozen_manifest_sha256", "frozen_target_samples_sha256", "config_sha256", "model_sha256"):
        assert manifests[0][key] == manifests[1][key], (case_id, key)
    targets = []
    for controller in ("hold", "differential-ik"):
        result = json.loads((root / f"{controller}-{case_id}/evaluation.json").read_text())
        targets.append([{key: row[key] for key in ("time_s", "stamp_s", "valid", "target_positions_m", "target_quaternions_wxyz", "position_valid", "orientation_valid")} for row in result["trace"]])
        metrics = result["metrics"]
        assert metrics["completed"] == int(case_id < 2), (controller, case_id, result)
        if case_id >= 2:
            assert result["termination_reason"] == ("target_dropout" if case_id == 2 else "target_timeout")
        summary["cases"].append({"controller": controller, "case": case_id,
            "completed": metrics["completed"], "requested_duration_s": metrics["requested_duration_s"],
            "observed_duration_s": metrics["observed_duration_s"], "reason": result["termination_reason"]})
    assert targets[0] == targets[1], (case_id, "raw reference traces differ")
recorded = json.loads((root / "recorded-synthetic/evaluation.json").read_text())
assert recorded["termination_reason"] == "target_timeout" and recorded["metrics"]["completed"] == 0
summary["identical_targets_and_constraints_verified"] = True
summary["synthetic_recording_eof_timeout_verified"] = True
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
