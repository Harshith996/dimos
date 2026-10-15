#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: R1PRO_RL_PYTHON=python scripts/r1pro_rl_smoke.sh OUTPUT_DIRECTORY" >&2
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
"$python_bin" -m dimos.robot.galaxea.r1pro.rl.train \
  --output "$output_dir/ppo" --config configs/r1pro_rl_smoke.json \
  --target-mode procedural --num-envs 2 --iterations 1 --seed 42 \
  | tee "$output_dir/train.log"

for case_id in 0 2 3; do
  "$python_bin" -m dimos.robot.galaxea.r1pro.rl.evaluate \
    --output "$output_dir/eval-hold-$case_id" --duration 0.4 --case "$case_id" \
    | tee "$output_dir/eval-hold-$case_id.log"
done
"$python_bin" -m dimos.robot.galaxea.r1pro.rl.evaluate \
  --output "$output_dir/eval-checkpoint-0" --duration 0.4 --case 0 \
  --checkpoint "$output_dir/ppo/model_0.pt" \
  | tee "$output_dir/eval-checkpoint-0.log"
