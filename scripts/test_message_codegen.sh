#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${DIMOS_CODEGEN_PYTHON:-$PWD/.venv/bin/python}"
prepare_args=()
if [[ "${DIMOS_OFFLINE:-0}" == "1" ]]; then prepare_args+=(--offline); fi
"$python" -m dimos.message_codegen.prepare_test_inputs "${prepare_args[@]}"
evidence="$PWD/build/message-codegen/demo/evidence"
mkdir -p "$evidence"
# Library regression tests live with dimos-cdrgen; exercise DimOS integration here.
"$python" -m pytest --noconftest -o addopts='' -q -ra \
  --cov=dimos/message_codegen --cov-report="xml:$evidence/coverage.xml" \
  dimos/message_codegen/test_ownership.py dimos/message_codegen/test_stubs.py \
  dimos/message_codegen/test_source_package.py \
  | tee "$evidence/pytest.txt"
DIMOS_NATIVE_ACCEPTANCE=1 DIMOS_OWNERSHIP_EVIDENCE="$evidence" \
  "$python" -m pytest --noconftest -o addopts='' -q \
  dimos/message_codegen/test_native_consumer.py | tee "$evidence/ownership.txt"
