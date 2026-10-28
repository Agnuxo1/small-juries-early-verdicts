#!/usr/bin/env bash
# End-to-end reproduction, in the order and with the arguments used for the paper.
# Requires: .venv (requirements-analysis.txt), .venv-vllm (requirements-inference.txt), one 24 GB GPU, ~30 GB disk.
set -euo pipefail
export SJEV_ROOT=${SJEV_ROOT:-$PWD/work}
PY=${PY:-.venv/bin/python}
PYV=${PYV:-.venv-vllm/bin/python}
mkdir -p "$SJEV_ROOT"/{data,models,out}

# 0. Data and models (public, no login required)
$PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
R = os.environ['SJEV_ROOT']
snapshot_download('nebius/SWE-agent-trajectories', repo_type='dataset', local_dir=f'{R}/data/nebius_raw')
for m in ('google/gemma-4-E4B-it', 'google/gemma-4-E2B-it', 'Qwen/Qwen3-4B-Instruct-2507'):
    snapshot_download(m, local_dir=f'{R}/models/' + m.split('/')[1])
EOF

cd src
# 1. Sampling (1 run per repository and outcome), trace features, leakage checks, E1 baseline
$PY 01_sample.py 1
$PY 02_features.py
$PY 04_checks.py
$PY 03_baseline.py
# 2. Repository-disjoint splits: test 600 / dev / pilot 50 (fixed before any judge is run)
$PY 05_splits.py 0.40
# 3. Judges (one model per job; resumable)
for spec in "e4b gemma-4-E4B-it" "e2b gemma-4-E2B-it" "q4b Qwen3-4B-Instruct-2507"; do
  set -- $spec
  $PYV 06_judge.py --model "$SJEV_ROOT/models/$2" --tag "$1" --stages "test:abs10,abs20;dev:abs10,abs20"
  $PYV 06_judge.py --model "$SJEV_ROOT/models/$2" --tag "$1" --stages "test:frac66;dev:frac66"
done
$PYV 06_judge.py --model "$SJEV_ROOT/models/gemma-4-E4B-it" --tag e4b --stages "test:frac33;dev:frac33"
$PYV 06_judge.py --model "$SJEV_ROOT/models/gemma-4-E2B-it" --tag e2b --stages "test:frac33;dev:frac33"
# 4. Pre-registered analysis (H1-H4) and supplementary analysis (p-values, Holm, H4 contrasts, control model)
$PY 07_analysis.py "$SJEV_ROOT/out/e2" "$SJEV_ROOT/out/e2_analysis"
$PY 08_supplementary.py "$SJEV_ROOT/out/e2" "$SJEV_ROOT/out/supplementary"
# 5. Blind validation set (annotate items/*.txt with the rubric, then compute agreement)
$PYV 09_export_validation.py
# $PY 10_validation.py "$SJEV_ROOT/out/validation/key.csv" ../validation/annotations "$SJEV_ROOT/out/e2" ../results
# 6. Figures: copy the CSV/JSON outputs into results/ and run
# cd .. && python scripts/make_figures.py
