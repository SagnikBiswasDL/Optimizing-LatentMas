#!/usr/bin/env bash
# LoopDelta stage 0 on one GPU. Run from the repo root on the pod.
# Each step writes its own directory; stage 0b reuses finished generations on restart.
set -ex
A=${A:-artifacts/loopdelta}
M=${M:-ByteDance/Ouro-1.4B}

pip install -q math-verify || echo "math-verify unavailable; grading falls back to string match"

python scripts/loopdelta_stage0a.py --model "$M" --per_level 200 --depth 4 --bs 64 --out "$A/stage0a_r4"
python scripts/loopdelta_analyze.py --stage0a "$A/stage0a_r4" --out "$A/analysis_0a_r4"

python scripts/loopdelta_stage0a.py --model "$M" --per_level 200 --depth 8 --extrapolate --bs 32 --out "$A/stage0a_r8"
python scripts/loopdelta_analyze.py --stage0a "$A/stage0a_r8" --out "$A/analysis_0a_r8"

python scripts/loopdelta_stage0b.py --model "$M" --per_level 60 --depths 1 2 3 4 --out "$A/stage0b"
python scripts/loopdelta_analyze.py --stage0a "$A/stage0a_r4" --stage0b "$A/stage0b" --out "$A/analysis"

touch "$A.DONE"
