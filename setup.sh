#!/usr/bin/env bash
set -euo pipefail

pip install zstandard sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 5

echo "Setup complete. Run: torchrun --standalone --nproc_per_node=1 experiment_phase2_combined.py"\n