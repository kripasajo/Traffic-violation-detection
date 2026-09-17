#!/bin/bash
# Final run: both tasks, end to end, every value recorded.
#
# CLASS ORDER MATTERS. features.py assigns label 0 to the first --classes entry
# and label 1 to the second. Label 1 must be the VIOLATION for both tasks, so
# that "positive", "precision" and "recall" mean the same thing throughout:
#     triple-riding : normal  -> 0,  triple    -> 1
#     helmet        : helmet  -> 0,  no_helmet -> 1
# Inverting either would silently report the metrics of the majority class.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

ts() { date '+%H:%M:%S'; }
say() { echo "[$(ts)] ===== $* ====="; }

say "STAGE 1/4  extract crops for both tasks (single video pass)"
python3 extract.py

say "STAGE 2/4  DINOv2 features"
python3 features.py --crops crops_triple --out feats/triple \
        --classes normal triple
python3 features.py --crops crops_helmet --out feats/helmet \
        --classes helmet no_helmet

say "STAGE 3/4  experiments"
python3 experiments.py --feats feats/triple --task triple \
        --pos-name "triple riding" --out metrics/triple.json
python3 experiments.py --feats feats/helmet --task helmet \
        --pos-name "no helmet" --out metrics/helmet.json

say "STAGE 4/4  figures and tables"
python3 figures.py --metrics metrics/triple.json metrics/helmet.json \
        --crops-triple crops_triple --crops-helmet crops_helmet \
        --out figures

say "COMPLETE"
ls -la figures/
