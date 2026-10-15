#!/bin/bash
# Isolated sign recognition on ASL Citizen.  usage: bash scripts/05_islr.sh [frozen|lora|full] [SEED]
set -euo pipefail
source "$(dirname "$0")/env.sh"
mode=${1:-lora}; seed=${2:-0}
python signdeg/islr.py --store $O/stores/store_aslcitizen --labels $O/lists/aslc_labels.json \
  --upstream $UP --mode $mode --seed $seed --out $O/runs/islr/aslcitizen_${mode}_s$seed
