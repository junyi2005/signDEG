#!/bin/bash
# Translation: source stage on the pre-training captions, then How2Sign and OpenASL target stages.
#   usage: bash scripts/04_translation.sh [live|frozen] [SEED]
set -euo pipefail
source "$(dirname "$0")/env.sh"
mode=${1:-live}; seed=${2:-0}
S=$O/runs/slt/$mode/s$seed
torchrun --nproc_per_node $NGPU signdeg/slt.py --stage source --store $O/stores/store_source --captions $YTASL_TABLE \
  --upstream $UP --upstream-mode $mode --backbone $W/byt5-base --seed $seed --out $S/source
torchrun --nproc_per_node $NGPU signdeg/slt.py --stage target --store $O/stores/store_h2s \
  --text $H2S/how2sign_train.csv $H2S/how2sign_val.csv $H2S/how2sign_test.csv --init $S/source/final.pt \
  --upstream $UP --upstream-mode $mode --backbone $W/byt5-base --seed $seed --out $S/how2sign
torchrun --nproc_per_node $NGPU signdeg/slt.py --stage target --store $O/stores/store_openasl --text $OPENASL_TABLE \
  --init $S/source/final.pt --upstream $UP --upstream-mode $mode --backbone $W/byt5-base --seed $seed --out $S/openasl
