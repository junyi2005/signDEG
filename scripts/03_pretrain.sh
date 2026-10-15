#!/bin/bash
# SignDEG pre-training, one run per stream on $NGPU GPUs (400K + 50K updates).  usage: bash scripts/03_pretrain.sh [lh rh face]
set -euo pipefail
source "$(dirname "$0")/env.sh"
for s in ${*:-lh rh face}; do
  cfg=configs/signdeg_hand.yaml; [ "$s" = face ] && cfg=configs/signdeg_face.yaml
  torchrun --nproc_per_node $NGPU train.py --config $cfg --stream $s --store $O/stores/store_source --output-root $R
done                                              # -> $R/signdeg_{lh,rh,face}/ckpt_final.pt
