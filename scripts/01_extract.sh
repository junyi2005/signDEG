#!/bin/bash
# Feature extraction of one shard on one GPU.  usage: bash scripts/01_extract.sh {source|h2s|openasl|aslc} SHARD NSHARDS
# Run all shards (in parallel on several GPUs or as cluster array jobs); finished parts are skipped on restart.
set -euo pipefail
source "$(dirname "$0")/env.sh"
set_=$1; i=$2; n=$3
case $set_ in
  source)  python signdeg/extract.py --segments $O/lists/source_segments.tsv --video-dir $SOURCE_VIDEOS \
             --out $O/parts/source/shard$i --shard $i --nshards $n --geometry --wilor-root $T/WiLoR $DET ;;
  h2s)     python signdeg/extract.py --segments $O/lists/h2s_segments.tsv --video-dir $H2S/clips \
             --out $O/parts/h2s/shard$i --shard $i --nshards $n $DET ;;
  openasl) python signdeg/extract.py --segments $O/lists/openasl_segments.tsv --video-dir $OPENASL_VIDEOS \
             --out $O/parts/openasl/shard$i --shard $i --nshards $n $DET ;;
  aslc)    python signdeg/extract.py --segments $O/lists/aslc_segments.tsv --video-dir $ASLC/videos \
             --out $O/parts/aslc/shard$i --shard $i --nshards $n $DET ;;
  *) echo "unknown set $set_"; exit 1 ;;
esac
