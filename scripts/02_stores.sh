#!/bin/bash
# FeatureStores from the extraction parts.  usage: bash scripts/02_stores.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
python signdeg/build_store.py --shards $O/parts/source/*.pt --out $O/stores/store_source --chunk-frames 256
python signdeg/build_store.py --shards $O/parts/h2s/*.pt --out $O/stores/store_h2s --slim --split-file $O/lists/h2s_split.json
python signdeg/build_store.py --shards $O/parts/openasl/*.pt --out $O/stores/store_openasl --slim --split-file $O/lists/openasl_split.json
python signdeg/build_store.py --shards $O/parts/aslc/*.pt --out $O/stores/store_aslcitizen --slim --split-file $O/lists/aslc_split.json
