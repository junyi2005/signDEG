#!/bin/bash
# Segment tables, splits and labels.  usage: bash scripts/00_lists.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
L=$O/lists; mkdir -p $L
for d in ${SOURCE_VIDEOS//,/ }; do python signdeg/make_splits.py durations --video-dir $d --out $L/dur_$(basename $(dirname $d)).tsv; done
(head -1 $(ls $L/dur_*.tsv | head -1); for f in $L/dur_*.tsv; do tail -n +2 $f; done) > $L/durations.tsv
python signdeg/make_splits.py source --ytasl-table $YTASL_TABLE --openasl-table $OPENASL_TABLE \
  --ytsl25-list $YTSL25_LIST --durations $L/durations.tsv --out $L
python signdeg/make_splits.py how2sign --csv-dir $H2S --out $L
python signdeg/make_splits.py openasl --table $OPENASL_TABLE --out $L
python signdeg/make_splits.py aslcitizen --root $ASLC --out $L
