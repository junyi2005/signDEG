# SignDEG

**DINO Eats Geometry: Self-Supervised Sign Representations that Absorb Articulated Hand Geometry**

Junyi Hu, Zhewen He, Haomian Huang, Zhenhua Li, Shuaihang Yuan, Zhifei Li, Yi Fang<sup>†</sup>
New York University Abu Dhabi · <sup>†</sup>corresponding author · jh10472@nyu.edu

[**Paper (arXiv)**](https://arxiv.org/abs/2609.06296) · [**Project page**](https://junyi2005.github.io/signdino/) · [**BibTeX**](#citation)

---

This repository is the code of SignDEG. It covers the method end to end: articulator streams and feature
extraction, self-supervised pre-training, and the downstream translator, isolated-sign recognizer and
fingerspelling detector that read the encoders. No data, model weights or checkpoints are included; Sections 2
and 3 list what to obtain. The earlier SignDino code of this repository is kept at the tag `signdino-v1`.

## Method in brief

SignDEG trains one temporal encoder per articulator stream (left hand, right hand, face).

* **Input: appearance.** YOLOv8n face and hand detectors with ByteTrack follow the face and the two hands; every
  crop (hands 1.4x, face 1.2x, 224 x 224, 12.5 fps grid) is embedded by a frozen DINOv3 ViT-B/16 into its 768-d
  class token. These features are computed once and cached.
* **Output: temporal organisation.** A 6-block temporal Transformer (D = 384) per stream is trained by
  temporal-axis self-distillation between a student and an EMA teacher: DINO loss between long (global) and
  short (local) temporal crops, iBOT loss on masked frames, KoLeo regulariser, Sinkhorn-Knopp normalisation of
  the teacher over the distributed batch.
* **Geometry absorption (hand streams, pre-training only).** A frozen WiLoR gives per-frame targets: the
  geometric latent `u` (mean of the 16 MANO pose-query tokens after backbone block 23, standardised), the MANO
  articulation `theta` (15 x 6D) and the wrist-relative 3D joints `J` (21 x 3), each frame weighted by a
  reliability score `w`. **GLP** makes block 4 of the student predict `u`, `theta`, `J` at every frame, masked
  frames included. **GRT** matches, at the last block, the student's similarity distribution over frames of other
  videos with the one induced by `[u ; theta]`. All terms are active from the first update.
* **Inference.** Only DINOv3 and the EMA teachers run; WiLoR and every geometry head are discarded. A downstream
  model reads each stream through a learned softmax over its L + 1 = 7 layers (the projected DINOv3 input and
  the six blocks).

Where each part lives: `model/geo.py` (GLP, GRT, reliability `w`), `model/meta_arch.py` (student/teacher,
temporal self-distillation, total objective), `model/temporal_encoder.py` (stream encoder), `dino/` (losses,
heads, EMA), `data/` (temporal multi-crop sampling, frame masking, cached features), `signdeg/` (extraction,
stores, downstream models).

## 0. Folder layout

| Path | Content |
| --- | --- |
| `train.py` | pre-training of one stream (torchrun, data-parallel) |
| `configs/signdeg_hand.yaml` | pre-training of a hand stream: temporal self-distillation + GLP + GRT; all hyperparameters |
| `configs/signdeg_face.yaml` | pre-training of the face stream: temporal self-distillation |
| `model/` | temporal Transformer, student/teacher model, GLP / GRT heads, reliability weights |
| `dino/` | DINO / iBOT / KoLeo losses, prototype heads, EMA, Sinkhorn-Knopp |
| `data/` | cached-feature reader (FeatureStore), temporal multi-crop sampler, collate with frame masking |
| `preprocess/yolo_bytetrack.py` | YOLOv8n face + hand detection, ByteTrack association, hand-slot mapping |
| `signdeg/make_splits.py` | segment tables, splits and labels (incl. the pre-training source split) |
| `signdeg/extract.py` | tracking, hand sides, crops, DINOv3 features and (optional) WiLoR targets |
| `signdeg/build_store.py` | extraction outputs -> FeatureStore (latent standardisation, reliability, chunking) |
| `signdeg/downstream.py` | encoders for downstream use (frozen / live / LoRA) and the layer-weighted fusion |
| `signdeg/slt.py` | translation with ByT5-Base: source stage, target stage, evaluation |
| `signdeg/islr.py` | isolated sign recognition: mask-aware average, BatchNorm, linear classifier |
| `signdeg/fingerspelling.py` | fingerspelling detection: two-layer Transformer, frame head, interval IoU |
| `scripts/` | the commands of Section 4 as shell scripts (paths in `scripts/env.sh`) |
| `requirements.txt` | pinned Python packages |
| `docs/` | project page |

Run everything from the root of this folder; every Python script prints its options with `--help`.

## 1. Environment

Linux with NVIDIA GPUs, Python 3.10, PyTorch 2.6.0 with CUDA 12.4. Pre-training and translation use four GPUs
through `torchrun`; any number of GPUs works (the effective batch scales with it).

```bash
git clone https://github.com/junyi2005/signdino.git && cd signdino
conda create -n signdeg python=3.10 -y && conda activate signdeg
pip install torch==2.6.0 torchvision==0.21.0      # the CUDA 12.4 builds, from the official PyTorch wheel index
pip install -r requirements.txt
```

* **chumpy** (used by the MANO layer inside WiLoR) does not import with numpy >= 1.24. In
  `chumpy/__init__.py` replace `from numpy import bool, int, float, complex, object, unicode, str, nan, inf` by
  `from numpy import nan, inf`. With Python >= 3.11, also replace `inspect.getargspec(` by
  `inspect.getfullargspec(` in `chumpy/ch.py`. Set `PYTHONNOUSERSITE=1` if another copy of chumpy is installed
  in the user site.
* WiLoR imports pyrender; on a machine without a display set `PYOPENGL_PLATFORM=egl` for extraction.
* Pre-training and the downstream models need only the "core" packages of `requirements.txt`; the
  detection/tracking and WiLoR groups are needed for extraction.

## 2. External models (not included)

Obtain each model from its official release, under its own licence. Paths are passed on the command line; the
names below are the ones used in Section 4 and `scripts/env.sh`.

| Path | Model |
| --- | --- |
| `weights/dinov3-vitb16/` | DINOv3 ViT-B/16 pre-trained on LVD-1689M, Hugging Face transformers format (`config.json`, `model.safetensors`, `preprocessor_config.json`) |
| `weights/yolo/yolov8n-face.pt` | YOLOv8n face detector (the public `yolov8n-face` checkpoint, trained on WIDER FACE) |
| `weights/yolo/hand_yolov8n.pt` | YOLOv8n hand detector (the public single-class `hand_yolov8n.pt` checkpoint of the ADetailer model collection) |
| `third_party/WiLoR/` | official WiLoR code release with `pretrained_models/wilor_final.ckpt`, `model_config.yaml`, `dataset_config.yaml` |
| `third_party/WiLoR/mano_data/` | MANO hand model (`MANO_RIGHT.pkl`, `MANO_LEFT.pkl`) from the official MANO release (registration required), and `mano_mean_params.npz` shipped with WiLoR |
| `weights/byt5-base/` | ByT5-Base, transformers format (translation decoder) |

## 3. Data (not included)

Obtain every dataset from its authors under its own terms of use.

* **Pre-training source split** (the public-data protocol of SHuBERT): the YouTube-ASL videos, with videos that
  contain OpenASL validation or test clips removed and the removed duration replaced by non-overlapping ASL videos
  of YouTube-SL-25. Needed: the videos (`<youtube_id>.mp4`), a caption table of these videos (TSV, one row per
  English caption: `vid`, `yid`, `start`, `end` as `hh:mm:ss.fff`, `raw-text`, `split` in train / validation /
  test; the column layout of the OpenASL table), the OpenASL table, and the list of YouTube-SL-25 ASL video ids.
  `signdeg/make_splits.py source` writes the split.
* **How2Sign**: the sentence-level clips of the green-screen RGB frontal view for train, validation and test
  (`<SENTENCE_NAME>.mp4`) and `how2sign_{train,val,test}.csv` (tab-separated, columns include `SENTENCE_NAME`
  and `SENTENCE`).
* **OpenASL**: the clips' source videos (`<youtube_id>.mp4`) and the OpenASL table (`vid`, `yid`, `start`, `end`,
  `raw-text`, `split`).
* **FLEURS-ASL** (zero-shot translation): sentence videos and their English text; list them with
  `make_splits.py clips` (columns `video`, `split`, `text`).
* **ASL Citizen**: `videos/` and `splits/{train,val,test}.csv` (columns `Video file`, `Gloss`).
* **Sem-Lex, WLASL2000**: list the clips with `make_splits.py clips` (columns `video`, `split`, `label`).
* **ASL-STEM-Wiki** (fingerspelling): a JSON `{video id: {"fold": k, "intervals": [[start_s, end_s], ...]}}`
  with the dataset's cross-validation folds and fingerspelling intervals.

## 4. Pipeline

`scripts/` runs these steps after the paths in `scripts/env.sh` are set; the commands below are the same.

```bash
export W=weights T=third_party D=/path/to/datasets O=/path/to/work PYOPENGL_PLATFORM=egl
DET="--dinov3 $W/dinov3-vitb16 --face-ckpt $W/yolo/yolov8n-face.pt --hand-ckpt $W/yolo/hand_yolov8n.pt"
```

**4.0 Smoke test** (no data): `python train.py --config configs/signdeg_hand.yaml --synthetic-smoke --max-steps 20`

**4.1 Segment tables** (`scripts/00_lists.sh`)

```bash
python signdeg/make_splits.py durations --video-dir $D/youtube_asl/videos --out $O/lists/dur_ytasl.tsv
python signdeg/make_splits.py durations --video-dir $D/youtube_sl25/videos --out $O/lists/dur_ytsl25.tsv
# concatenate the two duration tables into $O/lists/durations.tsv (one header line), then
python signdeg/make_splits.py source --ytasl-table $D/youtube_asl/youtube_asl.tsv --openasl-table $D/openasl/openasl-v1.0.tsv \
  --ytsl25-list $D/youtube_sl25/asl_video_ids.txt --durations $O/lists/durations.tsv --out $O/lists
python signdeg/make_splits.py how2sign --csv-dir $D/how2sign --out $O/lists
python signdeg/make_splits.py openasl --table $D/openasl/openasl-v1.0.tsv --out $O/lists
python signdeg/make_splits.py aslcitizen --root $D/ASL_Citizen --out $O/lists
```

**4.2 Extraction** (`scripts/01_extract.sh`). One call per shard (`--shard i --nshards n`) and GPU; finished
parts are skipped on restart. Pre-training videos also get the WiLoR targets (`--geometry`).

```bash
python signdeg/extract.py --segments $O/lists/source_segments.tsv --video-dir $D/youtube_asl/videos,$D/youtube_sl25/videos \
  --out $O/parts/source/shard0 --shard 0 --nshards 32 --geometry --wilor-root $T/WiLoR $DET
python signdeg/extract.py --segments $O/lists/h2s_segments.tsv --video-dir $D/how2sign/clips --out $O/parts/h2s/shard0 $DET
python signdeg/extract.py --segments $O/lists/openasl_segments.tsv --video-dir $D/openasl/videos --out $O/parts/openasl/shard0 $DET
python signdeg/extract.py --segments $O/lists/aslc_segments.tsv --video-dir $D/ASL_Citizen/videos --out $O/parts/aslc/shard0 $DET
```

Detection and tracking run on every native frame; tracker predictions fill detector gaps of at most 3 frames
and keep the last detector score. For a frontal signer the image-left track is the right hand; tracks keep their
identities through crossings, and a video whose side assignment is ambiguous gets an invalid hand mask. Left
hands are mirrored into WiLoR's right-hand convention; their joints are reflected back to the camera axes.

**4.3 FeatureStores** (`scripts/02_stores.sh`)

```bash
python signdeg/build_store.py --shards $O/parts/source/*.pt --out $O/stores/store_source --chunk-frames 256
python signdeg/build_store.py --shards $O/parts/h2s/*.pt --out $O/stores/store_h2s --slim --split-file $O/lists/h2s_split.json
python signdeg/build_store.py --shards $O/parts/openasl/*.pt --out $O/stores/store_openasl --slim --split-file $O/lists/openasl_split.json
python signdeg/build_store.py --shards $O/parts/aslc/*.pt --out $O/stores/store_aslcitizen --slim --split-file $O/lists/aslc_split.json
```

The pre-training store standardises `u` with its training statistics, computes the reliability weights over each
full source video, and stores long videos as consecutive items of at most 256 frames.

**4.4 Pre-training** (`scripts/03_pretrain.sh`). One run per stream on four GPUs (8 videos per GPU), 400K + 50K
updates as configured; the three runs are independent.

```bash
torchrun --nproc_per_node 4 train.py --config configs/signdeg_hand.yaml --stream lh --store $O/stores/store_source --output-root $O/runs/ssl
torchrun --nproc_per_node 4 train.py --config configs/signdeg_hand.yaml --stream rh --store $O/stores/store_source --output-root $O/runs/ssl
torchrun --nproc_per_node 4 train.py --config configs/signdeg_face.yaml           --store $O/stores/store_source --output-root $O/runs/ssl
# -> $O/runs/ssl/signdeg_{lh,rh,face}/ckpt_final.pt ; --resume CKPT continues an interrupted run
```

**4.5 Translation** (`scripts/04_translation.sh`). ByT5-Base on the fused features; `--upstream-mode live`
updates the temporal encoders at 1/10 of the decoder learning rate, `frozen` keeps them fixed. Source stage
(250K updates on the source captions), then a target stage per dataset (50K updates, best validation BLEU-4,
test decoding with beam 5):

```bash
UP="$O/runs/ssl/signdeg_lh/ckpt_final.pt $O/runs/ssl/signdeg_rh/ckpt_final.pt $O/runs/ssl/signdeg_face/ckpt_final.pt"
torchrun --nproc_per_node 4 signdeg/slt.py --stage source --store $O/stores/store_source --captions $D/youtube_asl/youtube_asl.tsv \
  --upstream $UP --upstream-mode live --backbone $W/byt5-base --out $O/runs/slt/source
torchrun --nproc_per_node 4 signdeg/slt.py --stage target --store $O/stores/store_h2s \
  --text $D/how2sign/how2sign_train.csv $D/how2sign/how2sign_val.csv $D/how2sign/how2sign_test.csv \
  --init $O/runs/slt/source/final.pt --upstream $UP --upstream-mode live --backbone $W/byt5-base --out $O/runs/slt/how2sign
torchrun --nproc_per_node 4 signdeg/slt.py --stage target --store $O/stores/store_openasl --text $D/openasl/openasl-v1.0.tsv \
  --init $O/runs/slt/source/final.pt --upstream $UP --upstream-mode live --backbone $W/byt5-base --out $O/runs/slt/openasl
# zero-shot FLEURS-ASL: extract and store its clips like How2Sign, then
python signdeg/slt.py --stage source --eval-only test --store $O/stores/store_fleurs --text $O/lists/fleurs_text.tsv \
  --init $O/runs/slt/source/final.pt --upstream $UP --backbone $W/byt5-base --out $O/runs/slt/fleurs
```

Effective batch: `--batch-per-gpu 2` x 4 GPUs x `--accum 8` = 64 utterances. Runs resume from `state.pt`.

**4.6 Isolated sign recognition** (`scripts/05_islr.sh`); `--mode frozen | lora | full` (frozen features with a
linear head, rank-1 LoRA on the temporal encoders, or full fine-tuning):

```bash
python signdeg/islr.py --store $O/stores/store_aslcitizen --labels $O/lists/aslc_labels.json --upstream $UP --mode lora \
  --out $O/runs/islr/aslcitizen
```

**4.7 Fingerspelling detection** (ASL-STEM-Wiki store built like the ASL Citizen store; one run per fold):

```bash
python signdeg/fingerspelling.py --store $O/stores/store_stemwiki --labels $D/asl_stem_wiki/fs_intervals.json \
  --upstream $UP --test-fold 0 --out $O/runs/fs/fold0
```

## 5. Data formats

* **Extraction output** (`*.pt`, read with `torch.load`): `{vid: record}`; a record holds `fps`, `frame_idx`
  (native frame indices on the sampling grid, relative to the segment start), `side_ambiguous`, and per stream
  `lh` / `rh` / `face`: `valid`, `box`, `dino_cls` (768, fp16); hands also `score`, `state` (`active` /
  `predicted`), and with `--geometry` `wilor_jtok23` (1280), `wilor_hand_pose` (15 x 6), `wilor_joints3d` (21 x 3).
* **FeatureStore** (a directory): `index.json` with one entry per item (`vid`, `split`, `offset`, `T`, `fps`,
  `src`, `start`) and one memory-mapped array per stream and key, `<stream>_<key>.npy`, frames of all items
  concatenated; keys `dino`, `valid`, `score`, `pred`, `geo_ok`, and for hands of the pre-training store `u`,
  `theta`, `joint`, `w` (see `data/feature_store.py`).
* **Pre-training checkpoint** (`ckpt_final.pt`): `{"model": state_dict, "optimizer", "step", "cfg"}`; the
  downstream models load the EMA teacher encoder (`teacher_encoder.*`).

## Citation

```bibtex
@article{hu2026signdeg,
  title   = {{DINO} Eats Geometry: Self-Supervised Sign Representations that Absorb
             Articulated Hand Geometry},
  author  = {Hu, Junyi and He, Zhewen and Huang, Haomian and Li, Zhenhua and
             Yuan, Shuaihang and Li, Zhifei and Fang, Yi},
  journal = {arXiv preprint arXiv:2609.06296},
  year    = {2026}
}
```

## License

[MIT](LICENSE). The datasets and third-party models listed above keep their own licences.
