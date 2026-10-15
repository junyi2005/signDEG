"""Articulator-stream feature extraction (paper Sec. 3.1, 3.3; Appendix B).

For every source video listed in a segment table (TSV with columns vid, yid, start, end; times hh:mm:ss.fff):
  1. tracking pass: YOLOv8n face and hand detection + ByteTrack on every native frame of the segment span
     (short detector gaps are filled by the tracker's motion model, see preprocess/yolo_bytetrack.py);
  2. hand sides: for a frontal signer, the anatomical left/right tracks are initialised from image-right /
     image-left and keep their ByteTrack identities through crossings; videos whose initial side assignment is
     ambiguous get an all-invalid hand mask;
  3. feature pass on the 12.5 fps grid: square crops (hands 1.4x, face 1.2x, 224^2) -> frozen DINOv3 ViT-B/16
     class token; with --geometry, a frozen WiLoR on each hand crop gives
        wilor_jtok23     mean of the 16 MANO pose-query tokens after backbone block 23 (1280),
        wilor_hand_pose  the 15 local finger-joint rotations in 6D (90),
        wilor_joints3d   wrist-relative 3D joints in metres (21 x 3);
     left hands are mirrored into WiLoR's right-hand convention; their joints are reflected back to the
     camera axes with S = diag(-1, 1, 1), their local rotations stay in the canonical convention.
Records are cut per segment and written as <out>.partK.pt = {vid: record}; see signdeg/build_store.py.

Clips (How2Sign, ASL Citizen, ...) are segment tables with one whole-video segment per clip
(signdeg/make_splits.py); pre-training videos use one whole-video segment per source video.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocess"))
from yolo_bytetrack import HandFaceDetector  # noqa: E402

WILOR_LAYER = 23        # zero-based backbone block whose pose-query tokens give the geometric latent u
N_POSE_TOKENS = 16      # MANO pose-query tokens of WiLoR
CROP, PAD_HAND, PAD_FACE = 224, 1.4, 1.2
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
REFLECT = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)   # S


def sec(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def square_crop(img: np.ndarray, xyxy, pad: float, out: int) -> np.ndarray | None:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * pad
    if s < 4:
        return None
    a = np.array([[out / s, 0, out / 2 - cx * out / s], [0, out / s, out / 2 - cy * out / s]], dtype=np.float32)
    return cv2.warpAffine(img, a, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    x = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0
    x = (x - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): the first two columns (Zhou et al. 2019)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def _finite(x: torch.Tensor, name: str) -> torch.Tensor:
    """Fail loudly instead of silently writing overflowed features (fp16 range is +-65504)."""
    if not torch.isfinite(x).all() or x.abs().max() > 6e4:
        raise FloatingPointError(f"{name}: non-finite or fp16-overflowing values")
    return x


class Models:
    def __init__(self, args):
        from transformers import AutoModel
        self.dev = "cuda"
        self.det = HandFaceDetector(face_ckpt=Path(args.face_ckpt), hand_ckpt=Path(args.hand_ckpt))
        self.dino = AutoModel.from_pretrained(args.dinov3).to(self.dev).half().eval()
        self.wilor = None
        if args.geometry:
            sys.path.insert(0, str(Path(args.wilor_root).parent))
            sys.path.insert(0, args.wilor_root)
            cwd = os.getcwd()
            os.chdir(args.wilor_root)
            from wilor.models import load_wilor
            self.wilor, self.wcfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                               cfg_path="./pretrained_models/model_config.yaml")
            os.chdir(cwd)
            self.wilor = self.wilor.to(self.dev).eval()
            self._tap = None
            self.wilor.backbone.blocks[WILOR_LAYER].register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        self._tap = out

    @torch.no_grad()
    def dino_cls(self, crops: list[np.ndarray], bs: int = 256) -> np.ndarray:
        out = []
        for k in range(0, len(crops), bs):
            x = to_tensor(crops[k:k + bs]).to(self.dev).half()
            out.append(self.dino(pixel_values=x).last_hidden_state[:, 0].float().cpu())
        if not out:
            return np.zeros((0, 768), np.float16)
        return _finite(torch.cat(out), "dino_cls").half().numpy()

    @torch.no_grad()
    def wilor_targets(self, frames: list[np.ndarray], boxes: list, right: int, bs: int = 64) -> dict:
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        out = {"hand_pose": [], "joints3d": [], f"jtok{WILOR_LAYER}": []}
        items = [torch.from_numpy(ViTDetDataset(self.wcfg, f, np.asarray([b], np.float32), np.asarray([right]), [0],
                                                rescale_factor=2.0)[0]["img"]) for f, b in zip(frames, boxes)]
        for k in range(0, len(items), bs):
            x = torch.stack(items[k:k + bs]).to(self.dev)
            o = self.wilor({"img": x})
            B = x.shape[0]
            out["hand_pose"].append(rot6d(o["pred_mano_params"]["hand_pose"].reshape(B, 15, 3, 3)).cpu())
            j = o["pred_keypoints_3d"].reshape(B, 21, 3)
            j = (j - j[:, :1]).cpu()
            if not right:                       # mirrored input: back to the original camera axes
                j = j @ torch.from_numpy(REFLECT)
            out["joints3d"].append(j)
            out[f"jtok{WILOR_LAYER}"].append(self._tap.float()[:, :N_POSE_TOKENS].mean(1).cpu())
        return {k: _finite(torch.cat(v), k).numpy() for k, v in out.items()}


def frames_in(path: Path, lo: int, hi: int, chunk: int = 32):
    """Yield (native index, frame) for native indices in [lo, hi]."""
    cap = cv2.VideoCapture(str(path))
    k = 0
    try:
        while k <= hi:
            if k < lo:
                if not cap.grab():
                    return
                k += 1
                continue
            ok, f = cap.read()
            if not ok:
                return
            yield k, f
            k += 1
    finally:
        cap.release()


def assign_sides(slot_box: dict, face_box: list) -> dict | None:
    """Anatomical side of hand slots 1/2 for a frontal signer: at the first sampled frame where both tracks
    exist, the image-left track is the right hand. If the two never co-occur, each track is placed by its side
    of the face centre; two tracks on the same side are one hand whose track was split (e.g. a one-handed sign).
    Returns {slot: is_right} or None if the assignment is ambiguous."""
    T = len(face_box)
    cx = lambda b: (b[0] + b[2]) / 2  # noqa: E731
    for t in range(T):
        b1, b2 = slot_box[1][t], slot_box[2][t]
        if b1 is not None and b2 is not None:
            if abs(cx(b1) - cx(b2)) < 1e-3:
                return None
            return {1: int(cx(b1) < cx(b2)), 2: int(cx(b2) < cx(b1))}
    sides = {}
    for s in (1, 2):
        votes = [cx(b) < cx(f) for b, f in zip(slot_box[s], face_box) if b is not None and f is not None]
        if votes:
            sides[s] = int(np.mean(votes) > 0.5)
    if len(sides) == 0:
        return None
    if len(sides) == 1:                         # one track only: the other slot gets the other side
        (s, r), = sides.items()
        sides[3 - s] = 1 - r
    return sides


def process_video(M: Models, path: Path, segs: list, target_fps: float, geometry: bool) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    step = max(1, int(round(fps / target_fps)))
    spans = [(sec(s["start"]), sec(s["end"])) for s in segs]
    lo = max(0, int((min(a for a, _ in spans) - 1.0) * fps))
    hi = int((max(b for _, b in spans) + 1.0) * fps)
    need = lambda k: any(a <= k / fps <= b for a, b in spans)  # noqa: E731

    # ---- pass 1: detection + tracking on every native frame, kept on the sampling grid -------------
    M.det.fps = fps
    keep, face_box, slot_box, slot_state, slot_score = [], [], {1: [], 2: []}, {1: [], 2: []}, {1: [], 2: []}
    H = W = None
    buf, buf_k = [], []

    def flush():
        for k, r in zip(buf_k, M.det.predict_batch(buf)):
            if k % step or not need(k):
                continue
            fb, sb = None, {1: None, 2: None}
            st, sc = {1: "", 2: ""}, {1: 0.0, 2: 0.0}
            for b in r.track_boxes:
                if b.label == "face" and fb is None:
                    fb = b.xyxy
                elif b.label == "hand" and b.track_id in sb:
                    sb[b.track_id], st[b.track_id], sc[b.track_id] = b.xyxy, b.state, b.score
            keep.append(k), face_box.append(fb)
            for s in (1, 2):
                slot_box[s].append(sb[s]), slot_state[s].append(st[s]), slot_score[s].append(sc[s])

    for k, f in frames_in(path, lo, hi):
        if H is None:
            H, W = f.shape[:2]
            M.det.reset(W, H)
        buf.append(f), buf_k.append(k)
        if len(buf) == 32:
            flush()
            buf, buf_k = [], []
    if buf:
        flush()
    if not keep:
        return {}
    sides = assign_sides(slot_box, face_box)

    # ---- pass 2: crops, DINOv3 and WiLoR on the kept frames ----------------------------------------
    T = len(keep)
    pos = {k: i for i, k in enumerate(keep)}
    streams = {"face": (face_box, PAD_FACE)}
    hand_slots = {"lh": [], "rh": []}
    if sides is not None:
        for s in (1, 2):
            hand_slots["rh" if sides[s] else "lh"].append(s)
    pick = {}                                   # per hand and frame: the slot that carries it (if any)
    for name, ss in hand_slots.items():
        pick[name] = [next((s for s in ss if slot_box[s][t] is not None), ss[0] if ss else None) for t in range(T)]
        if ss:
            streams[name] = ([slot_box[s][t] if s else None for t, s in enumerate(pick[name])], PAD_HAND)
    crops = {name: [None] * T for name in ("face", "lh", "rh")}
    frames = [None] * T if geometry else None
    for k, f in frames_in(path, keep[0], keep[-1]):
        i = pos.get(k)
        if i is None:
            continue
        for name, (boxes, pad) in streams.items():
            if boxes[i] is not None:
                crops[name][i] = square_crop(f, boxes[i], pad, CROP)
        if geometry:
            frames[i] = f
    feats = {}
    for name in ("face", "lh", "rh"):
        valid = np.asarray([c is not None for c in crops[name]])
        if name in ("lh", "rh") and sides is None:
            valid[:] = False                    # ambiguous side assignment: hand streams flagged invalid
        vi = np.nonzero(valid)[0]
        cls = np.zeros((T, 768), np.float16)
        if len(vi):
            cls[vi] = M.dino_cls([crops[name][i] for i in vi])
        boxes = streams[name][0] if name in streams else [None] * T
        d = {"valid": valid, "dino_cls": cls,
             "box": np.asarray([b if (b is not None and v) else (0, 0, 0, 0) for b, v in zip(boxes, valid)],
                               np.float32)}
        if name in ("lh", "rh"):
            d["state"] = np.asarray([slot_state[s][t] if s else "" for t, s in enumerate(pick[name])])
            d["score"] = np.asarray([slot_score[s][t] if s else 0.0 for t, s in enumerate(pick[name])], np.float32)
            if geometry and len(vi):
                w = M.wilor_targets([frames[i] for i in vi], [boxes[i] for i in vi], int(name == "rh"))
                for kk, v in w.items():
                    full = np.zeros((T,) + v.shape[1:], np.float16)
                    full[vi] = v.astype(np.float16)
                    d["wilor_" + kk] = full
        feats[name] = d

    fi = np.asarray(keep)
    out = {}
    for seg, (a, b) in zip(segs, spans):
        m = (fi / fps >= a) & (fi / fps <= b)
        if m.sum() < 2:
            continue
        r = {"fps": fps, "frame_idx": fi[m] - int(round(a * fps)), "hw": (H, W), "source": seg["yid"],
             "side_ambiguous": sides is None}
        for name, d in feats.items():
            r[name] = {kk: (v[m] if isinstance(v, np.ndarray) and len(v) == T else v) for kk, v in d.items()}
        out[seg["vid"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True, help="TSV: vid, yid, start, end (hh:mm:ss.fff)")
    ap.add_argument("--video-dir", required=True, help="directory of <yid>.mp4; a comma list is searched in order")
    ap.add_argument("--out", required=True, help="output prefix; parts are written as <out>.partK.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--videos-per-part", type=int, default=40)
    ap.add_argument("--target-fps", type=float, default=12.5)
    ap.add_argument("--dinov3", required=True)
    ap.add_argument("--face-ckpt", required=True)
    ap.add_argument("--hand-ckpt", required=True)
    ap.add_argument("--geometry", action="store_true", help="also extract WiLoR targets (pre-training data)")
    ap.add_argument("--wilor-root", default=None, help="WiLoR code release with pretrained_models/ and mano_data/")
    args = ap.parse_args()
    if args.geometry and not args.wilor_root:
        raise SystemExit("--geometry needs --wilor-root")

    by_video = defaultdict(list)
    with open(args.segments, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by_video[r["yid"]].append(r)
    src = {}
    for d in reversed(args.video_dir.split(",")):           # earlier directories take precedence
        src.update({y: Path(d) / f"{y}.mp4" for y in by_video if (Path(d) / f"{y}.mp4").exists()})
    yids = sorted(src)[args.shard::args.nshards]
    M = Models(args)
    t0 = time.time()
    for p0 in range(0, len(yids), args.videos_per_part):
        part = Path(f"{args.out}.part{p0 // args.videos_per_part:04d}.pt")
        if part.exists():
            continue
        out = {}
        for n, y in enumerate(yids[p0:p0 + args.videos_per_part], start=p0):
            try:
                out.update(process_video(M, src[y], by_video[y], args.target_fps, args.geometry))
            except Exception as e:  # keep going; failures are listed in the log
                print(f"[fail] {y}: {type(e).__name__}: {e}", flush=True)
            print(f"[{n + 1}/{len(yids)}] {y} segs={len(by_video[y])} {time.time() - t0:.0f}s", flush=True)
        part.parent.mkdir(parents=True, exist_ok=True)
        tmp = part.with_suffix(".tmp")
        torch.save(out, tmp)
        os.replace(tmp, part)
        print(f"saved {len(out)} segments -> {part}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
