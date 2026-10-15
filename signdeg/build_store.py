"""Merge extraction parts ({vid: record}, see signdeg/extract_clips.py / extract_segments.py) into a
FeatureStore -- streaming, one part in memory at a time.

  pass 1: index, and per-channel mean/std of the geometric latent u on *train* frames of this store
          (the source-training statistics used to standardise u, eps = 1e-6);
  pass 2: write every array into a preallocated .npy memmap.

Stored keys (see data/feature_store.py): dino, valid, score, pred, geo_ok for lh/rh/face; for hands with
geometry also u (WiLoR block-23 pose-query latent, standardised), theta, joint and the reliability weight w
(Eq. rel, computed over the full source video so that neighbours ignore item boundaries).

--chunk-frames splits long source videos (e.g. YouTube-ASL pre-training videos) into consecutive items of at
most that many frames; each item records its source video and start frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.geo import reliability_weights  # noqa: E402

HANDS = ("lh", "rh")
STREAMS = ("lh", "rh", "face")
U_KEY = "jtok23"         # WiLoR pose-query latent after block 23
NORM = {"u": "geo_ok"}   # standardised key -> mask selecting the frames used for its statistics
KAPPA, RHO = 3.0, 0.5    # reliability (Eq. rel)


def video_arrays(r: dict, slim: bool) -> dict:
    """All per-frame arrays of one source video, keyed '<stream>_<key>' (u not yet standardised)."""
    T = len(r["frame_idx"])
    z = lambda d: np.zeros((T, d), np.float16)  # noqa: E731
    out = {}
    for s in STREAMS:
        x = r[s]
        out[f"{s}_dino"] = x["dino_cls"]
        out[f"{s}_valid"] = x["valid"].astype(bool)
        if s in HANDS:
            out[f"{s}_score"] = x["score"].astype(np.float16)
            out[f"{s}_pred"] = x["state"] == "predicted"
            has = (not slim) and "wilor_hand_pose" in x
            out[f"{s}_geo_ok"] = x["valid"].astype(bool) if has else np.zeros(T, bool)
            if not slim:
                out[f"{s}_u"] = x[f"wilor_{U_KEY}"] if has else z(1280)
                out[f"{s}_theta"] = x["wilor_hand_pose"].reshape(T, -1) if has else z(90)
                out[f"{s}_joint"] = x["wilor_joints3d"].reshape(T, -1) if has else z(63)
                ok = out[f"{s}_valid"] & out[f"{s}_geo_ok"]
                w = reliability_weights(torch.from_numpy(out[f"{s}_theta"].astype(np.float32)),
                                        torch.from_numpy(ok), torch.from_numpy(x["score"].astype(np.float32)),
                                        torch.from_numpy(out[f"{s}_pred"]), kappa=KAPPA, predicted_scale=RHO)
                out[f"{s}_w"] = w.numpy().astype(np.float16)
        else:
            out[f"{s}_score"] = np.ones(T, np.float16)
            out[f"{s}_pred"] = np.zeros(T, bool)
            out[f"{s}_geo_ok"] = np.zeros(T, bool)
    return out


def chunks(T: int, n: int):
    if n <= 0 or T <= n:
        return [(0, T)]
    return [(a, min(a + n, T)) for a in range(0, T, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="extraction part files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-file", default=None, help="json {vid: split}; videos without an entry are 'train'")
    ap.add_argument("--slim", action="store_true", help="input features only (no geometric targets)")
    ap.add_argument("--chunk-frames", type=int, default=0,
                    help="split source videos into items of at most this many frames (0 = one item per video)")
    args = ap.parse_args()
    split = json.load(open(args.split_file)) if args.split_file else {}

    # ---- pass 1: index, shapes, train statistics of u ---------------------------------------
    index, off, shapes, dtypes, acc = [], 0, {}, {}, {}
    for p in args.shards:
        recs = torch.load(p, weights_only=False)
        for vid in sorted(recs):
            r = recs[vid]
            T = len(r["frame_idx"])
            if T == 0:
                continue
            sp = split.get(vid) or "train"
            a = video_arrays(r, args.slim)
            step = int(np.diff(r["frame_idx"][:2])[0]) if T > 1 else 1
            for k, (c0, c1) in enumerate(chunks(T, args.chunk_frames)):
                index.append({"vid": vid if args.chunk_frames <= 0 or T <= args.chunk_frames else f"{vid}#{k}",
                              "split": sp, "offset": off, "T": c1 - c0, "fps": r["fps"] / max(1, step),
                              "src": vid, "start": c0, "part": p})
                off += c1 - c0
            for k_, v in a.items():
                shapes.setdefault(k_, v.shape[1:])
                dtypes.setdefault(k_, v.dtype)
            if sp != "train":
                continue
            for k_, v in a.items():
                s, key = k_.split("_", 1)
                if key not in NORM:
                    continue
                x = v[a[f"{s}_{NORM[key]}"]].astype(np.float64)
                if not np.isfinite(x).all():
                    raise FloatingPointError(f"non-finite values in {k_} of {vid}")
                n, s1, s2 = acc.setdefault(k_, [0, 0.0, 0.0])
                acc[k_] = [n + len(x), s1 + x.sum(0), s2 + (x ** 2).sum(0)]
        del recs
        print(f"[pass1] {p}: {len(index)} items, {off} frames", flush=True)
    stats = {}
    for k_, (n, s1, s2) in acc.items():
        mu = s1 / max(n, 1)
        sd = np.sqrt(np.maximum(s2 / max(n, 1) - mu ** 2, 0)) + 1e-6
        stats[k_] = (mu.astype(np.float32), sd.astype(np.float32), n)

    # ---- pass 2: write memmaps ----------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mm = {k_: np.lib.format.open_memmap(out / f"{k_}.npy", mode="w+", dtype=dtypes[k_], shape=(off,) + shapes[k_])
          for k_ in shapes}
    by_part = {}
    for e in index:
        by_part.setdefault(e["part"], {}).setdefault(e["src"], []).append(e)
    for p, by_src in by_part.items():
        recs = torch.load(p, weights_only=False)
        for src, items in by_src.items():
            a = video_arrays(recs[src], args.slim)
            for k_, v in a.items():
                if k_ in stats:
                    mu, sd, _ = stats[k_]
                    v = ((v.astype(np.float32) - mu) / sd).astype(np.float16)
                for e in items:
                    mm[k_][e["offset"]:e["offset"] + e["T"]] = v[e["start"]:e["start"] + e["T"]]
        del recs
        print(f"[pass2] {p}", flush=True)
    for m in mm.values():
        m.flush()
    for k_, (mu, sd, n) in stats.items():
        np.savez(out / f"{k_}_stats.npz", mean=mu, std=sd, n=n)
    for e in index:
        e.pop("part")
    meta = {"videos": index, "u_key": U_KEY, "stats_frames": {k_: int(v[2]) for k_, v in stats.items()},
            "reliability": {"kappa": KAPPA, "rho": RHO}}
    (out / "index.json").write_text(json.dumps(meta))
    sp = dict(zip(*np.unique([e["split"] for e in index], return_counts=True)))
    print(f"store -> {out}: {len(index)} items, {off} frames; splits {sp}", flush=True)


if __name__ == "__main__":
    main()
