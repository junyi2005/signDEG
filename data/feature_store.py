"""Frame-feature store + multi-temporal-crop dataset with geometric targets.

Layout (one directory per corpus; a handful of large files -> HPC-inode friendly):
    index.json                    {"videos": [{"vid", "split", "offset", "T", "fps", "src", "start"}], ...}
                                  (src: source video of the item, start: its first frame in that video;
                                  long source videos are stored as several consecutive items)
    <stream>_<key>.npy            (N_frames_total, ...) float16 / bool, frames of all videos concatenated
with stream in {lh, rh, face} and key in
    dino      frozen DINOv3 CLS (768)                          -- the only input at inference
    valid     tracker produced a box                           -- bool
    score     detector / tracker score                          -- float
    pred      box is a Kalman fill (ByteTrack lost track)      -- bool
    geo_ok    geometric FM produced an estimate                -- bool
    u         geometric latent, already normalised with corpus statistics (hands only)
    theta     MANO articulation 15x6D  (hands only)
    joint     wrist-relative 3D joints 21x3 (hands only)
    w         reliability weight w_t of Eq. (rel), computed over the full source video (hands only)
Rows are frames on the store's sampling grid (`fps` per video).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .temporal_crop import MultiCropConfig, TemporalMultiCropSampler

GEO_KEYS = ("u", "theta", "joint")


class FeatureStore:
    def __init__(self, root: str):
        self.root = Path(root)
        meta = json.loads((self.root / "index.json").read_text())
        self.videos: List[dict] = meta["videos"]
        self._arr: Dict[str, np.ndarray] = {}

    def has(self, stream: str, key: str) -> bool:
        return (self.root / f"{stream}_{key}.npy").exists()

    def arr(self, stream: str, key: str) -> np.ndarray:
        name = f"{stream}_{key}"
        if name not in self._arr:
            self._arr[name] = np.load(self.root / f"{name}.npy", mmap_mode="r")
        return self._arr[name]

    def get(self, v: dict, stream: str, key: str) -> np.ndarray:
        return np.array(self.arr(stream, key)[v["offset"]: v["offset"] + v["T"]])


class MultiStore:
    """Several FeatureStores behind the FeatureStore interface (e.g. How2Sign + isolated-sign clips).
    A key counts as present only if every store has it, so all samples carry the same targets."""

    def __init__(self, roots):
        self.stores = [FeatureStore(r) for r in roots]
        self.videos = [dict(v, _si=i) for i, st in enumerate(self.stores) for v in st.videos]

    def has(self, stream: str, key: str) -> bool:
        return all(st.has(stream, key) for st in self.stores)

    def get(self, v: dict, stream: str, key: str) -> np.ndarray:
        return self.stores[v["_si"]].get(v, stream, key)


def open_store(root: str):
    """'path' -> FeatureStore; 'path1,path2' -> MultiStore."""
    roots = [r for r in root.split(",") if r]
    return FeatureStore(roots[0]) if len(roots) == 1 else MultiStore(roots)


@dataclass
class GeoDatasetConfig:
    store_root: str
    stream: str                          # 'lh' | 'rh' | 'face'
    split: str = "train"
    multi_crop: MultiCropConfig = None
    geo_target: str = "geometry"         # 'geometry' (WiLoR targets, hand streams) | 'none' (face stream)
    kappa: float = 3.0
    predicted_scale: float = 0.5


class GeoStreamClipDataset(Dataset):
    """Multi-temporal-crop samples of one stream (global/local windows of cached DINOv3 features), plus
    per-global-window geometric targets `global_geo_{u,theta,joint,w}` when geo_target != 'none'."""

    def __init__(self, cfg: GeoDatasetConfig):
        from model.geo import reliability_weights
        self._rw = reliability_weights
        self.cfg = cfg
        self.store = open_store(cfg.store_root)
        self.videos = [v for v in self.store.videos if v["split"] == cfg.split and v["T"] >= 4]
        # global integer id of each source video (identical on every rank): GRT relates frames of
        # different source videos, and deduplicates frames by (source video, frame index)
        srcs = sorted({v.get("src", v["vid"]) for v in self.store.videos})
        self.src_id = {s: i for i, s in enumerate(srcs)}
        self.sampler = TemporalMultiCropSampler(cfg.multi_crop or MultiCropConfig())
        s = cfg.stream
        self.keys = []
        if cfg.geo_target == "geometry":
            self.keys = [k for k in ("u", "theta", "joint") if self.store.has(s, k)]

    def __len__(self) -> int:
        return len(self.videos)

    def _weights(self, v: dict, valid: np.ndarray) -> torch.Tensor:
        s = self.cfg.stream
        if self.store.has(s, "w"):          # reliability precomputed over the full source video (build_store)
            return torch.from_numpy(self.store.get(v, s, "w").astype(np.float32) * valid)
        ok = valid & self.store.get(v, s, "geo_ok") if self.store.has(s, "geo_ok") else valid
        score = self.store.get(v, s, "score").astype(np.float32) if self.store.has(s, "score") else np.ones(len(valid), np.float32)
        pred = self.store.get(v, s, "pred") if self.store.has(s, "pred") else np.zeros(len(valid), bool)
        if self.store.has(s, "theta"):
            th = torch.from_numpy(self.store.get(v, s, "theta").reshape(len(valid), -1).astype(np.float32))
        else:  # no articulation: reliability = detection only
            th = torch.zeros(len(valid), 1)
        return self._rw(th, torch.from_numpy(ok), torch.from_numpy(score), torch.from_numpy(pred),
                        kappa=self.cfg.kappa if self.store.has(s, "theta") else 0.0,
                        predicted_scale=self.cfg.predicted_scale)

    def __getitem__(self, i: int):
        v = self.videos[i]
        s = self.cfg.stream
        full = torch.from_numpy(self.store.get(v, s, "dino").astype(np.float32))
        valid = self.store.get(v, s, "valid").astype(bool)
        T = len(valid)
        globals_, locals_ = self.sampler.sample(T, valid)

        def take(idx):
            m = valid[idx].copy()
            m[1:] &= idx[1:] != idx[:-1]   # clips shorter than the window are padded by repeating the last
            return full[idx], torch.from_numpy(m), torch.from_numpy(idx.astype(np.int64))  # frame: mask repeats

        g_pack = [take(idx) for idx in globals_]
        l_pack = [take(idx) for idx in locals_]
        out = {
            "video_id": v["vid"],
            "global_frames": [g[0] for g in g_pack], "global_valid": [g[1] for g in g_pack],
            "global_time": [g[2] for g in g_pack],
            "local_frames": [x[0] for x in l_pack], "local_valid": [x[1] for x in l_pack],
            "local_time": [x[2] for x in l_pack],
        }
        if self.keys:
            w = self._weights(v, valid)
            out["global_geo_w"] = [w[idx] for idx in globals_]
            sid, start = self.src_id[v.get("src", v["vid"])], int(v.get("start", 0))
            out["global_geo_video"] = [torch.full((len(idx),), sid, dtype=torch.long) for idx in globals_]
            out["global_geo_frame"] = [torch.from_numpy(idx.astype(np.int64) + start) for idx in globals_]
            for k in self.keys:
                a = torch.from_numpy(self.store.get(v, s, k).reshape(T, -1).astype(np.float32))
                out[f"global_geo_{k}"] = [a[idx] for idx in globals_]
        return out
