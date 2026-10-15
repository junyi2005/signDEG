"""Fingerspelling detection on SignDEG encoders (paper Sec. 4.3, Appendix E; ASL-STEM-Wiki).

Per-frame binary classification: the stream-fused representation (signdeg/downstream.py) passes through a
two-layer Transformer and a frame head; at test time the frame probabilities are thresholded (0.5), median
smoothed, and turned into intervals. The metric is the mean interval-IoU over test videos, under the
dataset's cross-validation folds (--test-fold; the next fold is used for validation).

usage: fingerspelling.py --store STORE --labels FS.json --upstream LH.pt RH.pt FACE.pt --test-fold 0 --out DIR
FS.json: {vid: {"fold": int, "intervals": [[start_s, end_s], ...]}} for every video in the store.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.feature_store import FeatureStore  # noqa: E402
from downstream import STREAMS, Fusion, Upstream, clip_features, collate  # noqa: E402


def frame_labels(v: dict, intervals: list) -> np.ndarray:
    t = np.arange(v["T"]) / v["fps"]
    y = np.zeros(v["T"], np.float32)
    for a, b in intervals:
        y[(t >= a) & (t <= b)] = 1.0
    return y


class Videos(Dataset):
    def __init__(self, store, labels, folds):
        self.store = store
        self.items = [(v, frame_labels(v, labels[v["vid"]]["intervals"])) for v in store.videos
                      if v["vid"] in labels and labels[v["vid"]]["fold"] in folds]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v, y = self.items[i]
        return {"vid": v["vid"], "y": torch.from_numpy(y), **clip_features(self.store, v)}


def collate_fs(b):
    out = collate(b)
    y = torch.zeros(out["attn"].shape)
    for i, x in enumerate(b):
        y[i, :len(x["y"])] = x["y"]
    out["y"] = y
    return out


class Detector(nn.Module):
    def __init__(self, upstream: Upstream, d: int = 256, heads: int = 4):
        super().__init__()
        self.up = upstream
        e = next(iter(upstream.enc.values()))
        self.fusion = Fusion(n_layers=e.depth + 1, dim=e.embed_dim, out_dim=d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(d, 1)

    def forward(self, b):
        h = self.tf(self.fusion(self.up(b), b), src_key_padding_mask=b["attn"] == 0)
        return self.head(h).squeeze(-1)


def median_smooth(p: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(p) == 0:
        return p
    pad = np.pad(p, (k // 2, k // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)


def interval_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU between the frames covered by predicted and by reference intervals of one video."""
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--upstream", nargs=3, required=True, metavar=("LH", "RH", "FACE"))
    ap.add_argument("--upstream-mode", choices=["frozen", "lora", "live"], default="frozen")
    ap.add_argument("--test-fold", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--median", type=int, default=5, help="median-filter width in frames")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed), np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(a.store)
    labels = json.load(open(a.labels))
    folds = sorted({x["fold"] for x in labels.values()})
    val_fold = folds[(folds.index(a.test_fold) + 1) % len(folds)]
    train_folds = [f for f in folds if f not in (a.test_fold, val_fold)]
    dl = {name: DataLoader(Videos(store, labels, fs), batch_size=a.bs if name == "train" else 1,
                           shuffle=name == "train", num_workers=4, collate_fn=collate_fs)
          for name, fs in (("train", train_folds), ("val", [val_fold]), ("test", [a.test_fold]))}
    model = Detector(Upstream(dict(zip(STREAMS, a.upstream)), a.upstream_mode)).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        ious = []
        for b in dl[split]:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            p = torch.sigmoid(model(b)[0].float()).cpu().numpy()
            pred = median_smooth((p >= a.threshold).astype(np.float32), a.median) >= 0.5
            ious.append(interval_iou(pred, b["y"][0].cpu().numpy() >= 0.5))
        model.train()
        return float(np.mean(ious)) if ious else 0.0

    best, t0 = -1.0, time.time()
    for ep in range(a.epochs):
        for b in dl["train"]:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            lo = model(b).float()
            m = b["attn"].float()
            loss = (F.binary_cross_entropy_with_logits(lo, b["y"], reduction="none") * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = evaluate("val")
        print(json.dumps({"ep": ep + 1, "loss": round(float(loss), 4), "val_miou": v, "t": int(time.time() - t0)}),
              flush=True)
        if v > best:
            best = v
            torch.save({"model": model.state_dict(), "ep": ep + 1}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=False)["model"])
    res = {"test_fold": a.test_fold, "val_fold": val_fold, "best_val_miou": best, "test_miou": evaluate("test")}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
