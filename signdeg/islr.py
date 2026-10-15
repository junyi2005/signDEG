"""Isolated sign recognition on SignDEG encoders (paper Sec. 4.3; ASL Citizen, Sem-Lex, WLASL2000).

Head: the stream-fused representation (signdeg/downstream.py) is averaged over valid frames (mask-aware),
then BatchNorm and a linear classifier. Regimes (--mode):
    frozen   frozen encoders, linear head
    lora     rank-1 LoRA on the temporal encoders
    full     full fine-tuning of the temporal encoders
The checkpoint with the best validation Recall@1 is evaluated once on the test split: Recall@1/5/10, and
per-instance / per-class top-1 accuracy (the WLASL2000 metrics).

usage: islr.py --store STORE --labels LABELS.json --upstream LH.pt RH.pt FACE.pt --mode lora --out DIR
LABELS.json: {vid: gloss}; the split of each clip is taken from the store index.
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


class Clips(Dataset):
    def __init__(self, store, labels, cls, split, max_frames):
        self.store, self.max_frames = store, max_frames
        self.items = [(v, cls[labels[v["vid"]]]) for v in store.videos if v["split"] == split and v["vid"] in labels]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v, y = self.items[i]
        return {"vid": v["vid"], "y": y, **clip_features(self.store, v, max_frames=self.max_frames)}


class Recognizer(nn.Module):
    def __init__(self, upstream: Upstream, n_cls: int):
        super().__init__()
        self.up = upstream
        e = next(iter(upstream.enc.values()))
        self.fusion = Fusion(n_layers=e.depth + 1, dim=e.embed_dim, out_dim=None)
        self.bn = nn.BatchNorm1d(self.fusion.out_dim)
        self.cls = nn.Linear(self.fusion.out_dim, n_cls)

    def forward(self, b):
        h = self.fusion(self.up(b), b)                              # (B, T, d)
        m = torch.stack([b[s + "_valid"] for s in STREAMS]).any(0).float()[..., None]
        h = (h * m).sum(1) / m.sum(1).clamp_min(1)                   # mask-aware temporal average
        return self.cls(self.bn(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--upstream", nargs=3, required=True, metavar=("LH", "RH", "FACE"))
    ap.add_argument("--mode", choices=["frozen", "lora", "full"], default="lora")
    ap.add_argument("--lora-rank", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=125)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed), np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(a.store)
    labels = json.load(open(a.labels))
    cls = {g: i for i, g in enumerate(sorted({labels[v["vid"]] for v in store.videos if v["vid"] in labels}))}

    def loader(split, shuffle):
        y = lambda b: {**collate(b), "y": torch.tensor([x["y"] for x in b])}  # noqa: E731
        return DataLoader(Clips(store, labels, cls, split, a.max_frames), batch_size=a.bs, shuffle=shuffle,
                          num_workers=a.workers, collate_fn=y, drop_last=shuffle)
    dl = {sp: loader(sp, sp == "train") for sp in ("train", "val", "test")}
    up = Upstream(dict(zip(STREAMS, a.upstream)), {"frozen": "frozen", "lora": "lora", "full": "live"}[a.mode],
                  a.lora_rank)
    model = Recognizer(up, len(cls)).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print({"classes": len(cls), **{k: len(v.dataset) for k, v in dl.items()}, "mode": a.mode,
           "lora_params": up.n_lora, "trainable_params": sum(p.numel() for p in params)}, flush=True)
    opt = torch.optim.Adam(params, lr=a.lr)

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        hits = np.zeros(3)
        per_cls_ok, per_cls_n = np.zeros(len(cls)), np.zeros(len(cls))
        n = 0
        for b in dl[split]:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lo = model(b).float()
            top = lo.topk(min(10, lo.shape[-1]), -1).indices
            ok = top == b["y"][:, None]
            hits += [ok[:, :k].any(1).sum().item() for k in (1, 5, 10)]
            y, t1 = b["y"].cpu().numpy(), ok[:, 0].cpu().numpy()
            np.add.at(per_cls_n, y, 1)
            np.add.at(per_cls_ok, y, t1)
            n += len(y)
        model.train()
        seen = per_cls_n > 0
        return {"R@1": hits[0] / max(n, 1), "R@5": hits[1] / max(n, 1), "R@10": hits[2] / max(n, 1),
                "per_instance_top1": 100 * hits[0] / max(n, 1),
                "per_class_top1": 100 * float((per_cls_ok[seen] / per_cls_n[seen]).mean()) if seen.any() else 0.0}

    best, t0, log = -1.0, time.time(), open(out / "log.jsonl", "a")
    for ep in range(a.epochs):
        for b in dl["train"]:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(b).float(), b["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = evaluate("val")
        rec = {"ep": ep + 1, "loss": round(float(loss), 4), **{f"val_{k}": x for k, x in v.items()},
               "t": int(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"), log.flush()
        if v["R@1"] > best:
            best = v["R@1"]
            torch.save({"model": model.state_dict(), "ep": ep + 1}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=False)["model"])
    res = {"best_val_R@1": best, **{f"test_{k}": x for k, x in evaluate("test").items()}, "mode": a.mode}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
