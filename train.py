"""SignDEG self-supervised pre-training of one articulator stream (paper Sec. 3, Appendix A/E).

    torchrun --nproc_per_node 4 train.py --config configs/signdeg_hand.yaml --stream lh --store STORE --output-root runs/ssl
    torchrun --nproc_per_node 4 train.py --config configs/signdeg_hand.yaml --stream rh --store STORE --output-root runs/ssl
    torchrun --nproc_per_node 4 train.py --config configs/signdeg_face.yaml           --store STORE --output-root runs/ssl
    python train.py --config configs/signdeg_hand.yaml --synthetic-smoke --max-steps 20      # no data needed

The student / EMA-teacher temporal Transformer is trained on cached frozen-DINOv3 crop features (a FeatureStore,
data/feature_store.py) with temporal self-distillation (DINO + iBOT + KoLeo) and, for the hand streams, GLP and
GRT (model/geo.py). All terms are active from the first update. The schedule is a list of optimisation segments
(configs: 400K + 50K updates); each segment has its own learning-rate, weight-decay, EMA and teacher-temperature
schedule. Data-parallel training uses torchrun (one process per GPU); Sinkhorn-Knopp and GRT operate on the
distributed batch. Checkpoints: <output-root>/<exp-name>/ckpt_step<N>.pt and ckpt_final.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data import multi_crop_collate, MultiCropConfig  # noqa: E402
from data.feature_store import GeoDatasetConfig, GeoStreamClipDataset  # noqa: E402
from model import SignTemporalDinoModel  # noqa: E402
from model.geo import GeoConfig  # noqa: E402
from model.meta_arch import SignDinoConfig  # noqa: E402


# --------- helpers ---------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed():
    """torchrun sets RANK / WORLD_SIZE / LOCAL_RANK; a plain `python train.py` runs as one process."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return world, rank, device


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, stream: str):
    mc = MultiCropConfig(**{k: cfg["multi_crop"][k] for k in (
        "n_global", "n_local", "global_len_min", "global_len_max", "local_len_min", "local_len_max",
        "min_valid_frames")})
    g = cfg.get("geo", {})
    return GeoStreamClipDataset(GeoDatasetConfig(
        store_root=cfg["data"]["store_root"], stream=stream, split=cfg["data"]["split"],
        multi_crop=mc, geo_target=g.get("target", "none"),
        kappa=g.get("kappa", 3.0), predicted_scale=g.get("predicted_scale", 0.5),
    ))


def build_model(cfg: dict) -> SignTemporalDinoModel:
    m = cfg["model"]
    keys = ("in_dim", "embed_dim", "depth", "num_heads", "mlp_ratio", "dropout", "max_len", "head_out_dim",
            "head_hidden", "head_bottleneck", "head_layers", "head_norm_last", "student_temp", "teacher_temp",
            "teacher_temp_start", "sinkhorn_iters", "dino_weight", "ibot_weight", "dkoleo_weight")
    geo = GeoConfig(**{k: v for k, v in cfg.get("geo", {}).items() if k in GeoConfig.__dataclass_fields__})
    return SignTemporalDinoModel(SignDinoConfig(**{k: m[k] for k in keys}, geo=geo))


def cosine(base: float, final: float, t: float) -> float:
    """Cosine interpolation from base (t=0) to final (t=1)."""
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))


def segment_values(seg: dict, i: int, min_lr: float) -> dict:
    """Schedules of one optimisation segment at its i-th update."""
    n = seg["steps"]
    warm = int(round(seg["warmup_frac"] * n))
    if i < warm:
        lr = seg["base_lr"] * (i + 1) / warm
    else:
        lr = cosine(seg["base_lr"], min_lr, (i - warm) / max(1, n - warm))
    tt0, tt1 = seg["teacher_temp"]
    tw = int(round(seg.get("teacher_temp_warmup_frac", 0.0) * n))
    tt = tt0 + (tt1 - tt0) * min(1.0, i / tw) if tw > 0 else tt1
    return {"lr": lr, "wd": cosine(seg["weight_decay"][0], seg["weight_decay"][1], i / n),
            "ema_m": cosine(seg["ema_momentum"][0], seg["ema_momentum"][1], i / n), "teacher_temp": tt}


class _Synth(Dataset):
    """Random features with the sample contract of GeoStreamClipDataset (no geometric targets)."""

    def __init__(self, cfg, n=64):
        self.n, self.D, self.mc = n, cfg["model"]["in_dim"], cfg["multi_crop"]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        def crop(lo, hi):
            t = int(torch.randint(lo, hi + 1, (1,)).item())
            return torch.randn(t, self.D), torch.ones(t, dtype=torch.bool), torch.arange(t)
        g = [crop(self.mc["global_len_min"], self.mc["global_len_max"]) for _ in range(self.mc["n_global"])]
        l_ = [crop(self.mc["local_len_min"], self.mc["local_len_max"]) for _ in range(self.mc["n_local"])]
        return {"video_id": f"synth_{i}",
                "global_frames": [x[0] for x in g], "global_valid": [x[1] for x in g],
                "global_time": [x[2] for x in g],
                "local_frames": [x[0] for x in l_], "local_valid": [x[1] for x in l_],
                "local_time": [x[2] for x in l_]}


# --------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stream", default=None, choices=["lh", "rh", "face"])
    ap.add_argument("--exp-name", default=None, help="default: signdeg_<stream>")
    ap.add_argument("--store", default=None, help="overrides data.store_root (comma-separated list allowed)")
    ap.add_argument("--output-root", default=None, help="overrides train.output_root")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (model, optimiser, step)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many updates (smoke tests)")
    ap.add_argument("--synthetic-smoke", action="store_true", help="random features instead of a store")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.stream is not None:
        cfg["stream"] = args.stream
    cfg["exp_name"] = args.exp_name or f"signdeg_{cfg['stream']}"
    if args.store is not None:
        cfg["data"]["store_root"] = args.store
    if args.output_root is not None:
        cfg["train"]["output_root"] = args.output_root
    if cfg["stream"] == "face" and cfg["geo"].get("target") == "geometry":
        raise SystemExit("the face stream has no hand geometry: use configs/signdeg_face.yaml")

    world, rank, device = setup_distributed()
    main_proc = rank == 0
    set_seed(cfg.get("seed", 0) + rank)
    tr = cfg["train"]

    ds = _Synth(cfg) if args.synthetic_smoke else build_dataset(cfg, cfg["stream"])
    sampler = DistributedSampler(ds, world, rank, shuffle=True, drop_last=True) if world > 1 else None
    loader = DataLoader(
        ds, batch_size=tr["batch_size_per_gpu"], shuffle=sampler is None, sampler=sampler,
        num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"], drop_last=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
        collate_fn=lambda b: multi_crop_collate(b, ibot_mask_min=cfg["ibot"]["mask_min"],
                                                ibot_mask_max=cfg["ibot"]["mask_max"]),
    )

    model = build_model(cfg).to(device)
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
    params = [p for p in core.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=tr["segments"][0]["base_lr"])

    segments = tr["segments"]
    total_steps = sum(s["steps"] for s in segments)
    step, epoch = 0, 0
    output_dir = Path(tr["output_root"]) / cfg["exp_name"]
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step, epoch = ck["step"], ck.get("epoch", 0)
        if main_proc:
            print(f"[train] resumed from {args.resume} at step {step}")
    if main_proc:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[train] {world} process(es), {total_steps} updates, outputs -> {output_dir}")
    log_jsonl = open(output_dir / "train.jsonl", "a") if main_proc else None

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[tr.get("amp", "bf16")]
    use_amp = amp is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == torch.float16)

    def save(name):
        if main_proc:
            torch.save({"step": step, "epoch": epoch, "model": core.state_dict(),
                        "optimizer": optimizer.state_dict(), "cfg": cfg}, output_dir / name)
            print(f"[ckpt] {output_dir / name}")

    t0 = time.time()
    stop = min(total_steps, args.max_steps) if args.max_steps > 0 else total_steps
    while step < stop:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= stop:
                break
            # which segment, and the update index inside it
            i, si = step, 0
            while i >= segments[si]["steps"]:
                i -= segments[si]["steps"]
                si += 1
            sv = segment_values(segments[si], i, tr["min_lr"])
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[1]["lr"] = sv["lr"]
            optimizer.param_groups[0]["weight_decay"] = sv["wd"]

            for k in [k for k in batch if torch.is_tensor(batch[k])]:
                batch[k] = batch[k].to(device, non_blocking=True)
            student_views = {k: batch[k] for k in (
                "global_frames", "global_valid", "global_time", "local_frames", "local_valid", "local_time",
                "ibot_mask")}
            student_views.update({k: batch[k] for k in batch if k.startswith("geo_")})
            teacher_views = {k: batch[k] for k in ("global_frames", "global_valid", "global_time")}

            with torch.autocast(device_type=device.type, dtype=amp or torch.float32, enabled=use_amp):
                losses = model(student_views, teacher_views, sv["teacher_temp"], step)
                loss = losses["loss_total"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tr["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            core.ema_step(sv["ema_m"])

            if main_proc and step % tr["log_every"] == 0:
                rec = {"step": step, "segment": si, **sv, **{k: float(v.detach()) for k, v in losses.items()},
                       "elapsed": time.time() - t0}
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}),
                      flush=True)
                log_jsonl.write(json.dumps(rec) + "\n")
                log_jsonl.flush()
            step += 1
            if step % tr["ckpt_every"] == 0 or step == sum(s["steps"] for s in segments[:si + 1]):
                save(f"ckpt_step{step}.pt")
        epoch += 1
    save("ckpt_final.pt")
    if log_jsonl:
        log_jsonl.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if main_proc:
        print("[train] done")


if __name__ == "__main__":
    main()
