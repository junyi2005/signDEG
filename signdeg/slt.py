"""Gloss-free sign-to-English translation on SignDEG encoders (paper Sec. 4.2, Table "slt_train").

The layer-weighted features of each stream are normalised, projected, concatenated and fed to ByT5-Base
(signdeg/downstream.py). Training has two stages:

  source stage   captioned segments of the source pre-training videos (YouTube-ASL split; the caption table
                 rows with split 'train'), read from the pre-training store by time span;
  target stage   How2Sign or OpenASL training clips, initialised from the source-stage checkpoint; the
                 checkpoint with the best validation BLEU-4 is decoded on the test split.

Variants: --upstream-mode live updates the temporal encoders at 1/10 of the decoder learning rate; 'frozen'
trains only the fusion projection and ByT5. --eval-only decodes a split with a trained checkpoint (e.g.
zero-shot FLEURS-ASL after the source stage). Data-parallel with torchrun; the effective batch is
batch-per-gpu x GPUs x accumulation steps.

  torchrun --nproc_per_node 4 signdeg/slt.py --stage source --store SOURCE_STORE --captions youtube_asl.tsv \
      --upstream LH.pt RH.pt FACE.pt --upstream-mode live --backbone weights/byt5-base --out runs/slt/source
  torchrun --nproc_per_node 4 signdeg/slt.py --stage target --store H2S_STORE --text how2sign_{train,val,test}.csv \
      --init runs/slt/source/final.pt --upstream LH.pt RH.pt FACE.pt --upstream-mode live \
      --backbone weights/byt5-base --out runs/slt/how2sign
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.feature_store import FeatureStore  # noqa: E402
from downstream import STREAMS, Fusion, Upstream, clip_features, collate  # noqa: E402

STAGE_DEFAULTS = {   # Table "slt_train"
    "source": dict(steps=250_000, lr=5e-4, betas=(0.9, 0.99), warmup=10_000),
    "target": dict(steps=50_000, lr=1e-4, betas=(0.9, 0.999), warmup=5_000),
}


def sec(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_text(paths) -> dict:
    """{vid: (split, sentence)} from How2Sign CSVs (SENTENCE_NAME / SENTENCE; split from the file name),
    OpenASL-style tables (vid / raw-text / split) or generic TSVs (vid / text / split)."""
    text = {}
    for p in paths:
        name = Path(p).stem.lower()
        file_split = next((s for s in ("train", "val", "test") if name.endswith(s)), None)
        with open(p, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if "SENTENCE_NAME" in r:
                    text[r["SENTENCE_NAME"]] = (file_split, r["SENTENCE"].strip())
                else:
                    sent = (r.get("raw-text") or r.get("text") or "").strip()
                    sp = {"validation": "val", "dev": "val"}.get(r.get("split", ""), r.get("split") or file_split)
                    if sent:
                        text[r["vid"]] = (sp, sent)
    return text


class ClipText(Dataset):
    """Store items with a sentence (target stage, evaluation)."""

    def __init__(self, store: FeatureStore, text: dict, split: str, max_frames: int):
        self.store, self.max_frames = store, max_frames
        self.items = [v for v in store.videos if v["vid"] in text and text[v["vid"]][0] == split]
        self.text = text

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v = self.items[i]
        return {"vid": v["vid"], "text": self.text[v["vid"]][1],
                **clip_features(self.store, v, max_frames=self.max_frames)}


class CaptionSpans(Dataset):
    """Caption segments (vid, yid, start, end, raw-text, split) read by time span from the pre-training store,
    whose items are consecutive chunks of whole source videos (fields src / start, signdeg/build_store.py)."""

    def __init__(self, store: FeatureStore, table: str, split: str, max_frames: int):
        self.store, self.max_frames = store, max_frames
        first = {}
        for v in store.videos:                              # chunk 0 of every source video
            if v.get("start", 0) == 0:
                first[v.get("src", v["vid"])] = v
        length = {}
        for v in store.videos:
            s = v.get("src", v["vid"])
            length[s] = length.get(s, 0) + v["T"]
        self.items = []
        with open(table, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                sent = (r.get("raw-text") or "").strip()
                sp = {"validation": "val"}.get(r.get("split", "train"), r.get("split", "train"))
                v = first.get(r["yid"])
                if not sent or sp != split or v is None:
                    continue
                a = int(round(sec(r["start"]) * v["fps"]))
                b = min(int(round(sec(r["end"]) * v["fps"])), length[r["yid"]])
                if b - a >= 2:
                    self.items.append((v, a, b, sent, r["vid"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v, a, b, sent, vid = self.items[i]
        # the chunks of one source video are contiguous in the store, so the span is one slice
        span = {"offset": v["offset"], "T": b}
        return {"vid": vid, "text": sent, **clip_features(self.store, span, a, b, self.max_frames)}


class Translator(nn.Module):
    def __init__(self, upstream: Upstream, backbone: str):
        super().__init__()
        from transformers import T5ForConditionalGeneration
        self.up = upstream
        self.t5 = T5ForConditionalGeneration.from_pretrained(backbone)
        e = next(iter(upstream.enc.values()))
        self.fusion = Fusion(n_layers=e.depth + 1, dim=e.embed_dim, out_dim=self.t5.config.d_model)

    def encode(self, batch):
        return self.fusion(self.up(batch), batch), batch["attn"]

    def forward(self, batch, labels, smoothing):
        emb, am = self.encode(batch)
        logits = self.t5(inputs_embeds=emb, attention_mask=am, labels=labels).logits
        return F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, label_smoothing=smoothing)


def bleu(hyps, refs):
    from sacrebleu.metrics import BLEU
    return BLEU().corpus_score(hyps, [refs]).score, BLEU(max_ngram_order=1).corpus_score(hyps, [refs]).score


@torch.no_grad()
def decode(model: Translator, tok, loader, dev, beams: int, max_len: int):
    model.eval()
    hyps, refs = [], []
    for b in loader:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            emb, am = model.encode(b)
            gen = model.t5.generate(inputs_embeds=emb, attention_mask=am, num_beams=beams, max_new_tokens=max_len)
        hyps += tok.batch_decode(gen, skip_special_tokens=True)
        refs += b["text"]
    model.train()
    return hyps, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["source", "target"], required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--captions", default=None, help="source stage: caption table (vid, yid, start, end, raw-text, split)")
    ap.add_argument("--text", nargs="+", default=None, help="target stage / evaluation: sentence files")
    ap.add_argument("--upstream", nargs=3, required=True, metavar=("LH", "RH", "FACE"),
                    help="SignDEG pre-training checkpoints of the three streams")
    ap.add_argument("--upstream-mode", choices=["frozen", "live"], default="live")
    ap.add_argument("--backbone", default="google/byt5-base")
    ap.add_argument("--init", default=None, help="checkpoint to start from (target stage: the source-stage final.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="ByT5 + fusion; live encoders use lr / 10")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--batch-per-gpu", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.2)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--max-out", type=int, default=384)
    ap.add_argument("--max-frames", type=int, default=512, help="uniform subsampling of longer clips")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-only", default=None, help="decode this split with --init and exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = STAGE_DEFAULTS[args.stage]
    steps, lr, warmup = args.steps or d["steps"], args.lr or d["lr"], args.warmup or d["warmup"]

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    dev = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.manual_seed(args.seed + rank), np.random.seed(args.seed + rank)
    main_proc = rank == 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = Translator(Upstream(dict(zip(STREAMS, args.upstream)), args.upstream_mode), args.backbone).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
    store = FeatureStore(args.store)
    text = load_text(args.text) if args.text else {}
    mk = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=args.batch_per_gpu, shuffle=shuffle and world == 1, num_workers=4, collate_fn=collate,
        sampler=DistributedSampler(ds, world, rank, shuffle=True, drop_last=True) if (shuffle and world > 1) else None,
        drop_last=shuffle)

    if args.eval_only:
        if main_proc:
            hyps, refs = decode(model, tok, mk(ClipText(store, text, args.eval_only, args.max_frames), False), dev,
                                args.beams, args.max_out)
            b4, b1 = bleu(hyps, refs)
            (out / f"{args.eval_only}_hyps.txt").write_text("\n".join(hyps))
            (out / f"{args.eval_only}_refs.txt").write_text("\n".join(refs))
            res = {"split": args.eval_only, "bleu4": b4, "bleu1": b1, "n": len(refs)}
            (out / f"{args.eval_only}_result.json").write_text(json.dumps(res, indent=1))
            print("RESULT", json.dumps(res), flush=True)
        if world > 1:
            dist.destroy_process_group()
        return

    train_ds = (CaptionSpans(store, args.captions, "train", args.max_frames) if args.stage == "source"
                else ClipText(store, text, "train", args.max_frames))
    val_loader = mk(ClipText(store, text, "val", args.max_frames), False) if args.stage == "target" else None
    loader = mk(train_ds, True)
    if main_proc:
        print({"stage": args.stage, "train_items": len(train_ds), "steps": steps, "world": world,
               "effective_batch": args.batch_per_gpu * world * args.accum}, flush=True)

    enc_params = [p for p in model.up.parameters() if p.requires_grad]
    head_params = list(model.fusion.parameters()) + list(model.t5.parameters())
    groups = [{"params": head_params, "lr": lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": lr / 10})
    opt = torch.optim.AdamW(groups, betas=d["betas"], weight_decay=args.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    sched = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))  # noqa: E731
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if dev.type == "cuda" else None,
                                                    find_unused_parameters=True) if world > 1 else model

    state = out / "state.pt"
    step, best = 0, -1.0
    if state.exists():
        st = torch.load(state, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        step, best = st["step"], st["best"]
    log = open(out / "log.jsonl", "a") if main_proc else None
    t0 = time.time()

    def stream():                       # endless batches; accumulation may span epoch boundaries
        epoch = 0
        while True:
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
            yield from loader
            epoch += 1
    it = stream()
    while step < steps:
        batches = [next(it) for _ in range(args.accum)]
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * sched(step)
        for i, b in enumerate(batches):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            lab = tok(b["text"], return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_out).input_ids.to(dev)
            lab[lab == tok.pad_token_id] = -100
            sync = (i == len(batches) - 1) or world == 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx, torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = ddp(b, lab, args.label_smoothing) / args.accum
                loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if main_proc and step % args.log_every == 0:
            rec = {"step": step, "loss": round(float(loss) * args.accum, 4), "lr": opt.param_groups[0]["lr"],
                   "t": int(time.time() - t0)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"), log.flush()
        if step % args.eval_every == 0 or step == steps:
            if main_proc:
                rec = {"step": step}
                if val_loader is not None:
                    hyps, refs = decode(model, tok, val_loader, dev, args.beams, args.max_out)
                    rec["val_bleu4"], rec["val_bleu1"] = bleu(hyps, refs)
                    if rec["val_bleu4"] > best:
                        best = rec["val_bleu4"]
                        torch.save({"model": model.state_dict(), "step": step, "val_bleu4": best}, out / "best.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best},
                           str(state) + ".tmp")
                os.replace(str(state) + ".tmp", state)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"), log.flush()
            if world > 1:
                dist.barrier()

    if main_proc:
        torch.save({"model": model.state_dict(), "step": step}, out / "final.pt")
        if args.stage == "target":
            model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model"])
            hyps, refs = decode(model, tok, mk(ClipText(store, text, "test", args.max_frames), False), dev,
                                args.beams, args.max_out)
            b4, b1 = bleu(hyps, refs)
            (out / "test_hyps.txt").write_text("\n".join(hyps))
            (out / "test_refs.txt").write_text("\n".join(refs))
            res = {"best_val_bleu4": best, "test_bleu4": b4, "test_bleu1": b1, "n_test": len(refs)}
            (out / "result.json").write_text(json.dumps(res, indent=1))
            print("RESULT", json.dumps(res), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
