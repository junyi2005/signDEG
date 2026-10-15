"""Shared pieces of the downstream models (translation, isolated signs, fingerspelling).

Upstream: the three EMA-teacher temporal encoders of a SignDEG pre-training run (one checkpoint per stream).
Each returns its L+1 layer stack over a clip (layer 0 = projected DINOv3 input, layers 1..L = blocks).
Modes: 'frozen' (no gradient), 'live' (all encoder weights trained) or 'lora' (rank-r LoRA on every weight
matrix of the encoders, base weights frozen). Clips longer than the encoder's positional range are encoded
in consecutive windows.

Fusion: per stream, a learned softmax over the L+1 layers, LayerNorm and a linear projection to 256; the
three streams are masked by their validity, concatenated and projected to the task width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.temporal_encoder import TemporalTransformer  # noqa: E402

STREAMS = ("lh", "rh", "face")


def load_teacher(path: str) -> TemporalTransformer:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["cfg"]["model"]
    enc = TemporalTransformer(in_dim=m["in_dim"], embed_dim=m["embed_dim"], depth=m["depth"],
                              num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"], dropout=0.0,
                              max_len=m["max_len"])
    enc.load_state_dict({k[len("teacher_encoder."):]: v for k, v in ck["model"].items()
                         if k.startswith("teacher_encoder.")})
    return enc


class _LoRA(nn.Module):
    """Parametrisation W -> W + (alpha / r) B A, with B initialised to zero."""

    def __init__(self, out_f: int, in_f: int, r: int = 1, alpha: float = 1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, in_f) / in_f ** 0.5)
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scale = alpha / r

    def forward(self, W):
        return W + self.scale * (self.B @ self.A)


def add_lora(enc: nn.Module, r: int = 1) -> int:
    """Rank-r LoRA on every 2-D weight of the encoder (attention in/out projections, MLP, input projection);
    all other encoder parameters are frozen. Returns the number of LoRA parameters."""
    for p in enc.parameters():
        p.requires_grad_(False)
    n = 0
    for mod in list(enc.modules()):
        for name, p in list(mod.named_parameters(recurse=False)):
            if p.ndim == 2 and name in ("weight", "in_proj_weight"):
                lora = _LoRA(p.shape[0], p.shape[1], r)
                P.register_parametrization(mod, name, lora)
                n += sum(q.numel() for q in lora.parameters())
    return n


class Upstream(nn.Module):
    def __init__(self, ckpts: dict, mode: str = "frozen", lora_rank: int = 1):
        super().__init__()
        assert mode in ("frozen", "live", "lora")
        self.mode = mode
        self.enc = nn.ModuleDict({s: load_teacher(ckpts[s]) for s in STREAMS})
        self.n_lora = 0
        for e in self.enc.values():
            if mode == "frozen":
                e.eval()
                for p in e.parameters():
                    p.requires_grad_(False)
            elif mode == "lora":
                self.n_lora += add_lora(e, lora_rank)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode == "frozen":
            for e in self.enc.values():
                e.eval()
        return self

    def _encode(self, enc, x, m):
        """(B, T, 768) features, (B, T) validity -> (L+1, B, T, D), windowed over max_len frames."""
        B, T, _ = x.shape
        outs = []
        for a in range(0, T, enc.max_len):
            xs, ms = x[:, a:a + enc.max_len], m[:, a:a + enc.max_len]
            t_idx = torch.arange(xs.shape[1], device=x.device)[None].expand(B, -1)
            _, _, st = enc(xs, ms, None, t_idx, return_all_layers=True)
            outs.append(st)
        return torch.cat(outs, dim=2)

    def forward(self, batch):
        out = {}
        for s in STREAMS:
            x, m = batch[s], batch[s + "_valid"]
            if self.mode == "frozen":
                with torch.no_grad():
                    out[s] = self._encode(self.enc[s], x, m)
            else:
                out[s] = self._encode(self.enc[s], x, m)
        return out


class Fusion(nn.Module):
    def __init__(self, n_layers: int, dim: int, out_dim: int | None, per_stream: int = 256, dropout: float = 0.0):
        super().__init__()
        self.w = nn.ParameterDict({s: nn.Parameter(torch.zeros(n_layers)) for s in STREAMS})
        self.proj = nn.ModuleDict({s: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, per_stream))
                                   for s in STREAMS})
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(per_stream * len(STREAMS), out_dim) if out_dim else nn.Identity()
        self.out_dim = out_dim or per_stream * len(STREAMS)

    def layer_weights(self) -> dict:
        return {s: self.w[s].softmax(0).detach().cpu().tolist() for s in STREAMS}

    def forward(self, stacks: dict, batch: dict) -> torch.Tensor:
        parts = []
        for s in STREAMS:
            h = (stacks[s] * self.w[s].softmax(0)[:, None, None, None]).sum(0)
            h = self.proj[s](self.drop(h)) * batch[s + "_valid"][..., None].float()
            parts.append(h)
        return self.out(torch.cat(parts, -1))


def collate(batch: list) -> dict:
    """Pad per-stream (T, 768) features and validity to the longest clip; 'attn' marks real frames."""
    T = max(b[STREAMS[0]].shape[0] for b in batch)
    out = {k: [b[k] for b in batch] for k in batch[0] if k not in STREAMS and not k.endswith("_valid")}
    for s in STREAMS:
        x = torch.zeros(len(batch), T, batch[0][s].shape[1])
        m = torch.zeros(len(batch), T, dtype=torch.bool)
        for i, b in enumerate(batch):
            t = b[s].shape[0]
            x[i, :t], m[i, :t] = b[s], b[s + "_valid"]
        out[s], out[s + "_valid"] = x, m
    out["attn"] = torch.zeros(len(batch), T, dtype=torch.long)
    for i, b in enumerate(batch):
        out["attn"][i, : b[STREAMS[0]].shape[0]] = 1
    return out


def clip_features(store, v: dict, a: int = 0, b: int | None = None, max_frames: int = 0) -> dict:
    """Per-stream features of frames [a, b) of store item v; uniformly subsampled to max_frames if longer."""
    b = v["T"] if b is None else b
    idx = np.arange(a, b)
    if max_frames and len(idx) > max_frames:
        idx = np.linspace(a, b - 1, max_frames).round().astype(int)
    out = {}
    for s in STREAMS:
        arr_x, arr_m = store.arr(s, "dino"), store.arr(s, "valid")
        o = v["offset"]
        out[s] = torch.from_numpy(np.asarray(arr_x[o + idx[0]:o + idx[-1] + 1])[idx - idx[0]].astype(np.float32))
        out[s + "_valid"] = torch.from_numpy(np.asarray(arr_m[o + idx[0]:o + idx[-1] + 1])[idx - idx[0]].astype(bool))
    return out
