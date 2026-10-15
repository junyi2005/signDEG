"""Per-stream temporal Transformer.

Takes a sequence of frozen-DINOv3 per-frame embeddings + a sinusoidal time
positional encoding, prepends a learnable CLS token, and returns

    cls:    (B, D)
    frames: (B, T, D)   -- contextualised per-frame tokens

Supports an iBOT mask: positions where `frame_mask` is True are replaced
with a learnable [MASK] embedding before the projection, and any
positions where `valid_mask` is False are masked out of attention.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding of shape (max_len, dim)."""
    assert dim % 2 == 0, "sinusoidal PE requires even dim"
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_len: int = 256,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_dim))  # in input space
        self.register_buffer("pos_embed", build_sinusoidal_pe(max_len + 1, embed_dim))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.depth = depth

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        frame_embeds: torch.Tensor,           # (B, T, in_dim)
        valid_mask: Optional[torch.Tensor] = None,   # (B, T) bool, True = real frame
        frame_mask: Optional[torch.Tensor] = None,   # (B, T) bool, True = iBOT-masked
        time_indices: Optional[torch.Tensor] = None, # (B, T) int, absolute frame idx
        return_all_layers: bool = False,             # if True, also return the L+1 layer stack (see below)
        tap_layer: Optional[int] = None,             # if set, also return frame tokens after block `tap_layer` (1-based)
    ):
        B, T, _ = frame_embeds.shape
        assert T <= self.max_len, f"temporal len {T} exceeds max_len {self.max_len}"

        x_in = frame_embeds
        if frame_mask is not None:
            # broadcast learnable mask token into masked positions
            mask_expand = self.mask_token.expand(B, T, -1)
            x_in = torch.where(frame_mask.unsqueeze(-1), mask_expand, x_in)

        x = self.input_proj(x_in)                                # (B, T, D)
        stem = x                                                 # layer 0 of the downstream weighted sum
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                            # (B, T+1, D)

        # positional encoding: CLS at position 0; frames at their absolute
        # time index when provided, else at 1..T
        if time_indices is None:
            time_indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pe = self.pos_embed[0].unsqueeze(0).expand(B, -1).unsqueeze(1)        # (B,1,D)
        frame_pe_idx = (time_indices.clamp(0, self.max_len - 1) + 1)          # shift past CLS
        frame_pe = self.pos_embed[frame_pe_idx]                               # (B,T,D)
        x = x + torch.cat([pe, frame_pe], dim=1)

        # attention key_padding_mask: True = ignore. CLS always attends.
        if valid_mask is not None:
            key_padding_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~valid_mask],
                dim=1,
            )
        else:
            key_padding_mask = None

        layer_outputs = []
        tapped = None
        for li, layer in enumerate(self.layers, start=1):
            x = layer(x, src_key_padding_mask=key_padding_mask)
            if return_all_layers and li < self.depth:
                layer_outputs.append(x[:, 1:])
            if tap_layer is not None and li == tap_layer:
                tapped = x[:, 1:]
        x = self.norm(x)
        if tap_layer is not None:
            return x[:, 0], x[:, 1:], tapped
        if return_all_layers:
            # frame tokens of the L+1 layers read by downstream heads, shape (L+1, B, T, D):
            # layer 0 = the projected DINOv3 input (post-stem), layers 1..L-1 = block outputs, layer L = the
            # final block after the output norm
            stacked = torch.stack([stem] + layer_outputs + [x[:, 1:]], dim=0)
            return x[:, 0], x[:, 1:], stacked
        return x[:, 0], x[:, 1:]
