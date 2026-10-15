"""Multi-temporal-crop collate.

Each sample yields `n_global` global windows + `n_local` local windows,
but per-window length is sampled independently in [len_min, len_max].
Collate therefore pads every global window to a common Tg (= max length
across the batch's Ng x B global windows) and every local window to a
common Tl. `valid_mask` is False on padded positions, so attention
masking and iBOT masking transparently ignore padding.

Produces an iBOT frame mask on the global crops only, with per-crop random
mask ratio in [ibot_mask_min, ibot_mask_max].
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _pad_view_to(tensor: torch.Tensor, T_target: int, pad_value=0) -> torch.Tensor:
    """Right-pad a (T, *) tensor along dim 0 to length T_target with `pad_value`."""
    T = tensor.shape[0]
    if T == T_target:
        return tensor
    pad_shape = (T_target - T, *tensor.shape[1:])
    pad = tensor.new_full(pad_shape, pad_value) if not tensor.dtype == torch.bool else tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad], dim=0)


def _stack_views(per_sample_lists: List[List[torch.Tensor]], pad_value=0) -> torch.Tensor:
    """`per_sample_lists[b][v]` is a tensor of shape (T_bv, *). Pad every window
    to T_max (= global max over (b, v)) and stack to (Nv, B, T_max, *)."""
    B = len(per_sample_lists)
    Nv = len(per_sample_lists[0])
    T_max = max(per_sample_lists[b][v].shape[0] for b in range(B) for v in range(Nv))
    per_view = []
    for v in range(Nv):
        stacked = torch.stack(
            [_pad_view_to(per_sample_lists[b][v], T_max, pad_value=pad_value) for b in range(B)],
            dim=0,
        )
        per_view.append(stacked)
    return torch.stack(per_view, dim=0)  # (Nv, B, T_max, *)


def multi_crop_collate(
    batch: List[Dict],
    ibot_mask_min: float = 0.25,
    ibot_mask_max: float = 0.50,
    rng: np.random.Generator | None = None,
) -> Dict:
    rng = rng or np.random.default_rng()

    g_frames = _stack_views([s["global_frames"] for s in batch], pad_value=0)    # (Ng, B, Tg, *)
    g_valid  = _stack_views([s["global_valid"]  for s in batch], pad_value=0)    # bool, pad = False
    g_time   = _stack_views([s["global_time"]   for s in batch], pad_value=0)
    l_frames = _stack_views([s["local_frames"]  for s in batch], pad_value=0)    # (Nl, B, Tl, *)
    l_valid  = _stack_views([s["local_valid"]   for s in batch], pad_value=0)
    l_time   = _stack_views([s["local_time"]    for s in batch], pad_value=0)

    Ng, B, Tg = g_valid.shape

    # iBOT mask is built on g_valid, which is already False at padded positions,
    # so padded frames are never selected.
    ibot_mask = torch.zeros(Ng, B, Tg, dtype=torch.bool)
    for v in range(Ng):
        for b in range(B):
            valid_idx = torch.where(g_valid[v, b])[0]
            if valid_idx.numel() == 0:
                continue
            ratio = float(rng.uniform(ibot_mask_min, ibot_mask_max))
            k = max(1, int(round(ratio * valid_idx.numel())))
            chosen = valid_idx[torch.randperm(valid_idx.numel())[:k]]
            ibot_mask[v, b, chosen] = True

    out = {
        "video_id": [s["video_id"] for s in batch],
        "global_frames": g_frames,
        "global_valid": g_valid,
        "global_time": g_time,
        "local_frames": l_frames,
        "local_valid": l_valid,
        "local_time": l_time,
        "ibot_mask": ibot_mask,
        "n_global": Ng,
    }
    # geometric targets (GLP / GRT): present only when the dataset serves them
    for k in ("w", "u", "theta", "joint", "video", "frame"):
        key = f"global_geo_{k}"
        if key in batch[0]:
            out[f"geo_{k}"] = _stack_views([s[key] for s in batch], pad_value=0)   # (Ng, B, Tg, *)
    return out
