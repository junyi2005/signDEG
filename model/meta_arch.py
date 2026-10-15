"""SignTemporalDinoModel -- one student/teacher pair for one stream.

This module is the SSL trainer's atomic unit. The trainer instantiates one
SignTemporalDinoModel per stream (lh, rh, face), wires it to a DataLoader
that produces multi-temporal-crop samples for that stream, and calls
`compute_losses(...)` per step.

Inputs to compute_losses:
    student_views: dict with
        'global_frames':  (Ng, B, Tg, D)
        'global_valid':   (Ng, B, Tg)
        'global_time':    (Ng, B, Tg)
        'local_frames':   (Nl, B, Tl, D)   # may have Nl=0
        'local_valid':    (Nl, B, Tl)
        'local_time':     (Nl, B, Tl)
        'ibot_mask':      (Ng, B, Tg)
        'geo_w', 'geo_u', 'geo_theta', 'geo_joint': (Ng, B, Tg[, d])  -- hand streams only
        'geo_video': (Ng, B, Tg) global source-video id, 'geo_frame': (Ng, B, Tg) frame index in that video
    teacher_views: dict with the same keys but only globals
        'global_frames', 'global_valid', 'global_time'

Returns: dict of losses (dino, ibot, dkoleo[, glp_*, grt]); the EMA teacher is updated by `ema_step`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from dino import (
    DINOHead,
    DINOLoss,
    KoLeoLoss,
    iBOTFrameLoss,
    EMAUpdater,
)
from .temporal_encoder import TemporalTransformer
from .geo import GeoConfig, GeoHeads


@dataclass
class SignDinoConfig:
    in_dim: int = 768                # DINOv3 ViT-B/16 CLS dim
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_len: int = 256

    # heads
    head_out_dim: int = 8192         # K (DINO prototypes)
    head_hidden: int = 512           # DINO / iBOT heads: D -> 512 -> 256 -> K
    head_bottleneck: int = 256
    head_layers: int = 2
    head_norm_last: bool = True

    # losses
    student_temp: float = 0.1
    teacher_temp: float = 0.07
    teacher_temp_start: float = 0.04
    sinkhorn_iters: int = 3
    dkoleo_weight: float = 0.1
    dino_weight: float = 1.0
    ibot_weight: float = 1.0
    geo: GeoConfig = field(default_factory=GeoConfig)  # GLP / GRT (training only)


class SignTemporalDinoModel(nn.Module):
    def __init__(self, cfg: SignDinoConfig):
        super().__init__()
        self.cfg = cfg

        self.student_encoder = TemporalTransformer(
            in_dim=cfg.in_dim, embed_dim=cfg.embed_dim,
            depth=cfg.depth, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            max_len=cfg.max_len,
        )
        self.teacher_encoder = TemporalTransformer(
            in_dim=cfg.in_dim, embed_dim=cfg.embed_dim,
            depth=cfg.depth, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            max_len=cfg.max_len,
        )

        self.student_cls_head = DINOHead(
            in_dim=cfg.embed_dim, out_dim=cfg.head_out_dim,
            nlayers=cfg.head_layers, hidden_dim=cfg.head_hidden,
            bottleneck_dim=cfg.head_bottleneck, norm_last_layer=cfg.head_norm_last,
        )
        self.teacher_cls_head = DINOHead(
            in_dim=cfg.embed_dim, out_dim=cfg.head_out_dim,
            nlayers=cfg.head_layers, hidden_dim=cfg.head_hidden,
            bottleneck_dim=cfg.head_bottleneck, norm_last_layer=cfg.head_norm_last,
        )
        self.student_patch_head = DINOHead(
            in_dim=cfg.embed_dim, out_dim=cfg.head_out_dim,
            nlayers=cfg.head_layers, hidden_dim=cfg.head_hidden,
            bottleneck_dim=cfg.head_bottleneck, norm_last_layer=cfg.head_norm_last,
        )
        self.teacher_patch_head = DINOHead(
            in_dim=cfg.embed_dim, out_dim=cfg.head_out_dim,
            nlayers=cfg.head_layers, hidden_dim=cfg.head_hidden,
            bottleneck_dim=cfg.head_bottleneck, norm_last_layer=cfg.head_norm_last,
        )

        self.dino_loss = DINOLoss(out_dim=cfg.head_out_dim, student_temp=cfg.student_temp)
        self.ibot_loss = iBOTFrameLoss(out_dim=cfg.head_out_dim, student_temp=cfg.student_temp)
        self.dkoleo_loss = KoLeoLoss()

        self.ema = EMAUpdater(self._student_modules(), self._teacher_modules())

        # geometric heads: student-only, not EMA'd, discarded at inference
        g = cfg.geo
        self.geo_heads = GeoHeads(g, cfg.embed_dim) if (g.glp_weight > 0 or g.grt_weight > 0) else None

    # --------- module groupings (EMA needs identical parameter shapes) -----
    def _student_modules(self) -> nn.Module:
        return _wrap(self.student_encoder, self.student_cls_head, self.student_patch_head)

    def _teacher_modules(self) -> nn.Module:
        return _wrap(self.teacher_encoder, self.teacher_cls_head, self.teacher_patch_head)

    # --------- main step ----------------------------------------------------
    def encode_student(self, frames, valid, time_idx, frame_mask=None):
        cls, patches = self.student_encoder(frames, valid, frame_mask, time_idx)
        return cls, patches

    @torch.no_grad()
    def encode_teacher(self, frames, valid, time_idx):
        cls, patches = self.teacher_encoder(frames, valid, None, time_idx)
        return cls, patches

    def compute_losses(
        self,
        student_views: Dict[str, torch.Tensor],
        teacher_views: Dict[str, torch.Tensor],
        teacher_temp: float,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # ----- teacher forward (global views only) ----------------------
        T_frames = teacher_views["global_frames"]        # (Ng, B, Tg, D)
        T_valid  = teacher_views["global_valid"]
        T_time   = teacher_views["global_time"]
        Ng, B, Tg, D_in = T_frames.shape
        T_cls, T_patches = self.encode_teacher(
            T_frames.reshape(Ng * B, Tg, D_in),
            T_valid.reshape(Ng * B, Tg),
            T_time.reshape(Ng * B, Tg),
        )
        t_cls_logits = self.teacher_cls_head(T_cls)              # (Ng*B, K)
        t_patch_logits = self.teacher_patch_head(T_patches)      # (Ng*B, Tg, K)

        t_cls_probs = self.dino_loss.sinkhorn_knopp_teacher(
            t_cls_logits, teacher_temp, n_iterations=self.cfg.sinkhorn_iters
        )                                                        # (Ng*B, K)
        t_patch_probs_flat = self.ibot_loss.sinkhorn_knopp_teacher(
            t_patch_logits.reshape(-1, t_patch_logits.shape[-1]),
            teacher_temp, n_iterations=self.cfg.sinkhorn_iters,
        ).reshape(Ng * B, Tg, -1)

        t_cls_probs = t_cls_probs.reshape(Ng, B, -1)              # (Ng, B, K)
        t_patch_probs = t_patch_probs_flat.reshape(Ng, B, Tg, -1) # (Ng, B, Tg, K)

        # ----- student global forward (with iBOT mask) ----------------
        s_global_frames = student_views["global_frames"]
        s_global_valid  = student_views["global_valid"]
        s_global_time   = student_views["global_time"]
        ibot_mask = student_views["ibot_mask"]
        Ng_s, B_s, Tg_s, _ = s_global_frames.shape
        use_geo = self.geo_heads is not None and "geo_w" in student_views
        enc_out = self.student_encoder(
            s_global_frames.reshape(Ng_s * B_s, Tg_s, D_in),
            s_global_valid.reshape(Ng_s * B_s, Tg_s),
            ibot_mask.reshape(Ng_s * B_s, Tg_s),
            s_global_time.reshape(Ng_s * B_s, Tg_s),
            tap_layer=self.cfg.geo.tap_layer if use_geo else None,
        )
        if use_geo:
            cls_g, patches_g, tap_g = enc_out
        else:
            cls_g, patches_g = enc_out
        s_cls_g_logits = self.student_cls_head(cls_g).reshape(Ng_s, B_s, -1)
        s_patch_g_logits = self.student_patch_head(patches_g).reshape(Ng_s, B_s, Tg_s, -1)

        # ----- student local forward (no iBOT mask) -------------------
        s_local_frames = student_views.get("local_frames")
        has_local = s_local_frames is not None and s_local_frames.numel() > 0 and s_local_frames.shape[0] > 0
        s_cls_l_logits = None
        if has_local:
            s_local_valid = student_views["local_valid"]
            s_local_time  = student_views["local_time"]
            Nl, B_l, Tl, _ = s_local_frames.shape
            cls_l, _ = self.encode_student(
                s_local_frames.reshape(Nl * B_l, Tl, D_in),
                s_local_valid.reshape(Nl * B_l, Tl),
                s_local_time.reshape(Nl * B_l, Tl),
            )
            s_cls_l_logits = self.student_cls_head(cls_l).reshape(Nl, B_l, -1)

        # ----- DINO loss ------------------------------------------------
        loss_dino_gg = self.dino_loss(s_cls_g_logits, t_cls_probs, ignore_diagonal=True)
        if has_local:
            loss_dino_lg = self.dino_loss(s_cls_l_logits, t_cls_probs, ignore_diagonal=False)
            loss_dino = 0.5 * (loss_dino_gg + loss_dino_lg)
        else:
            loss_dino = loss_dino_gg

        # ----- iBOT loss ------------------------------------------------
        # gather student masked + teacher masked at the same (view, frame)
        # positions.
        if ibot_mask.any():
            s_masked = s_patch_g_logits[ibot_mask]                # (Nmasked, K)
            t_masked = t_patch_probs[ibot_mask]                   # (Nmasked, K)
            loss_ibot = self.ibot_loss(s_masked, t_masked)
        else:
            loss_ibot = s_patch_g_logits.new_zeros(())

        # ----- DKoleo loss ----------------------------------------------
        # spread student global CLS features (pre-head, normalised internally)
        loss_dkoleo = self.dkoleo_loss(cls_g)

        out = {
            "loss_dino": self.cfg.dino_weight * loss_dino,
            "loss_ibot": self.cfg.ibot_weight * loss_ibot,
            "loss_dkoleo": self.cfg.dkoleo_weight * loss_dkoleo,
        }

        # ----- geometry absorption (GLP + GRT, hand streams) ------------
        if use_geo:
            M = Ng_s * B_s
            geo = {k: student_views["geo_" + k].reshape(M, Tg_s, -1)
                   for k in ("u", "theta", "joint") if ("geo_" + k) in student_views}
            w = student_views["geo_w"].reshape(M, Tg_s) * s_global_valid.reshape(M, Tg_s).float()
            gc = self.cfg.geo
            if gc.glp_weight > 0:
                glp = self.geo_heads.glp(tap_g, geo, w, ibot_mask.reshape(M, Tg_s))
                for k, v in glp.items():
                    out["loss_" + k] = gc.glp_weight * v
            if gc.grt_weight > 0:
                vid = student_views["geo_video"].reshape(M, Tg_s)
                frm = student_views["geo_frame"].reshape(M, Tg_s)
                out["loss_grt"] = gc.grt_weight * self.geo_heads.grt(patches_g, geo, w, vid, frm, step)

        out["loss_total"] = sum(out.values())
        return out

    def forward(self, student_views, teacher_views, teacher_temp: float, step: int = 0):
        """Same as compute_losses; called through forward so that DistributedDataParallel syncs gradients."""
        return self.compute_losses(student_views, teacher_views, teacher_temp, step)

    @torch.no_grad()
    def ema_step(self, momentum: float):
        self.ema.update(momentum)


class _Wrap(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)


def _wrap(*modules) -> nn.Module:
    return _Wrap(*modules)
