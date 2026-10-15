"""SignDEG training-only geometric objectives (paper Sec. 3.3-3.5).

Geometric Latent Prediction (GLP)
    From intermediate student block l_g, predict -- at every valid frame of the global temporal crops,
    including iBOT-masked frames whose appearance the student never saw -- the frozen WiLoR outputs:
        u      standardised geometric latent      (1 - cosine)
        theta  MANO articulation, 6D per joint    (L1, averaged over coordinates)
        J      wrist-relative 3D joints           (L1, averaged over coordinates)
    Frames are weighted by gamma_t * w_t (gamma_t = gamma_m for masked frames, 1 otherwise) and the loss is
    normalised by max(sum gamma_t w_t, eps).

Geometric Relation Transfer (GRT)
    At the last block: frames of the batch's global crops are deduplicated by (video, frame index), up to
    N valid frames (w > 0) are sampled uniformly across the distributed batch, and for every anchor the
    student's similarity distribution over frames of *other* videos (linear projection Q) is matched to the
    one induced by the geometric descriptor [u ; theta] (KL). Anchors without such candidates are omitted;
    candidate distributions are unweighted and w_i weights the anchor's KL term.

All heads here are discarded at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


@dataclass
class GeoConfig:
    glp_weight: float = 0.0          # lambda_P; 0 disables GLP
    grt_weight: float = 0.0          # lambda_R; 0 disables GRT
    tap_layer: int = 4               # l_g (1-based block index of the student encoder)
    u_dim: int = 0                   # 1280 for the WiLoR latent; 0 = no latent target
    theta_dim: int = 0               # 90 for MANO 15x6D; 0 = no articulation target
    joint_dim: int = 0               # 63 for 21x3; 0 = no joint target
    alpha_theta: float = 1.0
    alpha_joint: float = 1.0
    gamma_mask: float = 2.0          # gamma_m
    head_hidden: int = 256           # GLP heads: D -> 256 -> d (GELU)
    grt_frames: int = 1024           # N frames sampled per update across the distributed batch
    grt_proj_dim: int = 128          # Q: linear D -> 128
    grt_tau_g: float = 0.05
    grt_tau_z: float = 0.1


def _mlp(i: int, h: int, o: int) -> nn.Module:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _gather_rows(x: torch.Tensor, with_grad: bool) -> torch.Tensor:
    """Concatenate the (n_r, ...) tensors of all ranks along dim 0 (variable n_r). Autograd-aware if
    with_grad: the gradient of a loss computed identically on every rank reaches each rank's own rows."""
    if not _distributed():
        return x
    n = torch.tensor([x.shape[0]], device=x.device)
    sizes = [torch.zeros_like(n) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, n)
    sizes = [int(s.item()) for s in sizes]
    pad = max(sizes) - x.shape[0]
    xp = torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))]) if pad else x
    if with_grad:
        from torch.distributed.nn.functional import all_gather
        parts = all_gather(xp)
    else:
        parts = [torch.zeros_like(xp) for _ in sizes]
        dist.all_gather(parts, xp)
    return torch.cat([p[:s] for p, s in zip(parts, sizes)])


class GeoHeads(nn.Module):
    def __init__(self, cfg: GeoConfig, embed_dim: int):
        super().__init__()
        self.cfg = cfg
        self.p_u = _mlp(embed_dim, cfg.head_hidden, cfg.u_dim) if cfg.u_dim > 0 else None
        self.p_theta = _mlp(embed_dim, cfg.head_hidden, cfg.theta_dim) if cfg.theta_dim > 0 else None
        self.p_joint = _mlp(embed_dim, cfg.head_hidden, cfg.joint_dim) if cfg.joint_dim > 0 else None
        self.q = nn.Linear(embed_dim, cfg.grt_proj_dim) if cfg.grt_weight > 0 else None

    # ---- GLP ------------------------------------------------------------------
    def glp(self, z_tap: torch.Tensor, geo: Dict[str, torch.Tensor], w: torch.Tensor,
            frame_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """z_tap (M, T, D) student tokens after block l_g; geo[k] (M, T, d_k); w (M, T)."""
        c = self.cfg
        wt = w.float()
        if frame_mask is not None:
            wt = wt * torch.where(frame_mask, torch.full_like(wt, c.gamma_mask), torch.ones_like(wt))
        sel = wt > 0
        if not sel.any():                                   # empty valid-target set: zero geometry loss
            return {"glp": z_tap.sum() * 0.0}
        z = z_tap[sel].float()
        ws = wt[sel]
        norm = ws.sum().clamp_min(EPS)
        out: Dict[str, torch.Tensor] = {}
        if self.p_u is not None:
            cos = F.cosine_similarity(self.p_u(z), geo["u"][sel].float(), dim=-1)
            out["glp_u"] = ((1 - cos) * ws).sum() / norm
        if self.p_theta is not None:
            l1 = (self.p_theta(z) - geo["theta"][sel].float()).abs().mean(-1)
            out["glp_theta"] = c.alpha_theta * (l1 * ws).sum() / norm
        if self.p_joint is not None:
            l1 = (self.p_joint(z) - geo["joint"][sel].float()).abs().mean(-1)
            out["glp_joint"] = c.alpha_joint * (l1 * ws).sum() / norm
        return out

    # ---- GRT ------------------------------------------------------------------
    def grt(self, z_out: torch.Tensor, geo: Dict[str, torch.Tensor], w: torch.Tensor,
            video: torch.Tensor, frame: torch.Tensor, step: int) -> torch.Tensor:
        """z_out (M, T, D) last-block student tokens; video (M, T) global source-video id of each token;
        frame (M, T) its frame index within that video; step seeds the sampling identically on all ranks."""
        c = self.cfg
        sel = w > 0
        q = self.q(z_out[sel].float())                                  # (n, P) local rows
        g = F.normalize(torch.cat([F.normalize(geo["u"][sel].float(), dim=-1),
                                   F.normalize(geo["theta"][sel].float(), dim=-1)], -1), dim=-1)
        wi, vi, fi = w[sel].float(), video[sel], frame[sel]
        # concatenate the distributed batch (Q outputs keep their gradient path)
        q = _gather_rows(q, with_grad=True)
        g, wi = _gather_rows(g, False), _gather_rows(wi, False)
        vi, fi = _gather_rows(vi, False), _gather_rows(fi, False)
        if q.shape[0] < 2:
            return z_out.sum() * 0.0
        # deduplicate frames shared by overlapping global crops: keep the first (video, frame) occurrence
        key = vi.long() * (1 << 32) + fi.long()
        _, inv = torch.unique(key, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), key.numel(), device=key.device, dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(key.numel(), device=key.device), reduce="amin")
        keep = first
        # uniform sample of up to N frames, identical on every rank
        if keep.numel() > c.grt_frames:
            gen = torch.Generator().manual_seed(1_000_003 * (step + 1))
            keep = keep[torch.randperm(keep.numel(), generator=gen)[: c.grt_frames].to(keep.device)]
        q, g, wi, vi = F.normalize(q[keep], dim=-1), g[keep], wi[keep], vi[keep]
        cand = vi[:, None] != vi[None, :]                               # candidates: other videos only
        has = cand.any(1)                                               # anchors without candidates: omitted
        if not has.any():
            return z_out.sum() * 0.0
        # non-candidates: probability exactly 0 under P^g; a large finite negative logit keeps log P^z finite
        s_g = (g @ g.T / c.grt_tau_g).masked_fill(~cand, float("-inf"))[has]
        s_z = (q @ q.T / c.grt_tau_z).masked_fill(~cand, -1e4)[has]
        p_g = s_g.softmax(-1)
        logp_z = s_z.log_softmax(-1)
        kl = (p_g * (torch.log(p_g.clamp_min(1e-12)) - logp_z)).sum(-1)
        wa = wi[has]
        return (kl * wa).sum() / wa.sum().clamp_min(EPS)


def reliability_weights(theta: torch.Tensor, valid: torch.Tensor, score: torch.Tensor,
                        predicted: torch.Tensor, kappa: float = 3.0,
                        predicted_scale: float = 0.5) -> torch.Tensor:
    """Eq. (rel): w_t = m_t o_t s_t rho_t exp(-kappa * mean|theta_{t-1} - 2 theta_t + theta_{t+1}|).

    theta (T, d) in the source video's (downsampled) frame order; `valid` must already combine the tracker
    mask m_t with WiLoR's estimate flag. o_t = 1 only if the frame and both neighbours have estimates, so the
    first and last frame of a video always get o_t = 0. rho_t = predicted_scale for tracker-filled boxes."""
    T = theta.shape[0]
    w = torch.zeros(T)
    if T < 3:
        return w
    jit = (theta[:-2] - 2 * theta[1:-1] + theta[2:]).abs().mean(-1)
    o = valid[:-2] & valid[1:-1] & valid[2:]
    w[1:-1] = o.float() * score[1:-1].clamp(0, 1) * torch.exp(-kappa * jit)
    w = torch.where(predicted, w * predicted_scale, w)
    return w
