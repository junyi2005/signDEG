"""Modality-agnostic DINO core: loss, head, EMA, schedules.

Nothing in this package knows about images, frames or poses -- everything
operates on plain feature tensors. Ported from Meta's DINOv3 (single-GPU).
"""

from .head import DINOHead
from .loss import DINOLoss, KoLeoLoss, iBOTFrameLoss
from .ema import EMAUpdater, cosine_schedule, linear_warmup_cosine

__all__ = [
    "DINOHead",
    "DINOLoss",
    "KoLeoLoss",
    "iBOTFrameLoss",
    "EMAUpdater",
    "cosine_schedule",
    "linear_warmup_cosine",
]
