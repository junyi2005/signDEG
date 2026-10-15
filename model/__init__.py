"""Per-stream temporal model.

- TemporalTransformer: small Transformer over per-frame frozen-DINOv3 embeddings, with
  sinusoidal time PE and a learnable CLS token.
- SignTemporalDinoModel: student + EMA teacher + DINO/iBOT heads + training-only geometric heads.
"""

from .temporal_encoder import TemporalTransformer
from .meta_arch import SignTemporalDinoModel

__all__ = [
    "TemporalTransformer",
    "SignTemporalDinoModel",
]
