# Ported verbatim (with cosmetic trims) from dinov3/layers/dino_head.py.
# This module is modality-agnostic: it maps a [*, in_dim] feature to a
# [*, out_dim] vector of prototype logits.
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    layers = [nn.Linear(in_dim, hidden_dim, bias=bias), nn.GELU()]
    for _ in range(nlayers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    """MLP -> L2-normalize -> weight-normalized linear to `out_dim` prototypes.

    out_dim is the number of prototypes K (image DINOv3 uses 65536; we use
    8192 -- see configs).
    """

    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256, norm_last_layer=True):
        super().__init__()
        self.mlp = _build_mlp(max(nlayers, 1), in_dim, bottleneck_dim, hidden_dim)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)
