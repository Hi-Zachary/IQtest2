"""TAF -- Task-Aware Adaptive Fusion (小创新 3).

Design section 16: quality and alignment rely on different information, so we
learn per-task gates between the visual (F_visual) and cross-modal (F_cross)
streams:

    g_q = Sigmoid(MLP_q([F_visual; F_cross]))
    F_q = g_q * F_visual + (1 - g_q) * F_cross
    g_a = Sigmoid(MLP_a([F_visual; F_cross]))
    F_a = g_a * F_visual + (1 - g_a) * F_cross

    F_q -> Quality Head ;  F_a -> Alignment Head

Gate MLP last layer is zero-initialized so initial sigmoid is near 0.5.
"""

import torch
import torch.nn as nn


class _ZeroInitLinear(nn.Module):
    def __init__(self, in_features, out_features, zero_init=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class TAF(nn.Module):
    def __init__(self, dim=256, drop=0.1, gate_zero_init=True):
        super().__init__()
        self.dim = dim

        self.gate_q = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
            _ZeroInitLinear(dim, 1, zero_init=gate_zero_init),
        )
        self.gate_a = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
            _ZeroInitLinear(dim, 1, zero_init=gate_zero_init),
        )

    def forward(self, f_visual, f_cross):
        cat = torch.cat([f_visual, f_cross], dim=-1)
        g_q = torch.sigmoid(self.gate_q(cat))
        g_a = torch.sigmoid(self.gate_a(cat))

        f_q = g_q * f_visual + (1.0 - g_q) * f_cross
        f_a = g_a * f_visual + (1.0 - g_a) * f_cross

        return f_q, f_a
