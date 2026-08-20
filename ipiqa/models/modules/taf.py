"""TAF -- Task-Aware Adaptive Fusion (小创新 3), residual formulation (v2).

改进1.md 第 13-14 节：TAF 不再替代整个 B3 prediction path，而是在 B3 的
shared representation 上增加 task-specific residual。

    h = shared_fusion(concat[v, c])            (由 model 层计算)
    g_q = sigmoid(gate_q(concat[v, c]))
    g_a = sigmoid(gate_a(concat[v, c]))
    mix_q = g_q * v + (1 - g_q) * c
    mix_a = g_a * v + (1 - g_a) * c
    delta_q = quality_adapter(mix_q)
    delta_a = align_adapter(mix_a)

    h_q = h + tanh(lambda_taf_q) * delta_q     (lambda 在 model 层)
    h_a = h + tanh(lambda_taf_a) * delta_a

gate 默认初始化到 sigmoid(0)=0.5；adapter 输出与 h 同维度，作为残差。
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

        # task-specific residual adapters (输出与 h 同维度)
        self.quality_adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.align_adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

        self._last_gates = (None, None)

    def compute_gates(self, v, c):
        """Return (g_q, g_a) from concatenated [v, c]."""
        cat = torch.cat([v, c], dim=-1)
        g_q = torch.sigmoid(self.gate_q(cat))
        g_a = torch.sigmoid(self.gate_a(cat))
        self._last_gates = (g_q.detach().float(), g_a.detach().float())
        return g_q, g_a

    def forward(self, v, c):
        """Compute gates + residual deltas (model 层负责加 lambda 与 h)。"""
        g_q, g_a = self.compute_gates(v, c)
        mix_q = g_q * v + (1.0 - g_q) * c
        mix_a = g_a * v + (1.0 - g_a) * c
        delta_q = self.quality_adapter(mix_q)
        delta_a = self.align_adapter(mix_a)
        return g_q, g_a, delta_q, delta_a
