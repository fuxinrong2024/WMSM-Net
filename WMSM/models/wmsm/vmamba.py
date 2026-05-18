import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class DPI(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.fusion_weights = nn.Parameter(torch.ones(4) * 0.25)
        self.adaptive_conv = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x):
        B, H, W, C = x.shape
        x_input = x.permute(0, 3, 1, 2)
        
        if H % 2 != 0 or W % 2 != 0:
            x_input = F.pad(x_input, (0, W % 2, 0, H % 2))
            H, W = x_input.shape[2], x_input.shape[3]
        
        LL, LH, HL, HH = self._dpi_decompose(x_input)
        
        wavelet_coeffs = torch.cat([LL, LH, HL, HH], dim=1)
        fusion_weights_expanded = self.fusion_weights.view(1, 4, 1, 1).repeat(1, self.dim, 1, 1)
        fusion_weights_expanded = fusion_weights_expanded.reshape(1, 4 * self.dim, 1, 1)
        
        weighted_coeffs = wavelet_coeffs * fusion_weights_expanded
        fused = self.adaptive_conv(weighted_coeffs)
        
        return fused.permute(0, 2, 3, 1)

    def _dpi_decompose(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.reshape(B, C, H//2, 2, W//2, 2)
        
        LL = (
            0.50 * x_reshaped[:, :, :, 0, :, 0] +
            0.50 * x_reshaped[:, :, :, 0, :, 1] +
            0.50 * x_reshaped[:, :, :, 1, :, 0] +
            0.50 * x_reshaped[:, :, :, 1, :, 1]
        ) / 2.0

        LH = (
            0.75 * x_reshaped[:, :, :, 0, :, 0] - 0.75 * x_reshaped[:, :, :, 0, :, 1] +
            0.25 * x_reshaped[:, :, :, 1, :, 0] - 0.25 * x_reshaped[:, :, :, 1, :, 1]
        )

        HL = (
            0.75 * x_reshaped[:, :, :, 0, :, 0] + 0.25 * x_reshaped[:, :, :, 0, :, 1]
            - 0.75 * x_reshaped[:, :, :, 1, :, 0] - 0.25 * x_reshaped[:, :, :, 1, :, 1]
        )

        HH = (
            1.00 * x_reshaped[:, :, :, 0, :, 0] - 1.00 * x_reshaped[:, :, :, 0, :, 1]
            - 1.00 * x_reshaped[:, :, :, 1, :, 0] + 1.00 * x_reshaped[:, :, :, 1, :, 1]
        )

        return LL, LH, HL, HH

class WaveletEnhancedVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        use_wavelet: bool = True,
        wavelet_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.wavelet_ratio = wavelet_ratio
        
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        
        if use_wavelet:
            self.haar_transform = DPI(hidden_dim)
            self.wavelet_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            )
            self.wavelet_norm = norm_layer(hidden_dim)
            
            for m in self.wavelet_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        residual = input
        
        x_main = self.ln_1(input)
        x_main = self.self_attention(x_main)
        
        if self.use_wavelet and (self.training or torch.rand(1).item() < self.wavelet_ratio):
            B, H, W, C = x_main.shape
            
            with torch.no_grad():
                x_wavelet = self.haar_transform(x_main)
            
            x_wavelet_upsampled = F.interpolate(
                x_wavelet.permute(0, 3, 1, 2), 
                size=(H, W), 
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            gate_input = torch.cat([x_main, x_wavelet_upsampled], dim=-1)
            gate = self.wavelet_gate(gate_input)
            wavelet_contribution = self.wavelet_norm(x_wavelet_upsampled) * 0.1
            x_enhanced = x_main + gate * wavelet_contribution
        else:
            x_enhanced = x_main
        
        return residual + self.drop_path(x_enhanced)