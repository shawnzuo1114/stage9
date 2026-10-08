"""Stage9-HTFR: hierarchical temporal-frequency routing for video ReID.

It consumes the STH-refined Stage3/Stage4 feature sequences and the global
video representation ``z_g``.  It never needs a paired opposite-modality
sample at inference time.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warmup import linear_warmup


def _mean_vector_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).norm(dim=1).mean()


def _mean_cosine(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    return F.cosine_similarity(x.float(), y.float(), dim=-1, eps=eps).mean()


class SpatialHaarHFExtractor(nn.Module):
    """Extract per-frame spatial Haar high-frequency descriptors.

    Input is ``[B,T,C,H,W]`` and output is ``[B,T,C]``.  Time is deliberately
    preserved.  The spatial GeM exponent is fixed to keep this branch simple
    and easy to inspect.
    """

    def __init__(self, gem_p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.gem_p = float(gem_p)
        self.eps = float(eps)

    @staticmethod
    def _pad_even(x: torch.Tensor) -> torch.Tensor:
        pad_h = x.size(-2) % 2
        pad_w = x.size(-1) % 2
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def forward(self, feat_seq: torch.Tensor) -> torch.Tensor:
        if feat_seq.ndim != 5:
            raise ValueError(
                "SpatialHaarHFExtractor expects [B,T,C,H,W], got {}.".format(
                    tuple(feat_seq.shape)
                )
            )
        batch_size, seq_len, channels, height, width = feat_seq.shape
        flat = self._pad_even(
            feat_seq.contiguous().view(batch_size * seq_len, channels, height, width)
        )
        x00 = flat[..., 0::2, 0::2]
        x01 = flat[..., 0::2, 1::2]
        x10 = flat[..., 1::2, 0::2]
        x11 = flat[..., 1::2, 1::2]
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        spatial_hf = torch.maximum(lh.abs(), torch.maximum(hl.abs(), hh.abs()))
        pooled = spatial_hf.clamp_min(self.eps).pow(self.gem_p).mean(dim=(-2, -1))
        pooled = pooled.pow(1.0 / self.gem_p)
        return pooled.view(batch_size, seq_len, channels)


class TemporalHighPass(nn.Module):
    """Fixed temporal high-pass filter applied only to ``[B,T,C]`` tokens.

    The DC bin is removed, bin 1 receives weight 0.25, and all higher bins
    receive weight 1.0.  FFT work is local float32 for AMP compatibility.
    """

    def __init__(self, low_bin_weight: float = 0.25):
        super().__init__()
        self.low_bin_weight = float(low_bin_weight)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(
                "TemporalHighPass expects [B,T,C], got {}.".format(tuple(x.shape))
            )
        seq_len = int(x.size(1))
        if seq_len < 1:
            raise ValueError("TemporalHighPass requires T >= 1.")
        input_dtype = x.dtype
        spectrum = torch.fft.rfft(x.float(), dim=1)
        freq_bins = int(spectrum.size(1))
        mask = spectrum.real.new_ones(freq_bins)
        mask[0] = 0.0
        if freq_bins > 1:
            mask[1] = self.low_bin_weight
        filtered = spectrum * mask.view(1, freq_bins, 1)
        temporal_hf = torch.fft.irfft(filtered, n=seq_len, dim=1).to(input_dtype)

        energy = spectrum.abs().square().mean(dim=(0, 2))
        zero = energy.new_zeros(())
        stats = {
            "dc_energy": energy[0] if freq_bins > 0 else zero,
            "low_energy": energy[1] if freq_bins > 1 else zero,
            "high_energy": energy[2:].mean() if freq_bins > 2 else zero,
        }
        return temporal_hf, stats


class IdentityConditionedTemporalRouter(nn.Module):
    """Lightweight query-key temporal routing conditioned on ``z_g``."""

    def __init__(
        self,
        token_dim: int = 512,
        query_dim: int = 128,
        temperature: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("Temporal router temperature must be positive.")
        self.query_dim = int(query_dim)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.query_norm = nn.LayerNorm(token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, self.query_dim)
        self.k_proj = nn.Linear(token_dim, self.query_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)

    def forward(
        self, z_g: torch.Tensor, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if z_g.ndim != 2 or tokens.ndim != 3:
            raise ValueError("Temporal router expects z_g [B,D], tokens [B,T,D].")
        if z_g.size(0) != tokens.size(0) or z_g.size(1) != tokens.size(2):
            raise ValueError(
                "Temporal router shape mismatch: z_g={}, tokens={}.".format(
                    tuple(z_g.shape), tuple(tokens.shape)
                )
            )
        q = self.q_proj(self.query_norm(z_g))
        token_norm = self.token_norm(tokens)
        k = self.k_proj(token_norm)
        v = self.v_proj(token_norm)
        score = torch.einsum("bd,btd->bt", q, k) / math.sqrt(self.query_dim)
        score = torch.nan_to_num(score / self.temperature, nan=0.0, posinf=1e4, neginf=-1e4)
        attn = F.softmax(score, dim=1)
        routed = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        p = attn.float()
        entropy = -(p * (p + self.eps).log()).sum(dim=1)
        seq_len = int(tokens.size(1))
        if seq_len > 1:
            entropy_norm = entropy / math.log(seq_len)
        else:
            entropy_norm = torch.zeros_like(entropy)
        stats = {
            "attn_entropy": entropy.mean(),
            "attn_entropy_norm": entropy_norm.mean(),
            "effective_frames": entropy.exp().mean(),
            "attn_max": p.max(dim=1).values.mean(),
            "attn_min": p.min(dim=1).values.mean(),
        }
        return routed, attn, stats


class HierarchicalStageRouter(nn.Module):
    """Route two semantic stage experts without element-wise direct addition."""

    def __init__(self, token_dim: int = 512, hidden_dim: int = 256, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)
        self.z_norm = nn.LayerNorm(token_dim)
        self.v3_norm = nn.LayerNorm(token_dim)
        self.v4_norm = nn.LayerNorm(token_dim)
        self.router = nn.Sequential(
            nn.Linear(token_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(
        self, z_g: torch.Tensor, v3: torch.Tensor, v4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        z_norm = self.z_norm(z_g)
        v3_norm = self.v3_norm(v3)
        v4_norm = self.v4_norm(v4)
        weights = F.softmax(self.router(torch.cat([z_norm, v3_norm, v4_norm], dim=-1)), dim=-1)
        weighted3 = weights[:, 0:1] * v3
        weighted4 = weights[:, 1:2] * v4
        v_freq = self.fusion(torch.cat([weighted3, weighted4], dim=-1))
        entropy = -(weights.float() * (weights.float() + self.eps).log()).sum(dim=1)
        stats = {
            "pi3_mean": weights[:, 0].float().mean(),
            "pi4_mean": weights[:, 1].float().mean(),
            "pi3_std": weights[:, 0].float().std(unbiased=False),
            "pi4_std": weights[:, 1].float().std(unbiased=False),
            "stage_entropy": entropy.mean(),
            "stage_sample_std": weights.float().std(dim=0, unbiased=False).mean(),
            "v3_v4_cos": _mean_cosine(v3, v4, self.eps),
        }
        return v_freq, weights, stats


class DifferentiableFrequencyResidualGate(nn.Module):
    """Smooth bounded gate and zero-initialized residual projection.

    The actual gate is ``gate_max * sigmoid(logits)``.  There is no hard gate
    clipping, detached gate, or residual-ratio rescaling in this module.
    """

    def __init__(
        self,
        token_dim: int = 512,
        hidden_dim: int = 256,
        gate_max: float = 0.20,
        eps: float = 1e-8,
    ):
        super().__init__()
        if gate_max <= 0.0:
            raise ValueError("gate_max must be positive.")
        self.gate_max = float(gate_max)
        self.eps = float(eps)
        self.z_norm = nn.LayerNorm(token_dim)
        self.freq_norm = nn.LayerNorm(token_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.delta_norm = nn.LayerNorm(token_dim)
        self.final_norm = nn.LayerNorm(token_dim)
        self._reset_stable_parameters()

    def _reset_stable_parameters(self) -> None:
        gate_last = self.gate_net[-1]
        nn.init.normal_(gate_last.weight, mean=0.0, std=1e-3)
        nn.init.constant_(gate_last.bias, -1.38629436112)
        delta_last = self.delta_proj[-1]
        nn.init.zeros_(delta_last.weight)
        nn.init.zeros_(delta_last.bias)

    def forward(
        self, z_g: torch.Tensor, v_freq: torch.Tensor, fusion_warm: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        gate_input = torch.cat([self.z_norm(z_g), self.freq_norm(v_freq)], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gate_pre = torch.sigmoid(gate_logits)
        gate = self.gate_max * gate_pre
        delta_basis = self.delta_norm(self.delta_proj(v_freq))
        delta = gate * delta_basis
        z_final = self.final_norm(z_g + z_g.new_tensor(float(fusion_warm)) * delta)

        gate_float = gate.float()
        delta_norm_each = delta.float().norm(dim=1)
        zg_norm_each = z_g.float().norm(dim=1)
        delta_ratio = delta_norm_each / (zg_norm_each + self.eps)
        cos_zg_delta = F.cosine_similarity(z_g.float(), delta.float(), dim=1, eps=self.eps)
        cos_zg_zfinal = F.cosine_similarity(z_g.float(), z_final.float(), dim=1, eps=self.eps)
        angle = torch.acos(cos_zg_zfinal.detach().clamp(-1.0, 1.0)) * (180.0 / math.pi)
        stats = {
            "gate_mean": gate_float.mean(),
            "gate_std": gate_float.std(unbiased=False),
            "gate_min": gate_float.min(),
            "gate_max": gate_float.max(),
            "gate_p05": torch.quantile(gate_float, 0.05),
            "gate_p50": torch.quantile(gate_float, 0.50),
            "gate_p95": torch.quantile(gate_float, 0.95),
            "gate_channel_std": gate_float.std(dim=1, unbiased=False).mean(),
            "gate_sample_std": gate_float.std(dim=0, unbiased=False).mean(),
            "gate_logits_mean": gate_logits.float().mean(),
            "gate_logits_std": gate_logits.float().std(unbiased=False),
            "delta_norm": delta_norm_each.mean(),
            "delta_ratio_mean": delta_ratio.mean(),
            "delta_ratio_p95": torch.quantile(delta_ratio, 0.95),
            "delta_ratio_max": delta_ratio.max(),
            "cos_zg_delta": cos_zg_delta.mean(),
            "cos_zg_zfinal": cos_zg_zfinal.mean(),
            "zg_zfinal_angle": angle.mean(),
        }
        return z_final, gate, gate_logits, delta, stats


class HierarchicalTemporalFrequencyRouter(nn.Module):
    """Complete Stage9 hierarchical temporal-frequency router.

    Modes:
      - ``baseline``: exact ``z_final = z_g``; encoder may still run for diagnostics.
      - ``spatial``: temporal high-pass tensors are zero.
      - ``full``: fixed temporal high-pass is enabled.
    """

    VALID_MODES = {"baseline", "spatial", "full"}

    def __init__(
        self,
        stage3_channels: int,
        stage4_channels: int,
        token_dim: int = 512,
        query_dim: int = 128,
        temporal_temperature: float = 1.0,
        gate_max: float = 0.20,
        fusion_start_epoch: int = 15,
        fusion_warmup_epoch: int = 10,
        mode: str = "full",
        eps: float = 1e-8,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError("Unsupported HTFR mode: {}.".format(mode))
        self.stage3_channels = int(stage3_channels)
        self.stage4_channels = int(stage4_channels)
        self.token_dim = int(token_dim)
        self.mode = str(mode)
        self.fusion_start_epoch = int(fusion_start_epoch)
        self.fusion_warmup_epoch = int(fusion_warmup_epoch)
        self.eps = float(eps)

        self.spatial_hf_s3 = SpatialHaarHFExtractor(eps=eps)
        self.spatial_hf_s4 = SpatialHaarHFExtractor(eps=eps)
        self.temporal_hpf_s3 = TemporalHighPass()
        self.temporal_hpf_s4 = TemporalHighPass()
        self.token_proj_s3 = nn.Sequential(
            nn.Linear(3 * self.stage3_channels, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.SiLU(),
        )
        self.token_proj_s4 = nn.Sequential(
            nn.Linear(3 * self.stage4_channels, self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.SiLU(),
        )
        self.temporal_router_s3 = IdentityConditionedTemporalRouter(
            token_dim=self.token_dim,
            query_dim=query_dim,
            temperature=temporal_temperature,
            eps=eps,
        )
        self.temporal_router_s4 = IdentityConditionedTemporalRouter(
            token_dim=self.token_dim,
            query_dim=query_dim,
            temperature=temporal_temperature,
            eps=eps,
        )
        self.stage_router = HierarchicalStageRouter(token_dim=self.token_dim, eps=eps)
        self.residual_gate = DifferentiableFrequencyResidualGate(
            token_dim=self.token_dim, gate_max=gate_max, eps=eps
        )

    def fusion_weight(self, epoch: Optional[int]) -> float:
        if self.mode == "baseline":
            return 0.0
        return linear_warmup(epoch, self.fusion_start_epoch, self.fusion_warmup_epoch)

    def _stage_tokens(
        self,
        feat: torch.Tensor,
        spatial_extractor: SpatialHaarHFExtractor,
        temporal_filter: TemporalHighPass,
        projection: nn.Module,
        temporal_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h_spatial = spatial_extractor(feat)
        h_thf, spectral_stats = temporal_filter(h_spatial)
        if not temporal_enabled:
            h_thf = torch.zeros_like(h_spatial)
        token_input = torch.cat([h_spatial, h_thf, h_thf.abs()], dim=-1)
        tokens = projection(token_input)
        spatial_norm = _mean_vector_norm(h_spatial)
        temporal_norm = _mean_vector_norm(h_thf)
        stats = {
            "spatial_hf_norm": spatial_norm,
            "temporal_hf_norm": temporal_norm,
            "temporal_hf_ratio": temporal_norm / (spatial_norm + self.eps),
            **spectral_stats,
        }
        return tokens, h_spatial, h_thf, stats

    def forward(
        self,
        feat_s3: torch.Tensor,
        feat_s4: torch.Tensor,
        z_g: torch.Tensor,
        epoch: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, object]:
        active_mode = self.mode if mode is None else str(mode)
        if active_mode not in self.VALID_MODES:
            raise ValueError("Unsupported HTFR mode: {}.".format(active_mode))
        temporal_enabled = active_mode == "full"
        tokens3, _, _, freq3 = self._stage_tokens(
            feat_s3, self.spatial_hf_s3, self.temporal_hpf_s3, self.token_proj_s3,
            temporal_enabled,
        )
        tokens4, _, _, freq4 = self._stage_tokens(
            feat_s4, self.spatial_hf_s4, self.temporal_hpf_s4, self.token_proj_s4,
            temporal_enabled,
        )

        v3, attn3, attn3_stats = self.temporal_router_s3(z_g, tokens3)
        v4, attn4, attn4_stats = self.temporal_router_s4(z_g, tokens4)
        v_freq, stage_weights, stage_stats = self.stage_router(z_g, v3, v4)
        fusion_warm = 0.0 if active_mode == "baseline" else linear_warmup(
            epoch, self.fusion_start_epoch, self.fusion_warmup_epoch
        )
        z_final_normed, gate, gate_logits, delta, gate_stats = self.residual_gate(
            z_g, v_freq, fusion_warm
        )
        z_final = z_g if active_mode == "baseline" else z_final_normed

        diagnostics: Dict[str, float] = {"htfr_fusion_warm": float(fusion_warm)}
        for prefix, values in (
            ("htfr_s3_", {**freq3, **attn3_stats}),
            ("htfr_s4_", {**freq4, **attn4_stats}),
            ("htfr_", stage_stats),
            ("htfr_", gate_stats),
        ):
            for key, value in values.items():
                diagnostics[prefix + key] = float(value.detach().item())
        diagnostics.update(
            {
                "htfr_zg_norm": float(_mean_vector_norm(z_g.detach()).item()),
                "htfr_v_freq_norm": float(_mean_vector_norm(v_freq.detach()).item()),
                "htfr_zfinal_norm": float(_mean_vector_norm(z_final.detach()).item()),
            }
        )
        return {
            "z_final": z_final,
            "v_freq": v_freq,
            "v3": v3,
            "v4": v4,
            "attn3": attn3,
            "attn4": attn4,
            "stage_weights": stage_weights,
            "gate": gate,
            "gate_pre": torch.sigmoid(gate_logits),
            "gate_logits": gate_logits,
            "delta": delta,
            "fusion_warm": float(fusion_warm),
            "diagnostics": diagnostics,
        }


__all__ = [
    "SpatialHaarHFExtractor",
    "TemporalHighPass",
    "IdentityConditionedTemporalRouter",
    "HierarchicalStageRouter",
    "DifferentiableFrequencyResidualGate",
    "HierarchicalTemporalFrequencyRouter",
]
