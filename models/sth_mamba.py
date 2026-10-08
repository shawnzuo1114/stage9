import torch
import torch.nn as nn
import torch.nn.functional as F

from models.m3_fgsm_visual_cmip.multi_axis_mamba import ResidualAxisMamba
from utils.warmup import linear_warmup


class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(_DropPath, self).__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class GeMPooling(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p_init))
        self.eps = float(eps)

    def forward(self, x):
        p = self.p.clamp(min=1.0, max=6.0)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(p), x.shape[-2:]).pow(1.0 / p).flatten(1)


class STHMambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_hubs=4,
        drop_path=0.0,
        require_mamba=True,
        d_state=16,
        d_conv=4,
        expand=2,
        local_mamba_root="",
        local_causal_conv1d_root="",
        use_checkpoint=False,
        layer_idx_base=1000,
        module_name="stage9_sth",
    ):
        super(STHMambaBlock, self).__init__()
        self.dim = int(dim)
        self.num_hubs = int(num_hubs)
        self.require_mamba = bool(require_mamba)
        self.hub_init = nn.Parameter(torch.randn(1, 1, self.num_hubs, self.dim) * 0.02)
        self.spatial_norm = nn.LayerNorm(self.dim)
        self.temporal_norm = nn.LayerNorm(self.dim)
        self.spatial_mamba = ResidualAxisMamba(
            self.dim,
            d_state,
            d_conv,
            expand,
            False,
            False,
            local_mamba_root,
            local_causal_conv1d_root,
            use_checkpoint,
            layer_idx_base,
            module_name + "_spatial",
        )
        self.temporal_mamba = ResidualAxisMamba(
            self.dim,
            d_state,
            d_conv,
            expand,
            False,
            False,
            local_mamba_root,
            local_causal_conv1d_root,
            use_checkpoint,
            layer_idx_base + 1,
            module_name + "_temporal",
        )
        self.hub_proj = nn.Linear(self.dim, self.dim)
        self.gate = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.BatchNorm2d(self.dim)
        self.drop_path = _DropPath(drop_path)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat, epoch=None, warm=1.0):
        if feat.ndim != 5:
            raise ValueError("STHMambaBlock expects feat [B,T,C,H,W], got {}.".format(tuple(feat.shape)))
        batch_size, seq_len, channels, height, width = feat.shape
        if channels != self.dim:
            raise ValueError("STHMambaBlock dim mismatch: expected {}, got {}.".format(self.dim, channels))
        warm_value = float(warm)

        hubs = self.hub_init.expand(batch_size, seq_len, self.num_hubs, channels)
        patch = feat.permute(0, 1, 3, 4, 2).contiguous().view(batch_size * seq_len, height * width, channels)
        hubs_bt = hubs.contiguous().view(batch_size * seq_len, self.num_hubs, channels)

        seq = torch.cat([hubs_bt, patch], dim=1)
        seq = self.spatial_mamba(self.spatial_norm(seq))
        hubs = seq[:, : self.num_hubs].contiguous().view(batch_size, seq_len, self.num_hubs, channels)

        hubs_temporal = hubs.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_hubs, seq_len, channels)
        hubs_temporal = self.temporal_mamba(self.temporal_norm(hubs_temporal))
        hubs = hubs_temporal.view(batch_size, self.num_hubs, seq_len, channels).permute(0, 2, 1, 3).contiguous()

        hub_frame = hubs.mean(dim=2)
        hub_map = self.hub_proj(hub_frame).view(batch_size * seq_len, channels, 1, 1).expand(-1, -1, height, width)
        feat_bt = feat.contiguous().view(batch_size * seq_len, channels, height, width)
        gate = self.gate(torch.cat([feat_bt, hub_map], dim=1))
        residual = self.drop_path(gate * hub_map)
        refined = feat_bt + feat_bt.new_tensor(warm_value) * self.gamma * residual
        refined = self.out_norm(refined).view(batch_size, seq_len, channels, height, width)

        delta_ratio = (refined.contiguous().view(batch_size * seq_len, channels, height, width) - feat_bt).detach().norm()
        delta_ratio = delta_ratio / (feat_bt.detach().norm() + 1e-6)
        debug = {
            "sth_warm": warm_value,
            "sth_gamma": float(self.gamma.detach().item()),
            "sth_hub_norm": float(hubs.detach().norm(dim=-1).mean().item()),
            "sth_gate_mean": float(gate.detach().mean().item()),
            "sth_feat_delta_ratio": float(delta_ratio.item()),
        }
        return refined, hubs, debug


class STHVideoAggregator(nn.Module):
    def __init__(
        self,
        stage3_channels,
        stage4_channels,
        token_dim=512,
        hub_fusion_scale=1.0,
        use_stage3_hub=True,
        use_stage4_hub=True,
        gem_p_init=3.0,
    ):
        super(STHVideoAggregator, self).__init__()
        self.token_dim = int(token_dim)
        self.hub_fusion_scale = float(hub_fusion_scale)
        self.use_stage3_hub = bool(use_stage3_hub)
        self.use_stage4_hub = bool(use_stage4_hub)
        self.stage4_proj = nn.Conv2d(stage4_channels, self.token_dim, kernel_size=1, bias=False)
        self.gem = GeMPooling(p_init=gem_p_init)
        self.hub3_proj = nn.Linear(stage3_channels, self.token_dim) if self.use_stage3_hub else None
        self.hub4_proj = nn.Linear(stage4_channels, self.token_dim) if self.use_stage4_hub else None
        self.hub_norm = nn.LayerNorm(self.token_dim)
        self.out_norm = nn.LayerNorm(self.token_dim)

    def forward(self, stage4_feat, stage3_hubs=None, stage4_hubs=None):
        if stage4_feat.ndim != 5:
            raise ValueError("STHVideoAggregator expects stage4_feat [B,T,C,H,W], got {}.".format(tuple(stage4_feat.shape)))
        batch_size, seq_len, channels, height, width = stage4_feat.shape
        stage4_bt = stage4_feat.contiguous().view(batch_size * seq_len, channels, height, width)
        stage4_map = self.stage4_proj(stage4_bt)
        global_frame = self.gem(stage4_map).view(batch_size, seq_len, self.token_dim)
        global_video = global_frame.mean(dim=1)

        hub_terms = []
        if self.hub3_proj is not None and stage3_hubs is not None:
            if stage3_hubs.ndim != 4 or stage3_hubs.size(0) != batch_size or stage3_hubs.size(1) != seq_len:
                raise ValueError("stage3_hubs must be [B,T,M,C3], got {}.".format(tuple(stage3_hubs.shape)))
            hub_terms.append(self.hub3_proj(stage3_hubs.mean(dim=(1, 2))))
        if self.hub4_proj is not None and stage4_hubs is not None:
            if stage4_hubs.ndim != 4 or stage4_hubs.size(0) != batch_size or stage4_hubs.size(1) != seq_len:
                raise ValueError("stage4_hubs must be [B,T,M,C4], got {}.".format(tuple(stage4_hubs.shape)))
            hub_terms.append(self.hub4_proj(stage4_hubs.mean(dim=(1, 2))))

        if hub_terms:
            hub_video = self.hub_norm(torch.stack(hub_terms, dim=0).sum(dim=0))
        else:
            hub_video = torch.zeros_like(global_video)
        video_feat = self.out_norm(global_video + self.hub_fusion_scale * hub_video)
        debug = {
            "stage9_global_norm": float(global_video.detach().norm(dim=-1).mean().item()),
            "stage9_hub_norm": float(hub_video.detach().norm(dim=-1).mean().item()),
            "stage9_video_feat_norm": float(video_feat.detach().norm(dim=-1).mean().item()),
        }
        return video_feat, debug


__all__ = ["STHMambaBlock", "STHVideoAggregator", "linear_warmup"]
