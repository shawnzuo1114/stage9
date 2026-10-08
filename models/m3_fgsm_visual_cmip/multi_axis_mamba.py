import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.mamba_compat import build_mamba_block


class ResidualAxisMamba(nn.Module):
    def __init__(
        self,
        dim,
        d_state,
        d_conv,
        expand,
        bimamba,
        require_bimamba,
        local_mamba_root,
        local_causal_conv1d_root,
        use_checkpoint=False,
        layer_idx=0,
        module_name="axis_mamba",
    ):
        super(ResidualAxisMamba, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(dim)
        self.mamba = build_mamba_block(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba=bimamba,
            require_bimamba=require_bimamba,
            use_fast_path=True,
            layer_idx=layer_idx,
            module_name=module_name,
            allow_fallback=not require_bimamba,
            local_mamba_root=local_mamba_root,
            local_causal_conv1d_root=local_causal_conv1d_root,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def _forward_impl(self, x):
        x = x + self.mamba(self.norm(x))
        x = x + self.ffn(x)
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)


class LearnableAlpha(nn.Module):
    def __init__(self, init_value=0.1):
        super(LearnableAlpha, self).__init__()
        self.weight = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self):
        return self.weight
