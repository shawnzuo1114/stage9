"""The single Stage9-HTFR model used by this repository."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import VMambaMultiStageExtractor
from models.stage9_htfr_reid import HierarchicalTemporalFrequencyRouter
from models.sth_mamba import STHMambaBlock, STHVideoAggregator
from models.m3_fgsm_visual_cmip.heads import RetrievalHead
from models.m3_fgsm_visual_cmip.losses.batch_cross_modal_retrieval import BatchCrossModalRetrievalLoss
from models.m3_fgsm_visual_cmip.losses.cross_modal_prototype_retrieval import (
    CrossModalPrototypeMemory,
    CrossModalPrototypeRetrievalLoss,
)
from utils.warmup import linear_warmup


def _group_count(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class IRStructureRecovery(nn.Module):
    """Directional structure compensation shared by RGB and infrared frames."""

    def __init__(self, channels, alpha_ir=0.20, alpha_rgb=0.08, start_epoch=5, warmup_epoch=10):
        super().__init__()
        self.alpha_ir = float(alpha_ir)
        self.alpha_rgb = float(alpha_rgb)
        self.start_epoch = int(start_epoch)
        self.warmup_epoch = int(warmup_epoch)
        groups = _group_count(channels)
        self.diff_proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
        )
        self.gate_proj = nn.Conv2d(channels * 2, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, feat, modal_labels_frame=None, epoch=None, stage_id=None):
        dx = F.pad(feat[:, :, :, 1:] - feat[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
        dy = F.pad(feat[:, :, 1:, :] - feat[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
        center = feat - F.avg_pool2d(feat, 3, stride=1, padding=1)
        residual = self.out_proj(self.diff_proj(torch.cat((center, dx, dy), dim=1)))
        gate = torch.sigmoid(self.gate_proj(torch.cat((feat, residual), dim=1)))
        warm = linear_warmup(epoch or 0, self.start_epoch, self.warmup_epoch)
        if modal_labels_frame is None:
            alpha = feat.new_full((feat.size(0), 1, 1, 1), self.alpha_rgb)
        else:
            labels = modal_labels_frame.to(feat.device).view(-1, 1, 1, 1)
            alpha = torch.where(labels.eq(1), feat.new_tensor(self.alpha_ir), feat.new_tensor(self.alpha_rgb))
        delta = float(warm) * alpha * gate * residual
        output = feat + delta
        ratio = delta.detach().norm() / (feat.detach().norm() + 1e-6)
        prefix = "irm_srm_s{}_".format(stage_id)
        return output, {
            prefix + "res_ratio": float(ratio.item()),
            prefix + "gate_mean": float(gate.detach().mean().item()),
            "irm_srm_warm": float(warm),
        }


class Stage9Model(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        self.current_epoch = 0
        self.extractor = VMambaMultiStageExtractor(
            variant=args.vmamba_variant,
            pretrained_path=args.vmamba_pretrained,
            require_pretrained=args.require_vmamba_pretrained,
            local_vmamba_root=args.local_vmamba_root,
            out_indices=(2, 3),
            enable_sie=not args.disable_sie,
            sie_coeff=args.sie_coe,
            use_checkpoint=args.grad_checkpoint,
        )
        c3, c4 = self.extractor.stage_channels[2], self.extractor.stage_channels[3]
        block_args = dict(
            num_hubs=args.sth_num_hubs, require_mamba=True,
            d_state=args.mamba_d_state, d_conv=args.mamba_d_conv, expand=args.mamba_expand,
            local_mamba_root=args.local_mamba_root,
            local_causal_conv1d_root=args.local_causal_conv1d_root,
            use_checkpoint=args.grad_checkpoint,
        )
        self.sth_stage3_block = STHMambaBlock(c3, layer_idx_base=1100, module_name="stage9_sth_s3", **block_args)
        self.sth_stage4_block = STHMambaBlock(c4, layer_idx_base=1200, module_name="stage9_sth_s4", **block_args)
        srm_args = dict(alpha_ir=args.irm_srm_alpha_ir, alpha_rgb=args.irm_srm_alpha_rgb,
                        start_epoch=args.irm_srm_start_epoch, warmup_epoch=args.irm_srm_warmup_epoch)
        self.srm_stage3 = IRStructureRecovery(c3, **srm_args)
        self.srm_stage4 = IRStructureRecovery(c4, **srm_args)
        self.aggregator = STHVideoAggregator(c3, c4, token_dim=args.token_dim,
                                             hub_fusion_scale=args.sth_hub_fusion_scale)
        self.htfr = HierarchicalTemporalFrequencyRouter(
            stage3_channels=c3, stage4_channels=c4, token_dim=args.token_dim,
            temporal_temperature=args.htfr_temporal_temperature, gate_max=args.htfr_gate_max,
            fusion_start_epoch=args.htfr_fusion_start_epoch,
            fusion_warmup_epoch=args.htfr_fusion_warmup_epoch, mode=args.htfr_mode,
        )
        self.freq_classifier = nn.Linear(args.token_dim, num_classes, bias=False)
        nn.init.normal_(self.freq_classifier.weight, 0, 0.001)
        self.retrieval_head = RetrievalHead(args.token_dim, num_classes)
        self.batch_xm_loss = BatchCrossModalRetrievalLoss(
            args.batch_xm_tau, args.lambda_batch_xm, args.batch_xm_start_epoch, args.batch_xm_warmup_epoch)
        self.proto_memory = CrossModalPrototypeMemory(num_classes, args.token_dim, args.xm_proto_momentum)
        self.proto_loss = CrossModalPrototypeRetrievalLoss(
            args.xm_proto_tau, args.lambda_xm_proto, args.xm_proto_start_epoch, args.xm_proto_warmup_epoch)
        self.freq_xm_loss = BatchCrossModalRetrievalLoss(
            args.batch_xm_tau, args.htfr_freq_xm_lambda,
            args.htfr_freq_xm_start_epoch, args.htfr_freq_xm_warmup_epoch)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    @torch.no_grad()
    def update_prototypes(self, features, labels, modal_labels):
        self.proto_memory.update(features, labels, modal_labels, modal_labels.lt(2))

    @staticmethod
    def _frames(x, seq_len):
        if x.ndim != 4 or x.size(1) != 3 * seq_len:
            raise ValueError("Expected video tensor [B,3*T,H,W], got {} for T={}".format(tuple(x.shape), seq_len))
        return x.view(x.size(0) * seq_len, 3, x.size(2), x.size(3))

    def forward(self, x1, x2=None, modal=0, seq_len=None, labels=None, modal_labels=None, epoch=None, debug=False):
        seq_len = int(seq_len or self.args.seq_len)
        epoch = self.current_epoch if epoch is None else int(epoch)
        if modal == 0:
            if x2 is None or x1.size(0) != x2.size(0):
                raise ValueError("Stage9 training requires equally sized RGB and IR batches.")
            videos = torch.cat((x1, x2), dim=0)
            modal_labels = torch.cat((torch.zeros(x1.size(0), device=x1.device, dtype=torch.long),
                                      torch.ones(x2.size(0), device=x2.device, dtype=torch.long)))
        elif modal in (1, 2):
            videos = x1
            modal_labels = torch.full((x1.size(0),), modal - 1, device=x1.device, dtype=torch.long)
        else:
            raise ValueError("modal must be 0 (dual), 1 (RGB), or 2 (IR)")
        frames = self._frames(videos, seq_len)
        sth_warm = linear_warmup(epoch, self.args.sth_start_epoch, self.args.sth_warmup_epoch)
        features, hubs, extractor_debug = self.extractor.forward_stage9(
            frames, seq_len, modal_labels, self.sth_stage3_block, self.sth_stage4_block,
            self.srm_stage3, self.srm_stage4, epoch, sth_warm, debug)
        batch = videos.size(0)
        s3 = features["stage3"].view(batch, seq_len, *features["stage3"].shape[1:])
        s4 = features["stage4"].view(batch, seq_len, *features["stage4"].shape[1:])
        z_g, agg_debug = self.aggregator(s4, hubs["stage3"], hubs["stage4"])
        routed = self.htfr(s3, s4, z_g, epoch=epoch, mode=self.args.htfr_mode)
        head = self.retrieval_head(video_feat=routed["z_final"])
        htfr_debug = dict(routed["diagnostics"])
        htfr_debug.update(extractor_debug.get("sth", {}))
        htfr_debug.update(agg_debug)
        return {
            **head, "video_feat": routed["z_final"], "z_g": z_g,
            "z_final": routed["z_final"], "v_freq": routed["v_freq"],
            "freq_logits": self.freq_classifier(routed["v_freq"]),
            "modal_labels": modal_labels, "labels": labels,
            "htfr_debug": htfr_debug, "debug": {"head": head["debug"], "htfr": htfr_debug},
        }
