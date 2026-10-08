import os
import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.feature_debug import tensor_stats
from utils.pretrained_audit import print_pretrained_audit, summarize_load_state
from utils.progress import tqdm_write


def _add_vmamba_paths(local_vmamba_root=""):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    workspace_dir = os.path.dirname(project_dir)
    candidates = []
    if local_vmamba_root:
        candidates.append(local_vmamba_root)
    candidates.extend(
        [
            os.path.join(workspace_dir, "VMamba-main"),
            os.path.join(workspace_dir, "mamba-1p1p1"),
            os.path.join(workspace_dir, "causal-conv1d"),
        ]
    )
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


_VMAMBA_CONFIGS = {
    "vmamba_tiny": dict(depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v05_noz", mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False, patch_size=4, in_chans=3, num_classes=1000, patch_norm=True, norm_layer="ln2d", downsample_version="v3", patchembed_version="v2", use_checkpoint=False, posembed=False, imgsize=224),
    "vmamba_small": dict(depths=[2, 2, 20, 2], dims=96, drop_path_rate=0.3, ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v05_noz", mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False, patch_size=4, in_chans=3, num_classes=1000, patch_norm=True, norm_layer="ln2d", downsample_version="v3", patchembed_version="v2", use_checkpoint=False, posembed=False, imgsize=224),
    "vmamba_base": dict(depths=[2, 2, 20, 2], dims=128, drop_path_rate=0.5, ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu", ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v05_noz", mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False, patch_size=4, in_chans=3, num_classes=1000, patch_norm=True, norm_layer="ln2d", downsample_version="v3", patchembed_version="v2", use_checkpoint=False, posembed=False, imgsize=224),
}


class ModalitySIE(nn.Module):
    def __init__(self, channels, num_embeddings=2, coeff=1.0):
        super(ModalitySIE, self).__init__()
        self.coeff = coeff
        self.embedding = nn.Embedding(num_embeddings, channels)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, x, modality_id):
        if modality_id is None:
            return x
        emb = self.embedding.weight[modality_id].view(1, -1, 1, 1)
        return x + self.coeff * emb


class VMambaMultiStageExtractor(nn.Module):
    def __init__(
        self,
        variant="vmamba_small",
        pretrained_path="",
        require_pretrained=False,
        local_vmamba_root="",
        out_indices=(2, 3),
        enable_sie=False,
        sie_coeff=1.0,
        use_checkpoint=False,
    ):
        super(VMambaMultiStageExtractor, self).__init__()
        _add_vmamba_paths(local_vmamba_root=local_vmamba_root)
        from classification.models.vmamba import Backbone_VSSM
        if variant not in _VMAMBA_CONFIGS:
            raise ValueError("Unsupported VMamba variant: {}".format(variant))
        config = dict(_VMAMBA_CONFIGS[variant])
        config["use_checkpoint"] = bool(use_checkpoint)
        self.variant = variant
        self.out_indices = tuple(out_indices)
        self.backbone = Backbone_VSSM(out_indices=self.out_indices, pretrained=None, **config)
        self.stage_channels = list(self.backbone.dims)
        self.enable_sie = enable_sie
        self.sie_layers = nn.ModuleList([ModalitySIE(channels, coeff=sie_coeff) if enable_sie else nn.Identity() for channels in self.stage_channels])
        self.pretrained_report = None
        if pretrained_path:
            self.pretrained_report = self._load_checkpoint(pretrained_path)
            print_pretrained_audit(self.pretrained_report)
        elif require_pretrained:
            raise RuntimeError("require_vmamba_pretrained=True but no checkpoint path was provided.")
        tqdm_write("[VMambaExtractor] out_indices={}".format(self.out_indices))

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        for key in ("model", "state_dict", "net"):
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]
                break
        cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model_state = self.backbone.state_dict()
        merged_state = dict(model_state)
        loaded_keys = []
        viewed_keys = []
        skipped_shape_keys = []
        for key, value in cleaned.items():
            if key not in model_state:
                continue
            target = model_state[key]
            if target.shape == value.shape:
                merged_state[key] = value
                loaded_keys.append(key)
            elif target.numel() == value.numel():
                merged_state[key] = value.view_as(target)
                loaded_keys.append(key)
                viewed_keys.append(key)
            else:
                skipped_shape_keys.append({"key": key, "checkpoint_shape": tuple(value.shape), "model_shape": tuple(target.shape)})
        incompatible = self.backbone.load_state_dict(merged_state, strict=False)
        missing_keys = [key for key in model_state.keys() if key not in loaded_keys]
        unexpected_keys = [key for key in cleaned.keys() if key not in model_state]
        report = summarize_load_state(
            pretrained_path=checkpoint_path,
            incompatible=incompatible,
            model_state_dict=model_state,
            loaded_state_dict={key: cleaned[key] for key in loaded_keys if key in cleaned},
        )
        report["missing_keys"] = missing_keys
        report["unexpected_keys"] = unexpected_keys
        report["matched_keys"] = loaded_keys
        report["viewed_keys"] = viewed_keys
        report["skipped_shape_keys"] = skipped_shape_keys
        report["load_ratio"] = float(len(loaded_keys) / max(len(model_state), 1))
        if report["load_ratio"] < 0.5:
            tqdm_write("Warning: VMamba load_ratio is low ({:.4f}); verify checkpoint/variant alignment.".format(report["load_ratio"]))
        return report

    @staticmethod
    def _ensure_channel_first(feature, expected_channels):
        raw_shape = tuple(feature.shape)
        if feature.ndim != 4:
            raise ValueError("Expected 4D stage feature, got {}.".format(raw_shape))
        if feature.shape[1] == expected_channels:
            converted = feature
        elif feature.shape[-1] == expected_channels:
            converted = feature.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("Cannot infer channel layout for stage feature {}.".format(raw_shape))
        return raw_shape, converted

    @staticmethod
    def _restore_layout(channel_first, raw_shape):
        if len(raw_shape) != 4:
            raise ValueError("Expected 4D raw shape, got {}.".format(raw_shape))
        if raw_shape[1] == channel_first.shape[1]:
            return channel_first
        if raw_shape[-1] == channel_first.shape[1]:
            return channel_first.permute(0, 2, 3, 1).contiguous()
        raise ValueError("Cannot restore channel layout for raw shape {} and feature {}.".format(raw_shape, tuple(channel_first.shape)))

    def _apply_sie(self, feature, stage_idx, modality_id=None, seq_len=None):
        if not self.enable_sie or modality_id is None:
            return feature
        sie = self.sie_layers[stage_idx]
        if isinstance(sie, nn.Identity):
            return feature
        if torch.is_tensor(modality_id):
            ids = modality_id.to(device=feature.device, dtype=torch.long).flatten()
            if seq_len is not None and ids.numel() * int(seq_len) == feature.size(0):
                ids = ids.repeat_interleave(int(seq_len))
            if ids.numel() != feature.size(0):
                raise ValueError(
                    "SIE modality ids must match frames. got ids={} feature_N={} seq_len={}.".format(
                        ids.numel(), feature.size(0), seq_len
                    )
                )
            emb = sie.embedding(ids).view(feature.size(0), -1, 1, 1)
            return feature + sie.coeff * emb
        return sie(feature, modality_id)

    @staticmethod
    def _is_identity(module):
        return module is None or isinstance(module, nn.Identity)

    def _forward_layer_body_and_downsample(self, layer, feat):
        if not hasattr(layer, "blocks"):
            return layer(feat), None
        for block in layer.blocks:
            if bool(getattr(layer, "use_checkpoint", False)) and self.training and torch.is_tensor(feat) and feat.requires_grad:
                feat = checkpoint(block, feat, use_reentrant=True)
            else:
                feat = block(feat)
        return feat, getattr(layer, "downsample", None)

    def _apply_stage_output_norm(self, feat, stage_idx):
        for name in (
            "outnorm{}".format(stage_idx),
            "out_norm{}".format(stage_idx),
            "norm{}".format(stage_idx),
        ):
            norm = getattr(self.backbone, name, None)
            if norm is not None:
                return norm(feat)
        norms = getattr(self.backbone, "norms", None)
        if norms is not None and stage_idx < len(norms):
            return norms[stage_idx](feat)
        return feat

    def forward(self, x, modality_id=None, debug=False):
        raw_outputs = self.backbone(x)
        if len(raw_outputs) != len(self.out_indices):
            raise ValueError("Backbone returned {} stages, expected {}.".format(len(raw_outputs), len(self.out_indices)))

        features = {}
        debug_info = {"pretrained_report": self.pretrained_report}
        for list_idx, stage_idx in enumerate(self.out_indices):
            stage_name = "stage{}".format(stage_idx + 1)
            raw_shape, converted = self._ensure_channel_first(raw_outputs[list_idx], self.stage_channels[stage_idx])
            converted = self._apply_sie(converted, stage_idx, modality_id=modality_id)
            stats = tensor_stats(converted)
            features[stage_name] = converted
            debug_info[stage_name] = {
                "raw_shape": raw_shape,
                "converted_shape": tuple(converted.shape),
                "mean": stats["mean"],
                "std": stats["std"],
                "feature_norm": stats["norm"],
                "requires_grad": bool(converted.requires_grad),
            }
            if debug:
                print("[VMambaMultiStageExtractor] {} raw={} converted={} mean={:.6f} std={:.6f} norm={:.6f} requires_grad={}".format(
                    stage_name,
                    raw_shape,
                    tuple(converted.shape),
                    stats["mean"],
                    stats["std"],
                    stats["norm"],
                    converted.requires_grad,
                ))
        if self.pretrained_report is not None and debug:
            print("[VMambaMultiStageExtractor] checkpoint={} load_ratio={:.4f} missing={} unexpected={}".format(
                self.pretrained_report.get("pretrained_path"),
                self.pretrained_report.get("load_ratio", 0.0),
                len(self.pretrained_report.get("missing_keys", [])),
                len(self.pretrained_report.get("unexpected_keys", [])),
            ))
        return features, debug_info

    def forward_stage9(
        self,
        x,
        seq_len,
        modality_id=None,
        sth_stage3_block=None,
        sth_stage4_block=None,
        srm_stage3=None,
        srm_stage4=None,
        epoch=None,
        sth_warm=1.0,
        debug=False,
    ):
        if x.size(0) % int(seq_len) != 0:
            raise ValueError("Stage9 frame batch {} is not divisible by seq_len {}.".format(x.size(0), seq_len))
        if not hasattr(self.backbone, "patch_embed") or not hasattr(self.backbone, "layers"):
            raise RuntimeError(
                "Stage9 requires a VMamba backbone exposing sequential stage forward."
            )
        batch_size = x.size(0) // int(seq_len)
        layers = self.backbone.layers
        if len(layers) < 4:
            raise RuntimeError("Stage9 requires at least 4 VMamba stages, got {}.".format(len(layers)))

        feat = self.backbone.patch_embed(x)
        if hasattr(self.backbone, "pos_drop"):
            feat = self.backbone.pos_drop(feat)

        features = {}
        hubs = {"stage3": None, "stage4": None}
        sth_debug = {}
        srm_res_losses = []
        debug_info = {"pretrained_report": self.pretrained_report}
        modal_labels_frame = None
        if torch.is_tensor(modality_id):
            modal_video = modality_id.to(device=x.device).long().view(-1)
            if modal_video.numel() == batch_size:
                modal_labels_frame = modal_video.repeat_interleave(int(seq_len))
            elif modal_video.numel() == x.size(0):
                modal_labels_frame = modal_video
        elif modality_id is not None:
            modal_labels_frame = torch.full((x.size(0),), int(modality_id), dtype=torch.long, device=x.device)
        for stage_idx, layer in enumerate(layers):
            feat, downsample = self._forward_layer_body_and_downsample(layer, feat)
            if stage_idx not in (2, 3):
                if not self._is_identity(downsample):
                    feat = downsample(feat)
                continue
            stage_name = "stage{}".format(stage_idx + 1)
            raw_shape, converted = self._ensure_channel_first(feat, self.stage_channels[stage_idx])
            converted = self._apply_sie(converted, stage_idx, modality_id=modality_id, seq_len=seq_len)

            if stage_idx == 2 and srm_stage3 is not None:
                converted, srm_debug = srm_stage3(
                    converted,
                    modal_labels_frame=modal_labels_frame,
                    epoch=epoch,
                    stage_id=3,
                )
                feat = self._restore_layout(converted, raw_shape)
                sth_debug.update({k: v for k, v in srm_debug.items() if k != "res_ratio_loss"})
                if srm_debug.get("res_ratio_loss") is not None:
                    srm_res_losses.append(srm_debug["res_ratio_loss"])
            elif stage_idx == 3 and srm_stage4 is not None:
                converted, srm_debug = srm_stage4(
                    converted,
                    modal_labels_frame=modal_labels_frame,
                    epoch=epoch,
                    stage_id=4,
                )
                feat = self._restore_layout(converted, raw_shape)
                sth_debug.update({k: v for k, v in srm_debug.items() if k != "res_ratio_loss"})
                if srm_debug.get("res_ratio_loss") is not None:
                    srm_res_losses.append(srm_debug["res_ratio_loss"])

            if stage_idx == 2 and sth_stage3_block is not None:
                stage_video = converted.contiguous().view(batch_size, int(seq_len), converted.size(1), converted.size(2), converted.size(3))
                stage_video, stage_hubs, block_debug = sth_stage3_block(stage_video, epoch=epoch, warm=sth_warm)
                converted = stage_video.contiguous().view(batch_size * int(seq_len), converted.size(1), converted.size(2), converted.size(3))
                feat = self._restore_layout(converted, raw_shape)
                hubs["stage3"] = stage_hubs
                sth_debug.update({
                    "sth_warm": block_debug.get("sth_warm", float(sth_warm)),
                    "sth3_hub_norm": block_debug.get("sth_hub_norm", 0.0),
                    "sth3_gate_mean": block_debug.get("sth_gate_mean", 0.0),
                    "sth3_delta_ratio": block_debug.get("sth_feat_delta_ratio", 0.0),
                    "sth3_gamma": block_debug.get("sth_gamma", 0.0),
                })
            elif stage_idx == 3 and sth_stage4_block is not None:
                stage_video = converted.contiguous().view(batch_size, int(seq_len), converted.size(1), converted.size(2), converted.size(3))
                stage_video, stage_hubs, block_debug = sth_stage4_block(stage_video, epoch=epoch, warm=sth_warm)
                converted = stage_video.contiguous().view(batch_size * int(seq_len), converted.size(1), converted.size(2), converted.size(3))
                feat = self._restore_layout(converted, raw_shape)
                hubs["stage4"] = stage_hubs
                sth_debug.update({
                    "sth_warm": block_debug.get("sth_warm", float(sth_warm)),
                    "sth4_hub_norm": block_debug.get("sth_hub_norm", 0.0),
                    "sth4_gate_mean": block_debug.get("sth_gate_mean", 0.0),
                    "sth4_delta_ratio": block_debug.get("sth_feat_delta_ratio", 0.0),
                    "sth4_gamma": block_debug.get("sth_gamma", 0.0),
                })

            features[stage_name] = converted
            stats = tensor_stats(converted)
            debug_info[stage_name] = {
                "raw_shape": raw_shape,
                "converted_shape": tuple(converted.shape),
                "mean": stats["mean"],
                "std": stats["std"],
                "feature_norm": stats["norm"],
                "requires_grad": bool(converted.requires_grad),
            }
            if debug:
                print("[VMambaStage9] {} raw={} converted={} warm={:.4f}".format(
                    stage_name, raw_shape, tuple(converted.shape), float(sth_warm)
                ))
            if not self._is_identity(downsample):
                feat = downsample(feat)
        if "stage3" not in features or "stage4" not in features:
            raise RuntimeError("Stage9 requires stage3/stage4 features from sequential VMamba forward.")
        debug_info["sth"] = sth_debug
        if srm_res_losses:
            debug_info["srm_res_loss"] = torch.stack(srm_res_losses).mean()
        return features, hubs, debug_info
