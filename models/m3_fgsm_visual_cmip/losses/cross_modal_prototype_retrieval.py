import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warmup import linear_warmup


class CrossModalPrototypeMemory(nn.Module):
    def __init__(self, num_classes, feature_dim, momentum=0.2, eps=1e-12):
        super(CrossModalPrototypeMemory, self).__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("rgb_proto", torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("ir_proto", torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("uni_proto", torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("rgb_valid", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("ir_valid", torch.zeros(self.num_classes, dtype=torch.bool))
        self.register_buffer("uni_valid", torch.zeros(self.num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update_one(self, table, valid, label, feat):
        label_int = int(label)
        feat = F.normalize(feat.detach().float(), dim=0, eps=self.eps)
        if bool(valid[label_int]):
            table[label_int].mul_(self.momentum).add_(feat, alpha=1.0 - self.momentum)
            table[label_int].copy_(F.normalize(table[label_int], dim=0, eps=self.eps))
        else:
            table[label_int].copy_(feat)
            valid[label_int] = True

    @torch.no_grad()
    def update(self, features, labels, modal_labels, clean_mask=None):
        if features is None or labels is None or modal_labels is None or features.numel() == 0:
            return
        valid = modal_labels.lt(2)
        if clean_mask is not None:
            valid = valid & clean_mask.bool()
        if not bool(valid.any()):
            return
        z = F.normalize(features.detach().float(), dim=-1, eps=self.eps)
        for label in labels[valid].unique():
            label_int = int(label.item())
            if label_int < 0 or label_int >= self.num_classes:
                continue
            same = valid & labels.eq(label)
            rgb = same & modal_labels.eq(0)
            ir = same & modal_labels.eq(1)
            if bool(rgb.any()):
                self._update_one(self.rgb_proto, self.rgb_valid, label_int, z[rgb].mean(dim=0))
            if bool(ir.any()):
                self._update_one(self.ir_proto, self.ir_valid, label_int, z[ir].mean(dim=0))
            clean = rgb | ir
            if bool(clean.any()):
                self._update_one(self.uni_proto, self.uni_valid, label_int, z[clean].mean(dim=0))

    def summary_stats(self):
        return {
            "xm_proto_rgb_valid_count": float(self.rgb_valid.float().sum().item()),
            "xm_proto_ir_valid_count": float(self.ir_valid.float().sum().item()),
            "xm_proto_uni_valid_count": float(self.uni_valid.float().sum().item()),
        }


class CrossModalPrototypeRetrievalLoss(nn.Module):
    def __init__(self, tau=0.07, lambda_max=0.25, start_epoch=20, warmup_epoch=20, same_mod_weight=0.25, eps=1e-12):
        super(CrossModalPrototypeRetrievalLoss, self).__init__()
        self.tau = float(tau)
        self.lambda_max = float(lambda_max)
        self.start_epoch = int(start_epoch)
        self.warmup_epoch = int(warmup_epoch)
        self.same_mod_weight = float(same_mod_weight)
        self.eps = float(eps)

    def _zero(self, ref):
        return ref.sum() * 0.0

    @staticmethod
    def _masked_logsumexp(logits, mask):
        if not bool(mask.any()):
            return None
        return torch.logsumexp(logits[mask], dim=0)

    def forward(self, features, labels, modal_labels, memory, clean_mask=None, epoch=None):
        if epoch is None:
            epoch = 0
        stats = {
            "xm_proto_valid_samples": 0.0,
            "xm_proto_rgb_valid_count": 0.0,
            "xm_proto_ir_valid_count": 0.0,
            "xm_proto_uni_valid_count": 0.0,
            "xm_proto_pos_sim": 0.0,
            "xm_proto_neg_sim": 0.0,
            "xm_proto_gap": 0.0,
            "xm_proto_weight": 0.0,
        }
        if features is None:
            ref = labels if torch.is_tensor(labels) else modal_labels
            if torch.is_tensor(ref):
                return ref.new_tensor(0.0, dtype=torch.float32), stats
            return torch.tensor(0.0), stats
        zero = self._zero(features)
        if memory is None:
            return zero, stats
        if labels is None or modal_labels is None or features.numel() == 0:
            stats.update(memory.summary_stats())
            return zero, stats
        stats.update(memory.summary_stats())
        valid = modal_labels.lt(2)
        if clean_mask is not None:
            valid = valid & clean_mask.bool()
        if not bool(valid.any()):
            return zero, stats

        z = F.normalize(features.float(), dim=-1, eps=self.eps)
        rgb_proto = F.normalize(memory.rgb_proto.to(device=z.device, dtype=z.dtype), dim=-1, eps=self.eps)
        ir_proto = F.normalize(memory.ir_proto.to(device=z.device, dtype=z.dtype), dim=-1, eps=self.eps)
        uni_proto = F.normalize(memory.uni_proto.to(device=z.device, dtype=z.dtype), dim=-1, eps=self.eps)
        rgb_valid = memory.rgb_valid.to(device=z.device)
        ir_valid = memory.ir_valid.to(device=z.device)
        uni_valid = memory.uni_valid.to(device=z.device)

        all_proto = torch.cat([rgb_proto, ir_proto, uni_proto], dim=0)
        all_valid = torch.cat([rgb_valid, ir_valid, uni_valid], dim=0)
        losses = []
        pos_terms = []
        neg_terms = []
        for idx in torch.nonzero(valid, as_tuple=False).flatten():
            label = int(labels[idx].item())
            if label < 0 or label >= memory.num_classes:
                continue
            modal = int(modal_labels[idx].item())
            sim_all = torch.matmul(all_proto, z[idx]) / max(self.tau, self.eps)
            same_mask = torch.zeros_like(all_valid)
            same_mask[label] = True
            same_mask[memory.num_classes + label] = True
            same_mask[2 * memory.num_classes + label] = True
            neg_mask = all_valid & (~same_mask)
            if not bool(neg_mask.any()):
                continue

            pos_logs = []
            pos_sims = []
            if modal == 0 and bool(ir_valid[label]):
                pos_logs.append(torch.matmul(ir_proto[label], z[idx]) / max(self.tau, self.eps))
                pos_sims.append(torch.matmul(ir_proto[label], z[idx]).detach())
            if modal == 1 and bool(rgb_valid[label]):
                pos_logs.append(torch.matmul(rgb_proto[label], z[idx]) / max(self.tau, self.eps))
                pos_sims.append(torch.matmul(rgb_proto[label], z[idx]).detach())
            if bool(uni_valid[label]):
                pos_logs.append(torch.matmul(uni_proto[label], z[idx]) / max(self.tau, self.eps))
                pos_sims.append(torch.matmul(uni_proto[label], z[idx]).detach())
            if modal == 0 and bool(rgb_valid[label]) and self.same_mod_weight > 0.0:
                pos_logs.append(torch.matmul(rgb_proto[label], z[idx]) / max(self.tau, self.eps) + torch.log(z.new_tensor(self.same_mod_weight)))
            if modal == 1 and bool(ir_valid[label]) and self.same_mod_weight > 0.0:
                pos_logs.append(torch.matmul(ir_proto[label], z[idx]) / max(self.tau, self.eps) + torch.log(z.new_tensor(self.same_mod_weight)))
            if not pos_logs:
                continue

            log_pos = torch.logsumexp(torch.stack(pos_logs), dim=0)
            log_den = torch.logsumexp(sim_all[all_valid], dim=0)
            losses.append(-(log_pos - log_den))
            pos_terms.append(torch.stack(pos_sims).mean() if pos_sims else z.new_tensor(0.0))
            neg_terms.append((sim_all[neg_mask] * max(self.tau, self.eps)).mean().detach())

        if not losses:
            return zero, stats
        raw = torch.stack(losses).mean().to(dtype=features.dtype)
        weight = self.lambda_max * linear_warmup(epoch, self.start_epoch, self.warmup_epoch)
        loss = float(weight) * raw
        pos = torch.stack(pos_terms).mean()
        neg = torch.stack(neg_terms).mean()
        stats.update(
            {
                "xm_proto_valid_samples": float(len(losses)),
                "xm_proto_pos_sim": float(pos.item()),
                "xm_proto_neg_sim": float(neg.item()),
                "xm_proto_gap": float((pos - neg).item()),
                "xm_proto_weight": float(weight),
            }
        )
        return loss, stats
