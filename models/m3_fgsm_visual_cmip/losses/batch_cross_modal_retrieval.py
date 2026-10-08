import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warmup import linear_warmup


class BatchCrossModalRetrievalLoss(nn.Module):
    def __init__(self, tau=0.07, lambda_max=0.10, start_epoch=5, warmup_epoch=15, eps=1e-12):
        super(BatchCrossModalRetrievalLoss, self).__init__()
        self.tau = float(tau)
        self.lambda_max = float(lambda_max)
        self.start_epoch = int(start_epoch)
        self.warmup_epoch = int(warmup_epoch)
        self.eps = float(eps)

    def _zero(self, ref):
        return ref.sum() * 0.0

    def _direction(self, query, query_labels, cand, cand_labels):
        losses = []
        pos_terms = []
        neg_terms = []
        valid = 0
        if query.numel() == 0 or cand.numel() == 0:
            return losses, pos_terms, neg_terms, valid
        sim = torch.matmul(query, cand.t())
        for idx in range(query.size(0)):
            pos_mask = cand_labels.eq(query_labels[idx])
            neg_mask = ~pos_mask
            if not bool(pos_mask.any()) or not bool(neg_mask.any()):
                continue
            pos_sim = sim[idx][pos_mask]
            cand_sim = sim[idx]
            log_pos = torch.logsumexp(pos_sim / max(self.tau, self.eps), dim=0)
            log_den = torch.logsumexp(cand_sim / max(self.tau, self.eps), dim=0)
            losses.append(-(log_pos - log_den))
            pos_terms.append(pos_sim.mean().detach())
            neg_terms.append(sim[idx][neg_mask].mean().detach())
            valid += 1
        return losses, pos_terms, neg_terms, valid

    def forward(self, features, labels, modal_labels, clean_mask=None, epoch=None):
        if epoch is None:
            epoch = 0
        stats = {
            "batch_xm_valid_anchors": 0.0,
            "batch_xm_pos_sim": 0.0,
            "batch_xm_neg_sim": 0.0,
            "batch_xm_gap": 0.0,
            "batch_xm_weight": 0.0,
        }
        if features is None:
            ref = labels if torch.is_tensor(labels) else modal_labels
            if torch.is_tensor(ref):
                return ref.new_tensor(0.0, dtype=torch.float32), stats
            return torch.tensor(0.0), stats
        zero = self._zero(features)
        if labels is None or modal_labels is None or features.numel() == 0:
            return zero, stats
        valid = modal_labels.lt(2)
        if clean_mask is not None:
            valid = valid & clean_mask.bool()
        rgb = valid & modal_labels.eq(0)
        ir = valid & modal_labels.eq(1)
        if not bool(rgb.any()) or not bool(ir.any()):
            return zero, stats
        z = F.normalize(features.float(), dim=-1, eps=self.eps)
        losses, pos_terms, neg_terms, valid_count = self._direction(z[rgb], labels[rgb], z[ir], labels[ir])
        more_losses, more_pos, more_neg, more_valid = self._direction(z[ir], labels[ir], z[rgb], labels[rgb])
        losses.extend(more_losses)
        pos_terms.extend(more_pos)
        neg_terms.extend(more_neg)
        valid_count += more_valid
        if not losses:
            return zero, stats
        raw = torch.stack(losses).mean().to(dtype=features.dtype)
        weight = self.lambda_max * linear_warmup(epoch, self.start_epoch, self.warmup_epoch)
        loss = float(weight) * raw
        pos = torch.stack(pos_terms).mean()
        neg = torch.stack(neg_terms).mean()
        stats.update(
            {
                "batch_xm_valid_anchors": float(valid_count),
                "batch_xm_pos_sim": float(pos.item()),
                "batch_xm_neg_sim": float(neg.item()),
                "batch_xm_gap": float((pos - neg).item()),
                "batch_xm_weight": float(weight),
            }
        )
        return loss, stats
