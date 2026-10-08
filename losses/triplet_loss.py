import torch
import torch.nn as nn


class OriTripletLoss(nn.Module):
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # AMP can cast backbone outputs to fp16/bf16. Compute pairwise distances
        # in fp32 to avoid addmm dtype mismatches and improve numerical stability.
        inputs = inputs.float()
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1.0, alpha=-2.0)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap = []
        dist_an = []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][~mask[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
