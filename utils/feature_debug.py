import math

import torch
import torch.nn.functional as F


def tensor_stats(x):
    if x is None:
        return {"mean": None, "std": None, "min": None, "max": None, "norm": None}
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "norm": float(x.norm(dim=-1).mean().item()) if x.ndim >= 2 else float(x.norm().item()),
    }


def entropy_from_prob(prob, dim=-1, eps=1e-12):
    prob = prob.clamp_min(eps)
    entropy = -(prob * prob.log()).sum(dim=dim)
    normalizer = math.log(prob.size(dim)) if prob.size(dim) > 1 else 1.0
    return entropy / max(normalizer, eps)


def cosine_mean(x, y, dim=-1):
    return float(F.cosine_similarity(x, y, dim=dim).mean().item())


def safe_l2_normalize(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def token_norm_mean(x, dim=-1):
    return float(x.norm(dim=dim).mean().item())


def pairwise_cosine_stats(tokens):
    if tokens is None or tokens.numel() == 0:
        return {"mean": None, "abs_mean": None, "max": None, "min": None}
    if tokens.size(-2) <= 1:
        return {"mean": 0.0, "abs_mean": 0.0, "max": 0.0, "min": 0.0}
    normalized = safe_l2_normalize(tokens, dim=-1)
    sim = torch.matmul(normalized, normalized.transpose(-1, -2))
    diag = torch.eye(sim.size(-1), device=sim.device, dtype=torch.bool)
    while diag.ndim < sim.ndim:
        diag = diag.unsqueeze(0)
    off_diag = sim.masked_select(~diag)
    return {
        "mean": float(off_diag.mean().item()),
        "abs_mean": float(off_diag.abs().mean().item()),
        "max": float(off_diag.max().item()),
        "min": float(off_diag.min().item()),
    }


def part_diversity_score(tokens):
    stats = pairwise_cosine_stats(tokens)
    if stats["abs_mean"] is None:
        return 0.0
    return float(1.0 - stats["abs_mean"])


def topk_indices_from_scores(scores, k=2):
    if scores is None or scores.numel() == 0:
        return []
    k = min(k, scores.size(-1))
    return torch.topk(scores, k=k, dim=-1).indices.detach().cpu().tolist()
