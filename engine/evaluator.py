"""VCM evaluation for Stage9 representations."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from utils.progress import get_tqdm, is_main_process, tqdm_write

def evaluate_feature_set(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape; max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1); matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)
    cmcs, aps = [], []
    for i in range(num_q):
        keep = ~((g_pids[indices[i]] == q_pids[i]) & (g_camids[indices[i]] == q_camids[i]))
        raw = matches[i][keep]
        if not np.any(raw): continue
        cmc = raw.cumsum(); cmc[cmc > 1] = 1; cmcs.append(cmc[:max_rank])
        precision = raw.cumsum() / (np.arange(raw.size) + 1.0); aps.append(float((precision * raw).sum() / raw.sum()))
    if not cmcs: raise RuntimeError("No valid query identity is present in the gallery")
    padded = np.zeros((len(cmcs), max_rank), dtype=np.float32)
    for i, cmc in enumerate(cmcs): padded[i, :len(cmc)] = cmc; padded[i, len(cmc):] = cmc[-1]
    return padded.mean(0), float(np.mean(aps))

def extract_loader_stage9_feature_sets(model, loader, modality_mode, seq_len, device, desc="extract", quiet=False, use_tqdm=True):
    model.eval(); batches = {k: [] for k in ("z_g", "v_freq", "z_final", "x_pool", "bn_feat")}; pids, camids = [], []
    progress = get_tqdm()(loader, desc=desc, dynamic_ncols=True, leave=False,
                          disable=quiet or not use_tqdm or not is_main_process())
    with torch.no_grad():
        for imgs, ids, cams in progress:
            out = model(imgs.to(device), modal=modality_mode, seq_len=seq_len)
            for name in batches:
                if name not in out: raise KeyError("Stage9 output is missing '{}'".format(name))
                batches[name].append(F.normalize(out[name].float(), dim=-1).cpu().numpy())
            pids.extend(ids); camids.extend(cams)
    progress.close()
    return {k: np.concatenate(v) for k, v in batches.items()}, np.asarray(pids), np.asarray(camids)

def _line(label, name, cmc, mean_ap):
    at = lambda n: cmc[min(n - 1, len(cmc) - 1)]
    return "[Eval][{}][{}] R1={:.2%} R5={:.2%} R10={:.2%} R20={:.2%} mAP={:.2%}".format(label, name, at(1), at(5), at(10), at(20), mean_ap)

def run_vcm_evaluation(model, galleryloader, queryloader, gallery_mode, query_mode, seq_len, device, label="default", args=None, epoch=None):
    start = time.time(); quiet = bool(getattr(args, "quiet", False))
    use_tqdm = bool(getattr(args, "use_tqdm", True))
    gallery, gp, gc = extract_loader_stage9_feature_sets(
        model, galleryloader, gallery_mode, seq_len, device,
        desc="{} [gallery]".format(label), quiet=quiet, use_tqdm=use_tqdm)
    query, qp, qc = extract_loader_stage9_feature_sets(
        model, queryloader, query_mode, seq_len, device,
        desc="{} [query]".format(label), quiet=quiet, use_tqdm=use_tqdm)
    results = {}
    for name in gallery:
        cmc, mean_ap = evaluate_feature_set(-query[name].dot(gallery[name].T), qp, gp, qc, gc)
        results[name] = (cmc, mean_ap); tqdm_write(_line(label, name, cmc, mean_ap), quiet=quiet)
    final, mean_ap = results["z_final"]
    tqdm_write("[Eval][{}] completed in {:.1f}s".format(label, time.time() - start), quiet=quiet)
    return final, mean_ap
