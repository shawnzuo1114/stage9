from utils.progress import tqdm_write


def summarize_load_state(pretrained_path, incompatible, model_state_dict, loaded_state_dict):
    model_keys = set(model_state_dict.keys())
    checkpoint_keys = set(loaded_state_dict.keys())
    matched_keys = model_keys & checkpoint_keys
    return {
        "pretrained_path": pretrained_path,
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "matched_key_count": len(matched_keys),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "load_ratio": float(len(matched_keys) / max(len(model_keys), 1)),
    }


def _format_key_sample(keys, limit=20):
    keys = list(keys or [])
    if not keys:
        return "[]"
    if len(keys) <= limit:
        return str(keys)
    return "{} ... and {} more".format(keys[:limit], len(keys) - limit)


def print_pretrained_audit(report, prefix="VMamba"):
    tqdm_write(
        "[{}][Pretrained] path={} load_ratio={:.4f} matched={}/{} missing={} unexpected={}".format(
            prefix,
            report.get("pretrained_path"),
            float(report.get("load_ratio", 0.0)),
            int(report.get("matched_key_count", 0)),
            int(report.get("model_key_count", 0)),
            len(report.get("missing_keys", [])),
            len(report.get("unexpected_keys", [])),
        )
    )
    if report.get("missing_keys"):
        tqdm_write("[{}][Pretrained] missing sample: {}".format(prefix, _format_key_sample(report["missing_keys"], limit=20)))
    if report.get("unexpected_keys"):
        tqdm_write("[{}][Pretrained] unexpected sample: {}".format(prefix, _format_key_sample(report["unexpected_keys"], limit=20)))


def audit_new_module_parameters(named_parameters, module_prefixes):
    report = {}
    for prefix in module_prefixes:
        report[prefix] = [name for name, _param in named_parameters if name.startswith(prefix)]
    return report
