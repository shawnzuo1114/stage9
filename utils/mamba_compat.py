import importlib
import inspect
import os
import sys


def _normalize_path(path):
    return os.path.normcase(os.path.abspath(path)) if path else ""


def purge_module_prefix(prefixes):
    to_delete = []
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        module_path = getattr(module, "__file__", "")
        if module_path and "site-packages" in module_path.replace("\\", "/"):
            to_delete.append(name)
    for name in to_delete:
        del sys.modules[name]


def configure_local_imports(local_mamba_root="", local_causal_conv1d_root=""):
    roots = []
    if local_causal_conv1d_root:
        roots.append(local_causal_conv1d_root)
    if local_mamba_root:
        roots.append(local_mamba_root)
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for rel_path in ("causal-conv1d", "mamba-1p1p1"):
        roots.append(os.path.join(workspace_dir, rel_path))

    purge_module_prefix(("mamba_ssm", "causal_conv1d"))
    for root in reversed(roots):
        if root and os.path.isdir(root):
            normalized = _normalize_path(root)
            existing = {_normalize_path(item) for item in sys.path}
            if normalized not in existing:
                sys.path.insert(0, root)


def import_mamba(local_mamba_root="", local_causal_conv1d_root=""):
    configure_local_imports(local_mamba_root=local_mamba_root, local_causal_conv1d_root=local_causal_conv1d_root)
    mamba_module = importlib.import_module("mamba_ssm")
    causal_module = importlib.import_module("causal_conv1d")
    mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
    return mamba_module, causal_module, mamba_simple.Mamba


def inspect_mamba(local_mamba_root="", local_causal_conv1d_root=""):
    mamba_module, causal_module, mamba_cls = import_mamba(
        local_mamba_root=local_mamba_root,
        local_causal_conv1d_root=local_causal_conv1d_root,
    )
    signature = inspect.signature(mamba_cls.__init__)
    supports_bimamba = "bimamba" in signature.parameters
    supports_use_fast_path = "use_fast_path" in signature.parameters
    supports_layer_idx = "layer_idx" in signature.parameters
    return {
        "mamba_ssm_file": getattr(mamba_module, "__file__", ""),
        "causal_conv1d_file": getattr(causal_module, "__file__", ""),
        "mamba_signature": str(signature),
        "supports_bimamba": supports_bimamba,
        "supports_use_fast_path": supports_use_fast_path,
        "supports_layer_idx": supports_layer_idx,
        "mamba_cls": mamba_cls,
    }


def ensure_local_module_path(module_path, configured_root, module_name):
    if not configured_root:
        return
    normalized_module = _normalize_path(module_path)
    normalized_root = _normalize_path(configured_root)
    if normalized_root and normalized_root not in normalized_module:
        raise RuntimeError("{} resolved to non-local path: {}".format(module_name, module_path))


def build_mamba_block(
    d_model,
    d_state,
    d_conv,
    expand,
    bimamba=False,
    require_bimamba=False,
    use_fast_path=True,
    layer_idx=None,
    module_name="",
    allow_fallback=False,
    local_mamba_root="",
    local_causal_conv1d_root="",
):
    info = inspect_mamba(local_mamba_root=local_mamba_root, local_causal_conv1d_root=local_causal_conv1d_root)
    ensure_local_module_path(info["mamba_ssm_file"], local_mamba_root, "mamba_ssm")
    ensure_local_module_path(info["causal_conv1d_file"], local_causal_conv1d_root, "causal_conv1d")

    mamba_cls = info["mamba_cls"]
    kwargs = {
        "d_model": d_model,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
    }
    if info["supports_use_fast_path"]:
        kwargs["use_fast_path"] = use_fast_path
    elif not allow_fallback:
        raise RuntimeError("Local Mamba does not support use_fast_path for '{}'.".format(module_name))
    if info["supports_layer_idx"]:
        kwargs["layer_idx"] = layer_idx
    elif layer_idx is not None and not allow_fallback:
        raise RuntimeError("Local Mamba does not support layer_idx for '{}'.".format(module_name))
    actual_bimamba = False
    if bimamba:
        if info["supports_bimamba"]:
            kwargs["bimamba"] = True
            actual_bimamba = True
        elif require_bimamba or not allow_fallback:
            raise RuntimeError("Local Mamba does not support bimamba for '{}'.".format(module_name))
    elif info["supports_bimamba"]:
        kwargs["bimamba"] = False

    block = mamba_cls(**kwargs)
    block.actual_bimamba = actual_bimamba
    block.mamba_debug = {
        "mamba_ssm_file": info["mamba_ssm_file"],
        "causal_conv1d_file": info["causal_conv1d_file"],
        "mamba_signature": info["mamba_signature"],
        "supports_bimamba": info["supports_bimamba"],
        "actual_bimamba": actual_bimamba,
        "module_name": module_name,
    }

    if require_bimamba and not actual_bimamba:
        raise RuntimeError("require_bimamba=True but actual_bimamba=False for '{}'.".format(module_name))
    if bimamba and not actual_bimamba and not allow_fallback:
        raise RuntimeError("bimamba=True but actual_bimamba=False for '{}'.".format(module_name))
    return block
