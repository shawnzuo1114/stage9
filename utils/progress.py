try:
    import torch.distributed as dist
except Exception:
    dist = None

try:
    from tqdm.auto import tqdm
except Exception:
    class _TqdmFallback(object):
        @staticmethod
        def write(message):
            print(message)

    tqdm = _TqdmFallback()


def is_main_process():
    if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
        return True
    return dist.get_rank() == 0


def get_tqdm():
    return tqdm


def tqdm_write(message, quiet=False):
    if quiet or not is_main_process():
        return
    tqdm.write(str(message))


def format_float(value, digits=4, default="n/a"):
    if value is None:
        return default
    try:
        return ("{:.%df}" % int(digits)).format(float(value))
    except Exception:
        return default


def reduce_log_dict_if_needed(log_dict):
    return dict(log_dict)
