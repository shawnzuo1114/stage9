def linear_warmup(epoch, start, warmup):
    if epoch is None:
        return 1.0
    if epoch < start:
        return 0.0
    if warmup <= 0:
        return 1.0
    return min(1.0, float(epoch - start) / float(warmup))
