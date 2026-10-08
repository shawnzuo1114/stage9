import torch.nn as nn


def build_id_loss():
    return nn.CrossEntropyLoss()
