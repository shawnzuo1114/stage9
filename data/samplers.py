import numpy as np
from torch.utils.data.sampler import Sampler


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    for label in np.unique(train_color_label):
        color_pos.append([k for k, value in enumerate(train_color_label) if value == label])
    thermal_pos = []
    for label in np.unique(train_thermal_label):
        thermal_pos.append([k for k, value in enumerate(train_thermal_label) if value == label])
    return color_pos, thermal_pos


class IdentitySampler(Sampler):
    """Preserve the original identity-balanced RGB/IR batch organization."""

    def __init__(self, train_thermal_label, train_color_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        n = np.minimum(len(train_color_label), len(train_thermal_label))
        index1 = None
        index2 = None
        for batch_idx in range(int(n / (batchSize * num_pos))):
            picked = list(np.random.choice(uni_label, batchSize, replace=False))
            for identity in picked:
                sample_color = np.random.choice(color_pos[identity], num_pos)
                sample_thermal = np.random.choice(thermal_pos[identity], num_pos)
                if index1 is None:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        self.N = int(n / (batchSize * num_pos)) * (batchSize * num_pos)
        self.index1 = index1
        self.index2 = index2

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
