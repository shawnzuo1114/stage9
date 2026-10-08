from __future__ import absolute_import, print_function

import math
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}', retrying.".format(img_path))
    return img


def build_sample_clip(num_frames, seq_len):
    clip = []
    frame_indices = list(range(num_frames))
    if num_frames < seq_len:
        strip = list(range(num_frames)) + [frame_indices[-1]] * (seq_len - num_frames)
        for step in range(seq_len):
            clip.append(list(strip[step : step + 1]))
    else:
        interval = math.ceil(num_frames / seq_len)
        strip = list(range(num_frames)) + [frame_indices[-1]] * (interval * seq_len - num_frames)
        for step in range(seq_len):
            clip.append(list(strip[interval * step : interval * (step + 1)]))
    return np.array(clip)


def load_clip(img_paths, frame_indices, transform):
    imgs = []
    for index in frame_indices:
        img = np.array(read_image(img_paths[int(index)]))
        if transform is not None:
            img = transform(img)
        imgs.append(img)
    return torch.cat(imgs, dim=0)


class VideoDatasetTrain(Dataset):
    def __init__(self, dataset_ir, dataset_rgb, seq_len=6, sample="video_train", transform=None, index1=None, index2=None):
        self.dataset_ir = dataset_ir
        self.dataset_rgb = dataset_rgb
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.index1 = [] if index1 is None else index1
        self.index2 = [] if index2 is None else index2

    def __len__(self):
        return len(self.dataset_rgb)

    def __getitem__(self, index):
        img_ir_paths, pid_ir, camid_ir = self.dataset_ir[self.index2[index]]
        img_rgb_paths, pid_rgb, camid_rgb = self.dataset_rgb[self.index1[index]]

        if self.sample == "random":
            ir_indices = self._random_indices(len(img_ir_paths))
            rgb_indices = self._random_indices(len(img_rgb_paths))
        elif self.sample == "video_train":
            ir_clip = build_sample_clip(len(img_ir_paths), self.seq_len)
            rgb_clip = build_sample_clip(len(img_rgb_paths), self.seq_len)
            ir_pick = np.random.choice(ir_clip.shape[1], ir_clip.shape[0])
            rgb_pick = np.random.choice(rgb_clip.shape[1], rgb_clip.shape[0])
            ir_indices = ir_clip[np.arange(len(ir_clip)), ir_pick]
            rgb_indices = rgb_clip[np.arange(len(rgb_clip)), rgb_pick]
        else:
            raise KeyError("Unknown train sample method: {}".format(self.sample))

        imgs_ir = load_clip(img_ir_paths, ir_indices, self.transform)
        imgs_rgb = load_clip(img_rgb_paths, rgb_indices, self.transform)
        return imgs_ir, pid_ir, camid_ir, imgs_rgb, pid_rgb, camid_rgb

    def _random_indices(self, num_frames):
        frame_indices = range(num_frames)
        rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.seq_len, len(frame_indices))
        indices = list(frame_indices[begin_index:end_index])
        for idx in indices:
            if len(indices) >= self.seq_len:
                break
            indices.append(idx)
        return np.array(indices)


class VideoDatasetTest(Dataset):
    def __init__(self, dataset, seq_len=6, sample="video_test", transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num_frames = len(img_paths)

        if self.sample == "dense":
            return self._load_dense_clip(img_paths), pid, camid
        if self.sample == "random":
            indices = VideoDatasetTrain._random_indices(self, num_frames)
            return load_clip(img_paths, indices, self.transform), pid, camid
        if self.sample == "video_test":
            clip = build_sample_clip(num_frames, self.seq_len)
            return load_clip(img_paths, clip[:, 0], self.transform), pid, camid
        raise KeyError("Unknown test sample method: {}".format(self.sample))

    def _load_dense_clip(self, img_paths):
        num_frames = len(img_paths)
        current = 0
        frame_indices = range(num_frames)
        indices_list = []
        while num_frames - current > self.seq_len:
            indices_list.append(frame_indices[current : current + self.seq_len])
            current += self.seq_len
        last_seq = list(frame_indices[current:])
        for idx in last_seq:
            if len(last_seq) >= self.seq_len:
                break
            last_seq.append(idx)
        indices_list.append(last_seq)
        imgs_list = []
        for indices in indices_list:
            imgs = []
            for frame_idx in indices:
                img = np.array(read_image(img_paths[int(frame_idx)]))
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
            imgs_list.append(torch.cat(imgs, dim=0))
        return torch.stack(imgs_list)
