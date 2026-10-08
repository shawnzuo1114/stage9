from __future__ import absolute_import, print_function

import os.path as osp

import numpy as np


def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    modality_str = "ir" if modality == "1" else "rgb"
    t_pos = fname.find("T")
    d_pos = fname.find("D")
    f_pos = fname.find("F")
    camera = fname[d_pos:t_pos]
    picture = fname[f_pos + 1 :]
    return base + "/" + modality_str + "/" + camera + "/" + picture


class VCM(object):
    """Preserve the original MITML VCM parsing logic."""

    root = "../VCM_HIT/"
    train_name_path = osp.join(root, "info/train_name.txt")
    track_train_info_path = osp.join(root, "info/track_train_info.txt")
    test_name_path = osp.join(root, "info/test_name.txt")
    track_test_info_path = osp.join(root, "info/track_test_info.txt")
    query_IDX_path = osp.join(root, "info/query_IDX.txt")

    def __init__(self, min_seq_len=12):
        self._check_before_run()

        train_names = self._get_names(self.train_name_path)
        track_train = self._get_tracks(self.track_train_info_path)

        test_names = self._get_names(self.test_name_path)
        track_test = self._get_tracks(self.track_test_info_path)
        query_idx = self._get_query_idx(self.query_IDX_path) - 1

        track_query = track_test[query_idx, :]
        gallery_idx = [i for i in range(track_test.shape[0]) if i not in query_idx]
        track_gallery = track_test[gallery_idx, :]

        gallery_idx_1 = self._get_query_idx(self.query_IDX_path) - 1
        track_gallery_1 = track_test[gallery_idx_1, :]
        query_idx_1 = [j for j in range(track_test.shape[0]) if j not in gallery_idx_1]
        track_query_1 = track_test[query_idx_1, :]

        (
            train_ir,
            num_train_tracklets_ir,
            _num_train_imgs_ir,
            train_rgb,
            num_train_tracklets_rgb,
            _num_train_imgs_rgb,
            num_train_pids,
            ir_label,
            rgb_label,
        ) = self._process_data_train(train_names, track_train, relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, _num_query_imgs = self._process_data_test(
            test_names, track_query, relabel=False, min_seq_len=min_seq_len
        )
        gallery, num_gallery_tracklets, num_gallery_pids, _num_gallery_imgs = self._process_data_test(
            test_names, track_gallery, relabel=False, min_seq_len=min_seq_len
        )
        query_1, num_query_tracklets_1, num_query_pids_1, _num_query_imgs_1 = self._process_data_test(
            test_names, track_query_1, relabel=False, min_seq_len=min_seq_len
        )
        gallery_1, num_gallery_tracklets_1, num_gallery_pids_1, _num_gallery_imgs_1 = self._process_data_test(
            test_names, track_gallery_1, relabel=False, min_seq_len=min_seq_len
        )

        print("=> VCM loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_ir    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets_ir))
        print("train_rgb   | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets_rgb))
        print("query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("---------------------------------")

        self.train_ir = train_ir
        self.train_rgb = train_rgb
        self.ir_label = ir_label
        self.rgb_label = rgb_label
        self.query = query
        self.gallery = gallery
        self.query_1 = query_1
        self.gallery_1 = gallery_1
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_query_tracklets = num_query_tracklets
        self.num_gallery_tracklets = num_gallery_tracklets
        self.num_query_pids_1 = num_query_pids_1
        self.num_gallery_pids_1 = num_gallery_pids_1
        self.num_query_tracklets_1 = num_query_tracklets_1
        self.num_gallery_tracklets_1 = num_gallery_tracklets_1

    def _check_before_run(self):
        for path in (
            self.root,
            self.train_name_path,
            self.track_train_info_path,
            self.query_IDX_path,
            self.test_name_path,
            self.track_test_info_path,
        ):
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    @staticmethod
    def _get_names(fpath):
        with open(fpath, "r") as handle:
            return [line.rstrip() for line in handle]

    @staticmethod
    def _get_tracks(fpath):
        entries = []
        with open(fpath, "r") as handle:
            for line in handle:
                entries.append(list(map(int, line.rstrip().split(" "))))
        return np.array(entries)

    @staticmethod
    def _get_query_idx(fpath):
        with open(fpath, "r") as handle:
            for line in handle:
                idxs = list(map(int, line.rstrip().split(" ")))
        return np.array(idxs)

    def _process_data_train(self, names, meta_data, relabel=False, min_seq_len=0):
        pid_list = list(set(meta_data[:, 3].tolist()))
        num_pids = len(pid_list)
        pid2label = {pid: label for label, pid in enumerate(pid_list)} if relabel else {}

        tracklets_ir = []
        tracklets_rgb = []
        num_imgs_per_tracklet_ir = []
        num_imgs_per_tracklet_rgb = []
        ir_label = []
        rgb_label = []

        for data in meta_data:
            modality, start_index, end_index, pid, camid = data
            if relabel:
                pid = pid2label[pid]
            img_names = names[start_index - 1 : end_index]
            img_paths = tuple(osp.join(self.root, decoder_pic_path(img_name)) for img_name in img_names)
            if len(img_paths) < min_seq_len:
                continue
            if modality == 1:
                ir_label.append(pid)
                tracklets_ir.append((img_paths, pid, camid))
                num_imgs_per_tracklet_ir.append(len(img_paths))
            else:
                rgb_label.append(pid)
                tracklets_rgb.append((img_paths, pid, camid))
                num_imgs_per_tracklet_rgb.append(len(img_paths))

        return (
            tracklets_ir,
            len(tracklets_ir),
            num_imgs_per_tracklet_ir,
            tracklets_rgb,
            len(tracklets_rgb),
            num_imgs_per_tracklet_rgb,
            num_pids,
            ir_label,
            rgb_label,
        )

    def _process_data_test(self, names, meta_data, relabel=False, min_seq_len=0):
        pid_list = list(set(meta_data[:, 3].tolist()))
        num_pids = len(pid_list)
        pid2label = {pid: label for label, pid in enumerate(pid_list)} if relabel else {}

        tracklets = []
        num_imgs_per_tracklet = []
        for data in meta_data:
            _modality, start_index, end_index, pid, camid = data
            if relabel:
                pid = pid2label[pid]
            img_names = names[start_index - 1 : end_index]
            img_paths = tuple(osp.join(self.root, decoder_pic_path(img_name)) for img_name in img_names)
            if len(img_paths) < min_seq_len:
                continue
            tracklets.append((img_paths, pid, camid))
            num_imgs_per_tracklet.append(len(img_paths))
        return tracklets, len(tracklets), num_pids, num_imgs_per_tracklet
