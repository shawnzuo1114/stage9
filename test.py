from __future__ import print_function

import argparse
import os

import torch

from data import VCM
from engine import run_vcm_evaluation
from models import build_model
from train import apply_vram_mode_defaults, build_parser, build_test_loaders


def main():
    parser = build_parser()
    parser.add_argument("--checkpoint", required=True, type=str)
    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("Checkpoint not found: {}".format(args.checkpoint))

    apply_vram_mode_defaults(args)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = VCM()
    queryloader, galleryloader, queryloader_1, galleryloader_1 = build_test_loaders(dataset, args)
    model = build_model(args, num_classes=dataset.num_train_pids).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    epoch = checkpoint.get("epoch", 0)

    run_vcm_evaluation(
        model,
        galleryloader,
        queryloader,
        gallery_mode=1,
        query_mode=2,
        seq_len=args.seq_len,
        device=device,
        label="IR_to_RGB",
        args=args,
        epoch=epoch,
    )
    run_vcm_evaluation(
        model,
        galleryloader_1,
        queryloader_1,
        gallery_mode=2,
        query_mode=1,
        seq_len=args.seq_len,
        device=device,
        label="RGB_to_IR",
        args=args,
        epoch=epoch,
    )


if __name__ == "__main__":
    main()
