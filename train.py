"""Training entry point for the Stage9-HTFR video cross-modal ReID model."""
import argparse, datetime, os, random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data import GenIdx, IdentitySampler, VCM, VideoDatasetTest, VideoDatasetTrain, build_test_transform, build_train_transform
from engine import build_optimizers, print_mamba_audit_if_needed, run_vcm_evaluation, train_one_epoch
from losses import OriTripletLoss, build_id_loss
from models import build_model
from utils.logging_utils import Logger, mkdir_if_missing

def str2bool(value):
    if isinstance(value, bool): return value
    if value.lower() in {"true", "1", "yes", "y"}: return True
    if value.lower() in {"false", "0", "no", "n"}: return False
    raise argparse.ArgumentTypeError("invalid boolean: {}".format(value))

def build_parser():
    p = argparse.ArgumentParser(description="Stage9-HTFR training")
    def arg(name, default, kind=None, **kw): p.add_argument(name, default=default, type=kind or type(default), **kw)
    arg("--dataset", "VCM"); arg("--stage", "stage9_htfr"); arg("--arch", "stage9_htfr")
    arg("--batch-size", 8); arg("--seq-len", 12); arg("--num-pos", 2); arg("--lr", .005, float)
    arg("--max-epoch", 120); arg("--eval-step", 10); arg("--save-epoch", 20); arg("--workers", 4)
    arg("--img-h", 288); arg("--img-w", 144); arg("--test-batch", 32); arg("--margin", .3, float)
    arg("--gpu", "0"); arg("--seed", 3407); arg("--run-name", "stage9_htfr"); arg("--save-root", "save_model/stage9_htfr")
    arg("--resume", ""); arg("--amp", True, str2bool, dest="use_amp"); arg("--grad-checkpoint", True, str2bool)
    arg("--accum-steps", 1); arg("--quiet", False, str2bool); arg("--use-tqdm", True, str2bool); arg("--debug-one-batch", False, str2bool)
    arg("--print-mamba-audit", True, str2bool)
    arg("--vmamba-variant", "vmamba_tiny"); arg("--vmamba-pretrained", "../vm.pth")
    arg("--require-vmamba-pretrained", False, str2bool); arg("--local-vmamba-root", "../VMamba-main")
    arg("--local-mamba-root", "../mamba-1p1p1"); arg("--local-causal-conv1d-root", "../causal-conv1d")
    arg("--disable-sie", False, str2bool); arg("--sie-coe", 1.0, float); arg("--token-dim", 512)
    arg("--mamba-d-state", 16); arg("--mamba-d-conv", 4); arg("--mamba-expand", 2)
    arg("--sth-num-hubs", 4); arg("--sth-start-epoch", 5); arg("--sth-warmup-epoch", 10)
    arg("--sth-hub-fusion-scale", 1.0, float)
    arg("--irm-srm-alpha-ir", .20, float); arg("--irm-srm-alpha-rgb", .08, float)
    arg("--irm-srm-start-epoch", 5); arg("--irm-srm-warmup-epoch", 10); arg("--irm-lambda-struct", .005, float)
    arg("--htfr-mode", "full", choices=["baseline", "spatial", "full"]); arg("--htfr-gate-max", .20, float)
    arg("--htfr-temporal-temperature", 1.0, float); arg("--htfr-fusion-start-epoch", 15); arg("--htfr-fusion-warmup-epoch", 10)
    arg("--htfr-freq-id-lambda", .05, float); arg("--htfr-freq-id-start-epoch", 5); arg("--htfr-freq-id-warmup-epoch", 5)
    arg("--htfr-freq-xm-lambda", .05, float); arg("--htfr-freq-xm-start-epoch", 10); arg("--htfr-freq-xm-warmup-epoch", 10)
    arg("--lambda-batch-xm", .10, float); arg("--batch-xm-start-epoch", 5); arg("--batch-xm-warmup-epoch", 15); arg("--batch-xm-tau", .07, float)
    arg("--lambda-xm-proto", .10, float); arg("--xm-proto-start-epoch", 20); arg("--xm-proto-warmup-epoch", 20)
    arg("--xm-proto-tau", .07, float); arg("--xm-proto-momentum", .2, float)
    # Accepted fixed Stage9 markers keep the shell interface explicit.
    p.add_argument("--enable-stage9-htfr", type=str2bool, default=True)
    p.add_argument("--irm-enable-ir-srm", type=str2bool, default=True); p.add_argument("--enable-batch-xm-retrieval", type=str2bool, default=True)
    p.add_argument("--enable-xm-proto", type=str2bool, default=True); p.add_argument("--rerank", default="none")
    p.add_argument("--num-segments", default=4, type=int); p.add_argument("--segment-len", default=3, type=int)
    return p

def apply_vram_mode_defaults(args): return args

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_test_loaders(dataset, args):
    transform = build_test_transform(args.img_h, args.img_w)
    def loader(items): return DataLoader(VideoDatasetTest(items, args.seq_len, "video_test", transform), batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    return loader(dataset.query), loader(dataset.gallery), loader(dataset.query_1), loader(dataset.gallery_1)

def load_checkpoint(model, path):
    if not path: return 0
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    return int(state.get("epoch", 0))

def main():
    args = build_parser().parse_args()
    if args.stage != "stage9_htfr": raise ValueError("This repository supports only stage9_htfr")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu; set_seed(args.seed); cudnn.benchmark = True
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_root, args.run_name + "_" + stamp)
    mkdir_if_missing(os.path.join(args.save_dir, "log")); import sys; sys.stdout = Logger(os.path.join(args.save_dir, "log", "train.txt"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"; print_mamba_audit_if_needed(args)
    dataset = VCM(); rgb_pos, ir_pos = GenIdx(dataset.rgb_label, dataset.ir_label)
    query, gallery, query1, gallery1 = build_test_loaders(dataset, args)
    model = build_model(args, dataset.num_train_pids).to(device); start_epoch = load_checkpoint(model, args.resume)
    optimizer = build_optimizers(args, model); scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())
    criterion_id = build_id_loss().to(device); criterion_tri = OriTripletLoss(args.batch_size * args.num_pos, args.margin).to(device)
    for epoch in range(start_epoch, args.max_epoch):
        model.set_epoch(epoch); sampler = IdentitySampler(dataset.ir_label, dataset.rgb_label, rgb_pos, ir_pos, args.num_pos, args.batch_size)
        trainset = VideoDatasetTrain(dataset.train_ir, dataset.train_rgb, args.seq_len, "video_train", build_train_transform(args.img_h, args.img_w), sampler.index1, sampler.index2)
        trainloader = DataLoader(trainset, args.batch_size * args.num_pos, sampler=sampler, num_workers=args.workers, drop_last=True)
        train_one_epoch(model, trainloader, criterion_id, criterion_tri, optimizer, scaler, epoch, args, device)
        if args.debug_one_batch: break
        if (epoch + 1) % args.eval_step == 0 or epoch + 1 == args.max_epoch:
            run_vcm_evaluation(model, gallery, query, 1, 2, args.seq_len, device, "IR->RGB", args, epoch + 1)
            run_vcm_evaluation(model, gallery1, query1, 2, 1, args.seq_len, device, "RGB->IR", args, epoch + 1)
        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, os.path.join(args.save_dir, "latest.pth"))
        if (epoch + 1) % args.save_epoch == 0: torch.save({"model": model.state_dict(), "epoch": epoch + 1}, os.path.join(args.save_dir, "epoch_{:03d}.pth".format(epoch + 1)))

if __name__ == "__main__": main()
