"""Stage9-only optimization loop."""
import time
import torch
import torch.nn.functional as F
from utils.mamba_compat import inspect_mamba
from utils.meters import AverageMeter
from utils.progress import get_tqdm, is_main_process, tqdm_write
from utils.warmup import linear_warmup

def build_optimizers(args, model):
    return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)

def print_mamba_audit_if_needed(args):
    if args.print_mamba_audit:
        info = inspect_mamba(args.local_mamba_root, args.local_causal_conv1d_root)
        tqdm_write("Mamba audit: {}".format({k: v for k, v in info.items() if k != "mamba_cls"}), quiet=args.quiet)

def adjust_learning_rate(optimizer, epoch, args):
    factor = (epoch + 1) / 10.0 if epoch < 10 else 1.0 if epoch < 35 else 0.1 if epoch < 80 else 0.01
    lr = args.lr * factor
    for group in optimizer.param_groups: group["lr"] = lr
    return lr

def train_one_epoch(model, trainloader, criterion_id, criterion_tri, optimizer_model, scaler, epoch, args, device):
    model.train()
    lr = adjust_learning_rate(optimizer_model, epoch, args)
    names = ("total", "id", "tri", "batch_xm", "xm_proto", "freq_id", "freq_xm", "struct")
    meters = {name: AverageMeter() for name in names}
    optimizer_model.zero_grad(set_to_none=True)
    start = time.time()
    progress = get_tqdm()(
        trainloader,
        desc="Epoch {:03d}/{:03d} [train]".format(epoch + 1, args.max_epoch),
        dynamic_ncols=True,
        leave=False,
        disable=args.quiet or not args.use_tqdm or not is_main_process(),
    )
    for step, (imgs_ir, pids_ir, _cam_ir, imgs_rgb, pids_rgb, _cam_rgb) in enumerate(progress):
        rgb, ir = imgs_rgb.to(device), imgs_ir.to(device)
        labels = torch.cat((pids_rgb, pids_ir)).to(device)
        modal = torch.cat((torch.zeros(rgb.size(0), dtype=torch.long, device=device), torch.ones(ir.size(0), dtype=torch.long, device=device)))
        with torch.cuda.amp.autocast(enabled=args.use_amp and torch.cuda.is_available()):
            out = model(rgb, ir, modal=0, seq_len=args.seq_len, labels=labels, epoch=epoch, debug=args.debug_one_batch)
            loss_id = criterion_id(out["logits"], labels)
            loss_tri, _ = criterion_tri(out["x_pool"], labels)
            loss_batch, _ = model.batch_xm_loss(out["video_feat"], labels, modal, modal.lt(2), epoch)
            loss_proto, _ = model.proto_loss(out["video_feat"], labels, modal, model.proto_memory, modal.lt(2), epoch)
            freq_w = args.htfr_freq_id_lambda * linear_warmup(epoch, args.htfr_freq_id_start_epoch, args.htfr_freq_id_warmup_epoch)
            loss_freq_id = freq_w * F.cross_entropy(out["freq_logits"], labels)
            loss_freq_xm, _ = model.freq_xm_loss(out["v_freq"], labels, modal, modal.lt(2), epoch)
            dbg = out.get("htfr_debug", {})
            raw_struct = .5 * (float(dbg.get("irm_srm_s3_res_ratio", 0.0)) + float(dbg.get("irm_srm_s4_res_ratio", 0.0)))
            loss_struct = out["video_feat"].new_tensor(args.irm_lambda_struct * raw_struct)
            losses = (loss_id, loss_tri, loss_batch, loss_proto, loss_freq_id, loss_freq_xm, loss_struct)
            total = sum(losses)
        if not torch.isfinite(total): raise RuntimeError("Non-finite loss at epoch {}, step {}".format(epoch + 1, step + 1))
        scaler.scale(total / max(args.accum_steps, 1)).backward()
        if (step + 1) % max(args.accum_steps, 1) == 0:
            scaler.step(optimizer_model); scaler.update(); optimizer_model.zero_grad(set_to_none=True)
            (model.module if hasattr(model, "module") else model).update_prototypes(out["video_feat"].detach(), labels, modal)
        for name, value in zip(names, (total,) + losses): meters[name].update(float(value.detach().item()), labels.size(0))
        progress.set_postfix(
            loss="{:.4f}".format(meters["total"].avg),
            id="{:.4f}".format(meters["id"].avg),
            tri="{:.4f}".format(meters["tri"].avg),
            lr="{:.2e}".format(lr),
            refresh=False,
        )
        if args.debug_one_batch: break
    progress.close()
    elapsed = time.time() - start
    tqdm_write("[Epoch {:03d}][Stage9] total={:.6f} id={:.6f} tri={:.6f} batch_xm={:.6f} xm_proto={:.6f} freq_id={:.6f} freq_xm={:.6f} struct={:.6f} lr={:.2e} time={:.1f}s".format(epoch + 1, *(meters[n].avg for n in names), lr, elapsed), quiet=args.quiet)
    return {"loss_" + name: meter.avg for name, meter in meters.items()}
