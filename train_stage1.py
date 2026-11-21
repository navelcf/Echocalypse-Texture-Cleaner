#!/usr/bin/env python3
"""Train (or fine-tune) the stage-1 model on clean art damaged with the game's encoder.

  python train_stage1.py --name myrun --data CLEAN_DIR[:weight] [--data ...] --init models/echoclean_stage1.pt

Upgrades over v1.x: cosine LR decay, EMA weights, fp16 AMP, v2.0 defaults (384x12), optional L2/PSNR + L2 seam loss,
active-pixel loss normalization, and sparse/thin-piece patch sampling.
Writes runs/<name>/ckpt_last.pt, ckpt_best.pt (EMA weights at best validation PSNR) and log.txt.
"""
import os, sys, copy, math, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader


def _lr_at(step, total, lr, lr_min, warmup, cosine_from):
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    c0 = int(total * cosine_from)
    if step < c0:
        return lr
    t = (step - c0) / max(1, total - c0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True); ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--init', default=None, help='checkpoint to continue from (omit to train from scratch)')
    ap.add_argument('--channels', type=int, default=384); ap.add_argument('--blocks', type=int, default=12)
    ap.add_argument('--steps', type=int, default=40000); ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=None, help='default: 2e-4 from scratch, 5e-5 when fine-tuning')
    ap.add_argument('--lr-min', type=float, default=None, help='final cosine LR (default: lr / 10)')
    ap.add_argument('--cosine-from', type=float, default=0.2, help='fraction of steps before cosine decay begins')
    ap.add_argument('--warmup', type=int, default=None, help='linear warmup steps (default: 2000 from scratch, 200 fine-tune)')
    ap.add_argument('--ema', type=float, default=0.999, help='EMA decay (0 disables EMA)')
    ap.add_argument('--amp', action='store_true', default=True, help='use fp16 mixed precision on GPU')
    ap.add_argument('--no-amp', dest='amp', action='store_false')
    ap.add_argument('--batch', type=int, default=12); ap.add_argument('--patch', type=int, default=128)
    ap.add_argument('--sparse-prob', type=float, default=0.10, help='probability of sampling sparse/thin-alpha patches')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--pool', type=int, default=6, help='clean images each worker keeps in memory per data set')
    ap.add_argument('--l1-weight', type=float, default=1.0)
    ap.add_argument('--l2-weight', type=float, default=0.0, help='MSE/L2 weight (e.g. 5.0-15.0 for PSNR fine-tuning)')
    ap.add_argument('--hue-weight', type=float, default=2.0, help='straight-colour loss on see-through texels')
    ap.add_argument('--seam-weight', type=float, default=1.0, help='loss on the texture composited over its own colour')
    ap.add_argument('--seam-l2-weight', type=float, default=0.0, help='L2 coherent-seam weight matching seam RMS metric')
    ap.add_argument('--ssim-weight', type=float, default=1.0, help='SSIM loss on that composite')
    ap.add_argument('--val-raw'); ap.add_argument('--val-clean')
    a = ap.parse_args()
    from echoclean import blocks as B, models, metrics as MT, data as D, pipeline as PL
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join('runs', a.name); os.makedirs(out, exist_ok=True); log = open(os.path.join(out, 'log.txt'), 'a')
    def say(*x):
        s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
    say(f'=== {time.strftime("%Y-%m-%d %H:%M")}  ' + ' '.join(sys.argv[1:]))
    groups = [D.list_images(s) for s in a.data]; say('data:', ', '.join(f'{len(f)} images (weight {w:g})' for f, w in groups))
    net = models.load_stage1(a.init, dev) if a.init else models.SideNet(a.channels, a.blocks).to(dev)
    if not a.init: torch.nn.init.zeros_(net.tail.weight); torch.nn.init.zeros_(net.tail.bias)
    net.train()
    ema = copy.deepcopy(net).eval() if a.ema > 0 else None
    if ema is not None:
        for p in ema.parameters(): p.requires_grad_(False)
    lr = a.lr or (5e-5 if a.init else 2e-4); lr_min = a.lr_min or lr / 10.0
    warmup = a.warmup if a.warmup is not None else (200 if a.init else 2000)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    use_amp = bool(a.amp and dev.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    say(f'model {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters on {dev}, ' +
        (f'continuing from {a.init}' if a.init else 'from scratch') +
        f', lr {lr:g} -> {lr_min:g} (cosine from {a.cosine_from:.0%}, ema {a.ema}, amp {use_amp}, patch {a.patch})')
    stream = D.forever(DataLoader(D.PatchStream(groups, a.patch // 4 * 4, a.batch, a.pool, sparse_prob=a.sparse_prob),
                                  batch_size=None, num_workers=a.workers, prefetch_factor=6))
    val = None
    if a.val_raw and a.val_clean:
        vb, vw, vh = B.load_raw(a.val_raw); val = (vb, vw, vh, B.load_clean(a.val_clean))
    lumw = torch.tensor([0.3, 0.59, 0.11], device=dev).view(1, 3, 1, 1); best = -1.0

    def evaluate(step):
        nonlocal best
        ev_net = ema if (ema is not None and step > 0) else net
        models.save(ev_net, os.path.join(out, 'ckpt_last.pt'), step=step, kind='stage1')
        if val is None: return
        ev_net.eval()
        r = MT.score(PL.finish(PL.stage1(ev_net, val[0], dev, tta=False, half=False))[:val[2], :val[1]], val[3], dev)
        if ev_net is net: net.train()
        say(f'--- step {step}: validation pair ({"ema" if ev_net is ema else "raw"})  PSNR {r["psnr"]:.3f} dB / edges {r["edges"]:.3f} dB / seam {r["seam"]:.4f} / ssim {r["ssim"]:.4f}')
        if r['psnr'] > best:
            best = r['psnr']; models.save(ev_net, os.path.join(out, 'ckpt_best.pt'), step=step, kind='stage1')

    evaluate(0); t0 = time.time()
    for s in range(a.steps):
        cur_lr = _lr_at(s, a.steps, lr, lr_min, warmup, a.cosine_from)
        for g in opt.param_groups: g['lr'] = cur_lr
        x, y, _ = next(stream)
        x = x.permute(0, 3, 1, 2).to(dev, non_blocking=True)
        y = (y.permute(0, 3, 1, 2).float() / 255).to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            o = net(x)
        o = o.float()
        ya = y[:, 3:4]
        live = (ya > 0).float()
        w = (1 + 3 * ((ya > 0) & (ya < 1)).float()) * (0.1 + 0.9 * live)
        wnorm = w.sum().clamp(min=1.0)
        diff = o - y
        loss = torch.zeros((), device=dev)
        if a.l1_weight > 0:
            loss = loss + a.l1_weight * (w * diff.abs()).sum() / (wnorm * 4.0)
        if a.l2_weight > 0:
            loss = loss + a.l2_weight * ((w * diff * diff).sum() / (wnorm * 4.0) + 1e-10).sqrt()
        if a.hue_weight > 0:
            hm = (ya > 2 / 255).float()
            loss = loss + a.hue_weight * (hm * (o[:, :3] - y[:, :3]).abs() / ya.clamp(min=0.03)).sum() / (hm.sum().clamp(min=1.0) * 3.0)
        if a.seam_weight > 0 or a.seam_l2_weight > 0 or a.ssim_weight > 0:
            with torch.no_grad(): S, valid = MT.own_colour(y, sigmas=(1.5, 3, 6, 12))
            vwnorm = (valid * w).sum().clamp(min=1.0)
            if a.seam_weight > 0 or a.seam_l2_weight > 0:
                D_se = ((o[:, :3] - y[:, :3]) - S * (o[:, 3:4] - ya))
                e = D_se * valid * w
                if a.seam_weight > 0:
                    loss = loss + a.seam_weight * (e.abs().sum() / (vwnorm * 3.0) + 2.0 * MT.gauss((e * lumw).sum(1, keepdim=True), 1.5).abs().sum() / vwnorm)
                if a.seam_l2_weight > 0:
                    band = (torch.nn.functional.max_pool2d(((ya > 0) & (ya < 1)).float() * valid, 3, 1, 1) > 0).float() * valid
                    bn = band.sum().clamp(min=1.0)
                    lum = (D_se * lumw).sum(1, keepdim=True)
                    coh = MT.gauss(lum * band, 1.5) / MT.gauss(band, 1.5).clamp(min=1e-3)
                    loss = loss + a.seam_l2_weight * (((coh * coh) * band).sum() / bn + 1e-10).sqrt()
            if a.ssim_weight > 0:
                sm = MT.ssim_map(o[:, :3] + (1 - o[:, 3:4]) * S, y[:, :3] + (1 - ya) * S).mean(1, keepdim=True)
                loss = loss + a.ssim_weight * ((1 - sm) * valid * w).sum() / vwnorm
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            d = min(a.ema, (1 + s) / (10 + s))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(p, 1.0 - d)
        if (s + 1) % 500 == 0:
            el = time.time() - t0
            print(f'step {s + 1}/{a.steps}  lr {cur_lr:.2e}  loss {loss.item():.5f}  {(s + 1) / el:.2f} steps/s  ~{el / (s + 1) * (a.steps - s - 1) / 60:.1f} min left', flush=True)
        if (s + 1) % a.eval_every == 0 or s + 1 == a.steps:
            evaluate(s + 1)
    say(f'done in {(time.time() - t0) / 60:.1f} min' + (f'; best validation PSNR {best:.3f} dB' if val else ''))
    os._exit(0)


if __name__ == '__main__':
    main()
