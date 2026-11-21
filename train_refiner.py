#!/usr/bin/env python3
"""Train (or fine-tune) the stage-2 refiner on top of a finished stage-1 model.

  python train_refiner.py --name myrefiner --stage1 models/echoclean_stage1.pt --data CLEAN_DIR[:weight] [--init models/echoclean_refiner.pt]

Upgrades over v1.x:
- Native single-call pthreads batch encoding (no multiprocessing.Pool IPC, no redundant decode of bx)
- Cosine LR decay, EMA weights, fp16 AMP, SSIM loss, optional L2/MSE + L2 seam loss
- Optional unrolled Pass-2 training (--pass2-prob) so Pass 2 is in-distribution instead of drifting
- Optional 2x/4x TTA Stage-1 input (--s1-tta) so Refiner sees TTA-grade Stage-1 residuals
- Optional extended 17-channel input (--extended: adds split mm_rgb, mm_alpha, px, py; zero-init warm-start from 13ch)
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


def _encode_batch(u8_nhwc, prev_u8=None, prev_bl=None, nthreads=0):
    from echoclean import encoder
    B, P, W, _ = u8_nhwc.shape
    flat = np.ascontiguousarray(u8_nhwc.reshape(B * P, W, 4), np.uint8)
    pflat = np.ascontiguousarray(prev_u8.reshape(B * P, W, 4), np.uint8) if prev_u8 is not None else None
    pbl = np.ascontiguousarray(prev_bl.reshape(B * (P // 4), W // 4, 16), np.uint8) if prev_bl is not None else None
    out = encoder.encode_blocks(flat, nthreads=nthreads, prev_pm=pflat, prev_blocks=pbl)
    return out.reshape(B, P // 4, W // 4, 16)


def _decode_batch(bl_nbhbw16):
    from echoclean import blocks as B
    N, bh, bw, _ = bl_nbhbw16.shape
    dec = B.decode(bl_nbhbw16.reshape(N * bh, bw, 16))
    return dec.reshape(N, bh * 4, bw * 4, 4)


def _build_ref_input(x, o_cur, bo, bx, extended=False):
    """x, o_cur: (B, 4, P, P) float 0..1 on dev; bo, bx: (B, P/4, P/4, 16) uint8 numpy."""
    dev = x.device
    re_dec = torch.from_numpy(_decode_batch(bo)).to(dev).permute(0, 3, 1, 2).float().div_(255)
    fb = re_dec - x
    mm_any = torch.from_numpy((bo != bx).any(-1)).to(dev, dtype=torch.float32).repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1)
    if not extended:
        return torch.cat([x, o_cur, fb, mm_any], 1)
    mm_rgb = torch.from_numpy((bo[..., 8:] != bx[..., 8:]).any(-1)).to(dev, dtype=torch.float32).repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1)
    mm_a = torch.from_numpy((bo[..., :8] != bx[..., :8]).any(-1)).to(dev, dtype=torch.float32).repeat_interleave(4, 1).repeat_interleave(4, 2).unsqueeze(1)
    Bsz, _, H, W = x.shape
    px = (torch.arange(W, device=dev, dtype=torch.float32) % 4).div_(3).view(1, 1, 1, W).expand(Bsz, 1, H, W)
    py = (torch.arange(H, device=dev, dtype=torch.float32) % 4).div_(3).view(1, 1, H, 1).expand(Bsz, 1, H, W)
    return torch.cat([x, o_cur, fb, mm_any, mm_rgb, mm_a, px, py], 1)


@torch.no_grad()
def _stage1_pred(net1, x13, tta_k=1):
    """Run stage 1 on batch x13 (B, 13, P, P); if tta_k > 1, average tta_k random D4 orientations."""
    if tta_k <= 1:
        return net1(x13)
    dev = x13.device; Bsz, _, H, W = x13.shape
    # base without px, py (channels 4, 5)
    base = torch.cat([x13[:, :4], x13[:, 6:]], 1)
    px = (torch.arange(W, device=dev, dtype=torch.float32) % 4).div_(3).view(1, 1, 1, W).expand(Bsz, 1, H, W)
    py = (torch.arange(H, device=dev, dtype=torch.float32) % 4).div_(3).view(1, 1, H, 1).expand(Bsz, 1, H, W)
    acc = torch.zeros((Bsz, 4, H, W), device=dev, dtype=torch.float32)
    orients = [(k, f) for k in range(4) for f in (False, True)]
    if tta_k < 8:
        idx = np.random.choice(8, size=tta_k, replace=False)
        orients = [orients[i] for i in idx]
    for k, f in orients:
        t = torch.rot90(base, k, (2, 3))
        if f: t = torch.flip(t, (3,))
        inp = torch.cat([t[:, :4], px, py, t[:, 4:]], 1)
        out = net1(inp).float()
        if f: out = torch.flip(out, (3,))
        acc += torch.rot90(out, -k, (2, 3))
    return acc / len(orients)


def _u8(t):
    u = (t.clamp(0, 1) * 255 + 0.5).to(torch.uint8)
    u[:, :3] = torch.minimum(u[:, :3], u[:, 3:4])
    return u.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True); ap.add_argument('--stage1', required=True); ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--init', default=None); ap.add_argument('--channels', type=int, default=256); ap.add_argument('--blocks', type=int, default=10)
    ap.add_argument('--extended', action='store_true', help='17-channel refiner: add split mm_rgb, mm_alpha, px, py (zero-init warm-start from 13ch)')
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=None, help='default: 1e-4 from scratch, 2e-5 when fine-tuning')
    ap.add_argument('--lr-min', type=float, default=None)
    ap.add_argument('--cosine-from', type=float, default=0.2)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--amp', action='store_true', default=True); ap.add_argument('--no-amp', dest='amp', action='store_false')
    ap.add_argument('--batch', type=int, default=12); ap.add_argument('--patch', type=int, default=128)
    ap.add_argument('--sparse-prob', type=float, default=0.10)
    ap.add_argument('--pass2-prob', type=float, default=0.5, help='probability of unrolling pass 2 during training')
    ap.add_argument('--s1-tta', type=int, default=2, help='number of D4 views averaged for stage-1 input during training (1..8)')
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--pool', type=int, default=6)
    ap.add_argument('--l1-weight', type=float, default=1.0); ap.add_argument('--l2-weight', type=float, default=0.0)
    ap.add_argument('--hue-weight', type=float, default=2.0)
    ap.add_argument('--seam-weight', type=float, default=1.0); ap.add_argument('--seam-l2-weight', type=float, default=0.0)
    ap.add_argument('--ssim-weight', type=float, default=1.0)
    ap.add_argument('--val-raw'); ap.add_argument('--val-clean')
    a = ap.parse_args()
    from echoclean import blocks as B, models, metrics as MT, data as D, pipeline as PL
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join('runs', a.name); os.makedirs(out, exist_ok=True); log = open(os.path.join(out, 'log.txt'), 'a')
    def say(*x):
        s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
    say(f'=== {time.strftime("%Y-%m-%d %H:%M")}  ' + ' '.join(sys.argv[1:]))
    groups = [D.list_images(s) for s in a.data]; say('data:', ', '.join(f'{len(f)} images (weight {w:g})' for f, w in groups))
    net1 = models.load_stage1(a.stage1, dev)
    if dev.type == 'cuda': net1.half()
    in_ch = 17 if a.extended else 13
    if a.init:
        sd = torch.load(a.init, map_location='cpu', weights_only=True)['net']
        c, n = models._shape(sd); old_in = sd['head.weight'].shape[1] // 4
        ref = models.Refiner(c, n, in_ch=in_ch).to(dev)
        if old_in == in_ch:
            ref.load_state_dict(sd)
        elif old_in == 13 and in_ch == 17:
            w_new = torch.zeros((c, in_ch * 4, 3, 3))
            w_new[:, :52] = sd['head.weight']  # exact step-0 identity with 13ch checkpoint!
            sd = dict(sd); sd['head.weight'] = w_new
            ref.load_state_dict(sd)
            say(f'warm-started 17ch refiner from 13ch {a.init} (exact step-0 match)')
        else:
            raise ValueError(f'incompatible init channels {old_in} -> {in_ch}')
    else:
        ref = models.Refiner(a.channels, a.blocks, in_ch=in_ch).to(dev)
    ref.train()
    ema = copy.deepcopy(ref).eval() if a.ema > 0 else None
    if ema is not None:
        for p in ema.parameters(): p.requires_grad_(False)
    lr = a.lr or (2e-5 if a.init else 1e-4); lr_min = a.lr_min or lr / 10.0
    opt = torch.optim.Adam(ref.parameters(), lr=lr)
    use_amp = bool(a.amp and dev.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    say(f'refiner ({in_ch}ch) {sum(p.numel() for p in ref.parameters()) / 1e6:.2f}M params on {dev}, lr {lr:g}->{lr_min:g}, pass2_prob={a.pass2_prob}, s1_tta={a.s1_tta}, patch={a.patch}')
    stream = D.forever(DataLoader(D.PatchStream(groups, a.patch // 4 * 4, a.batch, a.pool, sparse_prob=a.sparse_prob),
                                  batch_size=None, num_workers=a.workers, prefetch_factor=6))
    val = None
    if a.val_raw and a.val_clean:
        vb, vw, vh = B.load_raw(a.val_raw); val = (vb, vw, vh, B.load_clean(a.val_clean))
        # Precompute 8x TTA stage-1 output on validation pair ONCE!
        val_o1 = PL.stage1(net1, vb, dev, tta=True, half=True)
        say(f'--- stage 1 (8x TTA):  {MT.fmt(MT.score(PL.finish(val_o1)[:vh, :vw], val[3], dev))}')
    lumw = torch.tensor([0.3, 0.59, 0.11], device=dev).view(1, 3, 1, 1); best = -1.0

    @torch.inference_mode()
    def evaluate(step):
        nonlocal best
        ev_ref = ema if (ema is not None and step > 0) else ref
        models.save(ev_ref, os.path.join(out, 'ckpt_last.pt'), step=step, kind='refiner')
        if val is None: return
        ev_ref.eval(); vb, vw, vh, vy = val
        # Score 2-pass 1x (fast) and 2-pass 8x (exact full pipeline!)
        o2_1x = PL.finish(PL.stage2(ev_ref, vb, val_o1, None, dev, passes=2, tta=False, half=True))[:vh, :vw]
        o2_8x = PL.finish(PL.stage2(ev_ref, vb, val_o1, None, dev, passes=2, tta=True, half=True))[:vh, :vw]
        if ev_ref is ref: ref.train()
        r1 = MT.score(o2_1x, vy, dev); r8 = MT.score(o2_8x, vy, dev)
        say(f'--- step {step}: val 2p-1x  PSNR {r1["psnr"]:.3f} / seam {r1["seam"]:.4f}  |  val 2p-8x  PSNR {r8["psnr"]:.3f} / edges {r8["edges"]:.3f} / seam {r8["seam"]:.4f} / ssim {r8["ssim"]:.4f}')
        if r8['psnr'] > best:
            best = r8['psnr']; models.save(ev_ref, os.path.join(out, 'ckpt_best.pt'), step=step, kind='refiner')

    def compute_loss(o, y):
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
        return loss

    evaluate(0); t0 = time.time()
    for s in range(a.steps):
        cur_lr = _lr_at(s, a.steps, lr, lr_min, a.warmup, a.cosine_from)
        for g in opt.param_groups: g['lr'] = cur_lr
        x13, y, bx = next(stream)
        x13 = x13.permute(0, 3, 1, 2).to(dev, non_blocking=True)
        y = (y.permute(0, 3, 1, 2).float() / 255).to(dev, non_blocking=True)
        x = x13[:, :4]; bx = bx.numpy()
        o1 = _stage1_pred(net1, x13, tta_k=a.s1_tta)
        u1 = _u8(o1)
        bo1 = _encode_batch(u1)
        inp1 = _build_ref_input(x, o1, bo1, bx, extended=a.extended)
        do_p2 = (a.pass2_prob > 0 and np.random.random() < a.pass2_prob)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            p1 = ref(inp1)
        p1_f = p1.float()
        if do_p2:
            o2_in = p1_f.detach()
            u2 = _u8(o2_in)
            bo2 = _encode_batch(u2, prev_u8=u1, prev_bl=bo1)
            inp2 = _build_ref_input(x, o2_in, bo2, bx, extended=a.extended)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                p2 = ref(inp2)
            p2_f = p2.float()
            loss = 0.5 * compute_loss(p1_f, y) + 0.5 * compute_loss(p2_f, y)
        else:
            loss = compute_loss(p1_f, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            d = min(a.ema, (1 + s) / (10 + s))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), ref.parameters()):
                    pe.lerp_(p, 1.0 - d)
        if (s + 1) % 500 == 0:
            el = time.time() - t0
            print(f'step {s + 1}/{a.steps}  lr {cur_lr:.2e}  loss {loss.item():.5f}  {(s + 1) / el:.2f} steps/s  ~{el / (s + 1) * (a.steps - s - 1) / 60:.1f} min left', flush=True)
        if (s + 1) % a.eval_every == 0 or s + 1 == a.steps:
            evaluate(s + 1)
    say(f'done in {(time.time() - t0) / 60:.1f} min' + (f'; best validation 2p-8x PSNR {best:.3f} dB' if val else ''))
    os._exit(0)


if __name__ == '__main__':
    main()
