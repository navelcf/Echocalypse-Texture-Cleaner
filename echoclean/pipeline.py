"""The cleaner: stage 1 with 8x self-ensemble, then the consistency refiner (two passes, each with 8x self-ensemble)."""
import os, sys, shutil, tempfile, multiprocessing as mp
import numpy as np, torch
from . import blocks as B, encoder, models

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1 = os.path.join(HERE, 'models', 'echoclean_stage1.pt'); REFINER = os.path.join(HERE, 'models', 'echoclean_refiner.pt')
PAD = 32
_ZERO_BLK = None
_ZIG_CC_WRAPPER = None


def _ensure_cc_for_compile():
    """Ensure Triton/Inductor has a C compiler; fall back to `python -m ziglang cc` when installed."""
    global _ZIG_CC_WRAPPER
    if os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang') or shutil.which('cc'):
        return
    try:
        import ziglang  # noqa: F401
    except ImportError:
        return
    if _ZIG_CC_WRAPPER is None or not os.path.exists(_ZIG_CC_WRAPPER):
        path = os.path.join(tempfile.gettempdir(), f'echoclean_zigcc_{os.getuid()}.sh')
        with open(path, 'w') as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" -m ziglang cc "$@"\n')
        os.chmod(path, 0o755)
        _ZIG_CC_WRAPPER = path
    os.environ['CC'] = _ZIG_CC_WRAPPER


def _zero_block():
    global _ZERO_BLK
    if _ZERO_BLK is None:
        if encoder.has_fast():
            _ZERO_BLK = encoder.encode_blocks(np.zeros((4, 4, 4), np.uint8))[0, 0]
        else:
            # Standard ETC2 RGBA8 all-zero block (EAC base=0, mult=0; ETC2 RGB=0)
            _ZERO_BLK = np.zeros(16, np.uint8)
    return _ZERO_BLK


def _net_dtype(net):
    return next(net.parameters()).dtype


def _tiles(net, inp, h, w, tile, half, pad=PAD, live_blk=None):
    out = torch.zeros((4, h, w), device=inp.device, dtype=torch.float32)
    bpad = (pad + 3) // 4
    bh, bw = h // 4, w // 4
    use_native_half = half and inp.device.type == 'cuda' and _net_dtype(net) == torch.float16
    use_autocast = half and inp.device.type == 'cuda' and not use_native_half
    for yy in range(0, h, tile):
        th = min(tile, h - yy)
        for xx in range(0, w, tile):
            tw = min(tile, w - xx)
            if live_blk is not None:
                by0 = max(0, yy // 4 - bpad); by1 = min(bh, (yy + th + 3) // 4 + bpad)
                bx0 = max(0, xx // 4 - bpad); bx1 = min(bw, (xx + tw + 3) // 4 + bpad)
                if not live_blk[by0:by1, bx0:bx1].any():
                    continue
            patch = inp[:, :, yy:yy + th + 2 * pad, xx:xx + tw + 2 * pad]
            if use_native_half:
                r = net(patch)[0]
            else:
                with torch.autocast(inp.device.type, dtype=torch.float16, enabled=use_autocast):
                    r = net(patch)[0].float()
            out[:, yy:yy + th, xx:xx + tw] = r[:, pad:pad + th, pad:pad + tw]
    return out


def finish(o):
    o = np.clip(o + 0.5, 0, 255).astype(np.uint8); o[..., :3] = np.minimum(o[..., :3], o[..., 3:4]); return o


def _finish_tensor(o):
    """(4, H, W) float 0..255 on GPU -> (H, W, 4) premultiplied uint8 on CPU."""
    u = (o + 0.5).clamp_(0, 255).to(torch.uint8)
    u[:3] = torch.minimum(u[:3], u[3:4])
    return u.permute(1, 2, 0).contiguous().cpu().numpy()


def _ensemble(net, base, tta, half, tile=1024, pos=False, pos_at=4, pad=None, live_blk=None):
    """net over a (1, C, H, W) input, in all 8 flips/rotations if tta, averaged -> (4, H, W) float 0..255.
    pos: insert the two position-in-block channels at index `pos_at` (they depend on the orientation).
    live_blk: optional (bh, bw) CPU numpy bool mask so empty tiles are skipped with zero GPU sync."""
    pad = PAD if pad is None else pad
    dev = base.device; acc = torch.zeros((4,) + tuple(base.shape[2:]), device=dev, dtype=torch.float32); n = 0
    for k in (range(4) if tta else (0,)):
        for f in ((False, True) if tta else (False,)):
            t = torch.rot90(base, k, (2, 3))
            lb = np.rot90(live_blk, k) if live_blk is not None else None
            if f:
                t = torch.flip(t, (3,))
                if lb is not None: lb = np.fliplr(lb)
            h, w = t.shape[2:]
            if pos:
                px = (torch.arange(w, device=dev, dtype=torch.float32) % 4).div_(3).view(1, 1, 1, w).expand(1, 1, h, w)
                py = (torch.arange(h, device=dev, dtype=torch.float32) % 4).div_(3).view(1, 1, h, 1).expand(1, 1, h, w)
                t = torch.cat([t[:, :pos_at], px, py, t[:, pos_at:]], 1)
            tp = torch.nn.functional.pad(t, (pad,) * 4)
            out = _tiles(net, tp, h, w, tile, half, pad=pad, live_blk=lb).clamp_(0, 1).mul_(255)
            out = torch.flip(out, (2,)) if f else out; acc += torch.rot90(out, -k, (1, 2)); n += 1
    return acc / n


@torch.inference_mode()
def stage1(net, blocks, dev, tta=True, tile=1024, half=True, pad=None, _x=None, _return_tensor=False):
    """blocks -> float32 RGBA 0..255 at the padded (multiple of 4) size. Not rounded: stage 2 wants the fractions."""
    x = B.decode(blocks) if _x is None else _x
    x_u8 = torch.from_numpy(x).to(dev)
    feat_t = B.coding_features_torch(blocks, x_u8[..., 3])
    x_f = x_u8.permute(2, 0, 1).float().div_(255)
    base = torch.cat([x_f, feat_t], 0).unsqueeze(0)
    bh, bw = blocks.shape[:2]
    live_blk = (x_u8.amax(dim=-1).view(bh, 4, bw, 4).amax(dim=(1, 3)) > 0).cpu().numpy()
    out = _ensemble(net, base, tta, half, tile, pos=True, pad=pad, live_blk=live_blk)
    if _return_tensor:
        return out, x, x_f, live_blk
    return out.permute(1, 2, 0).cpu().numpy()


def refiner_input(x, o1, blocks, reencoded):
    """x: the game texture, uint8; o1: the current result, 0..255 (float, as the refiner was trained; uint8 also works);
    blocks: the game's blocks; reencoded: blocks of encode(finish(o1)). -> (H, W, 13) float32"""
    inv255 = np.float32(1.0 / 255.0)
    fb = (B.decode(reencoded).astype(np.float32) - x.astype(np.float32)) * inv255
    mm = np.repeat(np.repeat((reencoded != blocks).any(-1).astype(np.float32), 4, 0), 4, 1)[..., None]
    return np.concatenate([x.astype(np.float32) * inv255, np.asarray(o1, np.float32) * inv255, fb, mm], -1)


def _refiner_input_gpu(x_f, o_t, blocks, reencoded, re_dec, dev, in_ch=13):
    """GPU-native refiner input builder: avoids ~650 MB float64 CPU churn and cuts PCIe upload ~9x."""
    re_f = torch.from_numpy(re_dec).to(dev).permute(2, 0, 1).float().div_(255)
    fb = re_f - x_f
    o_f = o_t * (1.0 / 255.0)
    if in_ch == 13:
        mm_blk = torch.from_numpy((reencoded != blocks).any(-1)).to(dev, dtype=torch.float32)
        mm = mm_blk.repeat_interleave(4, 0).repeat_interleave(4, 1).unsqueeze(0)
        return torch.cat([x_f, o_f, fb, mm], 0).unsqueeze(0), False, 12
    # Extended refiner (17ch): [x(4), o1(4), fb(4), mm_any(1), mm_rgb(1), mm_alpha(1), px(1), py(1)] -> px,py inserted at 15 per orientation
    mm_any = torch.from_numpy((reencoded != blocks).any(-1)).to(dev, dtype=torch.float32).repeat_interleave(4, 0).repeat_interleave(4, 1).unsqueeze(0)
    mm_rgb = torch.from_numpy((reencoded[..., 8:] != blocks[..., 8:]).any(-1)).to(dev, dtype=torch.float32).repeat_interleave(4, 0).repeat_interleave(4, 1).unsqueeze(0)
    mm_a = torch.from_numpy((reencoded[..., :8] != blocks[..., :8]).any(-1)).to(dev, dtype=torch.float32).repeat_interleave(4, 0).repeat_interleave(4, 1).unsqueeze(0)
    return torch.cat([x_f, o_f, fb, mm_any, mm_rgb, mm_a], 0).unsqueeze(0), True, 15


@torch.inference_mode()
def stage2(net, blocks, o, pool, dev, passes=2, tta=True, half=True, tile=1024, pad=None, _x=None, _x_f=None, _live_blk=None):
    """The refiner. o: stage-1 output, float 0..255 (numpy HWC or torch CHW), not rounded -> float 0..255.
    Each pass re-compresses the current result with the game's encoder, shows the refiner how that differs from the game's
    data, and takes its correction (8 flips/rotations averaged if tta). The results of the passes are averaged."""
    was_tensor = torch.is_tensor(o)
    o_t = o if was_tensor else torch.from_numpy(np.ascontiguousarray(o)).to(dev, dtype=torch.float32).permute(2, 0, 1)
    if _x_f is not None:
        x_f = _x_f
    else:
        x = B.decode(blocks) if _x is None else _x
        x_f = torch.from_numpy(x).to(dev).permute(2, 0, 1).float().div_(255)
    in_ch = getattr(net, 'in_ch', 13)
    weights = (1.0, 2.0) if (passes == 2 and in_ch > 13) else (1.0,) * passes
    acc = torch.zeros_like(o_t)
    prev_pm = prev_re = prev_dec = None
    for wi in weights:
        cur_pm = _finish_tensor(o_t)
        re = encoder.encode_blocks_mp(cur_pm, pool, prev_pm=prev_pm, prev_blocks=prev_re)
        re_dec = B.decode(re) if prev_re is None else B.decode_diff(re, prev_re, prev_dec)
        prev_pm, prev_re, prev_dec = cur_pm, re, re_dec
        inp, use_pos, pos_at = _refiner_input_gpu(x_f, o_t, blocks, re, re_dec, dev, in_ch=in_ch)
        o_t = _ensemble(net, inp, tta, half, tile=tile, pos=use_pos, pos_at=pos_at, pad=pad, live_blk=_live_blk)
        acc.add_(o_t, alpha=wi)
    out = acc.div_(sum(weights))
    return out if was_tensor else out.permute(1, 2, 0).cpu().numpy()


class Pipeline:
    def __init__(s, stage1_path=STAGE1, refiner_path=REFINER, device=None, procs=None, compile=False, half=True):
        s.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        for p in (stage1_path, refiner_path):
            if p and not os.path.exists(p): raise FileNotFoundError(f'{p} not found. Download the model files from the GitHub release into models/ (see README).')
        s.net1 = models.load_stage1(stage1_path, s.dev)
        s.ref = models.load_refiner(refiner_path, s.dev) if refiner_path else None
        s._half_ready = bool(half and s.dev.type == 'cuda')
        if s._half_ready:
            s.net1.half()
            if s.ref is not None: s.ref.half()
        if compile and s.dev.type == 'cuda':
            _ensure_cc_for_compile()
            s.net1 = torch.compile(s.net1, mode='default')
            if s.ref is not None: s.ref = torch.compile(s.ref, mode='default')
        s.pool = mp.get_context('spawn').Pool(procs or max(1, (os.cpu_count() or 4) - 2), maxtasksperchild=64) if (s.ref is not None and not encoder.has_fast()) else None

    @torch.inference_mode()
    def __call__(s, blocks, w=None, h=None, tta=True, refine=True, half=True, passes=2, tile=1024, pad=None):
        """blocks from blocks.load_raw -> cleaned premultiplied RGBA uint8."""
        if not half and s._half_ready:
            s.net1.float()
            if s.ref is not None: s.ref.float()
            s._half_ready = False
        o_t, x, x_f, live_blk = stage1(s.net1, blocks, s.dev, tta, tile=tile, half=half, pad=pad, _return_tensor=True)
        if refine and s.ref is not None:
            o_t = stage2(s.ref, blocks, o_t, s.pool, s.dev, passes, tta, half, tile=tile, pad=pad, _x=x, _x_f=x_f, _live_blk=live_blk)
        out = _finish_tensor(o_t)
        return out[:h or out.shape[0], :w or out.shape[1]]

    def close(s):
        if s.pool: s.pool.terminate()
