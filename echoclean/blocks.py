"""Reading the game's texture files and working with their ETC2 RGBA8 blocks.

A game texture (*.png in the game folders, despite the name) is  UF wrapper -> CCZ (zlib) -> PVR v3, pixel format 23
(ETC2 RGBA8: per 4x4 block 8 bytes EAC alpha + 8 bytes ETC2 colour), premultiplied alpha.
Block arrays here have shape (H/4, W/4, 16), uint8.
"""
import zlib, struct
import numpy as np
from PIL import Image
import texture2ddecoder

UF_KEY = bytes.fromhex('135b0c0d6616222b1119584024100e423157382c351c0b0574253a69140f4d071d')
_UF_KEY_ARR = np.frombuffer(UF_KEY, np.uint8)


def uf_decode(buf):
    """Remove the game's 'UF' wrapper (a repeating-key XOR). Data without the wrapper is returned unchanged."""
    if len(buf) < 5 or buf[0] != 0x55 or buf[1] != 0x46:
        return buf
    p = 2; extra = 0
    if buf[2] == 0x4f:
        p = 4; extra = 1
    if len(buf) <= p + 2:
        return buf
    mode = buf[p + 1]; L = len(UF_KEY)
    if mode == 1:
        off = buf[p + 2]; p += 3
        raw = np.frombuffer(buf, np.uint8, offset=p)
        if raw.size == 0:
            return b''
        key = np.roll(_UF_KEY_ARR, -(off % L))
        reps = (raw.size + L - 1) // L
        return np.bitwise_xor(raw, np.tile(key, reps)[:raw.size]).tobytes()
    if mode == 2:
        off = buf[p + 2]; p += 3; n = len(buf) - p
        if n < p:
            return buf
        out = bytearray(buf[:n])
        for j in range(p):
            out[j] = UF_KEY[(j + off) % L] ^ buf[n + j]
        for i in range(p, p + min(n - p, 0x5f)):
            out[i] ^= UF_KEY[(i + off) % L]
        return bytes(out)
    return buf[extra * 2 + 5:]


class NotEtc2(ValueError):
    pass


def load_raw(path):
    """-> (blocks, width, height). Raises NotEtc2 for files that are not ETC2 RGBA8 (plain PNGs, JPEGs, other formats)."""
    with open(path, 'rb') as f:
        d = uf_decode(f.read())
    if d[:4] == b'CCZ!':
        try:
            d = zlib.decompress(d[16:])
        except zlib.error as exc:
            raise NotEtc2(f'{path}: invalid CCZ stream ({exc})') from exc
    if len(d) < 52 or d[:4] != b'PVR\x03':
        raise NotEtc2(f'{path}: not a PVR texture')
    _, flags, pf, cs, ct, h, w, dep, ns, nf, nm, meta = struct.unpack('<IIQIIIIIIIII', d[:52])
    if pf != 23:
        raise NotEtc2(f'{path}: PVR pixel format {pf}, not ETC2 RGBA8')
    bw, bh = (w + 3) // 4, (h + 3) // 4
    need = 52 + meta + bw * bh * 16
    if len(d) < need:
        raise NotEtc2(f'{path}: truncated PVR data ({len(d)} < {need} bytes)')
    return np.frombuffer(d, np.uint8, count=bw * bh * 16, offset=52 + meta).reshape(bh, bw, 16).copy(), w, h


def decode(blocks, w=None, h=None):
    """blocks -> premultiplied RGBA uint8 image (cropped to w x h if given)."""
    bh, bw = blocks.shape[:2]
    px = np.frombuffer(
        texture2ddecoder.decode_etc2a8(np.ascontiguousarray(blocks).tobytes(), bw * 4, bh * 4),
        np.uint8,
    ).reshape(bh * 4, bw * 4, 4)[..., [2, 1, 0, 3]]
    if (h is None or h == bh * 4) and (w is None or w == bw * 4):
        return px
    return px[:h or bh * 4, :w or bw * 4].copy()


def decode_diff(blocks, prev_blocks, prev_dec):
    """Decode only blocks that differ from prev_blocks, updating a copy of prev_dec. Faster when few blocks changed."""
    diff = (blocks != prev_blocks).any(-1)
    n_chg = int(diff.sum())
    if n_chg == 0:
        return prev_dec
    bh, bw = blocks.shape[:2]
    if n_chg * 2 >= bh * bw:
        return decode(blocks)
    out = prev_dec.copy()
    ys, xs = np.where(diff)
    chg_blks = np.ascontiguousarray(blocks[ys, xs])  # (N, 16)
    dec_strip = np.frombuffer(
        texture2ddecoder.decode_etc2a8(chg_blks.tobytes(), n_chg * 4, 4),
        np.uint8,
    ).reshape(4, n_chg, 4, 4).transpose(1, 0, 2, 3)[..., [2, 1, 0, 3]]  # (N, 4, 4, 4)
    out_blk = out.reshape(bh, 4, bw, 4, 4)
    out_blk[ys, :, xs, :, :] = dec_strip.transpose(0, 1, 2, 3)
    return out


def load_clean(path):
    """Load a clean RGBA image as premultiplied uint8 (straight-alpha files are premultiplied; already premultiplied ones are kept)."""
    with Image.open(path) as img:
        a = np.asarray(img.convert('RGBA'))
    part = (a[..., 3] > 0) & (a[..., 3] < 255)
    over = (a[..., :3].max(2) > a[..., 3])[part].mean() if part.any() else 1.0
    if over < 0.02:
        return a.copy()
    f = a.astype(np.float32); f[..., :3] *= f[..., 3:] * np.float32(1.0 / 255.0)
    return f.round().astype(np.uint8)


MODES = ('individual', 'differential', 'T', 'H', 'planar')
def colour_mode(blocks):
    """ETC2 colour mode per block: 0 individual, 1 differential, 2 T, 3 H, 4 planar."""
    c = blocks[..., 8:12].astype(np.int32); diff = (blocks[..., 11] >> 1) & 1
    def over(b):
        d = b & 7; s = (b >> 3) + np.where(d > 3, d - 8, d); return (s < 0) | (s > 31)
    return np.where(diff == 0, 0, np.where(over(c[..., 0]), 2, np.where(over(c[..., 1]), 3, np.where(over(c[..., 2]), 4, 1))))


EAC = np.array([[-3, -6, -9, -15, 2, 5, 8, 14], [-3, -7, -10, -13, 2, 6, 9, 12], [-2, -5, -8, -13, 1, 4, 7, 12], [-2, -4, -6, -13, 1, 3, 5, 12],
                [-3, -6, -8, -12, 2, 5, 7, 11], [-3, -7, -9, -11, 2, 6, 8, 10], [-4, -7, -8, -11, 3, 6, 7, 10], [-3, -5, -8, -11, 2, 4, 7, 10],
                [-2, -6, -8, -10, 1, 5, 7, 9], [-2, -5, -8, -10, 1, 4, 7, 9], [-2, -4, -8, -10, 1, 3, 7, 9], [-2, -5, -7, -10, 1, 4, 6, 9],
                [-3, -4, -7, -10, 2, 3, 6, 9], [-1, -2, -3, -10, 0, 1, 2, 9], [-4, -6, -8, -9, 3, 5, 7, 8], [-3, -5, -7, -9, 2, 4, 6, 8]], dtype=np.int32)


def alpha_bounds(blocks, alpha):
    """Per texel: the interval of true alpha that the encoder would have coded as this texel's EAC level -> (lo, hi) float32 images."""
    bh, bw = blocks.shape[:2]
    base = blocks[..., 0].astype(np.int32); mult = (blocks[..., 1] >> 4).astype(np.int32); tab = blocks[..., 1] & 15
    lev = np.clip(base[..., None] + EAC[tab] * mult[..., None], 0, 255)[:, :, None, :]
    a = alpha.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(bh, bw, 16, 1).astype(np.int32)
    below = np.where(lev < a, lev, -1000000).max(-1)
    above = np.where(lev > a, lev, 1000000).min(-1)
    a = a[..., 0]
    lo = np.where(below > -100000, (a + below).astype(np.float32) * np.float32(0.5), np.float32(0.0))
    hi = np.where(above < 100000, (a + above).astype(np.float32) * np.float32(0.5), np.float32(255.0))
    return (
        lo.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4),
        hi.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4),
    )


NSIDE = 9
def coding_features(blocks, alpha):
    """Side information for the stage-1 model, (H, W, 7) float32: alpha interval lo/hi (0..1) and the colour mode one-hot.
    The two position-in-block channels are added on the fly (they depend on orientation)."""
    lo, hi = alpha_bounds(blocks, alpha); m = colour_mode(blocks); bh, bw = m.shape
    out = np.empty((bh * 4, bw * 4, 7), np.float32)
    out[..., 0] = lo * np.float32(1.0 / 255.0)
    out[..., 1] = hi * np.float32(1.0 / 255.0)
    oh = np.zeros((bh, bw, 5), np.float32)
    np.put_along_axis(oh, m[..., None], 1.0, -1)
    out[..., 2:] = oh[:, None, :, None, :].repeat(4, 1).repeat(4, 3).reshape(bh * 4, bw * 4, 5)
    return out


_EAC_TORCH = {}
def coding_features_torch(blocks, alpha_t):
    """GPU-native equivalent of coding_features(): blocks (bh, bw, 16) uint8 numpy, alpha_t (H, W) uint8/int32 torch -> (7, H, W) float32 on alpha_t.device."""
    import torch
    dev = alpha_t.device
    eac = _EAC_TORCH.get(dev)
    if eac is None:
        eac = torch.from_numpy(EAC).to(dev)
        _EAC_TORCH[dev] = eac
    bh, bw = blocks.shape[:2]
    # Only need bytes 0, 1 (EAC alpha) and 8..11 (ETC2 colour mode) -> 6 bytes/block (~1.5 MB for 2048x2048)
    hdr = torch.from_numpy(np.ascontiguousarray(blocks[..., [0, 1, 8, 9, 10, 11]])).to(dev, dtype=torch.int32)
    base = hdr[..., 0]; mult = hdr[..., 1] >> 4; tab = hdr[..., 1] & 15
    lev = (base.unsqueeze(-1) + eac[tab] * mult.unsqueeze(-1)).clamp_(0, 255).unsqueeze(2)  # (bh, bw, 1, 8)
    a = alpha_t.to(torch.int32).view(bh, 4, bw, 4).permute(0, 2, 1, 3).reshape(bh, bw, 16, 1)
    below = torch.where(lev < a, lev, -1000000).amax(-1)
    above = torch.where(lev > a, lev, 1000000).amin(-1)
    a0 = a.squeeze(-1)
    lo = torch.where(below > -100000, (a0 + below).float() * (0.5 / 255.0), 0.0)
    hi = torch.where(above < 100000, (a0 + above).float() * (0.5 / 255.0), 1.0)
    lo = lo.view(bh, bw, 4, 4).permute(0, 2, 1, 3).reshape(1, bh * 4, bw * 4)
    hi = hi.view(bh, bw, 4, 4).permute(0, 2, 1, 3).reshape(1, bh * 4, bw * 4)

    diff = (hdr[..., 5] >> 1) & 1
    def over(b):
        d = b & 7; s = (b >> 3) + torch.where(d > 3, d - 8, d); return (s < 0) | (s > 31)
    m = torch.where(diff == 0, 0, torch.where(over(hdr[..., 2]), 2, torch.where(over(hdr[..., 3]), 3, torch.where(over(hdr[..., 4]), 4, 1))))
    oh = torch.nn.functional.one_hot(m.long(), 5).float().permute(2, 0, 1)  # (5, bh, bw)
    oh = oh.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
    return torch.cat([lo, hi, oh], dim=0)


def with_pos(feat):
    h, w = feat.shape[:2]
    out = np.empty((h, w, feat.shape[2] + 2), np.float32)
    out[..., 0] = ((np.arange(w, dtype=np.float32) % 4) * np.float32(1.0 / 3.0))[None, :]
    out[..., 1] = ((np.arange(h, dtype=np.float32) % 4) * np.float32(1.0 / 3.0))[:, None]
    out[..., 2:] = feat
    return out
