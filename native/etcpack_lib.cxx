// Shared-library entry around Ericsson ETCPACK v2.74 (not included here; build_etcpack.sh fetches it).
// Exports:
//   etcpack_rgba8(rgba, W, H, out, fast, perceptual)                         -- single-threaded (legacy)
//   etcpack_rgba8_fast(rgba, W, H, out, fast, perceptual, nthreads, prev_rgba, prev_blocks)
//     -- native pthreads + stack 4x4 buffers + zero-block skip + constant-alpha LUT +
//        split RGB/alpha dirty-block skip + exact-zero-error early-out + thread-local constant-RGB cache
//        (100% byte-identical to Ericsson ETCPACK v2.74 with macOS rand())
#include <sys/timeb.h>
#define _timeb timeb
#define _ftime ftime
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

// ETCPACK calls srand(10000) and then rand() for every block it tries in T or H mode, so its output depends on the C library.
// The game's textures were encoded on macOS: with Apple's rand() below, a clean original re-encodes to the game's blocks
// 100% byte for byte (measured on hero 600090); with glibc's rand() it is 99.3%, all of the difference in T/H-mode decisions.
// The state is per thread, which also makes the encoder safe to call from several threads (libc rand() was its only shared state).
static __thread unsigned long etcpack_rand_state = 1;
static void etcpack_srand(unsigned seed) { etcpack_rand_state = seed; }
static int etcpack_rand(void)
{
	long hi, lo, x;
	if (etcpack_rand_state == 0) etcpack_rand_state = 123459876;
	hi = etcpack_rand_state / 127773; lo = etcpack_rand_state % 127773; x = 16807 * lo - 2836 * hi;      // 16807 * state mod (2^31 - 1)
	if (x < 0) x += 0x7fffffff;
	etcpack_rand_state = x;
	return (int)(x % 0x80000000UL);
}
#ifndef ETCPACK_LIBC_RAND      // build with -DETCPACK_LIBC_RAND for the old behaviour (this machine's own rand())
#define srand etcpack_srand
#define rand etcpack_rand
#undef RAND_MAX
#define RAND_MAX 0x7fffffff
#endif

#define main etcpack_main
#include "ETCPACK/source/etcpack.cxx"
#undef main

static pthread_once_t etcpack_init_once = PTHREAD_ONCE_INIT;
static unsigned char zero_blk_cache[4][16];
static unsigned char const_alpha_fast_cache[256][8]; // precomputed for fast=1 (256 constant alpha blocks)

// 100% byte-identical copy of compressBlockETC2FastPerceptual with exact-zero-error early-outs.
// Since calcBlockPerceptualErrorRGB >= 0.0 and srand(10000) is reset inside each T/H call,
// skipping subsequent modes once error == 0.0 cannot change any bit of this or any other block.
static inline void compress_rgb_fast_perceptual(uint8 *img, uint8 *imgdec, unsigned int &compressed1, unsigned int &compressed2)
{
	unsigned int etc1_word1, etc1_word2;
	compressBlockDiffFlipFastPerceptual(img, imgdec, 4, 4, 0, 0, etc1_word1, etc1_word2);
	decompressBlockDiffFlip(etc1_word1, etc1_word2, imgdec, 4, 4, 0, 0);
	double error_etc1 = 1000 * calcBlockPerceptualErrorRGB(img, imgdec, 4, 4, 0, 0);
	if (error_etc1 == 0.0) {
		compressed1 = etc1_word1;
		compressed2 = etc1_word2;
		return;
	}

	unsigned int planar57_word1, planar57_word2, planar_word1, planar_word2;
	compressBlockPlanar57(img, 4, 4, 0, 0, planar57_word1, planar57_word2);
	decompressBlockPlanar57(planar57_word1, planar57_word2, imgdec, 4, 4, 0, 0);
	double error_planar = 1000 * calcBlockPerceptualErrorRGB(img, imgdec, 4, 4, 0, 0);
	stuff57bits(planar57_word1, planar57_word2, planar_word1, planar_word2);
	if (error_planar == 0.0) {
		compressed1 = planar_word1;
		compressed2 = planar_word2;
		return;
	}

	unsigned int thumbT59_word1, thumbT59_word2, thumbT_word1, thumbT_word2;
	compressBlockTHUMB59TFastestPerceptual1000(img, 4, 4, 0, 0, thumbT59_word1, thumbT59_word2);
	decompressBlockTHUMB59T(thumbT59_word1, thumbT59_word2, imgdec, 4, 4, 0, 0);
	double error_thumbT = 1000 * calcBlockPerceptualErrorRGB(img, imgdec, 4, 4, 0, 0);
	stuff59bits(thumbT59_word1, thumbT59_word2, thumbT_word1, thumbT_word2);
	if (error_thumbT == 0.0) {
		compressed1 = thumbT_word1;
		compressed2 = thumbT_word2;
		return;
	}

	unsigned int thumbH58_word1, thumbH58_word2, thumbH_word1, thumbH_word2;
	compressBlockTHUMB58HFastestPerceptual1000(img, 4, 4, 0, 0, thumbH58_word1, thumbH58_word2);
	decompressBlockTHUMB58H(thumbH58_word1, thumbH58_word2, imgdec, 4, 4, 0, 0);
	double error_thumbH = 1000 * calcBlockPerceptualErrorRGB(img, imgdec, 4, 4, 0, 0);
	stuff58bits(thumbH58_word1, thumbH58_word2, thumbH_word1, thumbH_word2);

	double error_best = error_etc1;
	compressed1 = etc1_word1;
	compressed2 = etc1_word2;
	int best_mode = MODE_ETC1;

	if (error_planar < error_best) {
		compressed1 = planar_word1; compressed2 = planar_word2;
		error_best = error_planar; best_mode = MODE_PLANAR;
	}
	if (error_thumbT < error_best) {
		compressed1 = thumbT_word1; compressed2 = thumbT_word2;
		error_best = error_thumbT; best_mode = MODE_THUMB_T;
	}
	if (error_thumbH < error_best) {
		compressed1 = thumbH_word1; compressed2 = thumbH_word2;
		error_best = error_thumbH; best_mode = MODE_THUMB_H;
	}

	if (error_best == 0.0) return;

	switch (best_mode) {
	case MODE_THUMB_T:
		compressBlockTHUMB59TFast(img, 4, 4, 0, 0, thumbT59_word1, thumbT59_word2);
		decompressBlockTHUMB59T(thumbT59_word1, thumbT59_word2, imgdec, 4, 4, 0, 0);
		error_thumbT = calcBlockErrorRGB(img, imgdec, 4, 4, 0, 0);
		stuff59bits(thumbT59_word1, thumbT59_word2, thumbT_word1, thumbT_word2);
		if (error_thumbT < error_best) { compressed1 = thumbT_word1; compressed2 = thumbT_word2; }
		break;
	case MODE_THUMB_H:
		compressBlockTHUMB58HFast(img, 4, 4, 0, 0, thumbH58_word1, thumbH58_word2);
		decompressBlockTHUMB58H(thumbH58_word1, thumbH58_word2, imgdec, 4, 4, 0, 0);
		error_thumbH = calcBlockErrorRGB(img, imgdec, 4, 4, 0, 0);
		stuff58bits(thumbH58_word1, thumbH58_word2, thumbH_word1, thumbH_word2);
		if (error_thumbH < error_best) { compressed1 = thumbH_word1; compressed2 = thumbH_word2; }
		break;
	default:
		break;
	}
}

static inline void encode_rgb_only(const uint8 *rgb4, uint8 *dec4, uint8 *a4, int fast, int perceptual, unsigned char *out8)
{
	unsigned int b1, b2;
	if (!perceptual) {
		if (fast) compressBlockETC2Fast((uint8*)rgb4, a4, dec4, 4, 4, 0, 0, b1, b2);
		else compressBlockETC2Exhaustive((uint8*)rgb4, dec4, 4, 4, 0, 0, b1, b2);
	} else {
		if (fast) compress_rgb_fast_perceptual((uint8*)rgb4, dec4, b1, b2);
		else compressBlockETC2ExhaustivePerceptual((uint8*)rgb4, dec4, 4, 4, 0, 0, b1, b2);
	}
	out8[0] = b1 >> 24; out8[1] = b1 >> 16; out8[2] = b1 >> 8; out8[3] = b1;
	out8[4] = b2 >> 24; out8[5] = b2 >> 16; out8[6] = b2 >> 8; out8[7] = b2;
}

static inline void encode_alpha_only(uint8 *a4, int fast, unsigned char *out8)
{
	if (!fast) {
		compressBlockAlphaSlow(a4, 0, 0, 4, 4, out8);
		return;
	}
	uint8 a0 = a4[0];
	bool same = true;
	for (int i = 1; i < 16; i++) {
		if (a4[i] != a0) { same = false; break; }
	}
	if (same) {
		memcpy(out8, const_alpha_fast_cache[a0], 8);
		return;
	}
	compressBlockAlphaFast(a4, 0, 0, 4, 4, out8);
}

static void init_tables_once(void)
{
	readCompressParams();
	setupAlphaTableAndValtab();
	codec = CODEC_ETC2; format = ETC2PACKAGE_RGBA_NO_MIPMAPS; verbose = 0; formatSigned = 0;

	uint8 zrgb[48], zdec[48], za[16];
	memset(zrgb, 0, 48); memset(zdec, 0, 48);
	for (int a = 0; a < 256; a++) {
		memset(za, a, 16);
		compressBlockAlphaFast(za, 0, 0, 4, 4, const_alpha_fast_cache[a]);
	}
	memset(za, 0, 16);
	unsigned char zalpha_slow[8];
	compressBlockAlphaSlow(za, 0, 0, 4, 4, zalpha_slow);
	for (int f = 0; f < 2; f++) {
		for (int p = 0; p < 2; p++) {
			speed = f ? SPEED_FAST : SPEED_SLOW;
			metric = p ? METRIC_PERCEPTUAL : METRIC_NONPERCEPTUAL;
			int key = (f << 1) | p;
			memcpy(zero_blk_cache[key], f ? const_alpha_fast_cache[0] : zalpha_slow, 8);
			encode_rgb_only(zrgb, zdec, za, f, p, zero_blk_cache[key] + 8);
		}
	}
}

struct ConstRgbEntry {
	uint32_t key; // 0x01RRGGBB (high bit 1 = valid)
	unsigned char out8[8];
};

struct WorkCtx {
	const unsigned char *rgba;
	const unsigned char *prev_rgba;
	const unsigned char *prev_blocks;
	unsigned char *out;
	const unsigned char *zblk;
	int bw, bh, stride, fast, perceptual;
	volatile int next_row;
};

static void *worker_fn(void *arg)
{
	WorkCtx *ctx = (WorkCtx *)arg;
	const int bw = ctx->bw, bh = ctx->bh, stride = ctx->stride;
	const int fast = ctx->fast, perceptual = ctx->perceptual;
	const unsigned char *rgba = ctx->rgba;
	const unsigned char *prev_rgba = ctx->prev_rgba;
	const unsigned char *prev_blocks = ctx->prev_blocks;
	const unsigned char *zblk = ctx->zblk;
	unsigned char *out = ctx->out;
	uint8 rgb4[48], dec4[48], a4[16];
	ConstRgbEntry rgb_cache[1024];
	memset(rgb_cache, 0, sizeof(rgb_cache));

	for (;;) {
		int y0 = __sync_fetch_and_add(&ctx->next_row, 2);
		if (y0 >= bh) break;
		int y1 = y0 + 2;
		if (y1 > bh) y1 = bh;
		for (int by = y0; by < y1; by++) {
			for (int bx = 0; bx < bw; bx++) {
				size_t bidx = (size_t)by * bw + bx;
				unsigned char *dst = out + bidx * 16;
				const unsigned char *r0 = rgba + (size_t)(by * 4) * stride + bx * 16;
				const unsigned char *r1 = r0 + stride;
				const unsigned char *r2 = r1 + stride;
				const unsigned char *r3 = r2 + stride;

				uint64_t wbuf[8];
				memcpy(wbuf + 0, r0, 16);
				memcpy(wbuf + 2, r1, 16);
				memcpy(wbuf + 4, r2, 16);
				memcpy(wbuf + 6, r3, 16);
				if ((wbuf[0] | wbuf[1] | wbuf[2] | wbuf[3] | wbuf[4] | wbuf[5] | wbuf[6] | wbuf[7]) == 0) {
					memcpy(dst, zblk, 16);
					continue;
				}

				bool need_alpha = true, need_rgb = true;
				if (prev_rgba && prev_blocks) {
					const unsigned char *q0 = prev_rgba + (size_t)(by * 4) * stride + bx * 16;
					uint64_t pbuf[8];
					memcpy(pbuf + 0, q0, 16);
					memcpy(pbuf + 2, q0 + stride, 16);
					memcpy(pbuf + 4, q0 + 2 * stride, 16);
					memcpy(pbuf + 6, q0 + 3 * stride, 16);
					uint64_t diff = (wbuf[0] ^ pbuf[0]) | (wbuf[1] ^ pbuf[1]) |
					                (wbuf[2] ^ pbuf[2]) | (wbuf[3] ^ pbuf[3]) |
					                (wbuf[4] ^ pbuf[4]) | (wbuf[5] ^ pbuf[5]) |
					                (wbuf[6] ^ pbuf[6]) | (wbuf[7] ^ pbuf[7]);
					if (diff == 0) {
						memcpy(dst, prev_blocks + bidx * 16, 16);
						continue;
					}
					// Little-endian RGBA: bytes 0,1,2 are RGB (0x00FFFFFF), byte 3 is A (0xFF000000)
					need_rgb   = (diff & 0x00FFFFFF00FFFFFFULL) != 0;
					need_alpha = (diff & 0xFF000000FF000000ULL) != 0;
					if (!need_alpha) memcpy(dst, prev_blocks + bidx * 16, 8);
					if (!need_rgb)   memcpy(dst + 8, prev_blocks + bidx * 16 + 8, 8);
				}

				const unsigned char *rows[4] = {r0, r1, r2, r3};
				if (need_alpha || (!perceptual && fast)) {
					for (int ry = 0; ry < 4; ry++) {
						const unsigned char *rp = rows[ry];
						a4[ry * 4 + 0] = rp[3];
						a4[ry * 4 + 1] = rp[7];
						a4[ry * 4 + 2] = rp[11];
						a4[ry * 4 + 3] = rp[15];
					}
				}
				if (need_alpha) {
					encode_alpha_only(a4, fast, dst);
				}
				if (need_rgb) {
					uint8 r_0 = r0[0], g_0 = r0[1], b_0 = r0[2];
					bool const_rgb = true;
					for (int ry = 0; ry < 4; ry++) {
						const unsigned char *rp = rows[ry];
						for (int rx = 0; rx < 4; rx++) {
							int idx = ry * 4 + rx;
							uint8 cr = rp[4 * rx + 0], cg = rp[4 * rx + 1], cb = rp[4 * rx + 2];
							rgb4[3 * idx + 0] = cr;
							rgb4[3 * idx + 1] = cg;
							rgb4[3 * idx + 2] = cb;
							if (cr != r_0 || cg != g_0 || cb != b_0) const_rgb = false;
						}
					}
					if (const_rgb && (perceptual || !fast)) {
						uint32_t ckey = 0x01000000U | ((uint32_t)r_0 << 16) | ((uint32_t)g_0 << 8) | b_0;
						unsigned slot = ((uint32_t)r_0 * 73U + (uint32_t)g_0 * 151U + (uint32_t)b_0 * 233U) & 1023U;
						if (rgb_cache[slot].key == ckey) {
							memcpy(dst + 8, rgb_cache[slot].out8, 8);
						} else {
							encode_rgb_only(rgb4, dec4, a4, fast, perceptual, dst + 8);
							rgb_cache[slot].key = ckey;
							memcpy(rgb_cache[slot].out8, dst + 8, 8);
						}
					} else {
						encode_rgb_only(rgb4, dec4, a4, fast, perceptual, dst + 8);
					}
				}
			}
		}
	}
	return NULL;
}

extern "C" int etcpack_rgba8_fast(
	const unsigned char *rgba, int W, int H, unsigned char *out,
	int fast, int perceptual, int nthreads,
	const unsigned char *prev_rgba, const unsigned char *prev_blocks)
{
	if (W % 4 || H % 4) return 2;
	pthread_once(&etcpack_init_once, init_tables_once);
	// Note: ETCPACK globals speed/metric are only consulted in non-default paths; set for legacy compatibility.
	speed = fast ? SPEED_FAST : SPEED_SLOW;
	metric = perceptual ? METRIC_PERCEPTUAL : METRIC_NONPERCEPTUAL;

	int key = ((fast ? 1 : 0) << 1) | (perceptual ? 1 : 0);
	int bh = H / 4, bw = W / 4;
	if (nthreads <= 0) {
		long nc = sysconf(_SC_NPROCESSORS_ONLN);
		nthreads = nc > 2 ? (int)(nc - 2) : 1;
	}
	if (nthreads > bh / 2) nthreads = bh / 2;
	if (nthreads < 1) nthreads = 1;
	if (nthreads > 64) nthreads = 64;

	WorkCtx ctx;
	ctx.rgba = rgba; ctx.prev_rgba = prev_rgba; ctx.prev_blocks = prev_blocks;
	ctx.out = out; ctx.zblk = zero_blk_cache[key];
	ctx.bw = bw; ctx.bh = bh; ctx.stride = W * 4;
	ctx.fast = fast; ctx.perceptual = perceptual; ctx.next_row = 0;

	if (nthreads == 1) {
		worker_fn(&ctx);
	} else {
		pthread_t th[64];
		for (int i = 0; i < nthreads - 1; i++) pthread_create(&th[i], NULL, worker_fn, &ctx);
		worker_fn(&ctx);
		for (int i = 0; i < nthreads - 1; i++) pthread_join(th[i], NULL);
	}
	return 0;
}

extern "C" int etcpack_rgba8(const unsigned char *rgba, int W, int H, unsigned char *out, int fast, int perceptual)
{
	return etcpack_rgba8_fast(rgba, W, H, out, fast, perceptual, 1, NULL, NULL);
}
