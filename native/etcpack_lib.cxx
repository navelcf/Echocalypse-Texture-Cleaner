// Shared-library entry around Ericsson ETCPACK v2.74 (not included here; build_etcpack.sh fetches it).
// Exports:
//   etcpack_rgba8(rgba, W, H, out, fast, perceptual)                         -- single-threaded (legacy)
//   etcpack_rgba8_fast(rgba, W, H, out, fast, perceptual, nthreads, prev_rgba, prev_blocks)
//     -- native pthreads + stack 4x4 buffers + zero-block skip + dirty-block skip (100% byte-identical)
#include <sys/timeb.h>
#define _timeb timeb
#define _ftime ftime
#include <stdlib.h>
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

static int etcpack_inited = 0;
static unsigned char zero_blk_cache[4][16];
static int zero_blk_ready[4] = {0, 0, 0, 0};

static inline void encode_one_block(const uint8 *rgb4, uint8 *dec4, uint8 *a4, int fast, int perceptual, unsigned char *out16)
{
	unsigned int b1, b2;
	if (!perceptual) {
		if (fast) compressBlockETC2Fast((uint8*)rgb4, a4, dec4, 4, 4, 0, 0, b1, b2);
		else compressBlockETC2Exhaustive((uint8*)rgb4, dec4, 4, 4, 0, 0, b1, b2);
	} else {
		if (fast) compressBlockETC2FastPerceptual((uint8*)rgb4, dec4, 4, 4, 0, 0, b1, b2);
		else compressBlockETC2ExhaustivePerceptual((uint8*)rgb4, dec4, 4, 4, 0, 0, b1, b2);
	}
	if (!fast) compressBlockAlphaSlow(a4, 0, 0, 4, 4, out16);
	else compressBlockAlphaFast(a4, 0, 0, 4, 4, out16);
	out16[8] = b1 >> 24; out16[9] = b1 >> 16; out16[10] = b1 >> 8; out16[11] = b1;
	out16[12] = b2 >> 24; out16[13] = b2 >> 16; out16[14] = b2 >> 8; out16[15] = b2;
}

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

	for (;;) {
		int y0 = __sync_fetch_and_add(&ctx->next_row, 4);
		if (y0 >= bh) break;
		int y1 = y0 + 4;
		if (y1 > bh) y1 = bh;
		for (int by = y0; by < y1; by++) {
			for (int bx = 0; bx < bw; bx++) {
				unsigned char *dst = out + ((size_t)by * bw + bx) * 16;
				const unsigned char *r0 = rgba + (size_t)(by * 4) * stride + bx * 16;
				const unsigned char *r1 = r0 + stride;
				const unsigned char *r2 = r1 + stride;
				const unsigned char *r3 = r2 + stride;

				const unsigned long long *p0 = (const unsigned long long *)r0;
				const unsigned long long *p1 = (const unsigned long long *)r1;
				const unsigned long long *p2 = (const unsigned long long *)r2;
				const unsigned long long *p3 = (const unsigned long long *)r3;
				if ((p0[0] | p0[1] | p1[0] | p1[1] | p2[0] | p2[1] | p3[0] | p3[1]) == 0) {
					memcpy(dst, zblk, 16);
					continue;
				}
				if (prev_rgba && prev_blocks) {
					const unsigned char *q0 = prev_rgba + (size_t)(by * 4) * stride + bx * 16;
					if (memcmp(r0, q0, 16) == 0 &&
					    memcmp(r1, q0 + stride, 16) == 0 &&
					    memcmp(r2, q0 + 2 * stride, 16) == 0 &&
					    memcmp(r3, q0 + 3 * stride, 16) == 0) {
						memcpy(dst, prev_blocks + ((size_t)by * bw + bx) * 16, 16);
						continue;
					}
				}
				const unsigned char *rows[4] = {r0, r1, r2, r3};
				for (int ry = 0; ry < 4; ry++) {
					const unsigned char *rp = rows[ry];
					for (int rx = 0; rx < 4; rx++) {
						int idx = ry * 4 + rx;
						rgb4[3 * idx + 0] = rp[4 * rx + 0];
						rgb4[3 * idx + 1] = rp[4 * rx + 1];
						rgb4[3 * idx + 2] = rp[4 * rx + 2];
						a4[idx] = rp[4 * rx + 3];
					}
				}
				encode_one_block(rgb4, dec4, a4, fast, perceptual, dst);
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
	if (!etcpack_inited) {
		readCompressParams();
		setupAlphaTableAndValtab();
		codec = CODEC_ETC2; format = ETC2PACKAGE_RGBA_NO_MIPMAPS; verbose = 0; formatSigned = 0;
		etcpack_inited = 1;
	}
	speed = fast ? SPEED_FAST : SPEED_SLOW;
	metric = perceptual ? METRIC_PERCEPTUAL : METRIC_NONPERCEPTUAL;

	int key = ((fast ? 1 : 0) << 1) | (perceptual ? 1 : 0);
	if (!zero_blk_ready[key]) {
		uint8 zrgb[48], zdec[48], za[16];
		memset(zrgb, 0, 48); memset(zdec, 0, 48); memset(za, 0, 16);
		encode_one_block(zrgb, zdec, za, fast, perceptual, zero_blk_cache[key]);
		zero_blk_ready[key] = 1;
	}

	int bh = H / 4, bw = W / 4;
	if (nthreads <= 0) {
		long nc = sysconf(_SC_NPROCESSORS_ONLN);
		nthreads = nc > 2 ? (int)(nc - 2) : 1;
	}
	if (nthreads > bh / 4) nthreads = bh / 4;
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
