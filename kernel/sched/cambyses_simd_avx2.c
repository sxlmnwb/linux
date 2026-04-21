// SPDX-License-Identifier: GPL-2.0
/*
 * cambyses_simd_avx2.c — AVX2 batch scoring for Cambyses
 *
 * Computes migration scores for up to 32 candidates using VPMADDUBSW.
 * 8 candidates per ymm register, 4 features (u8) × 4 weights (s8).
 *
 * Must be called within kernel_fpu_begin/end.
 * Compiled with: CFLAGS += -mavx -mavx2
 *
 * Separate TU to prevent auto-vectorization contamination of scalar code.
 */

#include "cambyses.h"
#include <linux/types.h>

#ifdef CONFIG_SCHED_CAMBYSES_SIMD

/*
 * Score up to 32 candidates using AVX2 (8 candidates per ymm pass).
 *
 * @features_packed: AoS layout [F0,F1,F2,F3] × nr_cands, padded to
 *                   multiple of 32 bytes (8 candidates).
 * @scores:          Output s16 scores, one per candidate.
 * @nr_cands:        Number of candidates (9..32).
 */
void cambyses_simd_score_avx2(const u8 *features_packed,
			       s16 *scores, int nr_cands)
{
	int i;
	/* Round up to multiple of 8 for ymm processing */
	int nr_rounds = (nr_cands + 7) & ~7;

	for (i = 0; i < nr_rounds; i += 8) {
		v32qi feat;
		v32qi wt;
		v16hi partial;
		v16hi summed;

		/*
		 * Load 8 candidates' features: 32 bytes = 1 ymm
		 * Each candidate = [F0, F1, F2, F3] = 4 bytes
		 */
		feat = *(const v32qi *)&features_packed[i * 4];

		/* Broadcast weight pattern: [w0, w1, w2, -w3] × 8 */
		{
			s8 w0 = (s8)sysctl_cambyses_w0;
			s8 w1 = (s8)sysctl_cambyses_w1;
			s8 w2 = (s8)sysctl_cambyses_w2;
			s8 w3 = -(s8)sysctl_cambyses_w3;

			wt = (v32qi){w0, w1, w2, w3, w0, w1, w2, w3,
				     w0, w1, w2, w3, w0, w1, w2, w3,
				     w0, w1, w2, w3, w0, w1, w2, w3,
				     w0, w1, w2, w3, w0, w1, w2, w3};
		}

		/*
		 * VPMADDUBSW: u8 × s8 → s16 pairwise add
		 * 8 candidates → 16 × s16 partial sums (2 per candidate)
		 */
		partial = __builtin_ia32_pmaddubsw256(feat, wt);

		/*
		 * VPHADDW: horizontal add adjacent s16 pairs
		 * Result layout (AVX2 operates per 128-bit lane):
		 *   [score₀, score₁, score₂, score₃,
		 *    score₀, score₁, score₂, score₃,
		 *    score₄, score₅, score₆, score₇,
		 *    score₄, score₅, score₆, score₇]
		 *
		 * AVX2 VPHADDW operates within each 128-bit lane:
		 *   low 128:  hadd(partial[0:7], partial[0:7])
		 *   high 128: hadd(partial[8:15], partial[8:15])
		 *
		 * Scores for candidates 0-3 are in low lane [0:3],
		 * scores for candidates 4-7 are in high lane [8:11].
		 */
		summed = __builtin_ia32_phaddw256(partial, partial);

		/*
		 * Extract scores from lane-interleaved layout.
		 * Low lane [0..3] = candidates 0-3
		 * High lane [8..11] = candidates 4-7
		 */
		scores[i]     = summed[0];
		scores[i + 1] = summed[1];
		scores[i + 2] = summed[2];
		scores[i + 3] = summed[3];
		scores[i + 4] = summed[8];
		scores[i + 5] = summed[9];
		scores[i + 6] = summed[10];
		scores[i + 7] = summed[11];
	}
}

/*
 * AVX2 bitonic sort for 32 × s32 packed values (descending).
 *
 * Packed format: ((s32)score << 16) | (u16)index
 * VPMINSD/VPMAXSD on s32 naturally sorts by score, preserving indices.
 *
 * 32 elements in 4 ymm registers (R0-R3), 5-phase bitonic network,
 * 15 steps, ~204 SIMD operations total.
 *
 * Comparator primitives:
 *   d1: VPSHUFD 0xB1    — swap adjacent dwords
 *   d2: VPSHUFD 0x4E    — swap dword pairs
 *   d4: VPERM2I128 0x01 — swap 128-bit lanes
 *   d8/d16: inter-register VPMINSD/VPMAXSD
 *
 * Direction selection via VPBLENDD blend masks.
 * Blend semantics: pblendd(hi, lo, mask) — bit=0 takes hi, bit=1 takes lo.
 *
 * Must be called within kernel_fpu_begin/end.
 * @packed: 32 × s32 array, 32-byte aligned, padded with INT_MIN.
 */

/* Signed 32-bit min/max — guaranteed single VPMINSD/VPMAXSD on GCC and clang */
static __always_inline v8si v8si_min(v8si a, v8si b)
{
	v8si r;

	asm("vpminsd %2, %1, %0" : "=x"(r) : "x"(a), "x"(b));
	return r;
}

static __always_inline v8si v8si_max(v8si a, v8si b)
{
	v8si r;

	asm("vpmaxsd %2, %1, %0" : "=x"(r) : "x"(a), "x"(b));
	return r;
}

/* Intra-register compare-and-swap at distance 1 */
#define CAS_D1(r, blend) do {						\
	v8si _s = __builtin_ia32_pshufd256(r, 0xB1);			\
	v8si _lo = v8si_min(r, _s);					\
	v8si _hi = v8si_max(r, _s);					\
	r = __builtin_ia32_pblendd256(_hi, _lo, blend);			\
} while (0)

/* Intra-register compare-and-swap at distance 2 */
#define CAS_D2(r, blend) do {						\
	v8si _s = __builtin_ia32_pshufd256(r, 0x4E);			\
	v8si _lo = v8si_min(r, _s);					\
	v8si _hi = v8si_max(r, _s);					\
	r = __builtin_ia32_pblendd256(_hi, _lo, blend);			\
} while (0)

/* Intra-register compare-and-swap at distance 4 (cross 128-bit lane) */
#define CAS_D4(r, blend) do {						\
	v8si _s = (v8si)__builtin_ia32_permti256((v4di)(r),		\
						  (v4di)(r), 0x01);	\
	v8si _lo = v8si_min(r, _s);					\
	v8si _hi = v8si_max(r, _s);					\
	r = __builtin_ia32_pblendd256(_hi, _lo, blend);			\
} while (0)

/* Inter-register compare-and-swap: descending (lower reg gets max) */
#define CAS_REG_DESC(ra, rb) do {					\
	v8si _lo = v8si_min(ra, rb);					\
	v8si _hi = v8si_max(ra, rb);					\
	ra = _hi; rb = _lo;						\
} while (0)

/* Inter-register compare-and-swap: ascending (lower reg gets min) */
#define CAS_REG_ASC(ra, rb) do {					\
	v8si _lo = v8si_min(ra, rb);					\
	v8si _hi = v8si_max(ra, rb);					\
	ra = _lo; rb = _hi;						\
} while (0)

void cambyses_simd_sort_avx2(s32 *packed)
{
	v8si r0 = *(v8si *)&packed[0];
	v8si r1 = *(v8si *)&packed[8];
	v8si r2 = *(v8si *)&packed[16];
	v8si r3 = *(v8si *)&packed[24];

	/*
	 * Phase 1 (k=1): Sort pairs, alternating desc/asc.
	 * Blend 0x66: desc at even pairs (0,1)(4,5), asc at odd (2,3)(6,7).
	 */
	CAS_D1(r0, 0x66); CAS_D1(r1, 0x66);
	CAS_D1(r2, 0x66); CAS_D1(r3, 0x66);

	/*
	 * Phase 2 (k=2): Sort groups of 4, alternating desc/asc.
	 * Step 1: d=2, blend 0x3C (desc block 0-3, asc block 4-7).
	 * Step 2: d=1, blend 0x5A (cleanup within each 4-element block).
	 */
	CAS_D2(r0, 0x3C); CAS_D2(r1, 0x3C);
	CAS_D2(r2, 0x3C); CAS_D2(r3, 0x3C);
	CAS_D1(r0, 0x5A); CAS_D1(r1, 0x5A);
	CAS_D1(r2, 0x5A); CAS_D1(r3, 0x5A);

	/*
	 * Phase 3 (k=3): Sort groups of 8 (one register each).
	 * R0/R2 descending, R1/R3 ascending.
	 * Step 1: d=4, desc=0xF0, asc=0x0F.
	 * Step 2: d=2, desc=0xCC, asc=0x33.
	 * Step 3: d=1, desc=0xAA, asc=0x55.
	 */
	CAS_D4(r0, 0xF0); CAS_D4(r1, 0x0F);
	CAS_D4(r2, 0xF0); CAS_D4(r3, 0x0F);
	CAS_D2(r0, 0xCC); CAS_D2(r1, 0x33);
	CAS_D2(r2, 0xCC); CAS_D2(r3, 0x33);
	CAS_D1(r0, 0xAA); CAS_D1(r1, 0x55);
	CAS_D1(r2, 0xAA); CAS_D1(r3, 0x55);

	/*
	 * Phase 4 (k=4): Sort groups of 16.
	 * R0+R1 descending, R2+R3 ascending.
	 * Step 1: d=8, inter-register.
	 * Steps 2-4: d=4/2/1 within each register.
	 */
	CAS_REG_DESC(r0, r1);
	CAS_REG_ASC(r2, r3);
	CAS_D4(r0, 0xF0); CAS_D4(r1, 0xF0);
	CAS_D4(r2, 0x0F); CAS_D4(r3, 0x0F);
	CAS_D2(r0, 0xCC); CAS_D2(r1, 0xCC);
	CAS_D2(r2, 0x33); CAS_D2(r3, 0x33);
	CAS_D1(r0, 0xAA); CAS_D1(r1, 0xAA);
	CAS_D1(r2, 0x55); CAS_D1(r3, 0x55);

	/*
	 * Phase 5 (k=5): Final merge — all descending.
	 * Step 1: d=16, inter-register (R0↔R2, R1↔R3).
	 * Step 2: d=8, inter-register (R0↔R1, R2↔R3).
	 * Steps 3-5: d=4/2/1 within each register, all descending.
	 */
	CAS_REG_DESC(r0, r2);
	CAS_REG_DESC(r1, r3);
	CAS_REG_DESC(r0, r1);
	CAS_REG_DESC(r2, r3);
	CAS_D4(r0, 0xF0); CAS_D4(r1, 0xF0);
	CAS_D4(r2, 0xF0); CAS_D4(r3, 0xF0);
	CAS_D2(r0, 0xCC); CAS_D2(r1, 0xCC);
	CAS_D2(r2, 0xCC); CAS_D2(r3, 0xCC);
	CAS_D1(r0, 0xAA); CAS_D1(r1, 0xAA);
	CAS_D1(r2, 0xAA); CAS_D1(r3, 0xAA);

	/* Store sorted results */
	*(v8si *)&packed[0] = r0;
	*(v8si *)&packed[8] = r1;
	*(v8si *)&packed[16] = r2;
	*(v8si *)&packed[24] = r3;
}

#undef CAS_D1
#undef CAS_D2
#undef CAS_D4
#undef CAS_REG_DESC
#undef CAS_REG_ASC

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
