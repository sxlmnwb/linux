// SPDX-License-Identifier: GPL-2.0
/*
 * cambyses_simd_ssse3.c — SSSE3 batch scoring for Cambyses
 *
 * Computes migration scores for up to 32 candidates using PMADDUBSW.
 * 4 candidates per xmm register, 4 features (u8) × 4 weights (s8).
 *
 * Must be called within kernel_fpu_begin/end.
 * Compiled with: CFLAGS += -mssse3
 *
 * Separate TU to prevent auto-vectorization contamination of scalar code.
 */

#include "cambyses.h"
#include <linux/types.h>

#ifdef CONFIG_SCHED_CAMBYSES_SIMD

/*
 * PMADDUBSW: Multiply unsigned bytes by signed bytes, then add
 * adjacent pairs to produce signed words.
 *
 *   For each i in 0..7:
 *     dst[i] = saturate_s16(src1[2i] * src2[2i] + src1[2i+1] * src2[2i+1])
 *
 * With 4 features packed as [F0,F1,F2,F3] per candidate:
 *   After PMADDUBSW: [w0*F0+w1*F1, w2*F2+(-w3)*F3] per candidate (2 × s16)
 *   After PHADDW:    [score] per candidate (1 × s16)
 */

/*
 * Score up to 32 candidates using SSSE3 (4 candidates per xmm pass).
 *
 * @features_packed: AoS layout [F0,F1,F2,F3] × nr_cands, padded to
 *                   multiple of 16 bytes (4 candidates).
 * @scores:          Output s16 scores, one per candidate.
 * @nr_cands:        Number of candidates (9..32).
 * @weights_packed:  Pre-broadcast weight vector [w0,w1,w2,-w3] × 4.
 */
void cambyses_simd_score_ssse3(const u8 *features_packed,
				s16 *scores, int nr_cands)
{
	int i;
	/* Round up to multiple of 4 for xmm processing */
	int nr_rounds = (nr_cands + 3) & ~3;

	for (i = 0; i < nr_rounds; i += 4) {
		v16qi feat;
		v16qi wt;
		v8hi partial;
		v8hi summed;

		/*
		 * Load 4 candidates' features: 16 bytes = 1 xmm
		 * [F0₀ F1₀ F2₀ F3₀ | F0₁ F1₁ F2₁ F3₁ |
		 *  F0₂ F1₂ F2₂ F3₂ | F0₃ F1₃ F2₃ F3₃]
		 */
		feat = *(const v16qi *)&features_packed[i * 4];

		/*
		 * Broadcast weight pattern: [w0, w1, w2, -w3] × 4
		 * Loaded from the first 16 bytes of features_packed's
		 * companion weight array — but we construct it inline
		 * since weights are runtime sysctl values.
		 *
		 * We receive the weight vector pre-packed at the
		 * 16-byte-aligned area right after feature data.
		 * Actually, construct it from sysctl globals directly.
		 */
		{
			s8 w0 = (s8)sysctl_cambyses_w0;
			s8 w1 = (s8)sysctl_cambyses_w1;
			s8 w2 = (s8)sysctl_cambyses_w2;
			s8 w3 = -(s8)sysctl_cambyses_w3;

			wt = (v16qi){w0, w1, w2, w3, w0, w1, w2, w3,
				     w0, w1, w2, w3, w0, w1, w2, w3};
		}

		/*
		 * PMADDUBSW: u8 × s8 → s16 pairwise add
		 * Result: [w0*F0₀+w1*F1₀, w2*F2₀+w3'*F3₀,
		 *          w0*F0₁+w1*F1₁, w2*F2₁+w3'*F3₁,
		 *          w0*F0₂+w1*F1₂, w2*F2₂+w3'*F3₂,
		 *          w0*F0₃+w1*F1₃, w2*F2₃+w3'*F3₃]
		 */
		partial = __builtin_ia32_pmaddubsw128(feat, wt);

		/*
		 * PHADDW: horizontal add adjacent pairs
		 * Result: [score₀, score₁, score₂, score₃,
		 *          score₀, score₁, score₂, score₃]
		 * We only need the lower 4 words (indices 0-3).
		 */
		summed = __builtin_ia32_phaddw128(partial, partial);

		/* Store 4 scores — extract lower 4 s16 values */
		scores[i]     = summed[0];
		scores[i + 1] = summed[1];
		scores[i + 2] = summed[2];
		scores[i + 3] = summed[3];
	}
}

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */
