/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Cambyses — Context-Aware Migration Balancer Yielding Scored Entity Selection
 *
 * Scored migration selection for CFS load balancer Pull path.
 * See IMPLEMENTATION.md for design details.
 */
#ifndef _KERNEL_SCHED_CAMBYSES_H
#define _KERNEL_SCHED_CAMBYSES_H

#include <linux/types.h>
#include <linux/jump_label.h>

#ifdef CONFIG_SCHED_CAMBYSES

/*
 * Candidate entry for scored migration selection.
 * Stored on stack during detach_tasks_cambyses().
 *
 * Size: 16 bytes (pointer 8 + score 2 + padding 6)
 * Stack usage: 32 candidates = 512 bytes, 8 candidates = 128 bytes
 */
struct cambyses_candidate {
	struct task_struct	*p;
	s16			score;
};

/* Static key for zero-cost runtime disable (NOP patching) */
extern struct static_key_true sched_cambyses;

/* sysctl tunable weights (0–3, 2bit) */
extern u8 sysctl_cambyses_w0;	/* cache coldness weight (default: 1) */
extern u8 sysctl_cambyses_w1;	/* load contribution weight (default: 2) */
extern u8 sysctl_cambyses_w2;	/* vol switch ratio weight (default: 1) */
extern u8 sysctl_cambyses_w3;	/* wakee penalty weight (default: 1) */

/*
 * Scalar sort — separate TU (cambyses_sort.c), no SIMD flags.
 */
void sort_candidates_cambyses(struct cambyses_candidate *cands, int nr_cands);

/*
 * SIMD batch scoring — separate TUs to prevent auto-vectorization
 * contamination.  Each file is compiled with its own ISA flags.
 *
 * Scoring uses PMADDUBSW (u8 features × s8 weights → s16 scores).
 * Features are packed in AoS layout: [F0,F1,F2,F3] per candidate,
 * 4 bytes/candidate, aligned to xmm/ymm boundaries.
 */
#ifdef CONFIG_SCHED_CAMBYSES_SIMD

#define CAMBYSES_SIMD_THRESHOLD		8
#define CAMBYSES_NR_FEATURES		4

/* Static keys for ISA dispatch — enabled at boot based on CPUID */
extern struct static_key_false cambyses_has_avx2;
extern struct static_key_false cambyses_has_ssse3;

/* SSSE3: 4 candidates/xmm, PMADDUBSW + PHADDW */
void cambyses_simd_score_ssse3(const u8 *features_packed,
				s16 *scores, int nr_cands);

/* AVX2: 8 candidates/ymm, VPMADDUBSW + VPHADDW */
void cambyses_simd_score_avx2(const u8 *features_packed,
			       s16 *scores, int nr_cands);

/*
 * SIMD vector type definitions — only available in TUs compiled
 * with the corresponding ISA flags.
 *
 * Following Nap's pattern: GCC vector extensions + __builtin_ia32_*.
 * <immintrin.h> is a userspace header and cannot be used in kernel.
 */
#ifdef __SSSE3__
typedef short v8hi  __attribute__((__vector_size__(16)));  /* 8 × s16 */
typedef char  v16qi __attribute__((__vector_size__(16)));   /* 16 × s8/u8 */
#endif

#ifdef __AVX2__
typedef short v16hi __attribute__((__vector_size__(32)));   /* 16 × s16 */
typedef char  v32qi __attribute__((__vector_size__(32)));    /* 32 × s8/u8 */
typedef int   v8si  __attribute__((__vector_size__(32)));    /* 8 × s32 */
typedef long long v4di __attribute__((__vector_size__(32))); /* 4 × s64 */
#endif

/*
 * AVX2 bitonic sort — 32 elements via s32 packing.
 *
 * Input: array of s32 packed as ((s16 score) << 16) | (u16 index).
 * Sorting by s32 naturally sorts by score with index tracking for free.
 * Must be called within kernel_fpu_begin/end.  Array must be exactly
 * 32 elements, 32-byte aligned, padded with INT_MIN for unused slots.
 */
void cambyses_simd_sort_avx2(s32 *packed);

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */

#endif /* CONFIG_SCHED_CAMBYSES */
#endif /* _KERNEL_SCHED_CAMBYSES_H */
