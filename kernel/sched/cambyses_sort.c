// SPDX-License-Identifier: GPL-2.0
/*
 * Cambyses — Bitonic sort networks for candidate ranking
 *
 * Separate translation unit to prevent auto-vectorization contamination.
 * This file is compiled WITHOUT SIMD flags (-msse2, -mavx2).
 * Future SIMD sort variants live in cambyses_simd_{sse2,avx2}.c
 * with their own CC_FLAGS_FPU.
 */

#include "cambyses.h"

#include <linux/types.h>
#include <linux/limits.h>

#ifdef CONFIG_SCHED_CAMBYSES

/*
 * Compare-and-swap: ensure cands[a].score >= cands[b].score (descending).
 * Branch-free on most architectures via CMOV.
 */
#define CAMBYSES_CAS(a, b) do {					\
	if (cands[a].score < cands[b].score) {			\
		struct cambyses_candidate _tmp = cands[a];	\
		cands[a] = cands[b];				\
		cands[b] = _tmp;				\
	}							\
} while (0)

/*
 * Bitonic sort network for 8 elements — 19 comparators, 6 stages.
 * Sorts in descending order (highest score first).
 */
static void bitonic_sort_8(struct cambyses_candidate *cands)
{
	/* Stage 1: pairs */
	CAMBYSES_CAS(0, 1); CAMBYSES_CAS(2, 3);
	CAMBYSES_CAS(4, 5); CAMBYSES_CAS(6, 7);
	/* Stage 2: quads */
	CAMBYSES_CAS(0, 2); CAMBYSES_CAS(1, 3);
	CAMBYSES_CAS(4, 6); CAMBYSES_CAS(5, 7);
	/* Stage 3 */
	CAMBYSES_CAS(1, 2); CAMBYSES_CAS(5, 6);
	/* Stage 4: octets */
	CAMBYSES_CAS(0, 4); CAMBYSES_CAS(1, 5);
	CAMBYSES_CAS(2, 6); CAMBYSES_CAS(3, 7);
	/* Stage 5 */
	CAMBYSES_CAS(2, 4); CAMBYSES_CAS(3, 5);
	/* Stage 6 */
	CAMBYSES_CAS(1, 2); CAMBYSES_CAS(3, 4);
	CAMBYSES_CAS(5, 6);
}

/*
 * Bitonic sort network for 16 elements — 63 comparators.
 */
static void bitonic_sort_16(struct cambyses_candidate *cands)
{
	/* Sort each half of 8 */
	bitonic_sort_8(cands);
	bitonic_sort_8(cands + 8);

	/* Reverse the second half's order for bitonic merge */
	CAMBYSES_CAS(0, 15); CAMBYSES_CAS(1, 14);
	CAMBYSES_CAS(2, 13); CAMBYSES_CAS(3, 12);
	CAMBYSES_CAS(4, 11); CAMBYSES_CAS(5, 10);
	CAMBYSES_CAS(6, 9);  CAMBYSES_CAS(7, 8);

	/* Merge stage: size 8 */
	CAMBYSES_CAS(0, 8);  CAMBYSES_CAS(1, 9);
	CAMBYSES_CAS(2, 10); CAMBYSES_CAS(3, 11);
	CAMBYSES_CAS(4, 12); CAMBYSES_CAS(5, 13);
	CAMBYSES_CAS(6, 14); CAMBYSES_CAS(7, 15);

	/* Merge stage: size 4 */
	CAMBYSES_CAS(0, 4);  CAMBYSES_CAS(1, 5);
	CAMBYSES_CAS(2, 6);  CAMBYSES_CAS(3, 7);
	CAMBYSES_CAS(8, 12); CAMBYSES_CAS(9, 13);
	CAMBYSES_CAS(10, 14); CAMBYSES_CAS(11, 15);

	/* Merge stage: size 2 */
	CAMBYSES_CAS(0, 2);  CAMBYSES_CAS(1, 3);
	CAMBYSES_CAS(4, 6);  CAMBYSES_CAS(5, 7);
	CAMBYSES_CAS(8, 10); CAMBYSES_CAS(9, 11);
	CAMBYSES_CAS(12, 14); CAMBYSES_CAS(13, 15);

	/* Merge stage: size 1 */
	CAMBYSES_CAS(0, 1);  CAMBYSES_CAS(2, 3);
	CAMBYSES_CAS(4, 5);  CAMBYSES_CAS(6, 7);
	CAMBYSES_CAS(8, 9);  CAMBYSES_CAS(10, 11);
	CAMBYSES_CAS(12, 13); CAMBYSES_CAS(14, 15);
}

/*
 * Bitonic sort network for 32 elements — 159 comparators.
 */
static void bitonic_sort_32(struct cambyses_candidate *cands)
{
	/* Sort each half of 16 */
	bitonic_sort_16(cands);
	bitonic_sort_16(cands + 16);

	/* Reverse the second half's order for bitonic merge */
	CAMBYSES_CAS(0, 31); CAMBYSES_CAS(1, 30);
	CAMBYSES_CAS(2, 29); CAMBYSES_CAS(3, 28);
	CAMBYSES_CAS(4, 27); CAMBYSES_CAS(5, 26);
	CAMBYSES_CAS(6, 25); CAMBYSES_CAS(7, 24);
	CAMBYSES_CAS(8, 23); CAMBYSES_CAS(9, 22);
	CAMBYSES_CAS(10, 21); CAMBYSES_CAS(11, 20);
	CAMBYSES_CAS(12, 19); CAMBYSES_CAS(13, 18);
	CAMBYSES_CAS(14, 17); CAMBYSES_CAS(15, 16);

	/* Merge stage: size 16 */
	CAMBYSES_CAS(0, 16); CAMBYSES_CAS(1, 17);
	CAMBYSES_CAS(2, 18); CAMBYSES_CAS(3, 19);
	CAMBYSES_CAS(4, 20); CAMBYSES_CAS(5, 21);
	CAMBYSES_CAS(6, 22); CAMBYSES_CAS(7, 23);
	CAMBYSES_CAS(8, 24); CAMBYSES_CAS(9, 25);
	CAMBYSES_CAS(10, 26); CAMBYSES_CAS(11, 27);
	CAMBYSES_CAS(12, 28); CAMBYSES_CAS(13, 29);
	CAMBYSES_CAS(14, 30); CAMBYSES_CAS(15, 31);

	/* Merge stage: size 8 */
	CAMBYSES_CAS(0, 8);  CAMBYSES_CAS(1, 9);
	CAMBYSES_CAS(2, 10); CAMBYSES_CAS(3, 11);
	CAMBYSES_CAS(4, 12); CAMBYSES_CAS(5, 13);
	CAMBYSES_CAS(6, 14); CAMBYSES_CAS(7, 15);
	CAMBYSES_CAS(16, 24); CAMBYSES_CAS(17, 25);
	CAMBYSES_CAS(18, 26); CAMBYSES_CAS(19, 27);
	CAMBYSES_CAS(20, 28); CAMBYSES_CAS(21, 29);
	CAMBYSES_CAS(22, 30); CAMBYSES_CAS(23, 31);

	/* Merge stage: size 4 */
	CAMBYSES_CAS(0, 4);  CAMBYSES_CAS(1, 5);
	CAMBYSES_CAS(2, 6);  CAMBYSES_CAS(3, 7);
	CAMBYSES_CAS(8, 12); CAMBYSES_CAS(9, 13);
	CAMBYSES_CAS(10, 14); CAMBYSES_CAS(11, 15);
	CAMBYSES_CAS(16, 20); CAMBYSES_CAS(17, 21);
	CAMBYSES_CAS(18, 22); CAMBYSES_CAS(19, 23);
	CAMBYSES_CAS(24, 28); CAMBYSES_CAS(25, 29);
	CAMBYSES_CAS(26, 30); CAMBYSES_CAS(27, 31);

	/* Merge stage: size 2 */
	CAMBYSES_CAS(0, 2);  CAMBYSES_CAS(1, 3);
	CAMBYSES_CAS(4, 6);  CAMBYSES_CAS(5, 7);
	CAMBYSES_CAS(8, 10); CAMBYSES_CAS(9, 11);
	CAMBYSES_CAS(12, 14); CAMBYSES_CAS(13, 15);
	CAMBYSES_CAS(16, 18); CAMBYSES_CAS(17, 19);
	CAMBYSES_CAS(20, 22); CAMBYSES_CAS(21, 23);
	CAMBYSES_CAS(24, 26); CAMBYSES_CAS(25, 27);
	CAMBYSES_CAS(28, 30); CAMBYSES_CAS(29, 31);

	/* Merge stage: size 1 */
	CAMBYSES_CAS(0, 1);  CAMBYSES_CAS(2, 3);
	CAMBYSES_CAS(4, 5);  CAMBYSES_CAS(6, 7);
	CAMBYSES_CAS(8, 9);  CAMBYSES_CAS(10, 11);
	CAMBYSES_CAS(12, 13); CAMBYSES_CAS(14, 15);
	CAMBYSES_CAS(16, 17); CAMBYSES_CAS(18, 19);
	CAMBYSES_CAS(20, 21); CAMBYSES_CAS(22, 23);
	CAMBYSES_CAS(24, 25); CAMBYSES_CAS(26, 27);
	CAMBYSES_CAS(28, 29); CAMBYSES_CAS(30, 31);
}

#undef CAMBYSES_CAS

/*
 * sort_candidates_cambyses — variable-length bitonic sort dispatch
 *
 * Pads the candidate array to the smallest 2^k size (8/16/32) and
 * dispatches to the corresponding fixed-size bitonic sort network.
 * Padding entries (score=S16_MIN, p=NULL) sink to the tail after sort.
 */
void sort_candidates_cambyses(struct cambyses_candidate *cands, int nr_cands)
{
	int padded_size;
	int i;

	if (nr_cands <= 1)
		return;

	/* Determine smallest power-of-2 container */
	if (nr_cands <= 8)
		padded_size = 8;
	else if (nr_cands <= 16)
		padded_size = 16;
	else
		padded_size = 32;

	/* Pad unused slots — S16_MIN ensures they sink to the tail */
	for (i = nr_cands; i < padded_size; i++) {
		cands[i].p = NULL;
		cands[i].score = S16_MIN;
	}

	/* Dispatch to fixed-size network */
	switch (padded_size) {
	case 8:
		bitonic_sort_8(cands);
		break;
	case 16:
		bitonic_sort_16(cands);
		break;
	case 32:
		bitonic_sort_32(cands);
		break;
	}
}

#endif /* CONFIG_SCHED_CAMBYSES */
