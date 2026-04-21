// SPDX-License-Identifier: GPL-2.0
/*
 * Cambyses — Context-Aware Migration Balancer Yielding Scored Entity Selection
 *
 * Scored migration selection for CFS load balancer.
 * Replaces FIFO selection with a multi-feature scoring function that evaluates
 * cache coldness, load contribution, voluntary switch ratio, and wakee stability.
 */

/*
 * This file is #included from fair.c (not compiled separately)
 * to access static functions: can_migrate_task(), detach_task(),
 * task_h_load(), task_util_est(), task_fits_cpu(), etc.
 */

/**************************************************************
 * Version Information:
 */

#define CAMBYSES_PROGNAME "Cambyses Migration Selector"
#define CAMBYSES_AUTHOR   "Masahito Suzuki"

#define CAMBYSES_VERSION  "0.1.1"

/* Runtime toggle — NOP-patched when disabled */
DEFINE_STATIC_KEY_TRUE(sched_cambyses);

/* Default weights: w1=3 (load contribution dominant), w0=w2=w3=1 */
u8 sysctl_cambyses_w0 = 1;
u8 sysctl_cambyses_w1 = 2;
u8 sysctl_cambyses_w2 = 1;
u8 sysctl_cambyses_w3 = 1;

#ifdef CONFIG_SCHED_CAMBYSES_SIMD
DEFINE_STATIC_KEY_FALSE(cambyses_has_avx2);
DEFINE_STATIC_KEY_FALSE(cambyses_has_ssse3);
#endif

/*
 * log2p1_u64_u8fp2 — fixed-point log2(v+1) with 2-bit mantissa
 *
 * Returns (exponent << 2) | mantissa, giving 4× finer granularity than
 * fls64().  Based on BORE scheduler's log2p1_u64_u32fp().
 *
 * Result fits in u8 for all practical scheduler values (max ~163 for
 * nanosecond-scale deltas).
 */
static inline u8 log2p1_u64_u8fp2(u64 v)
{
	int clz, exponent;
	u8 mantissa;

	if (unlikely(!v))
		return 0;
	clz = __builtin_clzll(v);
	exponent = 64 - clz;
	mantissa = (u8)((v << clz) << 1 >> 62);
	return (u8)(exponent << 2 | mantissa);
}

/*
 * vol_switch_ratio_u6 — voluntary context switch ratio, 0–64
 *
 * Approximates (nvcsw / (nvcsw + nivcsw)) × 64 without division, using
 * log2 subtraction: log2(a/b) = log2(a) - log2(b).
 *
 * Both nvcsw and total are converted to pseudo-log2 via log2p1_u64_u8fp2
 * (CLZ + 2 shifts each), then subtracted.  The +1 offsets in log2p1 cancel.
 * Result is offset by +64 and clamped to [0, 64].
 *
 * Higher value = more I/O-bound = transient cache footprint = cheaper to migrate.
 */
static inline u8 vol_switch_ratio_u6(struct task_struct *p)
{
	unsigned long total = p->nvcsw + p->nivcsw;

	if (unlikely(!total))
		return 0;
	return (u8)clamp_t(int,
		(int)log2p1_u64_u8fp2(p->nvcsw)
		- (int)log2p1_u64_u8fp2(total) + 64,
		0, 64);
}

/*
 * score_task_cambyses — compute migration suitability score for a task
 *
 * Features (u8, range varies per feature):
 *   F0: cache coldness       — log2p1(time since last exec) (higher = colder = better)
 *   F1: load contribution    — log2p1(task hierarchical load) (higher = more effective)
 *   F2: vol switch ratio     — nvcsw/(nvcsw+nivcsw) × 64 (higher = I/O-bound = cheaper)
 *   F3: wakee penalty        — log2p1(wakee_flips + 1) (higher = riskier)
 *
 * Score range: max ~392, min ~-24 → fits in s16.
 *
 * When feat_out is non-NULL, also packs the raw u8 features for SIMD
 * batch scoring.  The scalar score is always returned for use when
 * nr_cands <= SIMD threshold or SIMD is unavailable.
 */
static s16 score_task_cambyses(struct task_struct *p, struct lb_env *env,
			       u8 *feat_out)
{
	int f0 = log2p1_u64_u8fp2(rq_clock_task(env->src_rq) - p->se.exec_start);
	int f1 = log2p1_u64_u8fp2(max_t(unsigned long, task_h_load(p), 1));
	int f2 = vol_switch_ratio_u6(p);
	int f3 = log2p1_u64_u8fp2(p->wakee_flips + 1);

	if (feat_out) {
		feat_out[0] = (u8)f0;
		feat_out[1] = (u8)f1;
		feat_out[2] = (u8)f2;
		feat_out[3] = (u8)f3;
	}

	return (s16)((int)sysctl_cambyses_w0 * f0
		   + (int)sysctl_cambyses_w1 * f1
		   + (int)sysctl_cambyses_w2 * f2
		   - (int)sysctl_cambyses_w3 * f3);
}

/*
 * check_imbalance_cambyses — coarse filter for Phase 1 candidate inclusion
 *
 * Mirrors the existing detach_tasks() switch logic but does NOT consume
 * imbalance (that happens in Phase 3 after sorting by score).
 */
static bool check_imbalance_cambyses(struct task_struct *p,
				     struct lb_env *env)
{
	switch (env->migration_type) {
	case migrate_load: {
		unsigned long load = max_t(unsigned long, task_h_load(p), 1);

		if (sched_feat(LB_MIN) &&
		    load < 16 && !env->sd->nr_balance_failed)
			return false;
		if (shr_bound(load, env->sd->nr_balance_failed) > env->imbalance)
			return false;
		return true;
	}
	case migrate_util: {
		unsigned long util = task_util_est(p);

		if (shr_bound(util, env->sd->nr_balance_failed) > env->imbalance)
			return false;
		return true;
	}
	case migrate_task:
		return true;
	case migrate_misfit:
		return !task_fits_cpu(p, env->src_cpu);
	}
	return false;
}

/*
 * consume_imbalance_cambyses — deduct task cost from imbalance budget
 *
 * Called in Phase 3 for each detached task, in score-descending order.
 * Matches Vanilla behavior: allows imbalance to go negative on the last task.
 */
static void consume_imbalance_cambyses(struct task_struct *p,
				       struct lb_env *env)
{
	switch (env->migration_type) {
	case migrate_load:
		env->imbalance -= max_t(unsigned long, task_h_load(p), 1);
		break;
	case migrate_util:
		env->imbalance -= task_util_est(p);
		break;
	case migrate_task:
		env->imbalance--;
		break;
	case migrate_misfit:
		env->imbalance = 0;
		break;
	}
}

#ifdef CONFIG_SCHED_CAMBYSES_SIMD
#include <asm/fpu/api.h>

/*
 * score_and_sort_cambyses — SIMD dispatch for scoring + sort
 *
 * For nr_cands <= CAMBYSES_SIMD_THRESHOLD (8): scalar path only.
 * Scores are already computed during collection.
 *
 * For nr_cands > 8 with AVX2: SIMD batch scoring + SIMD bitonic sort
 * within one kernel_fpu_begin/end context.  Scores are packed as
 * s32 = (score << 16) | index for the sort, then cands[] is reordered.
 *
 * For nr_cands > 8 with SSSE3: SIMD batch scoring, then scalar sort.
 * (SSSE3 lacks PMINSD — no s32 compare-and-swap for SIMD sort.)
 */
static void score_and_sort_cambyses(struct cambyses_candidate *cands,
				    u8 *features_packed, int nr_cands)
{
	if (nr_cands <= CAMBYSES_SIMD_THRESHOLD) {
		sort_candidates_cambyses(cands, nr_cands);
		return;
	}

	/*
	 * Pad feature buffer to alignment boundary (zeroed slots produce
	 * score 0, harmless — those slots have p=NULL in cands[]).
	 */
	{
		int padded = (nr_cands + 7) & ~7;
		int j;

		for (j = nr_cands * CAMBYSES_NR_FEATURES;
		     j < padded * CAMBYSES_NR_FEATURES; j++)
			features_packed[j] = 0;
	}

	kernel_fpu_begin();

	if (static_branch_likely(&cambyses_has_avx2)) {
		s16 scores[SCHED_NR_MIGRATE_BREAK];
		s32 packed[SCHED_NR_MIGRATE_BREAK] __aligned(32);
		struct cambyses_candidate tmp[SCHED_NR_MIGRATE_BREAK];
		int j;

		/* SIMD batch scoring */
		cambyses_simd_score_avx2(features_packed, scores, nr_cands);

		/*
		 * Pack: (score << 16) | index.  Sorting s32 descending
		 * automatically sorts by score, with index preserved for
		 * free in the low 16 bits.
		 */
		for (j = 0; j < nr_cands; j++)
			packed[j] = ((s32)scores[j] << 16) | (u16)j;
		for (j = nr_cands; j < SCHED_NR_MIGRATE_BREAK; j++)
			packed[j] = (s32)0x80000000; /* INT_MIN: sinks */

		/* SIMD bitonic sort — descending by s32 */
		cambyses_simd_sort_avx2(packed);

		kernel_fpu_end();

		/* Reorder cands[] to match sorted order */
		memcpy(tmp, cands, sizeof(cands[0]) * nr_cands);
		for (j = 0; j < nr_cands; j++) {
			int idx = packed[j] & 0xFFFF;

			cands[j] = tmp[idx];
		}

		return; /* AVX2 path: scoring + sorting complete */
	}

	if (static_branch_likely(&cambyses_has_ssse3)) {
		s16 scores[SCHED_NR_MIGRATE_BREAK];
		int j;

		cambyses_simd_score_ssse3(features_packed, scores, nr_cands);
		for (j = 0; j < nr_cands; j++)
			cands[j].score = scores[j];
	}
	/* else: keep scalar scores computed during collection */

	kernel_fpu_end();

	/* SSSE3 / scalar fallback: scalar bitonic sort */
	sort_candidates_cambyses(cands, nr_cands);
}
#endif /* CONFIG_SCHED_CAMBYSES_SIMD */

/*
 * detach_tasks_cambyses — scored migration selection for Pull path
 *
 * Called from detach_tasks() when sched_cambyses is active.
 * Operates under rq_lock_irqsave (inherited from caller).
 *
 * Phase 1: Sample candidates from cfs_tasks, score each one
 *          (also pack features for SIMD if enabled)
 * Phase 2: Sort candidates by score (descending) via bitonic sort
 *          (SIMD scoring if nr_cands > threshold)
 * Phase 3: Detach in score order, consuming imbalance budget
 */
static int detach_tasks_cambyses(struct lb_env *env)
{
	struct list_head *tasks = &env->src_rq->cfs_tasks;
	struct cambyses_candidate cands[SCHED_NR_MIGRATE_BREAK];
	LIST_HEAD(cand_tasks);
	int nr_cands = 0;
	int detached = 0;
	struct task_struct *p;
	int i;
#ifdef CONFIG_SCHED_CAMBYSES_SIMD
	/*
	 * Feature buffer for SIMD batch scoring (AoS layout).
	 * 32 candidates × 4 features = 128 bytes on stack.
	 */
	u8 features_packed[SCHED_NR_MIGRATE_BREAK * CAMBYSES_NR_FEATURES]
		__aligned(32);
#endif

	/*
	 * Phase 1: Sampling — collect eligible candidates and score them.
	 *
	 * Mirrors the loop structure of vanilla detach_tasks():
	 * same loop_max, loop_break, idle checks.
	 *
	 * Accepted candidates are moved to cand_tasks (off cfs_tasks)
	 * to prevent the list rotation from re-collecting the same task
	 * as the loop wraps around.  Skipped tasks stay on cfs_tasks.
	 */
	while (!list_empty(tasks)) {
		if (env->idle && env->src_rq->nr_running <= 1)
			break;

		env->loop++;
		if (env->loop > env->loop_max)
			break;
		if (env->loop > env->loop_break) {
			env->loop_break += SCHED_NR_MIGRATE_BREAK;
			env->flags |= LBF_NEED_BREAK;
			break;
		}

		p = list_last_entry(tasks, struct task_struct, se.group_node);

		if (!can_migrate_task(p, env)) {
			if (p->sched_task_hot)
				schedstat_inc(p->stats.nr_failed_migrations_hot);
			list_move(&p->se.group_node, tasks);
			continue;
		}

		if (!check_imbalance_cambyses(p, env))
			goto skip;

		/* Score and collect (also pack features for SIMD) */
		cands[nr_cands].p = p;
#ifdef CONFIG_SCHED_CAMBYSES_SIMD
		cands[nr_cands].score = score_task_cambyses(p, env,
			&features_packed[nr_cands * CAMBYSES_NR_FEATURES]);
#else
		cands[nr_cands].score = score_task_cambyses(p, env, NULL);
#endif
		nr_cands++;

		/*
		 * Remove from cfs_tasks so it cannot be re-scanned.
		 * detach_task() in Phase 3 will list_del this node;
		 * non-detached candidates are spliced back below.
		 */
		list_move(&p->se.group_node, &cand_tasks);

		if (nr_cands >= SCHED_NR_MIGRATE_BREAK)
			break;
		continue;
skip:
		list_move(&p->se.group_node, tasks);
	}

	if (!nr_cands) {
		/* cand_tasks is empty here — nothing to splice back */
		return 0;
	}

	/*
	 * Phase 2: Score (SIMD if applicable) + sort (descending).
	 * Bitonic sort network — branch-free, compile-time fixed comparisons.
	 */
#ifdef CONFIG_SCHED_CAMBYSES_SIMD
	score_and_sort_cambyses(cands, features_packed, nr_cands);
#else
	sort_candidates_cambyses(cands, nr_cands);
#endif

	/*
	 * Phase 3: Detach in score order, consuming imbalance budget.
	 *
	 * rq_lock + IRQ disabled from Phase 1 — candidates cannot be
	 * dequeued between phases.
	 *
	 * detach_task() removes the task from cand_tasks (via
	 * dequeue_entity → list_del on group_node), then list_add
	 * places it on env->tasks for attach_tasks().
	 */
	for (i = 0; i < nr_cands && env->imbalance > 0; i++) {
		p = cands[i].p;

		/* Skip padding entries */
		if (!p)
			continue;

		consume_imbalance_cambyses(p, env);
		detach_task(p, env);
		list_add(&p->se.group_node, &env->tasks);
		detached++;

#ifdef CONFIG_PREEMPTION
		if (env->idle == CPU_NEWLY_IDLE)
			break;
#endif
	}

	/*
	 * Return non-detached candidates to cfs_tasks.
	 * These were removed in Phase 1 but not selected in Phase 3
	 * (imbalance exhausted or preemption break).
	 */
	list_splice(&cand_tasks, tasks);

	schedstat_add(env->sd->lb_gained[env->idle], detached);
	return detached;
}

/*
 * detach_one_task_cambyses — scored selection for Push path (active balancing)
 *
 * Replaces the FIFO "first migratable task" policy in detach_one_task().
 * Scans all migratable tasks on src_rq and selects the one with the
 * highest migration score.
 *
 * Push moves exactly 1 task, so no sort is needed — simple max scan.
 * The stop_machine context is already heavy, so scoring overhead is negligible.
 */
static struct task_struct *detach_one_task_cambyses(struct lb_env *env)
{
	struct task_struct *p, *best = NULL;
	s16 best_score = S16_MIN;

	lockdep_assert_rq_held(env->src_rq);

	list_for_each_entry_reverse(p,
			&env->src_rq->cfs_tasks, se.group_node) {
		s16 score;

		if (!can_migrate_task(p, env))
			continue;

		score = score_task_cambyses(p, env, NULL);
		if (score > best_score) {
			best_score = score;
			best = p;
		}
	}

	if (!best)
		return NULL;

	detach_task(best, env);
	schedstat_inc(env->sd->lb_gained[env->idle]);
	return best;
}

/* ======== SIMD ISA detection ======== */

#ifdef CONFIG_SCHED_CAMBYSES_SIMD
#include <asm/cpufeature.h>

static int __init cambyses_simd_init(void)
{
	const char *simd_name;

	if (boot_cpu_has(X86_FEATURE_AVX2)) {
		static_branch_enable(&cambyses_has_avx2);
		simd_name = "AVX2";
	} else if (boot_cpu_has(X86_FEATURE_SSSE3)) {
		static_branch_enable(&cambyses_has_ssse3);
		simd_name = "SSSE3";
	} else {
		simd_name = "none";
	}

	pr_info("%s v%s by %s [SIMD: %s]\n",
		CAMBYSES_PROGNAME, CAMBYSES_VERSION,
		CAMBYSES_AUTHOR, simd_name);

	return 0;
}
late_initcall(cambyses_simd_init);

#else /* !CONFIG_SCHED_CAMBYSES_SIMD */

static int __init cambyses_init(void)
{
	pr_info("%s v%s by %s [SIMD: disabled]\n",
		CAMBYSES_PROGNAME, CAMBYSES_VERSION,
		CAMBYSES_AUTHOR);

	return 0;
}
late_initcall(cambyses_init);

#endif /* CONFIG_SCHED_CAMBYSES_SIMD */

/* ======== sysctl interface ======== */

#ifdef CONFIG_SYSCTL
static int sched_cambyses_handler(const struct ctl_table *table,
					  int write, void *buffer,
					  size_t *lenp, loff_t *ppos)
{
	static u8 sched_cambyses_val;
	struct ctl_table tmp = {
		.data	= &sched_cambyses_val,
		.maxlen	= sizeof(u8),
		.mode	= table->mode,
		.extra1	= SYSCTL_ZERO,
		.extra2	= SYSCTL_ONE,
	};
	int ret;

	if (!write)
		sched_cambyses_val = static_key_enabled(&sched_cambyses);

	ret = proc_dou8vec_minmax(&tmp, write, buffer, lenp, ppos);
	if (ret || !write)
		return ret;

	if (sched_cambyses_val)
		static_branch_enable(&sched_cambyses);
	else
		static_branch_disable(&sched_cambyses);

	return 0;
}

static struct ctl_table sched_cambyses_sysctls[] = {
	{
		.procname	= "sched_cambyses",
		.data		= NULL, /* handled by custom handler */
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= sched_cambyses_handler,
	},
	{
		.procname	= "sched_cambyses_w0",
		.data		= &sysctl_cambyses_w0,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE,
	},
	{
		.procname	= "sched_cambyses_w1",
		.data		= &sysctl_cambyses_w1,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE,
	},
	{
		.procname	= "sched_cambyses_w2",
		.data		= &sysctl_cambyses_w2,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE, /* vol switch ratio weight */
	},
	{
		.procname	= "sched_cambyses_w3",
		.data		= &sysctl_cambyses_w3,
		.maxlen		= sizeof(u8),
		.mode		= 0644,
		.proc_handler	= proc_dou8vec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_THREE,
	},
};

static int __init sched_cambyses_sysctl_init(void)
{
	register_sysctl_init("kernel", sched_cambyses_sysctls);
	return 0;
}
late_initcall(sched_cambyses_sysctl_init);
#endif /* CONFIG_SYSCTL */
