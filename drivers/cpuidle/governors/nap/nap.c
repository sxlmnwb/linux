// SPDX-License-Identifier: GPL-2.0
/*
 * nap.c — Neural Adaptive Predictor cpuidle governor
 *
 * A machine-learning-based cpuidle governor that uses a small MLP (8→8→1)
 * with 3 Mixture-of-Experts (short/long/deep) to predict a log2 correction
 * factor for sleep_length.  State selection is deterministic threshold
 * comparison.  Weights are Xavier-initialized at boot, then refined via
 * online learning (deferred backpropagation with SGD).
 *
 * IMPORTANT: This file is compiled WITHOUT FPU/SSE flags (normal kernel
 * compilation).  All floating-point and SIMD code lives in nap_fpu.c and
 * nap_nn_{sse2,avx2}.c, which are compiled with CC_FLAGS_FPU.
 * This separation ensures the compiler cannot emit SSE instructions in
 * governor callbacks (nap_select, nap_reflect, etc.), which would corrupt
 * userspace FPU register state.
 */

#include <linux/cpuidle.h>
#include <linux/cpu.h>
#include <linux/jump_label.h>
#include <linux/kobject.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/sched/clock.h>
#include <linux/sysfs.h>
#include <linux/string.h>
#include <linux/tick.h>
#include <asm/simd.h>
#include <asm/fpu/api.h>
#include <asm/processor.h>

#include "nap.h"

#include "../gov.h"

/**************************************************************
 * Version Information:
 */

#define CPUIDLE_NAP_PROGNAME "Nap CPUIdle Governor"
#define CPUIDLE_NAP_AUTHOR   "Masahito Suzuki"

#define CPUIDLE_NAP_VERSION  "0.4.0"

/* Governor defaults */
#define NAP_DEFAULT_LR_MILLTHS    1     /* 0.001 = 1 millths */
#define NAP_DEFAULT_INTERVAL      4     /* learn every 4 reflects */
#define NAP_DEFAULT_CLAMP_MILLTHS 1000  /* 1.0 = 1000 millths */
#define NAP_DEFAULT_PCTL_MILLTHS  100   /* 10th percentile */

/* ================================================================
 * ISA dispatch via static keys (definitions only; dispatch in nap_fpu.c)
 * ================================================================ */

DEFINE_STATIC_KEY_FALSE(nap_use_avx2);

static void __init nap_detect_simd(void)
{
	if (boot_cpu_has(X86_FEATURE_FMA) &&
	    boot_cpu_has(X86_FEATURE_AVX2)) {
		static_branch_enable(&nap_use_avx2);
		pr_info("nap: using AVX2+FMA\n");
	} else {
		pr_info("nap: using SSE2\n");
	}
}

/* ================================================================
 * Per-CPU data
 * ================================================================ */

DEFINE_PER_CPU(struct nap_cpu_data, nap_data);
static struct cpuidle_driver *nap_cached_drv;

/* ================================================================
 * Reflect-time updates (integer-only, no FPU needed)
 * ================================================================ */

static void nap_history_update(struct nap_cpu_data *d, u64 measured_ns)
{
	d->history[d->hist_idx] = measured_ns;
	d->hist_idx = (d->hist_idx + 1) % NAP_HISTORY_SIZE;
	if (d->hist_count < NAP_HISTORY_SIZE)
		d->hist_count++;

}

static void nap_update_external_signals(struct nap_cpu_data *d)
{
	d->prev_idle_exit = local_clock();
}

/* ================================================================
 * Governor callbacks
 * ================================================================ */

/*
 * Return the shallowest C-state index that is both enabled and
 * satisfies the current latency request.  Returns 0 if no such
 * state exists (caller must treat 0 as "POLL is the only option").
 *
 * Called from the short-circuit path to decide whether the predicted
 * sleep length is worth entering any C-state at all.  Does not
 * consult the NN.
 */
static int nap_find_min_valid_state(struct cpuidle_driver *drv,
				    struct cpuidle_device *dev,
				    s64 latency_req)
{
	int i;

	for (i = 1; i < drv->state_count; i++) {
		if (dev->states_usage[i].disable)
			continue;
		if (drv->states[i].exit_latency_ns > latency_req)
			continue;
		return i;
	}
	return 0;
}

/*
 * Cached wrapper around nap_find_min_valid_state().
 *
 * Invalidation triggers:
 *   1. latency_req changed since last cached value (immediate; PM QoS
 *      updates propagate on the next nap_select call).
 *   2. NAP_MIN_STATE_REFRESH_JIFFIES elapsed since last refresh
 *      (bounded staleness for sysfs-driven or runtime-driver state
 *      disable events, which are rare).
 *
 * Hot path cost when the cache is valid: ~5-7 cycles (one s64
 * compare, one time_after() check, one conditional return).  The
 * uncached loop runs at most once per HZ jiffies per CPU.
 */
static inline int nap_get_min_valid_state(struct nap_cpu_data *d,
					   struct cpuidle_driver *drv,
					   struct cpuidle_device *dev,
					   s64 latency_req)
{
	if (unlikely(latency_req != d->cached_min_state_latency ||
		     time_after(jiffies,
				d->cached_min_state_jiffies +
				NAP_MIN_STATE_REFRESH_JIFFIES))) {
		d->cached_min_state = nap_find_min_valid_state(drv, dev,
							       latency_req);
		d->cached_min_state_latency = latency_req;
		d->cached_min_state_jiffies = jiffies;
	}
	return d->cached_min_state;
}

/*
 * Compute dev->poll_limit_ns for the short-circuit path.
 *
 * Budget = predicted wake time (sleep_length) + 1 µs safety margin.
 * The margin absorbs timer jitter so a wake arriving slightly after
 * the predicted time does not trigger a select/enter/reflect retry
 * cycle.  It is consumed only when the wake is actually late; on-time
 * and early wakes exit POLL via need_resched without touching the
 * margin.
 *
 * Floor: NAP_POLL_LIMIT_MIN_NS (1 µs).  Below this, per-iteration
 * governor overhead exceeds actual polling, and POLL's own timeout
 * sampling granularity (~1.3 µs via POLL_IDLE_RELAX_COUNT cpu_relax
 * iterations) makes smaller limits indistinguishable in practice.
 *
 * Ceiling: min_state.target_residency_ns.  Beyond that point, the
 * C-state would have been a better choice than polling.
 */
static inline u64 nap_compute_poll_limit(u64 sleep_length_ns,
					 u64 min_state_target_ns)
{
	u64 budget = sleep_length_ns + NAP_POLL_LIMIT_MARGIN_NS;

	return clamp_t(u64, budget,
		       NAP_POLL_LIMIT_MIN_NS,
		       min_state_target_ns);
}

static int nap_fallback_heuristic(struct cpuidle_driver *drv,
				  struct cpuidle_device *dev)
{
	s64 latency_req = cpuidle_governor_latency_req(dev->cpu);
	ktime_t delta_tick;
	u64 sleep_length_ns;
	int i;

	sleep_length_ns = ktime_to_ns(tick_nohz_get_sleep_length(&delta_tick));

	for (i = drv->state_count - 1; i > 0; i--) {
		if (dev->states_usage[i].disable)
			continue;
		if (drv->states[i].exit_latency_ns > latency_req)
			continue;
		if (drv->states[i].target_residency_ns > sleep_length_ns)
			continue;
		return i;
	}
	return 0;
}

static int nap_select(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev,
		      bool *stop_tick)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	s64 latency_req;
	ktime_t delta_tick;
	u64 sleep_length_ns;
	int idx, min_state;

	if (unlikely(drv->state_count <= 1))
		return 0;

	latency_req = cpuidle_governor_latency_req(dev->cpu);
	sleep_length_ns = ktime_to_ns(tick_nohz_get_sleep_length(&delta_tick));

	min_state = nap_get_min_valid_state(d, drv, dev, latency_req);

	/*
	 * Fast path: when no C-state can amortize its target residency
	 * within the predicted sleep length, the answer is deterministically
	 * POLL.  Skip NN inference and feature extraction entirely.
	 * nap_reflect also skips history update and learning for
	 * short-circuited events (see the short_circuited check there).
	 * See spec §3.1.
	 */
	if (min_state == 0 ||
	    sleep_length_ns < drv->states[min_state].target_residency_ns) {

		if (min_state > 0)
			dev->poll_limit_ns = nap_compute_poll_limit(
				sleep_length_ns,
				drv->states[min_state].target_residency_ns);
		else
			dev->poll_limit_ns = max_t(u64, sleep_length_ns,
						   NAP_POLL_LIMIT_MIN_NS);

		*stop_tick = false;
		d->last_selected_idx = 0;
		d->short_circuited = true;
		d->stats.total_selects++;
		return 0;
	}

	/* Normal NN-driven path */
	d->short_circuited = false;

	if (likely(may_use_simd())) {
		kernel_fpu_begin();
		idx = nap_fpu_select(drv, dev, d);
		kernel_fpu_end();

		if (idx < 0)
			idx = nap_fallback_heuristic(drv, dev);
	} else {
		idx = nap_fallback_heuristic(drv, dev);
	}

	*stop_tick = (drv->states[idx].target_residency_ns >
		      RESIDENCY_THRESHOLD_NS);

	d->last_selected_idx = idx;
	d->stats.total_selects++;

	return idx;
}

static void nap_reflect(struct cpuidle_device *dev, int index)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct cpuidle_driver *drv = cpuidle_get_cpu_driver(dev);
	u64 measured_ns = dev->last_residency_ns;

	if (unlikely(!drv))
		return;

	/*
	 * Short-circuited POLL: NN was not invoked for this idle
	 * event, so the residency does not belong to the NN's
	 * training distribution.  Update the aggregate residency
	 * statistic and return — history, hit_intercept, prediction
	 * error, external signals, and learning are all skipped.
	 * See spec §3.4.
	 */
	if (d->short_circuited) {
		d->stats.total_residency_ns += measured_ns;
		return;
	}

	nap_history_update(d, measured_ns);

	d->last_prediction_error = d->last_predicted_ns - (s64)measured_ns;
	nap_update_external_signals(d);

	/*
	 * Dual gate: learn when both the per-N-reflect counter fires
	 * AND at least learn_jiffies_min jiffies have elapsed since
	 * the last learning step.  The time gate prevents sustained
	 * weight churn on workloads with very rapid idle bursts; a
	 * value of 0 disables it (restores the original counter-only
	 * behavior).  See spec §3.5.
	 */
	if (++d->learn_counter >= d->learn_interval &&
	    time_after_eq(jiffies,
			  d->last_learn_jiffies + d->learn_jiffies_min)) {
		d->learn_counter = 0;
		d->last_learn_jiffies = jiffies;
		d->learn_actual_ns = measured_ns;
		d->needs_learn = true;
	}

	d->stats.total_residency_ns += measured_ns;
	if (index > 0 && measured_ns < drv->states[index].target_residency_ns)
		d->stats.overshoot_count++;
}

static int nap_enable(struct cpuidle_driver *drv,
		      struct cpuidle_device *dev)
{
	struct nap_cpu_data *d = per_cpu_ptr(&nap_data, dev->cpu);

	memset(d, 0, sizeof(*d));

	/*
	 * Force first-call refresh of the min-valid-state cache.
	 * cached_min_state_latency = S64_MIN ensures the first
	 * nap_select() comparison will always trip the invalidation
	 * branch regardless of the actual latency_req value.
	 * cached_min_state itself is already zeroed by the memset above.
	 */
	d->cached_min_state_latency = S64_MIN;
	d->cached_min_state_jiffies = jiffies - NAP_MIN_STATE_REFRESH_JIFFIES;

	/* Default: allow at most one learning step per jiffy */
	d->learn_jiffies_min = 1;

	/*
	 * Defer weight initialization to the first nap_select() FPU path
	 * via reset_pending.  nap_enable() is called from cpuidle core
	 * (cpuidle_enable_device) which may run on a different CPU than
	 * dev->cpu during governor switch.  Deferring ensures FPU init
	 * happens on the correct CPU in its own idle context.
	 */
	WRITE_ONCE(nap_cached_drv, drv);
	d->learning_rate_millths  = NAP_DEFAULT_LR_MILLTHS;
	d->learn_interval = NAP_DEFAULT_INTERVAL;
	d->max_grad_norm_millths  = NAP_DEFAULT_CLAMP_MILLTHS;
	d->overshoot_pctl_millths = NAP_DEFAULT_PCTL_MILLTHS;
	d->reset_pending = true;

	return 0;
}

static void nap_disable(struct cpuidle_driver *drv,
			struct cpuidle_device *dev)
{
	WRITE_ONCE(nap_cached_drv, NULL);
}

/* ================================================================
 * sysfs interface  (/sys/devices/system/cpu/nap/)
 * ================================================================ */

static ssize_t stats_show(struct kobject *kobj,
			  struct kobj_attribute *attr, char *buf)
{
	int cpu, len = 0;
	u64 total_sel = 0, total_res = 0, total_under = 0, total_learn = 0;

	for_each_online_cpu(cpu) {
		struct nap_cpu_data *d = &per_cpu(nap_data, cpu);

		total_sel   += d->stats.total_selects;
		total_res   += d->stats.total_residency_ns;
		total_under += d->stats.overshoot_count;
		total_learn += d->stats.learn_count;
	}

	len += sysfs_emit_at(buf, len, "total_selects: %llu\n", total_sel);
	len += sysfs_emit_at(buf, len, "total_residency_ms: %llu\n",
			     div_u64(total_res, NSEC_PER_MSEC));
	len += sysfs_emit_at(buf, len, "overshoot_count: %llu\n", total_under);
	len += sysfs_emit_at(buf, len, "overshoot_rate_permil: %llu\n",
			     total_sel ? div_u64(total_under * 1000, total_sel) : 0);
	len += sysfs_emit_at(buf, len, "learn_count: %llu\n", total_learn);
	return len;
}

static ssize_t learning_rate_show(struct kobject *kobj,
				  struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).learning_rate_millths);
}

static ssize_t learning_rate_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 100)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learning_rate_millths = val;

	return count;
}

static ssize_t learn_interval_show(struct kobject *kobj,
				   struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%d\n",
			  per_cpu(nap_data, cpu).learn_interval);
}

static ssize_t learn_interval_store(struct kobject *kobj,
				    struct kobj_attribute *attr,
				    const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val == 0 || val > 10000)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learn_interval = val;

	return count;
}

static ssize_t learn_jiffies_min_show(struct kobject *kobj,
				      struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).learn_jiffies_min);
}

static ssize_t learn_jiffies_min_store(struct kobject *kobj,
				       struct kobj_attribute *attr,
				       const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val > HZ * 3600)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).learn_jiffies_min = val;

	return count;
}

static ssize_t reset_weights_store(struct kobject *kobj,
				   struct kobj_attribute *attr,
				   const char *buf, size_t count)
{
	cpumask_var_t mask;
	int cpu;

	if (!READ_ONCE(nap_cached_drv))
		return -ENODEV;

	/*
	 * Set a per-CPU flag; each CPU will reinitialize its own weights
	 * inside nap_select() within its own kernel_fpu_begin/end context.
	 * This avoids cross-CPU data races on the weight arrays.
	 *
	 * Accepts "all" to reset every online CPU, or a cpulist
	 * (e.g. "0-3,5,7") to reset specific CPUs.
	 */
	if (sysfs_streq(buf, "all")) {
		for_each_online_cpu(cpu)
			per_cpu(nap_data, cpu).reset_pending = true;
		pr_info("nap: weight reset scheduled for all CPUs\n");
		return count;
	}

	if (!alloc_cpumask_var(&mask, GFP_KERNEL))
		return -ENOMEM;

	if (cpulist_parse(buf, mask)) {
		free_cpumask_var(mask);
		return -EINVAL;
	}

	for_each_cpu_and(cpu, mask, cpu_online_mask)
		per_cpu(nap_data, cpu).reset_pending = true;

	pr_info("nap: weight reset scheduled for CPUs %*pbl\n",
		cpumask_pr_args(mask));
	free_cpumask_var(mask);
	return count;
}

static ssize_t reset_stats_store(struct kobject *kobj,
				 struct kobj_attribute *attr,
				 const char *buf, size_t count)
{
	int cpu;

	for_each_online_cpu(cpu)
		memset(&per_cpu(nap_data, cpu).stats, 0,
		       sizeof(struct nap_stats));

	return count;
}

static ssize_t overshoot_pctl_show(struct kobject *kobj,
				    struct kobj_attribute *attr, char *buf)
{
	int cpu;

	cpu = cpumask_first(cpu_online_mask);
	if (cpu >= nr_cpu_ids)
		return sysfs_emit(buf, "0\n");
	return sysfs_emit(buf, "%u\n",
			  per_cpu(nap_data, cpu).overshoot_pctl_millths);
}

static ssize_t overshoot_pctl_store(struct kobject *kobj,
				     struct kobj_attribute *attr,
				     const char *buf, size_t count)
{
	unsigned int val;
	int cpu;

	if (kstrtouint(buf, 10, &val) || val > 500)
		return -EINVAL;

	for_each_online_cpu(cpu)
		per_cpu(nap_data, cpu).overshoot_pctl_millths = val;

	return count;
}

static ssize_t version_show(struct kobject *kobj,
			    struct kobj_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%s\n", CPUIDLE_NAP_VERSION);
}

static ssize_t simd_show(struct kobject *kobj,
			 struct kobj_attribute *attr, char *buf)
{
	if (static_branch_unlikely(&nap_use_avx2))
		return sysfs_emit(buf, "avx2\n");
	else
		return sysfs_emit(buf, "sse2\n");
}

static struct kobj_attribute version_attr           = __ATTR_RO(version);
static struct kobj_attribute simd_attr              = __ATTR_RO(simd);
static struct kobj_attribute stats_attr             = __ATTR_RO(stats);
static struct kobj_attribute learning_rate_attr     = __ATTR_RW(learning_rate);
static struct kobj_attribute learn_interval_attr    = __ATTR_RW(learn_interval);
static struct kobj_attribute learn_jiffies_min_attr = __ATTR_RW(learn_jiffies_min);
static struct kobj_attribute overshoot_pctl_attr    = __ATTR_RW(overshoot_pctl);
static struct kobj_attribute reset_weights_attr     = __ATTR_WO(reset_weights);
static struct kobj_attribute reset_stats_attr       = __ATTR_WO(reset_stats);

static struct attribute *nap_attrs[] = {
	&version_attr.attr,
	&simd_attr.attr,
	&stats_attr.attr,
	&learning_rate_attr.attr,
	&learn_interval_attr.attr,
	&learn_jiffies_min_attr.attr,
	&overshoot_pctl_attr.attr,
	&reset_weights_attr.attr,
	&reset_stats_attr.attr,
	NULL,
};

static const struct attribute_group nap_attr_group = {
	.attrs = nap_attrs,
};

static struct kobject *cpuidle_kobj;

int nap_sysfs_init(void)
{
	struct device *dev_root;
	int ret;

	dev_root = bus_get_dev_root(&cpu_subsys);
	if (!dev_root)
		return -ENODEV;

	cpuidle_kobj = kobject_create_and_add("nap", &dev_root->kobj);
	put_device(dev_root);
	if (!cpuidle_kobj)
		return -ENOMEM;

	ret = sysfs_create_group(cpuidle_kobj, &nap_attr_group);
	if (ret) {
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
	return ret;
}

void nap_sysfs_exit(void)
{
	if (cpuidle_kobj) {
		sysfs_remove_group(cpuidle_kobj, &nap_attr_group);
		kobject_put(cpuidle_kobj);
		cpuidle_kobj = NULL;
	}
}

/* ================================================================
 * Governor registration
 * ================================================================ */

static struct cpuidle_governor nap_governor = {
	.name    = "nap",
	.rating  = 26,
	.enable  = nap_enable,
	.disable = nap_disable,
	.select  = nap_select,
	.reflect = nap_reflect,
};

static int __init nap_init(void)
{
	int ret;

	nap_detect_simd();

	ret = nap_sysfs_init();
	if (ret)
		pr_warn("nap: sysfs init failed: %d (continuing without sysfs)\n", ret);

	ret = cpuidle_register_governor(&nap_governor);
	if (ret) {
		pr_err("nap: register_governor failed: %d\n", ret);
		nap_sysfs_exit();
		return ret;
	}

	pr_info("%s v%s by %s registered (rating=%u)\n",
	       CPUIDLE_NAP_PROGNAME, CPUIDLE_NAP_VERSION,
	       CPUIDLE_NAP_AUTHOR, nap_governor.rating);
	return 0;
}
postcore_initcall(nap_init);
