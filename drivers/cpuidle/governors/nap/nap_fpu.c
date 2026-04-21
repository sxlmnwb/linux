// SPDX-License-Identifier: GPL-2.0
/*
 * nap_fpu.c — FPU/SIMD code for the NAP cpuidle governor
 *
 * This file is compiled with FPU/SSE flags enabled (CC_FLAGS_FPU).
 * ALL functions here MUST be called only from within
 * kernel_fpu_begin()/kernel_fpu_end() blocks.
 *
 * Keeping FPU code in a separate translation unit ensures the compiler
 * cannot emit SSE/x87 instructions in non-FPU code paths (nap.c),
 * which would silently corrupt userspace FPU register state.
 */

#include <linux/cpuidle.h>
#include <linux/math64.h>
#include <linux/percpu.h>
#include <linux/pm_qos.h>
#include <linux/sched/clock.h>
#include <linux/string.h>
#include <linux/tick.h>

#include "nap.h"

/* Clang lacks __builtin_ia32_movhlps; emulate with __builtin_shufflevector */
#ifdef __clang__
#define __builtin_ia32_movhlps(a, b) \
	__builtin_shufflevector(b, a, 2, 3, 6, 7)
#endif

/* ================================================================
 * Float math helpers
 * ================================================================ */

static inline float float_min(float a, float b) { return a < b ? a : b; }
static inline float float_max(float a, float b) { return a > b ? a : b; }

/*
 * Kernel-safe sqrtf using the SSE sqrtss instruction directly.
 * GCC may lower nap_sqrtf to a libm call, which is unavailable
 * in the kernel.  This file is always compiled with FPU/SSE enabled.
 */
static inline float nap_sqrtf(float x)
{
	asm("sqrtss %1, %0" : "=x"(x) : "x"(x));
	return x;
}

/* Scalar log2 approximation (same algorithm as fast_log2f_sse) */
static inline float fast_log2f(float x)
{
	union { float f; u32 i; } u = { .f = x };
	int exp = (int)((u.i >> 23) & 0xFFu) - 127;
	float e = (float)exp;
	float m, p;

	u.i = (u.i & 0x7FFFFFu) | (127u << 23);
	m = u.f - 1.0f;

	p = m * 0.4808f;
	p = 0.7213f - p;
	p = m * p;
	p = 1.4425f - p;
	p = m * p;

	return e + p;
}

/* ================================================================
 * Deterministic PRNG for weight initialization (LCG)
 * ================================================================ */

static inline float nap_prng_float(u32 *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (float)(s32)*state * (1.0f / 2147483648.0f);
}

/* ================================================================
 * ISA dispatch via static keys
 * ================================================================ */

static inline void nap_nn_forward(const float *input, float *output,
				  float *hidden_save,
				  const struct nap_weights *w)
{
	if (static_branch_unlikely(&nap_use_avx2))
		nap_nn_forward_avx2(input, output, hidden_save, w);
	else
		nap_nn_forward_sse2(input, output, hidden_save, w);
}

static inline void nap_nn_learn(struct nap_cpu_data *d)
{
	if (static_branch_unlikely(&nap_use_avx2))
		nap_nn_learn_avx2(d);
	else
		nap_nn_learn_sse2(d);
}

/* ================================================================
 * Weight initialization
 *
 * The NN directly outputs predicted sleep time in log2(ns) space.
 * Hidden neuron 0 is initialized as a pass-through for feature[0]
 * (log2(sleep_length)), so the initial output ≈ log2(sleep_length).
 * This matches the pre-learning behavior of selecting the deepest
 * state that fits within sleep_length.
 *
 * Other hidden neurons are Xavier-initialized with near-zero output
 * weights so their initial contribution is negligible.  Biases = 0.
 * ================================================================ */

#define NAP_PRNG_SEED 42u

static void nap_init_weights(struct nap_weights *w)
{
	u32 rng = NAP_PRNG_SEED;
	float scale_h1, scale_out;
	int i, j;

	/* Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(...))) */
	scale_h1  = nap_sqrtf(6.0f / (float)(NAP_INPUT_SIZE + NAP_HIDDEN_SIZE));
	scale_out = 0.01f;

	/* Hidden layer weights */
	for (i = 0; i < NAP_INPUT_SIZE; i++)
		for (j = 0; j < NAP_HIDDEN_SIZE; j++)
			w->w_h1[i][j] = nap_prng_float(&rng) * scale_h1;

	/* Hidden biases: zero (standard) */
	memset(w->b_h1, 0, sizeof(w->b_h1));

	/* Output weights: near-zero for ~0 initial contribution */
	for (j = 0; j < NAP_HIDDEN_SIZE; j++)
		w->w_out[j] = nap_prng_float(&rng) * scale_out;

	/* Output bias: zero */
	w->b_out = 0.0f;

	/*
	 * Neuron 0: pass-through for feature[0] = log2(sleep_length).
	 * hidden[0] = ReLU(1.0 * input[0] + 0) = input[0]  (always > 0)
	 * output += 1.0 * hidden[0] = log2(sleep_length)
	 *
	 * Override the random init above so initial output ≈ input[0].
	 */
	for (i = 0; i < NAP_INPUT_SIZE; i++)
		w->w_h1[i][0] = 0.0f;
	w->w_h1[0][0] = 1.0f;
	w->b_h1[0] = 0.0f;
	w->w_out[0] = 1.0f;
}

/*
 * Precompute log2(target_residency) per state for threshold-based selection.
 *
 * Used in the selection loop: pick deepest state where
 * log2_cost[i] <= nn_output (predicted sleep time in log2 space).
 *
 * Only target_residency_ns is used — exit_latency is a wakeup cost,
 * not a factor in whether the CPU can profitably stay in the state
 * for the predicted duration.
 */
static void nap_init_log2_cost(struct nap_cpu_data *d,
			       struct cpuidle_driver *drv)
{
	float log2_tick;
	int long_start, deep_idx;
	int i;

	for (i = 0; i < drv->state_count; i++) {
		float res = float_max(
			(float)drv->states[i].target_residency_ns, 1.0f);
		d->log2_cost[i] = fast_log2f(res);
	}

	/*
	 * MoE expert boundaries — 3-way split.
	 *
	 * Expert 0 (short): tick-bound idles where measured residency
	 *   is dominated by the next tick rather than the workload's
	 *   true idle duration.  Boundary: log2(TICK_NSEC).
	 *
	 * Expert 1 (long): nohz idles in intermediate C-states.
	 *
	 * Expert 2 (deep): idles targeting the deepest C-state.
	 *   The deepest state often has qualitatively different
	 *   residency characteristics (package C-state, longer
	 *   exit latency, power-gated domains) that warrant a
	 *   dedicated expert to avoid gradient interference with
	 *   intermediate states.
	 *
	 * Safety: with only 2 C-states (+ POLL), expert_deep is
	 * placed equal to expert_mid so the deep expert is never
	 * routed (same behavior as the old 2-expert split).
	 */
	if (drv->state_count <= 1) {
		d->expert_mid = 0.0f;
		d->expert_deep = 0.0f;
		return;
	}

	log2_tick = fast_log2f((float)TICK_NSEC);

	/* Default: deepest state belongs to long expert (safety) */
	long_start = drv->state_count - 1;

	/* Prefer the first state whose target_residency exceeds one jiffy */
	for (i = 1; i < drv->state_count; i++) {
		if (d->log2_cost[i] > log2_tick) {
			long_start = i;
			break;
		}
	}

	if (long_start > 1) {
		/* Normal case: boundary between last short and first long */
		d->expert_mid = (d->log2_cost[long_start - 1] +
				 d->log2_cost[long_start]) / 2.0f;
	} else {
		/*
		 * long_start == 1: even the shallowest C-state already
		 * exceeds one jiffy.  All NN-handled idles go to the
		 * long expert; place the boundary just below C1's
		 * residency so the short expert remains routable but
		 * unused.
		 */
		d->expert_mid = d->log2_cost[1] - 1.0f;
	}

	/*
	 * Deep expert boundary — deepest C-state split.
	 *
	 * When there are >= 3 C-states (state_count >= 4, counting POLL),
	 * place the boundary at the midpoint between the second-deepest
	 * and deepest state's log2(target_residency).  The deep expert
	 * then exclusively handles sleep durations long enough to reach
	 * the deepest state.
	 *
	 * With only 2 C-states, expert_deep == expert_mid collapses to
	 * the 2-expert regime (expert 2 is never selected).
	 */
	deep_idx = drv->state_count - 1;
	if (deep_idx >= 3) {
		/* >= 3 C-states: split before the deepest */
		d->expert_deep = (d->log2_cost[deep_idx - 1] +
				  d->log2_cost[deep_idx]) / 2.0f;
		/* Ensure deep > mid ordering */
		if (d->expert_deep <= d->expert_mid)
			d->expert_deep = d->expert_mid;
	} else {
		/* <= 2 C-states: collapse deep into long */
		d->expert_deep = d->expert_mid;
	}
}

/* ================================================================
 * Feature extraction helpers
 * ================================================================ */

struct logring_stats {
	float avg;
	float min;
	float max;
};

/*
 * Compute log_history statistics: avg, min, max.
 * SIMD fast path when the ring buffer is full (8 elements = 2 × xmm).
 */
static void logring_compute(const struct nap_cpu_data *d,
			    struct logring_stats *s)
{
	int i, n = d->hist_count;
	float sum;

	if (n == 0) {
		*s = (struct logring_stats){ 0 };
		return;
	}

	if (n == NAP_HISTORY_SIZE) {
		v4sf v0 = *(const v4sf *)&d->log_history[0];
		v4sf v1 = *(const v4sf *)&d->log_history[4];
		v4sf pmin, pmax, psum, t;

		pmin = __builtin_ia32_minps(v0, v1);
		pmax = __builtin_ia32_maxps(v0, v1);
		psum = v0 + v1;

		/* 4 → 2 */
		t = __builtin_ia32_movhlps(pmin, pmin);
		pmin = __builtin_ia32_minps(pmin, t);
		t = __builtin_ia32_movhlps(pmax, pmax);
		pmax = __builtin_ia32_maxps(pmax, t);
		t = __builtin_ia32_movhlps(psum, psum);
		psum = psum + t;

		/* 2 → 1 */
		t = __builtin_ia32_shufps(pmin, pmin, 0x55);
		pmin = __builtin_ia32_minps(pmin, t);
		t = __builtin_ia32_shufps(pmax, pmax, 0x55);
		pmax = __builtin_ia32_maxps(pmax, t);
		t = __builtin_ia32_shufps(psum, psum, 0x55);
		psum = psum + t;

		sum = psum[0];
		s->min = pmin[0];
		s->max = pmax[0];
	} else {
		float val;

		sum = d->log_history[0];
		s->min = sum;
		s->max = sum;

		for (i = 1; i < n; i++) {
			val = d->log_history[i];
			sum += val;
			s->min = float_min(s->min, val);
			s->max = float_max(s->max, val);
		}
	}

	s->avg = sum / (float)n;
}

/*
 * Extract 8 input features for the MLP.
 *
 *   [0] log2(sleep_length)           — next timer event
 *   [1] log2(last_residency)         — actual duration of last idle
 *   [2] log_hist avg                 — average recent idle duration
 *   [3] log_hist min                 — shortest recent idle
 *   [4] log_hist max                 — longest recent idle
 *   [5] signed log2(|pred_error|+1)  — prediction feedback
 *   [6] log2(busy_ns)               — pre-idle busy duration
 *   [7] log2(lat_req) - log2(deepest_lat) — PM QoS headroom
 */
static void nap_extract_features(struct cpuidle_driver *drv,
				 struct cpuidle_device *dev,
				 float out[NAP_INPUT_SIZE],
				 s64 latency_req)
{
	struct nap_cpu_data *d = this_cpu_ptr(&nap_data);
	struct logring_stats lr;
	ktime_t sleep_length, delta_tick;
	u64 busy_ns;
	float log_inputs[4] __aligned(16);
	float log_results[4] __aligned(16);

	sleep_length = tick_nohz_get_sleep_length(&delta_tick);
	busy_ns = local_clock() - d->prev_idle_exit;

	/*
	 * SSE log2 batch: 4 values in one fast_log2f_sse call.
	 *   [0] sleep_length   → out[0]
	 *   [1] last_residency → out[1], also stored to log_history
	 *   [2] busy_ns        → out[6]
	 *   [3] |pred_error_us| + 1 → out[5] (sign restored after)
	 */
	{
		float err_f = (float)(d->last_prediction_error / 1000);
		float abs_err = (err_f >= 0.0f) ? err_f : -err_f;

		log_inputs[0] = float_max((float)ktime_to_ns(sleep_length), 1.0f);
		log_inputs[1] = float_max((float)dev->last_residency_ns, 1.0f);
		log_inputs[2] = float_max((float)busy_ns, 1.0f);
		log_inputs[3] = abs_err + 1.0f;

		{
			v4sf log_in  = *(const v4sf *)log_inputs;
			v4sf log_out = fast_log2f_sse(log_in);
			*(v4sf *)log_results = log_out;
		}

		out[0] = log_results[0];
		out[1] = log_results[1];
		out[6] = log_results[2];

		/* out[5]: sign-preserving log2(|err_us| + 1) */
		{
			union { float f; u32 i; } res = { .f = log_results[3] };
			union { float f; u32 i; } sgn = { .f = err_f };

			res.i |= sgn.i & 0x80000000u;
			out[5] = res.f;
		}
	}

	/* Update log_history ring buffer */
	{
		int prev = (d->hist_idx - 1 + NAP_HISTORY_SIZE) % NAP_HISTORY_SIZE;
		d->log_history[prev] = log_results[1];
	}

	/* Compute log_history statistics: avg, min, max */
	logring_compute(d, &lr);
	out[2] = lr.avg;
	out[3] = lr.min;
	out[4] = lr.max;

	/* out[7]: log2(latency_req) - log2(deepest_lat), 0 if unconstrained */
	{
		u64 deepest_lat = drv->states[drv->state_count - 1]
				      .exit_latency_ns;
		bool lat_valid = (latency_req < PM_QOS_LATENCY_ANY_NS &&
				  deepest_lat > 0);

		if (lat_valid)
			out[7] = fast_log2f(float_max((float)latency_req, 1.0f))
			       - fast_log2f(float_max((float)deepest_lat, 1.0f));
		else
			out[7] = 0.0f;
	}

	d->last_predicted_ns = ktime_to_ns(sleep_length);
}

/* ================================================================
 * FPU entry point for nap_select
 *
 * Called within kernel_fpu_begin()/kernel_fpu_end().
 * Returns: selected idle state index (>= 0), or -1 to fall back
 *          to the integer heuristic.
 * ================================================================ */

int nap_fpu_select(struct cpuidle_driver *drv,
		   struct cpuidle_device *dev,
		   struct nap_cpu_data *d)
{
	s64 latency_req = cpuidle_governor_latency_req(dev->cpu);

	/* Handle deferred weight reset (set by sysfs or nap_enable) */
	if (unlikely(d->reset_pending)) {
		int e;

		for (e = 0; e < NAP_NUM_EXPERTS; e++)
			nap_init_weights(&d->expert_weights[e]);
		nap_init_log2_cost(d, drv);
		d->stats.learn_count = 0;
		d->needs_learn = false;
		d->reset_pending = false;
	}

	/* Deferred learning (always, even during warmup) */
	if (d->needs_learn) {
		float log2_eff = d->nn_output;
		float alpha = (float)d->overshoot_pctl_millths
			      / 1000.0f;
		int nn_selected = 0;
		bool is_overshoot;
		int i;

		/* Simulate which state the NN selected */
		for (i = drv->state_count - 1; i > 0; i--) {
			if (d->log2_cost[i] <= log2_eff) {
				nn_selected = i;
				break;
			}
		}

		/*
		 * Direct overshoot loss.
		 *
		 * Base the gradient on whether the simulated state
		 * selection actually caused overshoot
		 * (actual < target_residency).
		 *
		 * The asymmetric weight is encoded in the learning
		 * rate (not in d_out) so that gradient clamping
		 * cannot destroy the asymmetry.  d_out is ±1 and
		 * gets clipped symmetrically; the (1-α) vs α ratio
		 * is preserved through learn_lr.
		 *
		 * At equilibrium, P(overshoot) converges to α.
		 * α = overshoot_pctl / 1000.
		 */
		{
			float base_lr = (float)d->learning_rate_millths
					/ 1000.0f;

			is_overshoot = (nn_selected > 0 &&
				d->learn_actual_ns <
				drv->states[nn_selected].target_residency_ns);

			/*
			 * When the output was clamped at the upper
			 * limit (nn_output == features[0]), the NN
			 * is already predicting the maximum possible
			 * sleep time.  Non-overshoot events would
			 * push weights UP, but the output cannot
			 * actually increase.  Suppress this gradient
			 * to prevent unbounded weight growth in idle
			 * systems where natural overshoot rate < α.
			 *
			 * Overshoot events still learn normally
			 * (push DOWN) even when clamped.
			 */
			if (d->output_clamped && !is_overshoot) {
				d->learn_lr = 0;
				d->learn_d_out = 0;
			} else {
				d->learn_d_out = is_overshoot
					? 1.0f : -1.0f;
				d->learn_lr = is_overshoot
					? base_lr * (1.0f - alpha)
					: base_lr * alpha;
			}
		}

		d->stats.learn_count++;

		nap_nn_learn(d);
		d->needs_learn = false;
	}

	/*
	 * Feature extraction + NN forward pass.
	 * features_f32 is __aligned(64) in nap_cpu_data, satisfying
	 * AVX-512 vmovaps requirements.
	 */
	nap_extract_features(drv, dev, d->features_f32, latency_req);

	/* MoE: 3-way expert selection based on log2(sleep_length) */
	if (d->features_f32[0] >= d->expert_deep)
		d->active_expert = 2;		/* deep: deepest C-state */
	else if (d->features_f32[0] >= d->expert_mid)
		d->active_expert = 1;		/* long: nohz intermediate */
	else
		d->active_expert = 0;		/* short: tick-bound */
	d->active_w = &d->expert_weights[d->active_expert];

	nap_nn_forward(d->features_f32, &d->nn_output, d->hidden_out,
		       d->active_w);

	/*
	 * Clamp NN output: predicted sleep cannot exceed sleep_length
	 * (next timer event).  features_f32[0] = log2(sleep_length).
	 *
	 * Track whether the clamp was applied so the learning block
	 * can suppress "push up" gradients when the output is already
	 * at the maximum.  Without this, weights diverge unboundedly
	 * in idle systems where the natural overshoot rate < alpha.
	 */
	d->output_clamped = (d->nn_output > d->features_f32[0]);
	if (d->output_clamped)
		d->nn_output = d->features_f32[0];

	/*
	 * Threshold-based selection using NN predicted sleep time.
	 *
	 * The NN directly outputs log2(predicted_sleep) in ns.
	 * Select the deepest feasible state whose cost ≤ predicted_sleep.
	 */
	{
		float log2_eff = d->nn_output;
		int idx = 0, i;

		for (i = drv->state_count - 1; i > 0; i--) {
			if (dev->states_usage[i].disable)
				continue;
			if (drv->states[i].exit_latency_ns > latency_req)
				continue;
			if (d->log2_cost[i] <= log2_eff) {
				idx = i;
				break;
			}
		}
		return idx;
	}
}
