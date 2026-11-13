#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 — A3 entry (Jalur A3, LOCO-validated).

Blend 0.6 * V2 + 0.4 * V5-best-pure, then a gated slow asymmetric
bounce + down-continuation shift on q50 in Fisher space
f(p) = 2*arcsin(sqrt(p)).

Config (unanimous LOCO winner, all 6 folds; HistCV-confirmed):
  blend w(V2) = 0.6, V5-best-pure = V5Config(sum_alpha=0, bounce_beta=0,
  use_z_calib=False, winsor_k=10.0, core_scale=0.70, tail_scale=0.95,
  pi0_core_cap=0.0, jump_prob=0.0)
  asym: c_up = 1.0, c_dn = 0.4, anchor = mean(f[T-180:T-30]),
  tau_up = 2.4, tau_dn = 2.8, up_tau_h = 95.0, dn_tau_h = 110.0

Local validation (6 practice cases + 11 HistCV checkpoints):
  Practice 0.2651, LOCO-5 0.2480, p02 held-out 0.3635, HistCV 2.4167,
  mederr 0.7288, cov90 0.8977, Tier C 0.7176, ratio 0.7276.
  V2 baseline for comparison: 0.2830 / 0.2673 / 0.3729 / 2.5639 / 0.7675 /
  0.8337 / 0.8003 / 0.7374.
  Gates passed 5/6: LOCO-5 0.2480 <= 0.250, p02 0.3635 <= V2 0.3729,
  HistCV 2.4167 <= V2 2.5639, cov90 0.8977 in [0.87, 0.91],
  Tier C 0.7176 < V2 0.8003. The one thin miss: Practice pooled 0.2651
  vs the 0.262 target (gap +0.0031, pulled up by p02 0.3635).
  A2 (per-tier OOF multipliers on top of this blend) was REJECTED: it
  worsened p02 held-out (0.3946) and Tier C (0.8059) — overfit — so
  the blend below stays un-multiplied.

Pure Python 3.13 stdlib, no network, deterministic.
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes one <id>.out
per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import bisect
import glob
import json
import math
import os
from dataclasses import dataclass

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
HORIZON = 240

# ================= V2 backbone =================
V2_ANCHORS = [1, 2, 4, 8, 15, 30, 60, 100, 150, 200, 240]
V2_RECENCY_DECAY = 1.0
V2_WINSOR = 10.0
V2_EB_K = [300.0, 120.0, 80.0, 120.0, 300.0]
V2_REGIME_TAU = 60.0
V2_REGIME_LO = 0.55
V2_REGIME_HI = 1.85
V2_S_LOC_FLOOR_ABS = 5e-4
V2_S_LOC_FLOOR_FRAC = 0.25


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (+0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def clean_prices(prices):
    """Forward-fill missing/None/NaN, clamp to [0, 1]."""
    out = []
    last = 0.0
    for p in prices:
        if p is None:
            p = last
        else:
            p = float(p)
            if p != p:  # NaN
                p = last
            p = min(1.0, max(0.0, p))
        out.append(p)
        last = p
    return out


def v2_ewma(values, span):
    """Causal EWMA with the given span (weight halves every `span` minutes)."""
    alpha = 1.0 - 0.5 ** (1.0 / span)
    out = []
    cur = None
    for v in values:
        if cur is None:
            cur = v
        else:
            cur = alpha * v + (1.0 - alpha) * cur
        out.append(cur)
    return out


def v2_all_weighted_quantiles(pairs):
    """The 5 weighted empirical quantiles of (value, weight) pairs."""
    if not pairs:
        return [0.0] * 5
    sp = sorted(pairs, key=lambda t: t[0])
    total = 0.0
    for _, w in sp:
        total += w
    if total <= 0.0:
        mid = sp[len(sp) // 2][0]
        return [mid] * 5
    out = []
    acc = 0.0
    i = 0
    n = len(sp)
    for tau in QUANTILES:
        thresh = tau * total
        while i < n and acc < thresh:
            acc += sp[i][1]
            i += 1
        out.append(sp[i - 1][0] if i > 0 else sp[0][0])
    return out


def v2_ess(weights):
    """Effective sample size of a weighted sample."""
    s1 = 0.0
    s2 = 0.0
    for w in weights:
        s1 += w
        s2 += w * w
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def v2_g_factor(p):
    """Price-level variance factor sqrt(max(p(1-p), 0.0025*0.9975))."""
    return math.sqrt(max(p * (1.0 - p), 0.0025 * 0.9975))


def v2_sample_pairs(prices, s_loc, h, tier):
    """Historical (value, recency-weight) pairs at anchor horizon h.

    Tier 1/2: standardized residual r = (p[t+h] - p[t]) / s_loc[t],
    winsorized to [-10, +10]. Tier 3: raw price change dp (never divide
    by a tiny s_loc). Stride max(1, h//6) cuts autocorrelation overlap.
    """
    T = len(prices)
    step = h // 6
    if step < 1:
        step = 1
    pairs = []
    t = 0
    while t + h < T:
        if tier == 3:
            v = prices[t + h] - prices[t]
        else:
            v = (prices[t + h] - prices[t]) / s_loc[t]
            if v > V2_WINSOR:
                v = V2_WINSOR
            elif v < -V2_WINSOR:
                v = -V2_WINSOR
        age = (T - 1) - (t + h)
        w = math.exp(-V2_RECENCY_DECAY * age / 1440.0)
        pairs.append((v, w))
        t += step
    return pairs


def v2_prep_market(market):
    """Per-market prep: tier, local scale, volatility, anchor samples."""
    prices = clean_prices(market["prices"])
    T = len(prices)
    p0 = prices[-1] if T else 0.0
    d1 = [0.0] * T
    for t in range(1, T):
        d1[t] = prices[t] - prices[t - 1]
    absd1 = [abs(x) for x in d1]

    def act_rate(window):
        n = window if window < T - 1 else T - 1
        if n <= 0:
            return 0.0
        cnt = 0
        for t in range(T - n, T):
            if abs(d1[t]) > 1e-6:
                cnt += 1
        return cnt / n

    act_12h = act_rate(720)

    def mean_abs(window):
        n = window if window < T - 1 else T - 1
        if n <= 0:
            return 0.0
        s = 0.0
        for t in range(T - n, T):
            s += absd1[t]
        return s / n

    vol_fast = mean_abs(60)
    vol_slow = mean_abs(T - 1)

    if p0 >= 0.06 or act_12h >= 0.15:
        tier = 1
    elif p0 >= 0.015:
        tier = 2
    else:
        tier = 3

    p_mean = sum(prices) / T if T else 0.0
    g_mean = v2_g_factor(p_mean)
    ew = v2_ewma(absd1, 60)
    floor = V2_S_LOC_FLOOR_FRAC * vol_slow
    if floor < V2_S_LOC_FLOOR_ABS:
        floor = V2_S_LOC_FLOOR_ABS
    s_loc = [0.0] * T
    for t in range(T):
        v = 0.65 * ew[t] + 0.35 * vol_slow * (v2_g_factor(prices[t]) / g_mean if g_mean > 1e-4 else 1.0)
        if v < floor:
            v = floor
        s_loc[t] = v
    avg_s_loc = sum(s_loc) / T if T else 1.0
    s_now = s_loc[T - 1] if T else 1.0

    own = []
    for h in V2_ANCHORS:
        own.append(v2_sample_pairs(prices, s_loc, h, tier))

    return {
        "id": market["id"],
        "tier": tier,
        "p0": p0,
        "vol_fast": vol_fast,
        "avg_s_loc": avg_s_loc,
        "s_now": s_now,
        "own": own,
    }


def v2_regime_scale(vol_fast, avg_s_loc, h):
    """Conditional regime ratio, decaying to 1.0, clamped [0.55, 1.85]."""
    ratio = vol_fast / avg_s_loc if avg_s_loc > 0.0 else 1.0
    s = 1.0 + (ratio - 1.0) * math.exp(-h / V2_REGIME_TAU)
    if s < V2_REGIME_LO:
        s = V2_REGIME_LO
    elif s > V2_REGIME_HI:
        s = V2_REGIME_HI
    return s


def v2_interp_anchors(anchor_vals, h):
    """Linear interpolation over a sqrt(h) grid between anchor values."""
    if h <= V2_ANCHORS[0]:
        return list(anchor_vals[0])
    if h >= V2_ANCHORS[-1]:
        return list(anchor_vals[-1])
    sh = math.sqrt(h)
    for i in range(len(V2_ANCHORS) - 1):
        a0 = V2_ANCHORS[i]
        a1 = V2_ANCHORS[i + 1]
        if a0 <= h <= a1:
            s0 = math.sqrt(a0)
            s1 = math.sqrt(a1)
            u = (sh - s0) / (s1 - s0) if s1 > s0 else 0.0
            return [
                anchor_vals[i][j] * (1.0 - u) + anchor_vals[i + 1][j] * u
                for j in range(5)
            ]
    return list(anchor_vals[-1])


def v2_forecast_prep(prep, pool_q):
    """240 rows x 5 quantiles for one prepared market."""
    tier = prep["tier"]
    p0 = prep["p0"]
    s_now = prep["s_now"]
    vol_fast = prep["vol_fast"]
    avg_s_loc = prep["avg_s_loc"]

    anchor_q = []
    for ai, h in enumerate(V2_ANCHORS):
        own = prep["own"][ai]
        pq = pool_q.get((tier, ai))
        if pq is None:
            pq = [0.0] * 5
        if own:
            oq = v2_all_weighted_quantiles(own)
            e = v2_ess([w for _, w in own])
            blended = []
            for j in range(5):
                w_own = e / (e + V2_EB_K[j]) if (e + V2_EB_K[j]) > 0.0 else 0.0
                blended.append(w_own * oq[j] + (1.0 - w_own) * pq[j])
        else:
            blended = list(pq)
        med = blended[2]
        anchor_q.append([q - med for q in blended])

    rows = []
    for h in range(1, HORIZON + 1):
        qc = v2_interp_anchors(anchor_q, h)
        if tier == 3:
            vals = [p0 + qc[j] for j in range(5)]
        else:
            sc = v2_regime_scale(vol_fast, avg_s_loc, h)
            vals = [p0 + qc[j] * s_now * sc for j in range(5)]
        for j in range(5):
            if vals[j] < 0.0:
                vals[j] = 0.0
            elif vals[j] > 1.0:
                vals[j] = 1.0
        for j in range(1, 5):
            if vals[j] < vals[j - 1]:
                vals[j] = vals[j - 1]
        rows.append([price6(v) for v in vals])
    return rows


def v2_process_case(case):
    """Forecast every market of one case. Returns the forecasts list."""
    markets = case["markets"]
    preps = []
    pools = {}
    for m in markets:
        prep = v2_prep_market(m)
        preps.append(prep)
        for ai in range(len(V2_ANCHORS)):
            key = (prep["tier"], ai)
            if key in pools:
                pools[key].extend(prep["own"][ai])
            else:
                pools[key] = list(prep["own"][ai])
    pool_q = {}
    for key, pairs in pools.items():
        pool_q[key] = v2_all_weighted_quantiles(pairs)
    forecasts = []
    for prep in preps:
        forecasts.append({"id": prep["id"], "q": v2_forecast_prep(prep, pool_q)})
    return forecasts


# ================= V5 backbone (best-pure config) =================
@dataclass(frozen=True)
class V5Config:
    sum_eq_mode: str = "one"
    sum_med_hours: float = 24.0
    sum_blend_w1: float = 0.50
    sum_weight_mode: str = "prop"
    sum_alpha: float = 0.75
    sum_tau: float = 60.0
    sum_min_markets: int = 19
    bounce_mode: str = "ema"
    bounce_win: int = 3
    bounce_beta: float = 0.70
    bounce_er_max: float = 1.01
    bounce_tau: float = 30.0
    bounce_before_sum: bool = False
    hl_fast: float = 20.0
    hl_med: float = 150.0
    hl_slow: float = 1200.0
    winsor_k: float = 3.5
    tau_fast: float = 40.0
    tau_med: float = 300.0
    horizon_power: float = 0.0
    evt_gamma: float = 0.15
    asym_alpha: float = 0.35
    logit_bound_power: float = 0.0
    pi0_core_cap: float = 0.85
    pi0_tail_cap: float = 0.25
    jump_prob: float = 0.06
    jump_ratio: float = 3.0
    core_scale: float = 0.95
    tail_scale: float = 1.05
    use_z_calib: bool = True
    z_calib_shrink: float = 0.50
    z_calib_stride: int = 30
    tau_rec_hours: float = 18.0
    min_tail_floor_1m: float = 0.0004


# Best-pure (pure persistence median): LOCO-verified median_error == persistence.
V5_BEST_PURE = V5Config(sum_alpha=0.0, bounce_beta=0.0, use_z_calib=False,
                        winsor_k=10.0, core_scale=0.70, tail_scale=0.95,
                        pi0_core_cap=0.0, jump_prob=0.0)


def v5_median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n & 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def v5_weighted_quantile(pairs, tau):
    n = len(pairs)
    if n == 0:
        return 0.0
    pairs.sort(key=lambda vw: vw[0])
    tw = 0.0
    for _, w in pairs:
        tw += w
    if tw <= 0.0:
        return 0.0
    mids = [0.0] * n
    vals = [0.0] * n
    cum = 0.0
    inv_tw = 1.0 / tw
    for i, (v, w) in enumerate(pairs):
        vals[i] = v
        mids[i] = (cum + 0.5 * w) * inv_tw
        cum += w
    if tau <= mids[0]:
        return vals[0]
    if tau >= mids[-1]:
        return vals[-1]
    idx = bisect.bisect_right(mids, tau)
    m0 = mids[idx - 1]
    m1 = mids[idx]
    if m1 <= m0:
        return vals[idx]
    return vals[idx - 1] + ((tau - m0) / (m1 - m0)) * (vals[idx] - vals[idx - 1])


def v5_zi_laplace_jd_ppf(tau, pi0, b_minus, b_plus, p_jump, r_jump, cache=None):
    """Exact inverse CDF of Zero-Inflated Asymmetric Laplace + Jump-Diffusion
    Mixture. Returns offset in standardized scale (caller multiplies by s0)."""
    if pi0 < 0.0:
        pi0 = 0.0
    elif pi0 > 0.96:
        pi0 = 0.96
    half_p = 0.5 * pi0
    if abs(tau - 0.5) < 1e-9:
        return 0.0
    if 0.5 - half_p <= tau <= 0.5 + half_p:
        return 0.0

    # Pure Laplace closed form when p_jump <= 0
    if p_jump <= 1e-9:
        if tau < 0.5 - half_p:
            ratio = max(1e-8, (2.0 * tau) / (1.0 - pi0))
            return b_minus * math.log(ratio)
        ratio = max(1e-8, (2.0 * (1.0 - tau)) / (1.0 - pi0))
        return -b_plus * math.log(ratio)

    # 2-component Laplace jump-diffusion bisection (24 iterations)
    one_m_pi = 1.0 - pi0
    one_m_pj = 1.0 - p_jump
    if tau < 0.5 - half_p:
        lo, hi = -30.0 * b_minus * r_jump, 0.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            cdf = one_m_pi * 0.5 * (
                one_m_pj * math.exp(mid / b_minus) + p_jump * math.exp(mid / (b_minus * r_jump))
            )
            if cdf < tau:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lo, hi = 0.0, 30.0 * b_plus * r_jump
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        tail = 0.5 * (
            one_m_pj * math.exp(-mid / b_plus) + p_jump * math.exp(-mid / (b_plus * r_jump))
        )
        cdf = pi0 + one_m_pi * (1.0 - tail)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def v5_compute_q50_components(p_clean, vol_proxy, cfg):
    """Returns (p_start[m], sum_target_delta[m], bounce_delta[m]) at checkpoint t0."""
    m_count = len(p_clean)
    t_len = len(p_clean[0])
    t0 = t_len - 1
    p0 = [p_clean[m][t0] for m in range(m_count)]

    # Bounce delta per market
    bounce_delta = [0.0] * m_count
    w_b = max(2, min(cfg.bounce_win, t_len - 1))
    for m in range(m_count):
        pm = p_clean[m]
        if cfg.bounce_mode == "lag":
            ref = pm[t0 - w_b]
        elif cfg.bounce_mode == "ema":
            alpha_b = 2.0 / (w_b + 1.0)
            ema = pm[max(0, t0 - 3 * w_b)]
            for s in range(max(1, t0 - 3 * w_b + 1), t0 + 1):
                ema = alpha_b * pm[s] + (1.0 - alpha_b) * ema
            ref = ema
        else:
            ref = v5_median(pm[t0 - w_b + 1 : t0 + 1])

        path_len = 0.0
        for s in range(t0 - w_b + 1, t0 + 1):
            path_len += abs(pm[s] - pm[s - 1])
        net = abs(pm[t0] - pm[t0 - w_b])
        er = (net / path_len) if path_len > 1e-9 else 1.0
        if er <= cfg.bounce_er_max:
            bounce_delta[m] = cfg.bounce_beta * (ref - p0[m])

    # Base prices before sum reversion
    if cfg.bounce_before_sum and cfg.bounce_tau <= 1.0:
        p_for_sum = [min(1.0, max(0.0, p0[m] + bounce_delta[m])) for m in range(m_count)]
    else:
        p_for_sum = list(p0)

    # Equilibrium sum S_eq (completeness gate: skip sum-reversion when the
    # market set is incomplete, i.e. fewer than sum_min_markets markets).
    if m_count < cfg.sum_min_markets:
        return p0, [0.0] * m_count, bounce_delta
    s_cur = sum(p_for_sum)
    if cfg.sum_eq_mode == "one":
        s_eq = 1.0
    else:
        look = min(t_len, max(60, int(cfg.sum_med_hours * 60.0)))
        step = max(1, look // 120)
        s_hist = [sum(p_clean[m][t] for m in range(m_count)) for t in range(t_len - look, t_len, step)]
        s_med = v5_median(s_hist)
        if cfg.sum_eq_mode == "rolling_med":
            s_eq = s_med
        else:
            s_eq = cfg.sum_blend_w1 * 1.0 + (1.0 - cfg.sum_blend_w1) * s_med

    excess = s_cur - s_eq
    sum_delta = [0.0] * m_count
    if abs(excess) > 1e-7 and s_cur > 1e-6:
        if cfg.sum_weight_mode == "prop":
            for m in range(m_count):
                sum_delta[m] = p_for_sum[m] * (s_eq / s_cur - 1.0)
        elif cfg.sum_weight_mode == "pvar":
            ws = [p_for_sum[m] * (1.0 - p_for_sum[m]) for m in range(m_count)]
            sw = sum(ws)
            if sw > 1e-9:
                for m in range(m_count):
                    sum_delta[m] = -excess * (ws[m] / sw)
        else:
            sw = sum(vol_proxy)
            if sw > 1e-9:
                for m in range(m_count):
                    sum_delta[m] = -excess * (vol_proxy[m] / sw)

    return p0, sum_delta, bounce_delta


def v5_forecast_case(raw_prices, horizon=240, cfg=None):
    if cfg is None:
        cfg = V5Config()
    m_count = len(raw_prices)
    if m_count == 0:
        return []
    t_len = len(raw_prices[0])
    if t_len < 2:
        out = []
        for m in range(m_count):
            p0 = min(1.0, max(0.0, raw_prices[m][-1] if raw_prices[m] else 0.5))
            out.append([[p0] * 5 for _ in range(horizon)])
        return out

    p_clean = [[min(1.0, max(0.0, v)) for v in raw_prices[m]] for m in range(m_count)]
    t0 = t_len - 1

    # 1. Winsorized Multi-Scale EWMA Volatility (fast, med, slow + up/down asymmetry)
    alpha_f = 1.0 - math.exp(-math.log(2.0) / max(1.0, cfg.hl_fast))
    alpha_m = 1.0 - math.exp(-math.log(2.0) / max(1.0, cfg.hl_med))
    alpha_s = 1.0 - math.exp(-math.log(2.0) / max(1.0, cfg.hl_slow))
    alpha_act = 1.0 - math.exp(-math.log(2.0) / 180.0)
    inv_sqrt5 = 1.0 / math.sqrt(5.0)
    inv_sqrt15 = 1.0 / math.sqrt(15.0)
    inv_sqrt60 = 1.0 / math.sqrt(60.0)

    vf_all = [[0.0] * t_len for _ in range(m_count)]
    vm_all = [[0.0] * t_len for _ in range(m_count)]
    vs_all = [[0.0] * t_len for _ in range(m_count)]
    asym_ratio_t0 = [1.0] * m_count
    act_t0 = [0.0] * m_count
    sf_sum = [0.0] * t_len
    ss_sum = [0.0] * t_len

    for m in range(m_count):
        pm = p_clean[m]
        init_n = min(t_len, 360)
        init_d = [abs(pm[t] - pm[t - 5]) * inv_sqrt5 for t in range(5, init_n)]
        base_init = (sum(init_d) / len(init_d)) if init_d else 0.001
        vf = base_init
        vm = base_init
        vs = base_init
        v_up = base_init
        v_dn = base_init
        nz_ewma = 0.0
        vf_row = vf_all[m]
        vm_row = vm_all[m]
        vs_row = vs_all[m]
        af = alpha_f
        am = alpha_m
        a_s = alpha_s
        om_f = 1.0 - af
        om_m = 1.0 - am
        om_s = 1.0 - a_s
        wk = cfg.winsor_k
        for t in range(t_len):
            r5 = (pm[t] - pm[t - 5]) * inv_sqrt5 if t >= 5 else (pm[t] - pm[0]) / math.sqrt(max(1, t))
            d5 = abs(r5)
            d15 = abs(pm[t] - pm[t - 15]) * inv_sqrt15 if t >= 15 else d5
            d60 = abs(pm[t] - pm[t - 60]) * inv_sqrt60 if t >= 60 else d15

            cap = wk * (vs if vs > vm else vm)
            if cap < 0.0005 * wk:
                cap = 0.0005 * wk
            d5_w = d5 if d5 < cap else cap
            d15_w = d15 if d15 < cap else cap
            d60_w = d60 if d60 < cap else cap

            vf = af * d5_w + om_f * vf
            vm = am * d15_w + om_m * vm
            vs = a_s * d60_w + om_s * vs
            vf_row[t] = vf
            vm_row[t] = vm
            vs_row[t] = vs
            sf_sum[t] += vf
            ss_sum[t] += vs

            if t >= 5:
                if r5 > 0.0:
                    v_up = am * (d5 if d5 < cap else cap) + om_m * v_up
                elif r5 < 0.0:
                    v_dn = am * (d5 if d5 < cap else cap) + om_m * v_dn
            if t >= 1:
                moved = 1.0 if abs(pm[t] - pm[t - 1]) > 1e-6 else 0.0
                nz_ewma = alpha_act * moved + (1.0 - alpha_act) * nz_ewma

        act_t0[m] = nz_ewma
        v_avg = 0.5 * (v_up + v_dn)
        if v_avg > 1e-7:
            asym_ratio_t0[m] = max(0.65, min(1.50, v_up / v_avg))

    # 2. Cross-market event excitement multiplier (from inline sums)
    evt_mult_t = [1.0] * t_len
    if cfg.evt_gamma > 0.0:
        eg = cfg.evt_gamma
        for t in range(t_len):
            ss = ss_sum[t]
            ratio = (sf_sum[t] / ss) if ss > 1e-12 else 1.0
            if ratio < 0.7:
                ratio = 0.7
            elif ratio > 2.5:
                ratio = 2.5
            evt_mult_t[t] = ratio ** eg

    # 3. q50 components (Sum-Reverting Median + Bounce Reversal)
    vol_proxy = [vm_all[m][t0] for m in range(m_count)]
    p0, sum_delta, bounce_delta = v5_compute_q50_components(p_clean, vol_proxy, cfg)

    # 4. Precompute Heston variance integrals for h = 1..horizon
    tau_f = max(1.0, cfg.tau_fast)
    tau_m = max(tau_f + 1.0, cfg.tau_med)
    h_coeffs = []
    for h in range(1, horizon + 1):
        i_f = tau_f * (1.0 - math.exp(-h / tau_f))
        i_m = max(0.0, tau_m * (1.0 - math.exp(-h / tau_m)) - i_f)
        i_s = max(0.0, float(h) - tau_m * (1.0 - math.exp(-h / tau_m)))
        tilt = ((h / 60.0) ** cfg.horizon_power) if cfg.horizon_power != 0.0 else 1.0
        h_coeffs.append((i_f, i_m, i_s, tilt))

    def _scale(m, t, h_idx):
        i_f, i_m, i_s, tilt = h_coeffs[h_idx]
        vf = vf_all[m][t]
        vm = vm_all[m][t]
        vs = vs_all[m][t]
        var = vf * vf * i_f + vm * vm * i_m + vs * vs * i_s
        return (math.sqrt(var) * evt_mult_t[t] * tilt) if var > 0.0 else 0.0

    # 5. Causal zero-inflation rate & optional z-calibration across knot horizons
    knot_hs = (5, 20, 60, 120, 240)
    knot_hs = tuple(h for h in knot_hs if h <= horizon)
    if not knot_hs or knot_hs[-1] != horizon:
        knot_hs = knot_hs + (horizon,)

    inv_tau_rec = 1.0 / max(60.0, cfg.tau_rec_hours * 60.0)
    warmup = min(180, t_len // 5)
    z_cal_mult = [1.0, 1.0, 1.0, 1.0, 1.0]

    pi0_knots = [[0.0] * len(knot_hs) for _ in range(m_count)]
    z_pool_by_qi = [[], [], [], [], []]

    for ki, h in enumerate(knot_hs):
        end_t = t0 - h
        if end_t <= warmup + 10:
            continue
        h_idx = h - 1
        step = cfg.z_calib_stride
        for m in range(m_count):
            pm = p_clean[m]
            z_w = 0.0
            tot_w = 0.0
            for t in range(warmup, end_t + 1, step):
                w = math.exp(-(end_t - t) * inv_tau_rec)
                tot_w += w
                dp = pm[t + h] - pm[t]
                if abs(dp) < 1e-6:
                    z_w += w
                if cfg.use_z_calib:
                    sc = _scale(m, t, h_idx)
                    if sc > 1e-6:
                        z = max(-10.0, min(10.0, dp / sc))
                        for qi in (0, 1, 3, 4):
                            z_pool_by_qi[qi].append((z, w))
            pi0_knots[m][ki] = (z_w / tot_w) if tot_w > 0.0 else 0.0

    theo_lap = [
        v5_zi_laplace_jd_ppf(tau, 0.0, 1.0, 1.0, cfg.jump_prob, cfg.jump_ratio)
        for tau in QUANTILES
    ]
    if cfg.use_z_calib and z_pool_by_qi[0]:
        for qi in (0, 1, 3, 4):
            emp_z = v5_weighted_quantile(z_pool_by_qi[qi], QUANTILES[qi])
            if abs(theo_lap[qi]) > 1e-6:
                ratio = emp_z / theo_lap[qi]
                ratio = max(0.55, min(1.85, ratio))
                z_cal_mult[qi] = (1.0 - cfg.z_calib_shrink) * 1.0 + cfg.z_calib_shrink * ratio

    # 6. Generate final [M][horizon][5] predictions
    sum_tau = max(1.0, cfg.sum_tau)
    b_tau = max(0.0, cfg.bounce_tau)
    if b_tau <= 1.0:
        b_fac_list = [1.0] * horizon
    else:
        b_fac_list = [1.0 - math.exp(-(h + 1) / b_tau) for h in range(horizon)]
    s_fac_list = [cfg.sum_alpha * (1.0 - math.exp(-(h + 1) / sum_tau)) for h in range(horizon)]
    out = []

    for m in range(m_count):
        p_base = p0[m]
        sd = sum_delta[m]
        bd = bounce_delta[m]

        a_up = 1.0 + cfg.asym_alpha * (asym_ratio_t0[m] - 1.0)
        a_dn = 1.0 - cfg.asym_alpha * (asym_ratio_t0[m] - 1.0)
        if cfg.logit_bound_power > 0.0:
            up_room = min(1.8, max(0.35, ((1.0 - p_base + 0.03) / (p_base + 0.03)) ** (0.25 * cfg.logit_bound_power)))
            dn_room = min(1.8, max(0.35, ((p_base + 0.03) / (1.0 - p_base + 0.03)) ** (0.25 * cfg.logit_bound_power)))
            b_plus = a_up * up_room
            b_minus = a_dn * dn_room
        else:
            b_plus = a_up
            b_minus = a_dn

        m_rows = []
        z_knots = []
        for ki in range(len(knot_hs)):
            pi0_k = pi0_knots[m][ki]
            pi_c = min(cfg.pi0_core_cap, pi0_k)
            pi_t = min(cfg.pi0_tail_cap, pi0_k)
            z_knots.append([
                v5_zi_laplace_jd_ppf(0.05, pi_t, b_minus, b_plus, cfg.jump_prob, cfg.jump_ratio),
                v5_zi_laplace_jd_ppf(0.25, pi_c, b_minus, b_plus, cfg.jump_prob, cfg.jump_ratio),
                v5_zi_laplace_jd_ppf(0.75, pi_c, b_minus, b_plus, cfg.jump_prob, cfg.jump_ratio),
                v5_zi_laplace_jd_ppf(0.95, pi_t, b_minus, b_plus, cfg.jump_prob, cfg.jump_ratio),
            ])
        k_idx = 0
        for h in range(1, horizon + 1):
            h_idx = h - 1
            b_fac = b_fac_list[h_idx]
            s_fac = s_fac_list[h_idx]
            q50 = p_base + b_fac * bd + s_fac * sd
            if q50 < 0.0:
                q50 = 0.0
            elif q50 > 1.0:
                q50 = 1.0

            while k_idx + 1 < len(knot_hs) and knot_hs[k_idx + 1] < h:
                k_idx += 1
            if h <= knot_hs[0]:
                z05, z25, z75, z95 = z_knots[0]
            elif k_idx + 1 >= len(knot_hs):
                z05, z25, z75, z95 = z_knots[-1]
            else:
                h0 = knot_hs[k_idx]
                h1 = knot_hs[k_idx + 1]
                frac = (h - h0) / (h1 - h0) if h1 > h0 else 0.0
                zk0 = z_knots[k_idx]
                zk1 = z_knots[k_idx + 1]
                z05 = zk0[0] + frac * (zk1[0] - zk0[0])
                z25 = zk0[1] + frac * (zk1[1] - zk0[1])
                z75 = zk0[2] + frac * (zk1[2] - zk0[2])
                z95 = zk0[3] + frac * (zk1[3] - zk0[3])

            s_h = _scale(m, t0, h_idx)

            tail_fl = cfg.min_tail_floor_1m * math.sqrt(float(h))
            q05 = q50 + z05 * cfg.tail_scale * z_cal_mult[0] * s_h
            q25 = q50 + z25 * cfg.core_scale * z_cal_mult[1] * s_h
            q75 = q50 + z75 * cfg.core_scale * z_cal_mult[3] * s_h
            q95 = q50 + z95 * cfg.tail_scale * z_cal_mult[4] * s_h

            if tail_fl > 0.0:
                if q05 > q50 - tail_fl:
                    q05 = q50 - tail_fl
                if q95 < q50 + tail_fl:
                    q95 = q50 + tail_fl

            row = [q05, q25, q50, q75, q95]
            for qi in range(5):
                if row[qi] < 0.0:
                    row[qi] = 0.0
                elif row[qi] > 1.0:
                    row[qi] = 1.0
            if row[1] > row[2]:
                row[1] = row[2]
            if row[0] > row[1]:
                row[0] = row[1]
            if row[3] < row[2]:
                row[3] = row[2]
            if row[4] < row[3]:
                row[4] = row[3]
            m_rows.append(row)
        out.append(m_rows)
    return out


# ================= A3 pipeline: blend + gated slow asymmetric shift =================
A3_W_V2 = 0.6
A3_C_UP = 1.0
A3_C_DN = 0.4
A3_ANCHOR = "mean"
A3_TAU_UP = 2.4
A3_TAU_DN = 2.8
A3_UP_TAU_H = 95.0
A3_DN_TAU_H = 110.0
A3_LOG2 = math.log(2.0)
A3_PI = math.pi


def a3_fisher(p):
    if p <= 1e-6:
        p = 1e-6
    elif p >= 1.0 - 1e-6:
        p = 1.0 - 1e-6
    return 2.0 * math.asin(math.sqrt(p))


def a3_inv_fisher(x):
    if x <= 0.0:
        return 0.0
    if x >= A3_PI:
        return 1.0
    s = math.sin(0.5 * x)
    return s * s


def a3_ewma_series(vals, hl):
    alpha = 1.0 - math.exp(-A3_LOG2 / hl)
    one_m = 1.0 - alpha
    out = [0.0] * len(vals)
    cur = vals[0]
    out[0] = cur
    for i in range(1, len(vals)):
        cur = alpha * vals[i] + one_m * cur
        out[i] = cur
    return out


def a3_vol_fisher(rows):
    """Per market (f_list, v_ref, s0): v_ref is the local Fisher vol scale.

    Winsorized |5-min Fisher moves| (96th-pct cap), EWMA(60)/EWMA(240)
    average, floored at max(0.10 * mean, 2e-4). s0 is the first index where
    the market starts moving (used to bound the anchor window).
    """
    out = []
    for arr in rows:
        fm = [a3_fisher(x) for x in arr]
        T = len(fm)
        s0 = max(0, T - 720)
        v0 = arr[0]
        for t in range(1, T):
            if abs(arr[t] - v0) > 1e-9:
                s0 = max(0, t - 60)
                break
        df5 = [0.0] * T
        active = []
        for t in range(5, T):
            v = abs(fm[t] - fm[t - 5])
            df5[t] = v
            if t >= s0 + 5:
                active.append(v)
        if len(active) < 20:
            cap = 0.01
            floor_v = 5e-4
        else:
            sd = sorted(active)
            cap = max(sd[int(0.96 * (len(sd) - 1))], 2e-4)
            mean_act = sum(min(v, cap) for v in active) / len(active)
            floor_v = max(0.10 * mean_act, 2e-4)
        df5_w = [min(v, cap) for v in df5]
        ew_f = a3_ewma_series(df5_w, 60.0)
        ew_m = a3_ewma_series(df5_w, 240.0)
        v_ref = max(0.5 * ew_f[-1] + 0.5 * ew_m[-1], floor_v)
        out.append((fm, v_ref, s0))
    return out


def a3_apply_asym(fc, rows, ids):
    """Shift all quantiles in Fisher space by the gated slow asym signal.

    anchor = mean(f[T-180:T-30]); delta = f0 - anchor.
    If delta > tau_up * v_ref: up-signal = -c_up * sc * tanh((delta - t_up)/sc)
    (bounce back down after a slow run-up). If delta < -tau_dn * v_ref:
    down-signal = -c_dn * sc * tanh((|delta| - t_dn)/sc) (down-continuation:
    a slow slide keeps sliding). Both shifts are <= 0; the shift ramps in
    with horizon (1 - exp(-h/tau_h)). Applied to every quantile row, then
    re-sorted for non-crossing.
    """
    vols = a3_vol_fisher(rows)
    out = {}
    for k, mid in enumerate(ids):
        fm, v_ref, s0 = vols[k]
        T = len(fm)
        f0 = fm[-1]
        ws, we = max(s0, T - 180), max(s0 + 1, T - 30)
        anch = sum(fm[ws:we]) / (we - ws)
        delta = f0 - anch
        t_up = max(0.020, A3_TAU_UP * v_ref)
        t_dn = max(0.025, A3_TAU_DN * v_ref)
        up_sig = dn_sig = 0.0
        if delta > t_up and A3_C_UP > 0.0:
            sc = max(0.04, 4.0 * v_ref)
            up_sig = -A3_C_UP * sc * math.tanh((delta - t_up) / sc)
        elif delta < -t_dn and A3_C_DN > 0.0:
            sc = max(0.04, 4.0 * v_ref)
            dn_sig = -A3_C_DN * sc * math.tanh(((-delta) - t_dn) / sc)
        rows_out = []
        for h in range(1, 241):
            b_up = 1.0 - math.exp(-h / A3_UP_TAU_H)
            b_dn = 1.0 - math.exp(-h / A3_DN_TAU_H)
            sh = up_sig * b_up + dn_sig * b_dn
            if sh == 0.0:
                rows_out.append(list(fc[mid][h - 1]))
                continue
            r = [a3_inv_fisher(a3_fisher(v) + sh) for v in fc[mid][h - 1]]
            r.sort()
            rows_out.append([min(1.0, max(0.0, v)) for v in r])
        out[mid] = rows_out
    return out


def a3_blend(fc_a, fc_b, w, ids):
    """q = w*a + (1-w)*b per (market, minute); sort each row for non-crossing.

    Called with w = A3_W_V2 = 0.6 on the V2 backbone (fc_a) and 0.4 on
    V5-best-pure (fc_b, pure-persistence median from the LOCO-validated
    162-config grid: winsor_k=10, core_scale=0.70, tail_scale=0.95,
    pi0_core_cap=0, jump_prob=0). V2 supplies the interval shape, V5 the
    persistence median; the 0.6/0.4 split is the unanimous LOCO winner
    (Practice 0.2754, LOCO-5 0.2593 on the blend alone, before the asym
    shift). No per-tier multipliers are applied here: A2's 12-DOF OOF
    multipliers looked better in-sample (0.2692) but worsened the p02
    held-out (0.3946 vs V2 0.3729) and Tier C (0.8059 vs 0.8003), so
    they were rejected and the blend stays un-multiplied.
    """
    out = {}
    for mid in ids:
        qa, qb = fc_a[mid], fc_b[mid]
        rows = []
        for h in range(240):
            r = sorted(w * qa[h][j] + (1.0 - w) * qb[h][j] for j in range(5))
            rows.append([min(1.0, max(0.0, v)) for v in r])
        out[mid] = rows
    return out


def a3_pad_rows(case_markets):
    """Clean each market's prices; left-pad shorter ones with their first
    value so all rows share the max length (V5 requires equal lengths)."""
    cleaned = [clean_prices(m["prices"]) for m in case_markets]
    T = max(len(p) for p in cleaned)
    rows = []
    for p in cleaned:
        if len(p) < T:
            p = [p[0]] * (T - len(p)) + p
        rows.append(p)
    ids = [m["id"] for m in case_markets]
    return rows, ids


def process_case_a3(case):
    """V2 + V5-best-pure -> blend w=0.6 -> gated slow asym shift on q50.

    Returns the forecasts list in the same format as V2's process_case.
    """
    markets = case["markets"]
    rows, ids = a3_pad_rows(markets)

    # V2 backbone (already price6-rounded per row)
    v2_out = v2_process_case(case)
    fc_v2 = {f["id"]: f["q"] for f in v2_out}

    # V5-best-pure (pure persistence median), rounded to 6dp like the harness
    v5_rows = v5_forecast_case(rows, horizon=HORIZON, cfg=V5_BEST_PURE)
    fc_v5 = {}
    for i, mid in enumerate(ids):
        fc_v5[mid] = [[round(min(1.0, max(0.0, v)), 6) + 0.0 for v in row]
                      for row in v5_rows[i]]

    # Blend 0.6 * V2 + 0.4 * V5-best-pure
    fc_b = a3_blend(fc_v2, fc_v5, A3_W_V2, ids)

    # Gated slow asymmetric bounce + down-continuation (Fisher space)
    fc_a = a3_apply_asym(fc_b, rows, ids)

    forecasts = []
    for mid in ids:
        forecasts.append({"id": mid,
                          "q": [[price6(v) for v in row] for row in fc_a[mid]]})
    return forecasts


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    inputs = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    for path in inputs:
        name = os.path.basename(path)
        cid = name[:-3] if name.endswith(".in") else name
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        forecasts = process_case_a3(case)
        out_path = os.path.join(output_dir, cid + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
