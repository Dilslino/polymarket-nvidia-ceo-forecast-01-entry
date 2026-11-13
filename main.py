# polymarket-nvidia-ceo-forecast-01 -- V2 entry (Python 3.13, stdlib only)
#
# Model: Self-Calibrating Multi-Scale Empirical Quantile Forecaster.
#
# For each market, at 11 log-spaced anchor horizons, compute weighted
# empirical quantiles of historical price changes (standardized by a
# causal, floored local scale for active/mid markets; raw bounded
# changes for dormant/longshot markets), shrunk toward the tier-pooled
# distribution by Empirical Bayes, scaled by a conditional regime
# ratio that decays to 1.0 at long horizons, then interpolated over a
# sqrt(h) grid to all 240 minutes. Median anchored to the checkpoint
# price (markets are near-martingales); no sqrt(h) multiplication
# (the empirical residuals already capture horizon expansion); no
# forcing sum(q50) == 1 (Polymarket vig).
#
# Deterministic: no randomness, no wall-clock, no network, no dict-order
# dependence. Reads $MAKEFUN_INPUT_DIR (else /input), writes
# $MAKEFUN_OUTPUT_DIR (else /output). One <id>.out per <id>.in.

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
ANCHORS = [1, 2, 4, 8, 15, 30, 60, 100, 150, 200, 240]
HORIZON = 240

# --- Tunable constants (tuned by causal walk-forward CV, see DESIGN_V2.md) ---
RECENCY_DECAY = 1.0       # recency weight exp(-decay * age_min / 1440); ~17h half-life
EWMA_ALPHA = 1.0 / 60.0   # causal EWMA of |d1| (60-min memory)
LEVEL_BLEND = 0.35        # weight of level-adjusted long-run scale in s_loc
SLOC_FLOOR_MULT = 0.25    # s_loc floor = max(0.25 * vol_slow, SLOC_FLOOR_ABS)
SLOC_FLOOR_ABS = 5e-4
REGIME_DECAY_H = 60.0     # scale(m,h) decays to 1.0 as exp(-h / REGIME_DECAY_H)
SCALE_LO = 0.55           # regime scale clamp
SCALE_HI = 1.85
# Empirical-Bayes prior strength per quantile (larger = more pool weight).
# Tails are noisier, so they get a stronger prior toward the pool.
# Strong pooling won the walk-forward ablations (smoother, more robust).
K_SHRINK = [300.0, 120.0, 80.0, 120.0, 300.0]
# Winsorize standardized residuals to [-WINSOR_R, +WINSOR_R] (None = off).
# Mild winsorizing keeps one-off crashes from dominating the empirical
# tails; the spec allows it ("only mildly if needed"). +-10 won the
# walk-forward ablations (tune 0.3206, holdout 0.1056, WF 1.6081).
WINSOR_R = 10.0
# Tier-3 (dormant) shrinkage: same K_SHRINK, but on raw dp.
# Regime scaling is applied to Tier 1/2 only; Tier 3 uses scale 1.0
# (its raw dp quantiles already price the jump risk; shrinking them
# by a dormant vol_fast ratio would underprice jumps).


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def g_level(p):
    """Price-level variance factor sqrt(max(p(1-p), 0.0025*0.9975))."""
    return math.sqrt(max(p * (1.0 - p), 0.0025 * 0.9975))


def clean_prices(raw):
    """Forward-fill None/missing, clamp to [0, 1]. Returns list of floats."""
    out = []
    last = None
    for v in raw:
        if v is None:
            v = last
        if v is None:
            v = 0.0
        v = float(v)
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out.append(v)
        last = v
    return out


def activity_rate(prices, window):
    """Fraction of the last `window` minutes with a nonzero 1-min move."""
    n = len(prices)
    if n < 2:
        return 0.0
    w = min(window, n - 1)
    moves = 0
    for i in range(n - 1 - w, n - 1):
        if prices[i + 1] != prices[i]:
            moves += 1
    return moves / w if w > 0 else 0.0


def local_scales(prices):
    """Return (s_loc, vol_fast, vol_med, vol_slow, s_loc_now).
    s_loc[t] is the causal local scale at minute t (for t >= 1)."""
    n = len(prices)
    if n < 2:
        return [SLOC_FLOOR_ABS], SLOC_FLOOR_ABS, SLOC_FLOOR_ABS, SLOC_FLOOR_ABS, SLOC_FLOOR_ABS
    # 1-min absolute changes; d1[i] = prices[i] - prices[i-1] for i>=1.
    abs_d1 = [0.0] * n
    for i in range(1, n):
        abs_d1[i] = abs(prices[i] - prices[i - 1])
    # L1 volatilities.
    vol_slow = sum(abs_d1[1:]) / (n - 1)
    w_med = min(360, n - 1)
    vol_med = sum(abs_d1[n - w_med:]) / w_med if w_med > 0 else vol_slow
    w_fast = min(60, n - 1)
    vol_fast = sum(abs_d1[n - w_fast:]) / w_fast if w_fast > 0 else vol_slow
    # Price-level mean for the g(p) ratio.
    p_mean = sum(prices) / n
    g_mean = max(g_level(p_mean), 1e-4)
    floor = max(SLOC_FLOOR_MULT * vol_slow, SLOC_FLOOR_ABS)
    # Causal EWMA of |d1|, initialized at vol_slow.
    s_loc = [floor] * n
    ewma = vol_slow
    for t in range(1, n):
        ewma = EWMA_ALPHA * abs_d1[t] + (1.0 - EWMA_ALPHA) * ewma
        level = LEVEL_BLEND * vol_slow * (g_level(prices[t]) / g_mean)
        s = (1.0 - LEVEL_BLEND) * ewma + level
        if s < floor:
            s = floor
        s_loc[t] = s
    return s_loc, vol_fast, vol_med, vol_slow, s_loc[n - 1]


def weighted_quantiles(pairs, taus):
    """pairs: list of (value, weight). Returns list of quantiles, one per tau.
    Linear interpolation on cumulative weight (deterministic)."""
    if not pairs:
        return [0.0] * len(taus)
    # Tuples sort lexicographically by value first, which is the order we
    # need (no key lambda: much faster on thousands of samples).
    pairs = sorted(pairs)
    total = 0.0
    for _, w in pairs:
        total += w
    if total <= 0.0:
        return [pairs[len(pairs) // 2][0]] * len(taus)
    # Cumulative weights.
    cum = []
    c = 0.0
    for v, w in pairs:
        c += w
        cum.append(c)
    out = []
    for tau in taus:
        target = tau * total
        # First index with cum >= target.
        lo, hi = 0, len(pairs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        if i == 0 or cum[i] == cum[i - 1]:
            out.append(pairs[i][0])
        else:
            # Interpolate between pairs[i-1] and pairs[i].
            f = (target - cum[i - 1]) / (cum[i] - cum[i - 1])
            out.append(pairs[i - 1][0] + f * (pairs[i][0] - pairs[i - 1][0]))
    return out


def tier_of(p0, act12):
    if p0 >= 0.06 or act12 >= 0.15:
        return 1
    if p0 >= 0.015:
        return 2
    return 3


def collect_market(prices, s_loc, tier):
    """Precompute per-anchor sample lists for one market.
    Only builds what the tier needs: standardized residuals r for T1/T2
    (own quantiles + tier pool), raw dp for T3. Returns (out_r, out_dp)
    with the unneeded one empty."""
    n = len(prices)
    # Precompute recency weight by end-index j = t+h (avoids an exp() call
    # per sample in the inner loop).
    w_end = [0.0] * n
    for j in range(n):
        w_end[j] = math.exp(-RECENCY_DECAY * (n - 1 - j) / 1440.0)
    out_r = {}
    out_dp = {}
    need_r = tier in (1, 2)
    need_dp = tier == 3
    for h in ANCHORS:
        step = max(1, h // 6)
        rpairs = []
        dppairs = []
        # t ranges so that t+h <= n-1 (last index). Sample t = 0..n-1-h.
        for t in range(0, n - h, step):
            w = w_end[t + h]
            dp = prices[t + h] - prices[t]
            if need_dp:
                dppairs.append((dp, w))
            if need_r:
                s = s_loc[t] if t < len(s_loc) else s_loc[-1]
                r = dp / s
                if WINSOR_R is not None:
                    if r > WINSOR_R:
                        r = WINSOR_R
                    elif r < -WINSOR_R:
                        r = -WINSOR_R
                rpairs.append((r, w))
        out_r[h] = rpairs
        out_dp[h] = dppairs
    return out_r, out_dp


def ess(pairs):
    """Effective sample size from weights."""
    sw = 0.0
    sw2 = 0.0
    for _, w in pairs:
        sw += w
        sw2 += w * w
    if sw2 <= 0.0:
        return 0.0
    return (sw * sw) / sw2


def blend_quantiles(q_own, q_pool, own_pairs):
    """Empirical-Bayes blend of own and pooled quantiles.
    w_own = ESS / (ESS + K_tau); tails get a stronger pool prior."""
    e = ess(own_pairs)
    out = []
    for j in range(5):
        k = K_SHRINK[j]
        w_own = e / (e + k) if (e + k) > 0 else 0.0
        out.append(w_own * q_own[j] + (1.0 - w_own) * q_pool[j])
    return out


def forecast_case(case):
    markets = case["markets"]
    # First pass: per-market stats + sample collection for pooling.
    infos = []
    pool_r = {1: {}, 2: {}, 3: {}}
    pool_dp = {1: {}, 2: {}, 3: {}}
    for m in markets:
        prices = clean_prices(m["prices"])
        p0 = prices[-1]
        act12 = activity_rate(prices, 720)
        tier = tier_of(p0, act12)
        s_loc, vol_fast, vol_med, vol_slow, s_loc_now = local_scales(prices)
        own_r, own_dp = collect_market(prices, s_loc, tier)
        infos.append((m, prices, tier, s_loc, vol_fast, vol_med, vol_slow,
                      s_loc_now, own_r, own_dp))
        for h in ANCHORS:
            pool_r[tier].setdefault(h, []).extend(own_r[h])
            pool_dp[tier].setdefault(h, []).extend(own_dp[h])
    # Precompute tier-pooled quantiles ONCE per (tier, anchor) instead of
    # re-sorting the pooled samples for every market (big speedup).
    pool_qr = {}
    pool_qdp = {}
    for tier in (1, 2, 3):
        pool_qr[tier] = {}
        pool_qdp[tier] = {}
        for h in ANCHORS:
            pool_qr[tier][h] = weighted_quantiles(pool_r[tier].get(h, []),
                                                  QUANTILES)
            pool_qdp[tier][h] = weighted_quantiles(pool_dp[tier].get(h, []),
                                                   QUANTILES)
    forecasts = []
    for (m, prices, tier, s_loc, vol_fast, vol_med, vol_slow, s_loc_now,
         own_r, own_dp) in infos:
        rows = forecast_market_from_parts(
            m, prices, tier, s_loc, vol_fast, vol_med, vol_slow, s_loc_now,
            own_r, own_dp, pool_qr, pool_qdp)
        forecasts.append({"id": m["id"], "q": rows})
    return {"forecasts": forecasts}


def forecast_market_from_parts(market, prices, tier, s_loc, vol_fast,
                               vol_med, vol_slow, s_loc_now, own_r, own_dp,
                               pool_qr, pool_qdp):
    """Same as forecast_market but reuses precomputed parts (for pooling).
    pool_qr / pool_qdp are precomputed tier-pooled quantiles per anchor."""
    n = len(prices)
    p0 = prices[-1]
    avg_sloc = sum(s_loc[1:]) / max(1, n - 1) if n > 1 else s_loc_now
    if avg_sloc <= 0.0:
        avg_sloc = s_loc_now if s_loc_now > 0 else SLOC_FLOOR_ABS
    scale1 = vol_fast / avg_sloc
    if scale1 < SCALE_LO:
        scale1 = SCALE_LO
    elif scale1 > SCALE_HI:
        scale1 = SCALE_HI
    anchor_q = {}
    for h in ANCHORS:
        if tier == 3:
            q_own = weighted_quantiles(own_dp[h], QUANTILES)
            q_pool = pool_qdp[tier][h]
            q = blend_quantiles(q_own, q_pool, own_dp[h])
            med = q[2]
            anchor_q[h] = [q[j] - med for j in range(5)]
        else:
            q_own = weighted_quantiles(own_r[h], QUANTILES)
            q_pool = pool_qr[tier][h]
            q = blend_quantiles(q_own, q_pool, own_r[h])
            med = q[2]
            anchor_q[h] = [(q[j] - med) * s_loc_now for j in range(5)]
    rows = []
    sqrt_anchors = [math.sqrt(h) for h in ANCHORS]
    for h in range(1, HORIZON + 1):
        sh = math.sqrt(h)
        if sh <= sqrt_anchors[0]:
            offs = list(anchor_q[ANCHORS[0]])
        elif sh >= sqrt_anchors[-1]:
            offs = list(anchor_q[ANCHORS[-1]])
        else:
            lo_i = 0
            while lo_i < len(ANCHORS) - 2 and sqrt_anchors[lo_i + 1] < sh:
                lo_i += 1
            hi_i = lo_i + 1
            span = sqrt_anchors[hi_i] - sqrt_anchors[lo_i]
            frac = (sh - sqrt_anchors[lo_i]) / span if span > 0 else 0.0
            qa = anchor_q[ANCHORS[lo_i]]
            qb = anchor_q[ANCHORS[hi_i]]
            offs = [qa[j] + frac * (qb[j] - qa[j]) for j in range(5)]
        if tier == 3:
            scale = 1.0
        else:
            scale = 1.0 + (scale1 - 1.0) * math.exp(-h / REGIME_DECAY_H)
        row = [price6(p0 + scale * offs[j]) for j in range(5)]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(p.name for p in in_dir.iterdir()
                       if p.name.endswith(".in")):
        case_id = name[:-3]
        with open(in_dir / name, "r", encoding="utf-8") as f:
            case = json.load(f)
        if (case["quantiles"] != QUANTILES
                or case["horizon_minutes"] != HORIZON):
            raise ValueError(f"{name}: unexpected quantiles/horizon")
        out = forecast_case(case)
        with open(out_dir / f"{case_id}.out", "w", encoding="utf-8") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
