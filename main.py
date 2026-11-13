#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 — V2 entry.

Self-Calibrating Multi-Scale Empirical Quantile Forecaster (Python 3.13,
standard library only, no network, deterministic).

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes one <id>.out
per case to $MAKEFUN_OUTPUT_DIR (else /output) with
{"forecasts": [{"id": <conditionId>, "q": [[5 numbers] x 240]}]}.

Model (see DESIGN_V2.md): multi-scale robust (L1/MAD) volatility with a
causal, floored local scale; 3 market tiers (active / mid / dormant);
weighted empirical quantiles of historical price changes at 11 log-spaced
anchor horizons, shrunk toward the tier-pooled distribution by Empirical
Bayes; mild winsorizing of standardized residuals (+/-10); conditional
regime scaling decaying to 1.0 at long horizons; median anchored to the
checkpoint price; no sqrt(h) double-counting; no forcing sum(q50) == 1.
"""

import glob
import json
import math
import os

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
ANCHORS = [1, 2, 4, 8, 15, 30, 60, 100, 150, 200, 240]
HORIZON = 240
RECENCY_DECAY = 1.0
WINSOR = 10.0
EB_K = [300.0, 120.0, 80.0, 120.0, 300.0]
REGIME_TAU = 60.0
REGIME_LO = 0.55
REGIME_HI = 1.85
S_LOC_FLOOR_ABS = 5e-4
S_LOC_FLOOR_FRAC = 0.25


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


def ewma(values, span):
    """Causal EWMA with the given span. Returns a list of the same length.

    Uses a 60-minute half-life (alpha = 1 - 0.5**(1/span)): the weight
    on a minute halves every `span` minutes, so the local scale tracks
    the recent volatility regime without chasing single-minute spikes.
    This is the "EWMA_60" of the design doc.
    """
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


def all_weighted_quantiles(pairs):
    """The 5 weighted empirical quantiles of (value, weight) pairs.

    Sorts by value once and walks the cumulative weight; returns the
    smallest value whose cumulative weight reaches tau * total for each
    tau in QUANTILES. Empty input returns all zeros.
    """
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


def ess(weights):
    """Effective sample size of a weighted sample."""
    s1 = 0.0
    s2 = 0.0
    for w in weights:
        s1 += w
        s2 += w * w
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def g_factor(p):
    """Price-level variance factor sqrt(max(p(1-p), 0.0025*0.9975))."""
    return math.sqrt(max(p * (1.0 - p), 0.0025 * 0.9975))


def sample_pairs(prices, s_loc, h, tier):
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
            if v > WINSOR:
                v = WINSOR
            elif v < -WINSOR:
                v = -WINSOR
        age = (T - 1) - (t + h)
        w = math.exp(-RECENCY_DECAY * age / 1440.0)
        pairs.append((v, w))
        t += step
    return pairs


def prep_market(market):
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
    g_mean = g_factor(p_mean)
    ew = ewma(absd1, 60)
    floor = S_LOC_FLOOR_FRAC * vol_slow
    if floor < S_LOC_FLOOR_ABS:
        floor = S_LOC_FLOOR_ABS
    s_loc = [0.0] * T
    for t in range(T):
        v = 0.65 * ew[t] + 0.35 * vol_slow * (g_factor(prices[t]) / g_mean if g_mean > 1e-4 else 1.0)
        if v < floor:
            v = floor
        s_loc[t] = v
    avg_s_loc = sum(s_loc) / T if T else 1.0
    s_now = s_loc[T - 1] if T else 1.0

    own = []
    for h in ANCHORS:
        own.append(sample_pairs(prices, s_loc, h, tier))

    return {
        "id": market["id"],
        "tier": tier,
        "p0": p0,
        "vol_fast": vol_fast,
        "avg_s_loc": avg_s_loc,
        "s_now": s_now,
        "own": own,
    }


def regime_scale(vol_fast, avg_s_loc, h):
    """Conditional regime ratio, decaying to 1.0, clamped [0.55, 1.85]."""
    ratio = vol_fast / avg_s_loc if avg_s_loc > 0.0 else 1.0
    s = 1.0 + (ratio - 1.0) * math.exp(-h / REGIME_TAU)
    if s < REGIME_LO:
        s = REGIME_LO
    elif s > REGIME_HI:
        s = REGIME_HI
    return s


def interp_anchors(anchor_vals, h):
    """Linear interpolation over a sqrt(h) grid between anchor values."""
    if h <= ANCHORS[0]:
        return list(anchor_vals[0])
    if h >= ANCHORS[-1]:
        return list(anchor_vals[-1])
    sh = math.sqrt(h)
    for i in range(len(ANCHORS) - 1):
        a0 = ANCHORS[i]
        a1 = ANCHORS[i + 1]
        if a0 <= h <= a1:
            s0 = math.sqrt(a0)
            s1 = math.sqrt(a1)
            u = (sh - s0) / (s1 - s0) if s1 > s0 else 0.0
            return [
                anchor_vals[i][j] * (1.0 - u) + anchor_vals[i + 1][j] * u
                for j in range(5)
            ]
    return list(anchor_vals[-1])


def forecast_prep(prep, pool_q):
    """240 rows x 5 quantiles for one prepared market."""
    tier = prep["tier"]
    p0 = prep["p0"]
    s_now = prep["s_now"]
    vol_fast = prep["vol_fast"]
    avg_s_loc = prep["avg_s_loc"]

    anchor_q = []
    for ai, h in enumerate(ANCHORS):
        own = prep["own"][ai]
        pq = pool_q.get((tier, ai))
        if pq is None:
            pq = [0.0] * 5
        if own:
            oq = all_weighted_quantiles(own)
            e = ess([w for _, w in own])
            blended = []
            for j in range(5):
                w_own = e / (e + EB_K[j]) if (e + EB_K[j]) > 0.0 else 0.0
                blended.append(w_own * oq[j] + (1.0 - w_own) * pq[j])
        else:
            blended = list(pq)
        med = blended[2]
        anchor_q.append([q - med for q in blended])

    rows = []
    for h in range(1, HORIZON + 1):
        qc = interp_anchors(anchor_q, h)
        if tier == 3:
            vals = [p0 + qc[j] for j in range(5)]
        else:
            sc = regime_scale(vol_fast, avg_s_loc, h)
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


def process_case(case):
    """Forecast every market of one case. Returns the forecasts list."""
    markets = case["markets"]
    preps = []
    pools = {}
    for m in markets:
        prep = prep_market(m)
        preps.append(prep)
        for ai in range(len(ANCHORS)):
            key = (prep["tier"], ai)
            if key in pools:
                pools[key].extend(prep["own"][ai])
            else:
                pools[key] = list(prep["own"][ai])
    pool_q = {}
    for key, pairs in pools.items():
        pool_q[key] = all_weighted_quantiles(pairs)
    forecasts = []
    for prep in preps:
        forecasts.append({"id": prep["id"], "q": forecast_prep(prep, pool_q)})
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
        forecasts = process_case(case)
        out_path = os.path.join(output_dir, cid + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
