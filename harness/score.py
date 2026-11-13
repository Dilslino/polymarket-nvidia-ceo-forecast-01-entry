"""Local harness scorer for polymarket-nvidia-ceo-forecast-01 (V2 build).

Scores a forecast dict {market_id: [[5 floats] x 240]} against the kit's
verified practice answers (data/answers-practice.json). Also computes
per-tier losses, empirical quantile coverage, and the house baseline proxy
for direct comparison. Uses the kit's authoritative answers (same as
data/score-practice.py, which reproduces house baseline 0.363589).

Usage (from v2 workspace):
  python3 harness/score.py <out-dir>            # score *.out in out-dir
  (import score_forecast / baseline_forecast from other harness scripts)
"""
import json, math, os, sys, glob, datetime as dt

KIT = '/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-kit'
ANS = json.load(open(f'{KIT}/data/answers-practice.json'))
GRID = ANS['markets']  # conditionId -> {'name', 'grid': {epoch_str: price}}
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

def ts(s):
    return int(dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp())

def load_case(cid, indir=f'{KIT}/input'):
    return json.load(open(f'{indir}/{cid}.in'))

def score_forecast(case, fc):
    """fc: {market_id: [[5 floats] x 240]}.
    Returns dict(loss, mederr, cov90, n_items, per_case, per_tier, coverage).
    coverage[q] = share of scored market-minutes with y < q_tau (empirical)."""
    at = ts(case['at'])
    tot = 0.0; n = 0; med = 0.0; nm = 0; cov = 0.0; nc = 0
    # coverage counters per quantile: count y < q_tau
    cov_lt = [0] * 5
    # per-tier accumulators
    tier_tot = {1: 0.0, 2: 0.0, 3: 0.0}
    tier_n = {1: 0, 2: 0, 3: 0}
    tier_med = {1: 0.0, 2: 0.0, 3: 0.0}
    tier_nm = {1: 0, 2: 0, 3: 0}
    for m in case['markets']:
        g = GRID[m['id']]['grid']
        q = fc[m['id']]
        assert len(q) == 240, f"{m['name']}: {len(q)} rows"
        # tier from checkpoint price + 12h activity (same rule as main.py)
        p0 = m['prices'][-1]
        prices = m['prices']
        w = min(720, len(prices) - 1)
        moves = sum(1 for i in range(len(prices) - 1 - w, len(prices) - 1)
                    if prices[i + 1] != prices[i])
        act12 = moves / w if w > 0 else 0.0
        if p0 >= 0.06 or act12 >= 0.15:
            tier = 1
        elif p0 >= 0.015:
            tier = 2
        else:
            tier = 3
        for h in range(1, 241):
            y = g[str(at + h * 60)]
            assert y is not None, f"{m['name']}: no answer at +{h}m"
            row = q[h - 1]
            for j, tau in enumerate(TAUS):
                d = y - row[j]
                l = 100 * max(tau * d, (tau - 1) * d)
                tot += l; n += 1
                tier_tot[tier] += l; tier_n[tier] += 1
                if y < row[j]:
                    cov_lt[j] += 1
            med += 100 * abs(y - row[2]); nm += 1
            tier_med[tier] += 100 * abs(y - row[2]); tier_nm[tier] += 1
            if row[0] <= y <= row[4]:
                cov += 1
            nc += 1
    per_tier = {}
    for t in (1, 2, 3):
        per_tier[t] = (tier_tot[t] / tier_n[t] if tier_n[t] else None,
                       tier_med[t] / tier_nm[t] if tier_nm[t] else None,
                       tier_n[t])
    return {
        'loss': tot / n, 'mederr': med / nm, 'cov90': cov / nc, 'n': n,
        'per_tier': per_tier,
        'coverage': [cov_lt[j] / nc for j in range(5)],
        'n_minutes': nc,
    }

def baseline_forecast(case, sigma=0.0025):
    """House baseline proxy: q = clamp(x + z*sigma*sqrt(h))."""
    fc = {}
    for m in case['markets']:
        x = m['prices'][-1]
        rows = []
        for h in range(1, 241):
            rows.append([round(min(1.0, max(0.0, x + z * sigma * math.sqrt(h))), 6) + 0.0
                         for z in Z])
        fc[m['id']] = rows
    return fc

def score_outdir(outdir, cases=('p00', 'p01', 'p02', 'p03', 'p04', 'p05')):
    """Score *.out files in outdir. Returns (per_case dict, overall dict)."""
    per_case = {}
    tot_l = 0.0; tot_n = 0
    for cid in cases:
        c = load_case(cid)
        op = os.path.join(outdir, f'{cid}.out')
        if not os.path.exists(op):
            print(f'{cid}: MISSING {op}')
            continue
        out = json.load(open(op))
        fc = {f['id']: f['q'] for f in out['forecasts']}
        r = score_forecast(c, fc)
        per_case[cid] = r
        tot_l += r['loss'] * r['n']; tot_n += r['n']
        print(f"{cid}: loss {r['loss']:.6f} mederr {r['mederr']:.6f} "
              f"cov90 {r['cov90']:.4f} cov[{', '.join(f'{v:.3f}' for v in r['coverage'])}] "
              f"T1 {r['per_tier'][1][0]:.4f} T2 {r['per_tier'][2][0]:.4f} T3 {r['per_tier'][3][0]:.4f}")
    overall = tot_l / tot_n if tot_n else None
    print(f'OVERALL loss {overall:.6f} over {tot_n} items (house baseline 0.363589)')
    return per_case, overall

if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'out'
    # baseline first for reference
    print('--- house baseline (proxy) ---')
    tot_l = 0.0; tot_n = 0
    for cid in ['p00', 'p01', 'p02', 'p03', 'p04', 'p05']:
        c = load_case(cid)
        r = score_forecast(c, baseline_forecast(c))
        print(f"{cid}: baseline loss {r['loss']:.6f}")
        tot_l += r['loss'] * r['n']; tot_n += r['n']
    print(f'baseline OVERALL {tot_l / tot_n:.6f}')
    print('--- model outputs ---')
    score_outdir(outdir)
