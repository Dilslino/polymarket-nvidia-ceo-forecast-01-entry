"""Causal walk-forward validation for polymarket-nvidia-ceo-forecast-01 (V2).

Builds (context -> 4h future) splits from harness/history.json at 4h-spaced
checkpoints BEFORE p00, plus the 6 practice cases, and scores a forecast
function on all of them. Used for tuning WITHOUT overfitting to p00-p05.

A split at checkpoint C uses each market's grid from its first price to C
(causal, variable length) and scores minutes C+1..C+240 against the grid.

Usage:
  python3 harness/walkforward.py            # score main.py's forecast_case
  (import build_splits / score_fn from other scripts)
"""
import json, math, sys, datetime as dt

V2 = '/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-v2'
HIST = json.load(open(f'{V2}/harness/history.json'))
T0 = dt.datetime(2026, 11, 10, 2, 0, 0, tzinfo=dt.timezone.utc)
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

def idx(ts):
    return int((ts - T0).total_seconds() // 60)

def build_case(checkpoint):
    """checkpoint: datetime (UTC, whole minute). Returns (case, answers).
    case: {'at': iso, 'horizon_minutes': 240, 'markets': [{'id','name','prices'}]}
    answers: {market_id: [240 actual prices]} (None where grid missing)."""
    ci = idx(checkpoint)
    markets = []
    answers = {}
    for mid, m in HIST.items():
        grid = m['grid']
        # first non-None index
        first = next((i for i, v in enumerate(grid) if v is not None), None)
        if first is None or first > ci:
            continue
        # price at checkpoint must exist
        if grid[ci] is None:
            continue
        prices = grid[first:ci + 1]
        markets.append({'id': mid, 'name': m['name'], 'prices': prices})
        fut = []
        for h in range(1, 241):
            j = ci + h
            fut.append(grid[j] if j < len(grid) else None)
        answers[mid] = fut
    case = {'at': checkpoint.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'horizon_minutes': 240, 'markets': markets}
    return case, answers

def score_fn(forecast_case, case, answers):
    """forecast_case: fn(case) -> {'forecasts': [{'id','q': [[5]x240]}]}.
    Returns dict(loss, mederr, cov90, coverage, per_tier)."""
    out = forecast_case(case)
    fc = {f['id']: f['q'] for f in out['forecasts']}
    tot = 0.0; n = 0; med = 0.0; nm = 0; cov = 0.0; nc = 0
    cov_lt = [0] * 5
    tier_tot = {1: 0.0, 2: 0.0, 3: 0.0}
    tier_n = {1: 0, 2: 0, 3: 0}
    for m in case['markets']:
        y = answers[m['id']]
        q = fc[m['id']]
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
        for h in range(240):
            yy = y[h]
            if yy is None:
                continue
            row = q[h]
            for j, tau in enumerate(TAUS):
                d = yy - row[j]
                l = 100 * max(tau * d, (tau - 1) * d)
                tot += l; n += 1
                tier_tot[tier] += l; tier_n[tier] += 1
                if yy < row[j]:
                    cov_lt[j] += 1
            med += 100 * abs(yy - row[2]); nm += 1
            if row[0] <= yy <= row[4]:
                cov += 1
            nc += 1
    return {
        'loss': tot / n if n else None, 'mederr': med / nm if nm else None,
        'cov90': cov / nc if nc else None, 'n': n,
        'coverage': [cov_lt[j] / nc if nc else None for j in range(5)],
        'per_tier': {t: (tier_tot[t] / tier_n[t] if tier_n[t] else None)
                     for t in (1, 2, 3)},
    }

def baseline_forecast(case, sigma=0.0025):
    fc = []
    for m in case['markets']:
        x = m['prices'][-1]
        rows = [[round(min(1.0, max(0.0, x + z * sigma * math.sqrt(h))), 6) + 0.0
                 for z in Z] for h in range(1, 241)]
        fc.append({'id': m['id'], 'q': rows})
    return {'forecasts': fc}

CHECKPOINTS = [
    dt.datetime(2026, 11, 11, 6, 0, tzinfo=dt.timezone.utc),
    dt.datetime(2026, 11, 11, 10, 0, tzinfo=dt.timezone.utc),
    dt.datetime(2026, 11, 11, 14, 0, tzinfo=dt.timezone.utc),
    dt.datetime(2026, 11, 11, 18, 0, tzinfo=dt.timezone.utc),
    dt.datetime(2026, 11, 11, 22, 0, tzinfo=dt.timezone.utc),
    dt.datetime(2026, 11, 12, 2, 0, tzinfo=dt.timezone.utc),
]

def main():
    sys.path.insert(0, V2)
    import main as entry
    print('=== walk-forward: extra splits (before p00) ===')
    tot_l = 0.0; tot_n = 0; base_l = 0.0; base_n = 0
    for cp in CHECKPOINTS:
        case, answers = build_case(cp)
        r = score_fn(entry.forecast_case, case, answers)
        b = score_fn(baseline_forecast, case, answers)
        tot_l += r['loss'] * r['n']; tot_n += r['n']
        base_l += b['loss'] * b['n']; base_n += b['n']
        print(f"{cp.strftime('%m-%d %H:%M')} n_mk={len(case['markets']):2d} "
              f"model {r['loss']:.6f} base {b['loss']:.6f} "
              f"cov[{', '.join(f'{v:.2f}' for v in r['coverage'])}] "
              f"T1 {r['per_tier'][1]:.4f} T2 {r['per_tier'][2]:.4f} T3 {r['per_tier'][3]:.4f}")
    print(f'EXTRA mean model {tot_l / tot_n:.6f} vs baseline {base_l / base_n:.6f}')

if __name__ == '__main__':
    main()
