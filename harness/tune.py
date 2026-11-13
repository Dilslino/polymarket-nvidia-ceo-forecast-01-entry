"""Ablation / tuning harness for V2 (spec) model.

Tests config variants on:
  - practice p00-p04 (tuning set) + p05 (holdout, reported separately)
  - walk-forward extra splits (before p00) from harness/history.json
Compares each variant to the house baseline proxy on the same splits.

Usage: python3 harness/tune.py
"""
import sys, math, importlib
sys.path.insert(0, '/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-v2')
sys.path.insert(0, '/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-v2/harness')
import main as M
import score as S
import walkforward as W

def run_config(name, overrides):
    """overrides: dict of main.py constant -> value. Returns (tune_loss,
    holdout_loss, wf_loss, tune_base, wf_base)."""
    saved = {}
    for k, v in overrides.items():
        saved[k] = getattr(M, k)
        setattr(M, k, v)
    try:
        # practice p00-p04 (tune) + p05 (holdout)
        tune_l = 0.0; tune_n = 0; base_l = 0.0; base_n = 0
        hold_l = None; hold_n = 0
        for cid in ['p00', 'p01', 'p02', 'p03', 'p04']:
            c = S.load_case(cid)
            r = S.score_forecast(c, {f['id']: f['q'] for f in M.forecast_case(c)['forecasts']})
            b = S.score_forecast(c, S.baseline_forecast(c))
            tune_l += r['loss'] * r['n']; tune_n += r['n']
            base_l += b['loss'] * b['n']; base_n += b['n']
        c = S.load_case('p05')
        r = S.score_forecast(c, {f['id']: f['q'] for f in M.forecast_case(c)['forecasts']})
        hold_l = r['loss']; hold_n = r['n']
        # walk-forward extra splits
        wf_l = 0.0; wf_n = 0; wfb_l = 0.0; wfb_n = 0
        for cp in W.CHECKPOINTS:
            case, answers = W.build_case(cp)
            r = W.score_fn(M.forecast_case, case, answers)
            b = W.score_fn(W.baseline_forecast, case, answers)
            wf_l += r['loss'] * r['n']; wf_n += r['n']
            wfb_l += b['loss'] * b['n']; wfb_n += b['n']
        tune = tune_l / tune_n
        hold = hold_l
        wf = wf_l / wf_n
        print(f"{name:34s} tune(p00-04) {tune:.6f}  holdout(p05) {hold:.6f}  "
              f"WF-extra {wf:.6f}  (base tune {base_l/base_n:.6f}, base WF {wfb_l/wfb_n:.6f})")
        return tune, hold, wf
    finally:
        for k, v in saved.items():
            setattr(M, k, v)

if __name__ == '__main__':
    print('=== V2 ablations round 4 (final) ===')
    run_config('winsor +-12', {'WINSOR_R': 12.0})
    run_config('winsor +-10 + vstrong + rec 1.0', {'WINSOR_R': 10.0, 'K_SHRINK': [300.0, 120.0, 80.0, 120.0, 300.0], 'RECENCY_DECAY': 1.0})
    run_config('winsor +-10 + strong + rec 1.0', {'WINSOR_R': 10.0, 'K_SHRINK': [150.0, 60.0, 40.0, 60.0, 150.0], 'RECENCY_DECAY': 1.0})
