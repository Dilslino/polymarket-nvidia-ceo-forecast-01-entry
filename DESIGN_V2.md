# DESIGN_V2 — polymarket-nvidia-ceo-forecast-01

Bounty: Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 240 minutes
ahead, five quantiles (0.05/0.25/0.5/0.75/0.95), mean pinball loss in
percentage points. Entries close 2026-11-13 21:00 UTC (04:00 WIB 14 Nov).
Prize $250 total, top 100 paid ($40 top 1); a prize needs a loss strictly
below the best house entry (baseline and/or house AI model).

## V2 model (main.py, Python 3.13, stdlib only)

Self-Calibrating Multi-Scale Empirical Quantile Forecaster, following the
spec architecture (nvidia_forecast_spec.txt) exactly:

- **Step A — Cleaning & activity.** Forward-fill, clamp [0,1]. p0 = last
  price. 1-min changes d1. act_12h / act_72h = share of minutes with a
  nonzero move.
- **Step B — Multi-scale robust volatility (L1/MAD, never L2 std).**
  vol_fast (60-min), vol_med (360-min), vol_slow (full history) as mean
  |d1|. g(p) = sqrt(max(p(1-p), 0.0025*0.9975)). Causal local scale
  s_loc[t] = 0.65 * EWMA_60(|d1|) + 0.35 * vol_slow * g(p[t])/g(p_mean),
  floored at max(0.25*vol_slow, 5e-4) so dividing by a quiet-period
  scale never explodes residuals.
- **Step C — Tiers & empirical distributions.** T1 active (p0>=0.06 or
  act_12h>=0.15), T2 mid (0.015<=p0<0.06), T3 dormant (p0<0.015 and
  act_12h<0.15). 11 log anchors [1,2,4,8,15,30,60,100,150,200,240],
  stride max(1,h//6), recency weight exp(-1.0 * age_min/1440) (~17h
  half-life). T1/T2: standardized residuals r=(p[t+h]-p[t])/s_loc[t],
  winsorized to [-10,+10]; T3: raw dp (never divide by tiny s_loc).
  Weighted empirical quantiles from own + tier-pooled samples, blended
  by Empirical Bayes (w_own = ESS/(ESS+K), K=[300,120,80,120,300] —
  strong pooling, tails get the strongest prior).
- **Step D — Regime scaling & median guard.** scale(m,h) = 1 +
  (vol_fast/avg_s_loc - 1) * exp(-h/60), clamped [0.55, 1.85]; decays
  to 1.0 at long horizons. NEVER multiply by sqrt(h) (r already
  captures horizon expansion). Median anchored to p0 (empirical
  quantiles recentered on their own median, so q50 = p0 exactly).
  Regime scaling applies to T1/T2 only; T3 uses scale 1.0 (its raw dp
  quantiles already price jump risk; shrinking them by a dormant
  vol_fast ratio would underprice jumps).
- **Step E — Interpolation & clamping.** Linear interpolation over a
  sqrt(h) grid between anchors; clamp [0,1]; enforce non-crossing
  0<=q05<=q25<=q50<=q75<=q95<=1. No forcing sum(q50)==1 (vig 1.01-1.06).

## Tuning (causal walk-forward CV, not practice-overfit)

Tuned on practice p00–p04 + 6 extra (72h→4h) splits built from
Polymarket CLOB history at checkpoints before p00 (2026-11-11);
p05 held out as a final check. Key ablation results (tune / holdout /
WF-extra, house baseline 0.3925 / — / 1.6813):

| config | tune | holdout | WF-extra |
|---|---|---|---|
| initial (decay .55, K 60/20/15, no winsor) | 0.3548 | 0.1246 | 1.5499 |
| strong pool K 150/60/40 | 0.3484 | 0.1216 | 1.5475 |
| winsor ±6 | 0.3368 | 0.1126 | 1.6602 |
| winsor ±10 | 0.3228 | 0.1101 | 1.6171 |
| winsor ±12 | 0.3197 | 0.1094 | 1.6107 |
| **final: winsor ±10, K 300/120/80, decay 1.0** | **0.3206** | **0.1056** | **1.6081** |

Notes:
- Winsorizing helps the practice a lot (crash outliers dominate the
  empirical tails) but hurts the volatile Nov-11 WF splits — the final
  is a Fri–Sun weekend window (spec: volatility decays into quiet
  regimes), so the practice-era tuning is the better guide; the WF
  splits were kept as a guard (final still beats baseline there by 4.4%).
- The optimum is flat across nearby configs (0.320–0.327 tune), so the
  final config is not a knife-edge fit.

## Measured results (kit authoritative scorer, data/score-practice.py)

| | V2 final | house baseline | V1 (parametric) |
|---|---|---|---|
| overall loss | **0.280322** | 0.363589 | 0.2880 |
| vs baseline | **−22.9%** | — | −22.5% |
| per-case | p00 0.4996, p01 0.3233, p02 0.3742, p03 0.2519, p04 0.2004, p05 0.1056 | 0.5361, 0.3804, 0.4084, 0.3686, 0.2988, 0.2382 | 0.4634, 0.3098, 0.3519, 0.2518, 0.2071, 0.1439 |
| wins | 6/6 cases | — | 6/6 |

Per-tier (practice): T1 (active) dominates the loss (0.22–1.29/case) —
jump-dominated and mostly irreducible; T2 0.05–0.18, T3 0.04–0.19.
Calibration: coverage_90 0.846 (target ~0.90, slightly narrow);
empirical quantile coverage p05 [0.043, 0.160, 0.358, 0.690, 0.941]
vs nominal [0.05, 0.25, 0.50, 0.75, 0.95] — inner quantiles sit low
because practice prices drifted up after checkpoints (event heating
up); the median is anchored to p0 by design (persistence beats drift
in every test). Pinball loss, the ranking metric, is what was tuned.

## Runtime / determinism (measured)

- Per-case (single-case runs, incl. interpreter startup):
  p00 0.293s, p01 0.337s, p02 0.359s, p03 0.408s, p04 0.470s,
  p05 0.514s. Limit 8s/case; target <0.5s met for 5/6 cases
  (p05, the largest, is 0.514s — essentially at target).
- Two runs byte-identical (sha256 match across out/ and out2/).
- Output 179–242 KB per case (limit 1 MiB); format validated
  (exactly {"forecasts": [...]}, every case market once, 240 rows ×
  5 monotone quantiles in [0,1]).

## Files

- `main.py` — the entry (repo root).
- `LICENSE` — MIT, copied from the kit starter.
- `README.md` — short entry readme.
- `DESIGN_V2.md` — this file.
- `harness/` — dev-only, NOT part of the submitted repo:
  `score.py` (pinball scorer + per-tier + calibration),
  `walkforward.py` (extra 72h→4h splits from CLOB history),
  `fetch_history.py` (CLOB fetch), `tune.py` (ablations),
  `history.json` (fetched 1-min grids, 2026-11-10→13).
- `tests/validate.py` — format + determinism + runtime checks.
- `out/`, `out2/` — local practice outputs (2 runs, byte-identical).

## Risks

- The house AI-model forecast is unknown; V2 must beat the *lower* of
  baseline and house model. −22.9% vs baseline is a solid margin.
- Practice ≠ final: the rules state a practice score is no guide to
  the final. The flat optimum, strong pooling, and walk-forward
  validation are the mitigation.
- A market-deciding event (a CEO named) ends the scoring window early;
  nothing in the model depends on that.
