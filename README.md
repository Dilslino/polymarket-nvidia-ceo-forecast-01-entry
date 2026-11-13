# polymarket-nvidia-ceo-forecast-01 — entry

Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 240 minutes ahead,
as five quantiles (0.05 / 0.25 / 0.50 / 0.75 / 0.95) per market per minute.

## Model

Self-Calibrating Multi-Scale Empirical Quantile Forecaster (Python 3.13,
standard library only, no network, deterministic):

- Multi-scale robust (L1/MAD) volatility with a causal, floored local
  scale; 3 market tiers (active / mid / dormant-longshot).
- Weighted empirical quantiles of historical price changes at 11
  log-spaced anchor horizons, shrunk toward the tier-pooled
  distribution by Empirical Bayes; mild winsorizing of standardized
  residuals (±10).
- Conditional regime scaling that decays to 1.0 at long horizons;
  median anchored to the checkpoint price (near-martingale); no
  sqrt(h) double-counting; no forcing sum(q50) == 1.

See `DESIGN_V2.md` for the full design, tuning (causal walk-forward
CV), and measured results.

## Practice score (local, kit scorer)

Mean pinball loss **0.280322** over the 6 practice cases (153,600
items) vs house baseline **0.363589** (−22.9%), winning all 6 cases.
Per-case: p00 0.4996, p01 0.3233, p02 0.3742, p03 0.2519, p04 0.2004,
p05 0.1056. Runtime 0.29–0.51 s per case (limit 8 s); two runs
byte-identical.

## Run

```sh
MAKEFUN_INPUT_DIR=<input-dir> MAKEFUN_OUTPUT_DIR=<output-dir> python3 main.py
```

Reads every `*.in` in the input dir, writes one `<id>.out` per case
(`{"forecasts": [{"id": ..., "q": [[5 numbers] x 240]}]}`).

## License

MIT (see LICENSE).
