# FinOptix — Portfolio Optimization

[![tests](https://github.com/Architathakur/FinOptix---Portfolio-Optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/Architathakur/FinOptix---Portfolio-Optimization/actions/workflows/tests.yml)

ML-assisted Black-Litterman portfolio optimization for NSE large-cap equities.

> **The deliverable here is the evaluation harness, not a profitable strategy.** The strategy shows no predictive skill, and the project's value is that it can demonstrate that cleanly rather than report a large number produced by a bug.

> For education and portfolio demonstration only. Not financial advice; historical backtests do not predict future performance.

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [The look-ahead bug, and how it was caught](#2-the-look-ahead-bug-and-how-it-was-caught)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Limitations](#5-limitations)
6. [How to run it](#6-how-to-run-it)

---

## 1. What this project is

An end-to-end Python pipeline that forecasts monthly returns for NSE large caps, builds a Black-Litterman portfolio from those forecasts, and evaluates the result rigorously enough to establish whether the forecasts are worth anything.

It downloads market data from Yahoo Finance, engineers strictly backward-looking technical features, trains one XGBoost model per stock, ranks and selects a universe, converts the predictions into Black-Litterman views, optimizes for maximum Sharpe, and backtests against three benchmarks with transaction costs and bootstrap significance tests. It runs either as a single held-out window or as an 18-step quarterly walk-forward.

**The finding is negative, and it is reported as such.** Pooled across 52 non-overlapping out-of-sample windows, the model's cross-sectional information coefficient is 0.0096 with a t-statistic of 0.37 — indistinguishable from zero. This is the expected result for technical indicators on cross-sectional equity returns, and [section 4](#4-results) states it plainly rather than dressing it up.

What is reusable is the harness: purged splits, non-overlapping IC sampling, drift-aware turnover costs, bootstrap significance testing, walk-forward evaluation, and a regression test that fails if look-ahead bias is reintroduced.

### Every number in this README is traceable

Figures come from committed artifacts in `outputs/`, from tests you can run, or from a named commit. Two historical figures describe superseded runs and are cited to the commits that hold them:

| Figure | Source |
|---|---|
| Walk-forward and single-window results | `outputs/walkforward_*.csv`, `outputs/*.csv` |
| Random-walk leakage scores | `tests/test_no_leakage.py` (seeded, deterministic) |
| Pre-fix correlations of 0.212–0.516 | `c95e942:outputs/ml_model_metrics.csv` |
| Earlier single-window result of −1.88% | `ae16350:outputs/performance_stats.csv` |

---

## 2. The look-ahead bug, and how it was caught

An earlier version of this pipeline reported per-ticker prediction correlations of **0.212 to 0.516** and a portfolio that beat its benchmark by ten percentage points of CAGR. Both were artifacts of look-ahead bias. Those numbers were deleted from this README rather than corrected in place, because they were not measurements of anything.

### What the bug was

The prediction target was the return *ending* at day `t`:

```python
feat["returns"] = df["Close"].pct_change()          # return ending at t
feat["ma_10"] = df["Close"].rolling(10).mean()      # window contains Close[t]
feat["volatility_20"] = df["Close"].rolling(20).std()
feat["upper_band"] = ma_20 + 2 * std_20
```

`ma_10`, `ma_50`, `volatility_20`, `upper_band` and `lower_band` were all rolling windows that included `Close[t]`. The model was handed today's closing price and asked to predict today's move. That is the entire source of the reported correlations; a genuine daily-return model built on technical indicators scores roughly 0.02–0.05.

### What tipped it off

Not the code — the number. Published results for daily-return models on technical indicators sit around 0.02–0.05. A weekend project on free data reporting 0.21–0.52 is not four to twenty times better than the field; it has a bug. **A result far above what a field achieves has essentially one boring explanation, and checking it first is cheaper than believing it.**

### The proof

Feed the feature pipeline a seeded pure random walk — iid normal returns, 8000 days — where predictability of the target is exactly zero by construction. Anything scoring meaningfully above zero is reading the future, not finding signal.

| Setup | \|corr\| on a random walk |
|---|---:|
| Original features + same-day target + original XGB params | 0.418 |
| Original features + same-day target + **regularized** XGB params | 0.404 |
| Current features + 21-day forward target + purged split | 0.044 |

Mean of five seeds; per-seed the original set spans 0.385–0.445 and the current set 0.008–0.123.

**The middle row is the control that matters.** The first objection to "the old model scored 0.418" is that it was simply overfitting. Re-running the same leaky features under the current, heavily regularized hyperparameters still scores 0.404 — barely moved. The cause was the construction of the features and target, not model tuning. Per feature, the original set had `returns_20` correlating 0.226 with its own target on a random walk; the current set peaks at 0.080 (`dist_52w_high`), with none of its 14 features above 0.15.

All three rows are computed by a committed regression test, not a one-off script. `tests/test_no_leakage.py` keeps a working reconstruction of the original buggy pipeline as a live positive control and asserts that the current feature set scores below a quarter of it — a ratio rather than a fixed threshold, because the current set's score is a noise floor that rises as the series shortens, while the leaky control sits near 0.4 at every length. The test also asserts the control still leaks under the regularized hyperparameters, so the diagnosis above cannot silently become wrong.

### How it propagated

The bug did not stay in the model. `scoring.py` ranked stocks by mean *predicted* return over the test window, and those predictions echoed realized test-window returns. So the pipeline selected the stocks that had already performed well over the exact window it then backtested, and reported that the resulting portfolio beat equal weighting. The old outperformance was circular: the benchmark was compared against a portfolio built with knowledge of the answer.

### What changed

| Area | Before | After |
|---|---|---|
| Target | Return ending at `t` | `close.pct_change(21).shift(-21)` — the forward 21-day return |
| Features | Raw rupee levels (`ma_10`, `upper_band`, `volatility_20`) containing `Close[t]` | 14 scale-free, strictly trailing features — ratios, z-scores, correlations |
| Split | Two-way TRAIN/TEST, with TEST used for evaluation *and* selection *and* covariance | Three-way TRAIN / VALID / TEST; VALID makes every decision, TEST is opened once, after weights are frozen |
| Split hygiene | None | Purged at every inner boundary, counted in trading days |
| Feature computation | Built separately per download, so every rolling window warmed up again inside the test slice | Computed once over the continuous price history, then sliced |
| Headline metric | Per-ticker time-series correlation | Cross-sectional rank IC, sampled every 21 days so forward windows do not overlap |
| Benchmarks | Equal weighting of the model's own picks | Selected EW, **universe EW**, and the NIFTY 50 |
| Costs | None | Basis points on turnover, charged against drifted weights |
| Return maths | Log returns summed by weight, then compounded as if simple | Simple returns throughout; buy-and-hold weights drift instead of being silently rebalanced daily for free |
| Significance | None | Bootstrapped Sharpe differences with CIs and p-values, resampling dates jointly |
| Evaluation | One split, 11 IC observations | 18-step quarterly walk-forward, 52 pooled IC observations |
| Fundamentals in selection | Current snapshot, 60% of the score | Off by default — a current snapshot is look-ahead bias |

---

## 3. Methodology

### The three-way split

| Window | Job |
|---|---|
| **TRAIN** | Fit the models. Nothing else. |
| **VALID** | Make every decision: stock selection, Black-Litterman views, covariance. |
| **TEST** | Score the result. Opened once, after the weights are frozen. |

Nothing reads TEST until the weights are final. The moment a parameter is chosen by looking at TEST, TEST stops being held out.

### Purging

The model predicts the return over the next 21 trading days, so the label attached to the last day of TRAIN describes what happens *inside* the following window. A purge drops those rows:

```text
TRAIN ────────────────┐         ┌──────────── VALID
                      └─ purge ─┘
                    21 trading days, discarded
```

Gaps are counted on the actual exchange calendar, not in calendar days, so they do not silently shrink across a holiday week. Without the purge, the model trains on the first month of the window used to evaluate it.

### Single-window pipeline

```text
NSE ticker universe
        |
        v
ONE download covering TRAIN_START..TEST_END
        |
        v
Engineer scale-free, trailing-only features over the continuous history
        |
        v
Train one XGBoost model per ticker on TRAIN (purged)
        |
        v
Predict forward returns over VALID and TEST; measure rank IC on both
        |
        v
Rank on VALID predictions -> select top N
        |
        v
Build Black-Litterman prior, views and covariance -- all from VALID
        |
        v
Optimize portfolio weights  ->  WEIGHTS FROZEN
        |
        v
Backtest on TEST vs. selected EW, universe EW and NIFTY 50, with costs
        |
        v
Bootstrap the Sharpe differences
```

### Walk-forward

One split yields one allocation decision and about eleven non-overlapping IC observations — not enough to distinguish a real effect from noise in either direction, which is why every confidence interval it produced straddled zero. `src/walkforward.py` re-runs the entire decision process quarterly:

```text
At each rebalance date t:
  TRAIN   WALKFORWARD_START -> (t - 1y - PURGE_DAYS)    expanding, fixed start
  VALID   (t - 1y)          -> (t - PURGE_DAYS)         makes every decision
  HOLD    t                 -> next rebalance date      strictly out of sample
```

`PURGE_DAYS` trading days are removed at **both** inner boundaries: without the first, the tail of TRAIN carries labels reaching into VALID; without the second, the tail of VALID carries labels reaching past `t` into the window being held.

Weights are held through the quarter and drift with performance. At the next rebalance the cost is charged on **turnover against the drifted weights**, `(COST_BPS / 10000) * sum|w_new - w_drifted|` — charging against the original targets would invent a trade that never happened.

The feature panel is computed once over the full history and sliced per rebalance. Recomputing inside the loop would restart every rolling window in each slice, and would be 18x the work.

### Why cross-sectional rank IC

The strategy does not trade one stock against its own history; on a given day it ranks the universe and buys the top of that ranking. Per-ticker time-series correlation answers a question the strategy never asks — a model that correctly predicts everything rises in bull markets scores well on it and says nothing about *which* stock to buy.

The IC ranks all tickers by prediction on a given date, ranks them by realized forward return, and correlates the two orderings. It is sampled every 21 trading days because a 21-day forward return measured today and one measured tomorrow share 20 of their 21 days; treating those as independent inflates the sample ~21x and the t-statistic by roughly √21.

### ML-driven Black-Litterman views

A common weakness in portfolio notebooks is that ML predictions drive screening while Black-Litterman views are hand-specified and disconnected from the model. FinOptix wires them together: each selected stock's mean predicted return becomes an absolute view, annualized so views, prior and risk aversion share units. View uncertainty uses the He-Litterman proportional rule scaled by `VIEW_CONFIDENCE`.

That is a statement about the plumbing, not the forecasts. On the runs below the views carry no measurable predictive content, so `VIEW_CONFIDENCE` controls how much weight the optimizer gives a signal that has not been shown to work.

---

## 4. Results

Both backtests were generated from live Yahoo Finance runs on 2026-09-05, with `USE_FUNDAMENTALS_IN_SELECTION = False`, 20bps of transaction cost and `TOP_N_STOCKS = 10`. Reported exactly as produced.

### The decomposition — the strongest finding

Three benchmarks split the result into layers. "Equal-Weight (Selected)" inherits the model's own stock picks, so it can only benchmark the *weighting*; "Equal-Weight (Universe)" holds all 47 tradeable tickers and is what isolates *selection*.

Bootstrapped Sharpe differences, 5000 resamples, dates resampled jointly:

| Step | Isolates | Sharpe diff | 95% CI | p |
|---|---|---:|---:|---:|
| Universe EW vs NIFTY 50 | equal- vs cap-weighting | **+0.399** | [+0.102, +0.699] | **0.007** |
| Selected EW vs Universe EW | the selection layer | −0.145 | [−0.603, +0.314] | 0.536 |
| BL vs Selected EW | the weighting layer | −0.033 | [−0.296, +0.231] | 0.816 |
| BL vs NIFTY 50 | everything combined | +0.221 | [−0.417, +0.869] | 0.495 |

The chain is additive and lands exactly on the observed Sharpes: `0.611 + 0.399 − 0.145 − 0.033 = 0.830`.

**The only effect distinguishable from noise is equal-weighting versus cap-weighting the index.** It is worth +0.399 Sharpe at p = 0.007, and it is not a machine-learning result — it is the equal-weight premium, available to anyone willing to hold 47 stocks in equal proportion and rebalance quarterly.

**Both model layers subtract.** Selection costs −0.145 Sharpe, weighting a further −0.033. Neither is statistically distinguishable from zero, but neither shows any sign of adding value. Black-Litterman finishes +0.221 ahead of the NIFTY 50, and the entire margin — more than the entire margin — comes from the equal-weighting effect that the universe benchmark isolates. **Without that benchmark, +0.221 reads as the strategy working. It is not.**

### Walk-forward (headline)

18 quarterly rebalances, 2022-04-01 to 2026-09-05, expanding training window, 47 tradeable tickers. Every allocation is strictly out of sample.

**Pooled rank IC across all hold windows**

| n_periods | Mean IC | IC std | t-stat | Hit rate |
|---:|---:|---:|---:|---:|
| 52 | 0.0096 | 0.1868 | **0.37** | 0.577 |

Pooling was the point of the exercise: it takes the IC sample from 11 observations to 52. **The answer did not change.** A mean IC of 0.0096 with a t-statistic of 0.37 is zero. The model does not rank stocks better than chance, and that conclusion now rests on a sample large enough to state it rather than on a window too short to distinguish anything. Individual quarters ranged from −0.23 to +0.13, positive in only 7 of 18.

**Performance on the stitched daily series**

| Stream | CAGR | AnnVol | Sharpe | Max Drawdown | Turnover/rebalance | Total cost |
|---|---:|---:|---:|---:|---:|---:|
| Black-Litterman | 11.75% | 14.69% | 0.830 | −22.13% | 0.759 | 2.73% |
| Equal-Weight (Selected) | 12.98% | 15.55% | 0.863 | −24.12% | 0.711 | 2.56% |
| Equal-Weight (Universe) | **13.00%** | 12.96% | **1.008** | **−17.40%** | 0.135 | 0.49% |
| NIFTY 50 | 7.44% | 13.18% | 0.611 | −15.77% | 0.056 | 0.20% |

The cost side reinforces the decomposition. The Black-Litterman portfolio turns over 76% of the book per quarter and pays 2.73% in transaction costs, against 0.49% for the passive universe portfolio that beat it on every measure — higher CAGR, lower volatility, shallower drawdown, higher Sharpe.

**This is the expected outcome.** Cross-sectional equity returns are close to unpredictable from technical indicators, so a correctly evaluated pipeline should show an IC near zero and no reliable edge from selection or weighting. That is what it shows.

### Single-window (for comparison)

TRAIN 2020-01-01 to 2024-09-05, VALID to 2025-09-05, TEST 2025-09-05 to 2026-09-05. Selected on VALID: `INDUSINDBK.NS`, `NTPC.NS`, `COALINDIA.NS`, `ASIANPAINT.NS`, `SBIN.NS`, `AXISBANK.NS`, `ADANIPORTS.NS`, `SUNPHARMA.NS`, `NESTLEIND.NS`, `ITC.NS`.

| Window | n_periods | Mean IC | IC std | t-stat | Hit rate |
|---|---:|---:|---:|---:|---:|
| VALID | 12 | −0.1073 | 0.1783 | −2.09 | 0.333 |
| TEST | 11 | 0.0369 | 0.1648 | 0.74 | 0.545 |

| Stream | CAGR | AnnVol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Black-Litterman | 7.28% | 12.09% | 0.641 | −9.94% |
| Equal-Weight (Selected) | 11.26% | 12.89% | 0.893 | −10.99% |
| Equal-Weight (Universe) | 3.31% | 12.24% | 0.327 | −12.42% |
| NIFTY 50 | −3.69% | 13.12% | −0.221 | −15.18% |

| Step | Sharpe diff | 95% CI | p | n |
|---|---:|---:|---:|---:|
| Universe EW vs NIFTY 50 | +0.572 | [+0.035, +1.166] | 0.047 | 246 |
| Selected EW vs Universe EW | +0.566 | [−0.322, +1.470] | 0.220 | 251 |
| BL vs Selected EW | −0.251 | [−0.547, +0.048] | 0.097 | 251 |
| BL vs NIFTY 50 | +0.902 | [−0.233, +2.079] | 0.136 | 246 |

The TEST IC t-statistic of 0.74 is again indistinguishable from zero. The VALID IC is *negative* at −2.09, meaning that over the window used to choose the stocks, the model ranked them backwards.

**Why this table should not be trusted, and why walk-forward exists.** An earlier run of this same single-window backtest — before fundamentals were removed from selection, and dated one day earlier — reported a Black-Litterman CAGR of −1.88% and a Sharpe of −0.115 (`ae16350`). Changing the selection rule and moving the window by a single day swung the headline return by nine percentage points. Nothing about the strategy changed; the sample was simply too small to hold still.

---

## 5. Limitations

**Survivorship bias — this one qualifies the headline result.** The universe is today's NIFTY 50 constituents, so both backtests only ever hold names that were still in the index in 2026. Equal weighting overweights the smaller names, which is precisely where the bias bites hardest, so it most likely inflates the single significant finding. **Read +0.399 as a ceiling, not an estimate.** Removing it requires building the universe from historical index membership.

**A single market regime.** The walk-forward covers 2022–2026, a mostly-rising period for Indian equities. 18 rebalances is enough to say the IC is not distinguishable from zero; it is not enough to characterise behaviour across regimes, and nothing here has been tested through a sustained drawdown.

**No point-in-time fundamentals.** `download_fundamentals` returns today's trailing P/E, D/E and market cap — a single current snapshot — and `SCORE_WEIGHTS` gives those three 60% of the selection score. Using a 2026 balance sheet to decide what to buy in 2022 is look-ahead bias, and walk-forward makes it worse because the same snapshot would drive all 18 rebalances. Doing it properly needs a paid data source, so `USE_FUNDAMENTALS_IN_SELECTION` defaults to `False` and selection ranks on the ML score alone. **Setting that flag to `True` contaminates selection and the backtest stops being a measurement.** This is a documented workaround, not a fix.

**Daily closing prices only.** No intraday data, no bid-ask spreads, no volume-weighted execution. Every trade is assumed to happen at the close at no impact, which is optimistic at 76% quarterly turnover.

**Transaction costs are a basis-point charge on turnover.** Slippage, market impact, taxes, liquidity constraints and borrow are not modelled, and the understatement is larger for the high-turnover strategy than for the benchmarks.

**Other constraints**

- Yahoo Finance is a free, unofficial source; ticker availability, schemas and rate limits change without notice. `LTIM.NS` and `TATAMOTORS.NS` returned 404s in the 2026-09-05 runs and were skipped, leaving 47 tickers.
- Market-implied equilibrium returns use an equal-weight proxy rather than float-adjusted market-cap weights.
- Models are refit quarterly but every hyperparameter is fixed across the whole period. Nothing is tuned per rebalance — tuning inside the loop on VALID would need its own nested split.
- This is a research and portfolio project, not production trading infrastructure.

### Next steps

- Source point-in-time fundamentals, closing the last known leak.
- Build the universe from historical index membership, which would also test whether the equal-weight premium survives.
- Extend the walk-forward past a single regime.
- Model slippage and market impact rather than basis points on turnover.
- Replace the equal-weight prior with market-cap weights; add sector and concentration constraints.

---

## 6. How to run it

### Installation

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### The two entry points

**Single held-out window:**

```bash
python main.py
```

**Quarterly walk-forward — this is the one to read:**

```bash
python -m src.walkforward
```

Both accept the same flags; the walk-forward adds `--use-fundamentals`.

| Flag | Description |
|---|---|
| `--top-n` | Number of ranked stocks passed into the optimizer |
| `--confidence` | Black-Litterman view confidence in `(0, 1]` |
| `--cost-bps` | Transaction cost in basis points. Charged on gross exposure at entry in single-window mode; on turnover at each rebalance in walk-forward |
| `--tickers-file` | Optional newline- or comma-separated ticker file |
| `--refresh-cache` | Re-download market data and update the local cache |
| `--use-fundamentals` | Walk-forward only. Re-enables the fundamental blend, which contaminates selection |

### Tests

```bash
pytest tests/ -v
```

32 tests, all on synthetic data, no network access required. CI runs the same command on every push and pull request.

Three suites carry the methodological guarantees:

- **`tests/test_no_leakage.py`** — the regression guard. Feeds the feature pipeline seeded random walks and asserts no feature correlates with the future, that the target is exactly `close[t+21]/close[t] - 1`, and that the model cannot predict an unpredictable series. It scores the original leaky feature set alongside the current one across five seeds, so the assertion is a ratio against a live control rather than a threshold that would only hold at one series length.
- **`tests/test_walkforward.py`** — the split audit. For every rebalance it materializes the actual TRAIN/VALID/HOLD date sets through the same `window_slice` the pipeline uses, then asserts the purge gap is exactly `PURGE_DAYS` trading days at both inner boundaries, that the forward-looking *label* of the last row of each window closes before the next opens, that no hold-window date appears anywhere in that rebalance's training or validation data, and that the hold windows tile the period exactly once.
- **`tests/test_pipeline_smoke.py`** — runs the whole pipeline end to end against synthetic generators with no network.

### Repository layout

```text
finoptix/
├── main.py                  # Single-window entry point and CLI
├── config.py                # Universe, split dates, model and backtest parameters
├── src/
│   ├── data.py              # yfinance downloads, retries, cache handling
│   ├── features.py          # Trailing-only, scale-free features; forward-return target
│   ├── ml_returns.py        # Feature panel, purged training, multi-window prediction
│   ├── evaluation.py        # Rank IC and bootstrapped Sharpe-difference test
│   ├── walkforward.py       # Quarterly walk-forward backtest (own entry point)
│   ├── scoring.py           # Ranking and top-N selection
│   ├── black_litterman.py   # Prior, views, Omega, posterior
│   ├── optimizer.py         # Max-Sharpe optimization
│   └── backtest.py          # Simple-return buy-and-hold with drift and costs
├── tests/                   # 32 synthetic-data tests
├── outputs/                 # Committed run artifacts (CSVs and plots)
├── resources_study/         # Concept notes and interview guide
└── data_cache/              # Local market-data cache, generated at runtime
```

### Generated artifacts

Single-window run:

| File | Contents |
|---|---|
| `ml_model_metrics.csv` | Per-ticker training rows, per-window correlation and RMSE |
| `information_coefficient.csv` | Rank IC summary for VALID and TEST |
| `stock_scores.csv` | Composite score and intermediate scoring features |
| `portfolio_weights.csv` | Final optimized weights |
| `performance_stats.csv` | CAGR, volatility, Sharpe, max drawdown per stream |
| `significance.csv` | Bootstrapped Sharpe differences |
| `cumulative_returns.png`, `portfolio_weights.png` | Plots |

Walk-forward run:

| File | Contents |
|---|---|
| `walkforward_summary.csv` | Stitched-series stats per stream, plus turnover and total cost |
| `walkforward_by_period.csv` | Per-rebalance date, n selected, turnover, return per stream, window IC |
| `walkforward_ic.csv` | Pooled IC observations, one row per non-overlapping window |
| `walkforward_ic_summary.csv` | Pooled IC mean, std, t-stat, hit rate, n |
| `walkforward_significance.csv` | Bootstrapped Sharpe differences for the decomposition chain |
| `walkforward_cumulative_returns.png` | Cumulative returns across the full stitched period |

### Configuration

Strategy parameters live in `config.py`.

| Parameter | Purpose |
|---|---|
| `TICKERS` | NSE ticker universe |
| `TRAIN_START` / `TRAIN_END` | Window used to fit models, and nothing else |
| `VALID_START` / `VALID_END` | Window that drives selection, views and covariance |
| `TEST_START` / `TEST_END` | Backtest-only window, untouched until weights are frozen |
| `PRED_HORIZON` | Forward return horizon in trading days (21) |
| `PURGE_DAYS` | Trading days dropped at each inner split boundary |
| `FEATURE_COLUMNS` | The 14 trailing-only, scale-free features |
| `XGB_PARAMS` | Model hyperparameters, deliberately heavily regularized |
| `TOP_N_STOCKS` | Number of stocks selected |
| `RISK_AVERSION`, `TAU`, `VIEW_CONFIDENCE` | Black-Litterman parameters |
| `WALKFORWARD_START` | First date of the expanding training window |
| `WALKFORWARD_VALID_DAYS` | Length of VALID at each rebalance (365 days) |
| `WALKFORWARD_FREQ` | Rebalance frequency (`QS`, quarterly) |
| `WALKFORWARD_WARMUP_DAYS` | Pre-history downloaded so features are warm on day one of TRAIN |
| `USE_FUNDAMENTALS_IN_SELECTION` | **False.** Turning it on contaminates selection |
| `SCORE_WEIGHTS` | Return/PE/DE/market-cap blend, used only when the flag above is on |
| `BENCHMARK_TICKER`, `COST_BPS`, `TRADING_DAYS` | Benchmark, cost and annualization settings |

### Further reading

`resources_study/` contains two documents: a plain-language explanation of the concepts the project turns on, and a project walkthrough with the questions the work invites.

---

## License

MIT. See [LICENSE](LICENSE).
