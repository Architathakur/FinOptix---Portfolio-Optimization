# FinOptix - Portfolio Optimization

ML-assisted Black-Litterman portfolio optimization for NSE large-cap equities.

FinOptix is an end-to-end Python pipeline that combines technical-feature return forecasting, fundamental ranking, Black-Litterman posterior return estimation, and mean-variance optimization. It downloads market data from Yahoo Finance, trains one XGBoost return model per stock, selects a portfolio universe, builds ML-driven investor views, optimizes allocations, and backtests the result against equal weighting of its picks, equal weighting of the whole universe, and the NIFTY 50, with transaction costs and bootstrap significance tests. It runs either as a single held-out window or as an 18-step quarterly walk-forward.

> This project is for education and portfolio demonstration only. It is not financial advice, and historical backtests do not predict future performance.

## The Look-Ahead Bug

An earlier version of this pipeline reported per-ticker prediction correlations of 0.212 to 0.516 and a Black-Litterman portfolio that beat equal weighting by ten percentage points of CAGR. Both were artifacts of look-ahead bias. Those numbers have been deleted from this README rather than corrected in place, because they were not measurements of anything.

### What the bug was

The prediction target was the return *ending* at day `t`:

```python
feat["returns"] = df["Close"].pct_change()          # return ending at t
feat["ma_10"] = df["Close"].rolling(10).mean()      # window contains Close[t]
feat["volatility_20"] = df["Close"].rolling(20).std()
feat["upper_band"] = ma_20 + 2 * std_20
```

`ma_10`, `ma_50`, `volatility_20`, `upper_band` and `lower_band` were all rolling windows that included `Close[t]`. The model was handed today's closing price and asked to predict today's move. That is the entire source of the reported 0.21-0.52 correlations; a genuine daily-return model built on technical indicators scores roughly 0.02-0.05.

### The evidence

Feed the feature pipeline a seeded pure random walk — iid normal returns, 8000 days — where the predictability of the target is exactly zero by construction. Anything scoring meaningfully above zero is reading the future, not finding signal.

| Setup | \|corr\| on a random walk |
|---|---:|
| Original features + same-day target + original XGB params | 0.418 |
| Original features + same-day target + **regularized** XGB params | 0.404 |
| Current features + 21-day forward target + purged split | 0.044 |

Mean of five seeds; per-seed the original set spans 0.385-0.445 and the current set 0.008-0.123.

The middle row is the control that matters. Swapping in the current, heavily regularized hyperparameters barely moves the score, so the cause was the construction of the features and target, not model tuning. Per feature, the original set had `returns_20` correlating 0.226 with its own target on a random walk; the current set peaks at 0.080 (`dist_52w_high`), with none of its 14 features above 0.15.

All three rows are computed by a committed regression test, not a one-off script. `tests/test_no_leakage.py` keeps a working reconstruction of the original buggy pipeline as a positive control and asserts that the current feature set scores below a quarter of it — a ratio rather than a fixed threshold, because the current set's score is a noise floor that rises as the series shortens, while the leaky control sits near 0.4 at every length. The test also asserts that the control still leaks under the regularized hyperparameters, so the diagnosis above cannot silently become wrong.

### How it propagated

The bug did not stay in the model. `scoring.py` ranked stocks by mean *predicted* return over the test window, and those predictions echoed realized test-window returns. So the pipeline selected the stocks that had already performed well over the exact window it then backtested, and then reported that the resulting portfolio beat equal weighting. The old outperformance was circular: the benchmark was being compared against a portfolio built with knowledge of the answer.

### What changed

| Area | Before | After |
|---|---|---|
| Target | Return ending at `t` | `close.pct_change(21).shift(-21)` — the forward 21-day return |
| Features | Raw rupee levels (`ma_10`, `upper_band`, `volatility_20`) containing `Close[t]` | 14 scale-free, strictly trailing features — ratios, z-scores, correlations |
| Split | Two-way TRAIN/TEST, with TEST used for evaluation *and* selection *and* covariance | Three-way TRAIN / VALID / TEST; VALID makes every decision, TEST is opened once, after weights are frozen |
| Split hygiene | None | TRAIN purged by 21 rows so a forward label cannot straddle the boundary |
| Feature computation | Built separately per download, so every rolling window warmed up again inside the test slice | Computed once over the continuous price history, then sliced |
| Headline metric | Per-ticker time-series correlation | Cross-sectional rank IC, sampled every 21 days so forward windows do not overlap |
| Benchmark | Equal weighting of the model's own picks | Equal weight **and** NIFTY 50 |
| Costs | None | One-off entry cost, 20bps of gross exposure |
| Return maths | Log returns summed by weight, then compounded as if simple | Simple returns throughout; buy-and-hold weights drift instead of being silently rebalanced daily for free |
| Significance | None | Bootstrapped Sharpe differences with CIs and p-values, resampling dates jointly |

**The deliverable of this project is the evaluation harness, not the return number.** Purged splits, non-overlapping IC sampling, transaction costs and significance testing are the parts worth reusing. The portfolio's performance over any single 12-month window is close to meaningless either way, and the results below say so explicitly.

## Highlights

- Live NSE price and fundamentals ingestion through `yfinance`
- Local data caching in `data_cache/` for repeatable development runs
- Per-ticker XGBoost models trained on strictly backward-looking, scale-free technical features
- Three-way TRAIN / VALID / TEST split with a purge gap at each boundary
- Cross-sectional rank information coefficient with non-overlapping forward windows
- Composite stock ranking using ML expected returns, P/E, debt-to-equity, and market cap
- Black-Litterman posterior returns with absolute views derived from ML predictions
- Max-Sharpe portfolio optimization via PyPortfolioOpt
- Walk-forward backtesting: 18 quarterly rebalances, expanding training window, purged at both inner boundaries
- Three benchmarks, which decompose the result into an equal-weighting effect, a selection effect and a weighting effect
- Turnover-based transaction costs, charged against drifted weights at each rebalance
- Bootstrapped Sharpe-difference tests with confidence intervals and p-values
- Fundamentals excluded from selection by default, because a current snapshot is look-ahead bias
- Synthetic-data unit tests that do not require network access, including a random-walk leakage guard and a strict walk-forward split audit

## Methodology

The pipeline follows this flow:

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
Blend VALID predictions with fundamental scores -> select top N
        |
        v
Build Black-Litterman prior, views and covariance -- all from VALID
        |
        v
Optimize portfolio weights  ->  WEIGHTS FROZEN
        |
        v
Backtest on TEST vs. equal weight and NIFTY 50, with costs
        |
        v
Bootstrap the Sharpe differences
```

Nothing in the pipeline reads the TEST window until the weights are final. The moment a parameter is chosen by looking at TEST, TEST stops being held out.

### Walk-forward

The single-window scheme above yields one allocation decision and about eleven non-overlapping IC observations — not enough to distinguish a real effect from noise in either direction, which is why every confidence interval it produced straddled zero. `src/walkforward.py` re-runs the entire decision process quarterly:

```text
At each rebalance date t:
  TRAIN   WALKFORWARD_START -> (t - 1y - PURGE_DAYS)    expanding, fixed start
  VALID   (t - 1y)          -> (t - PURGE_DAYS)         makes every decision
  HOLD    t                 -> next rebalance date      strictly out of sample
```

`PURGE_DAYS` trading days are removed at **both** inner boundaries. A label at time `t` spans `(t, t + PRED_HORIZON]`, so without the first gap the tail of TRAIN carries labels reaching into VALID, and without the second the tail of VALID carries labels reaching past `t` into the window being held. The gaps are counted on the actual trading calendar, so they survive holidays.

Weights are held through the quarter and drift with performance. At the next rebalance the cost is charged on **turnover against the drifted weights**, `(COST_BPS / 10000) * sum|w_new - w_drifted|`, not on gross exposure — charging against the original targets would invent a trade that never happened.

The feature panel is computed once over the full history and sliced per rebalance. Recomputing inside the loop would restart every rolling window in each slice, and would also be 18x the work.

### ML-Driven Black-Litterman Views

A common weakness in portfolio notebooks is that ML predictions are used for screening while Black-Litterman views are manually specified and disconnected from the model. FinOptix wires the two together by using each selected stock's mean XGBoost-predicted return as an absolute Black-Litterman view, annualized so that the views, the prior and the risk-aversion parameter share units.

That is a statement about the plumbing, not about the forecasts. On the run reported below the views carry no measurable predictive content, and `VIEW_CONFIDENCE` therefore controls how much weight the optimizer gives to a signal that has not been shown to work. View uncertainty is computed with the He-Litterman proportional rule and scaled by `VIEW_CONFIDENCE`; lower confidence pulls the posterior closer to the market-implied prior.

## Repository Structure

```text
finoptix/
├── main.py                  # Pipeline entry point and CLI
├── config.py                # Tickers, split dates, model parameters, scoring weights
├── requirements.txt
├── src/
│   ├── data.py              # yfinance downloads, retries, cache handling
│   ├── features.py          # Trailing-only, scale-free feature engineering
│   ├── ml_returns.py        # Feature panel, purged training, multi-window prediction
│   ├── evaluation.py        # Rank IC and bootstrapped Sharpe-difference test
│   ├── walkforward.py       # Quarterly walk-forward backtest (own entry point)
│   ├── scoring.py           # ML + fundamentals composite ranking
│   ├── black_litterman.py   # Prior, views, Omega, posterior calculations
│   ├── optimizer.py         # Max-Sharpe optimization
│   └── backtest.py          # Simple-return buy-and-hold backtest with costs
├── tests/                   # Unit tests using synthetic data
├── outputs/                 # Generated CSVs and plots
└── data_cache/              # Local market-data cache, generated at runtime
```

## Installation

FinOptix requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline with default settings:

```bash
python main.py
```

Useful CLI options:

```bash
python main.py --top-n 10
python main.py --confidence 0.5
python main.py --cost-bps 20
python main.py --tickers-file tickers.txt
python main.py --refresh-cache
```

Run the walk-forward backtest (this is the one to read):

```bash
python -m src.walkforward
```

It accepts the same flags, plus `--use-fundamentals` to re-enable the fundamental blend in selection (off by default — see Configuration).

Options:

| Flag | Description |
|---|---|
| `--top-n` | Number of ranked stocks passed into the optimizer |
| `--confidence` | Black-Litterman view confidence in `(0, 1]` |
| `--cost-bps` | One-off entry cost in basis points of gross exposure, charged on day 0 |
| `--tickers-file` | Optional newline- or comma-separated ticker file |
| `--refresh-cache` | Re-download market data and update the local cache |

The pipeline writes these artifacts to `outputs/`:

| File | Description |
|---|---|
| `ml_model_metrics.csv` | Per-ticker training rows and per-window prediction correlation and RMSE |
| `information_coefficient.csv` | Rank IC summary for VALID and TEST: n_periods, mean, std, t-stat, hit rate |
| `stock_scores.csv` | Composite score and intermediate scoring features |
| `portfolio_weights.csv` | Final optimized portfolio weights |
| `performance_stats.csv` | CAGR, annualized volatility, Sharpe, max drawdown per stream |
| `significance.csv` | Bootstrapped Sharpe differences with 95% CIs and p-values |
| `cumulative_returns.png` | Cumulative returns for all four streams |
| `portfolio_weights.png` | Final portfolio allocation chart |

The walk-forward run writes:

| File | Description |
|---|---|
| `walkforward_summary.csv` | Stitched-series stats per stream, plus turnover and total cost |
| `walkforward_by_period.csv` | Per-rebalance breakdown: date, n selected, turnover, return per stream, window IC |
| `walkforward_ic.csv` | The pooled IC observations, one row per non-overlapping window |
| `walkforward_ic_summary.csv` | Pooled IC mean, std, t-stat, hit rate, n |
| `walkforward_significance.csv` | Bootstrapped Sharpe differences for the decomposition chain |
| `walkforward_cumulative_returns.png` | Cumulative returns across the full stitched period |

## Configuration

Most strategy parameters live in `config.py`.

| Parameter | Purpose |
|---|---|
| `TICKERS` | NSE ticker universe |
| `TRAIN_START`, `TRAIN_END` | Window used to fit the models, and nothing else |
| `VALID_START`, `VALID_END` | Window that drives selection, BL views and covariance |
| `TEST_START`, `TEST_END` | Backtest-only window, untouched until weights are frozen |
| `PRED_HORIZON` | Forward return horizon in trading days (21) |
| `PURGE_DAYS` | Rows dropped from the end of the training slice |
| `FEATURE_COLUMNS` | Trailing-only, scale-free technical features |
| `XGB_PARAMS` | XGBoost hyperparameters |
| `SCORE_WEIGHTS` | Blend of return, P/E, D/E, and market-cap scores |
| `TOP_N_STOCKS` | Default number of selected stocks |
| `RISK_AVERSION`, `TAU` | Black-Litterman model parameters |
| `VIEW_CONFIDENCE` | Weight assigned to ML-driven views |
| `WALKFORWARD_START` | First date of the expanding training window |
| `WALKFORWARD_VALID_DAYS` | Length of the VALID window at each rebalance (365 days) |
| `WALKFORWARD_FREQ` | Rebalance frequency (`QS`, quarterly) |
| `WALKFORWARD_WARMUP_DAYS` | Pre-history downloaded so features are warm on day one of TRAIN |
| `USE_FUNDAMENTALS_IN_SELECTION` | **False.** Turning it on contaminates selection with look-ahead bias |
| `BENCHMARK_TICKER` | Index benchmark, default `^NSEI` |
| `COST_BPS` | Default one-off entry cost in basis points |
| `TRADING_DAYS` | Annualization factor (252) |

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

The tests use synthetic data and do not require internet access. They cover Black-Litterman behavior, optimizer outputs, backtest statistics including weight drift and cost accounting, CLI parsing, retry logic, and cache refresh behavior.

Two suites are worth calling out:

- `tests/test_no_leakage.py` — the regression guard. It feeds the feature pipeline a seeded random walk and asserts that no feature correlates with the future, that the target is exactly `close[t + 21] / close[t] - 1`, and that the configured model cannot predict an unpredictable series. It scores the original leaky feature set alongside the current one on the same random walks, across five seeds, so the assertion is a ratio against a live control rather than an absolute threshold that would only hold at one series length.
- `tests/test_pipeline_smoke.py` — runs the entire pipeline end to end against synthetic generators with no network access, and asserts the purge gap between the end of training and `VALID_START`.
- `tests/test_walkforward.py` — the walk-forward split audit. For every rebalance it materializes the actual TRAIN/VALID/HOLD date sets through the same `window_slice` the pipeline uses and asserts that the purge gap is exactly `PURGE_DAYS` trading days at both inner boundaries, that the forward-looking *label* of the last row of each window closes before the next window opens, that no hold-window date appears anywhere in that rebalance's training or validation data, and that the hold windows tile the period exactly once with no date double-counted or skipped.

## Results

Two backtests are reported. **The walk-forward result is the headline**; the
single-window one is kept because it is what the rest of this README's
methodology section describes, and because comparing the two is instructive.

Both were generated from live Yahoo Finance runs on 2026-09-05, with
`USE_FUNDAMENTALS_IN_SELECTION = False`, 20bps of transaction cost, and
`TOP_N_STOCKS = 10`. Reported exactly as produced.

### Walk-forward (headline)

18 quarterly rebalances, 2022-04-01 to 2026-09-05, expanding training window,
47 tradeable tickers. Every allocation is strictly out of sample: at each
rebalance the models are refit on TRAIN, the stocks and weights are chosen on
VALID, and the weights are then held through the following quarter with 21
trading days purged at both inner boundaries.

**Pooled rank IC across all hold windows**

| n_periods | Mean IC | IC std | t-stat | Hit rate |
|---:|---:|---:|---:|---:|
| 52 | 0.0096 | 0.1868 | **0.37** | 0.577 |

Pooling was the point of this exercise: it takes the IC sample from 11
observations to 52. The answer did not change. A mean IC of 0.0096 with a
t-stat of 0.37 is zero. **The model does not rank stocks better than chance,
and now that conclusion rests on a sample large enough to say so** rather than
on a window too short to distinguish anything. Individual quarters ranged from
-0.23 to +0.13 and only 7 of 18 had a positive mean IC.

**Performance on the stitched daily series**

| Stream | CAGR | AnnVol | Sharpe | Max Drawdown | Avg turnover/rebalance | Total cost |
|---|---:|---:|---:|---:|---:|---:|
| Black-Litterman | 11.75% | 14.69% | 0.830 | -22.13% | 0.759 | 2.73% |
| Equal-Weight (Selected) | 12.98% | 15.55% | 0.863 | -24.12% | 0.711 | 2.56% |
| Equal-Weight (Universe) | **13.00%** | 12.96% | **1.008** | **-17.40%** | 0.135 | 0.49% |
| NIFTY 50 | 7.44% | 13.18% | 0.611 | -15.77% | 0.056 | 0.20% |

**The decomposition, and the actual finding**

Bootstrapped Sharpe differences, 5000 resamples, dates resampled jointly:

| Step | Isolates | Sharpe diff | 95% CI | p |
|---|---|---:|---:|---:|
| Universe EW vs NIFTY 50 | equal- vs cap-weighting | **+0.399** | [+0.102, +0.699] | **0.007** |
| Selected EW vs Universe EW | the selection layer | -0.145 | [-0.603, +0.314] | 0.536 |
| BL vs Selected EW | the weighting layer | -0.033 | [-0.296, +0.231] | 0.816 |
| BL vs NIFTY 50 | everything combined | +0.221 | [-0.417, +0.869] | 0.495 |

The chain is additive and lands exactly on the observed Sharpes:
`0.611 (NIFTY) + 0.399 - 0.145 - 0.033 = 0.830 (BL)`.

**The only effect in this pipeline that is distinguishable from noise is
equal-weighting versus cap-weighting the index.** It is worth +0.40 Sharpe at
p = 0.007, and it is not a machine-learning result — it is the well-documented
equal-weight premium, available to anyone willing to hold 47 stocks in equal
proportion and rebalance quarterly.

Both ML layers subtract. Selection costs -0.145 Sharpe, weighting a further
-0.033; neither is statistically distinguishable from zero, but neither shows
any sign of adding value either. Black-Litterman finishes ahead of the NIFTY 50
by +0.221 Sharpe, and the entire margin -- more than the entire margin -- comes
from the equal-weighting effect that the universe benchmark isolates. Without
that benchmark the +0.221 looks like the strategy working. It is not.

The cost side reinforces it. The Black-Litterman portfolio turns over 76% of
the book per quarter and pays 2.73% in transaction costs over the period,
against 0.49% for the universe benchmark that beat it on every measure --
higher CAGR, lower volatility, shallower drawdown, higher Sharpe.

This is the expected outcome. Cross-sectional equity returns are close to
unpredictable from technical indicators, so a correctly evaluated pipeline
should show an IC near zero and no reliable edge from selection or weighting.
That is what it shows.

### Single-window (for comparison)

TRAIN 2020-01-01 to 2024-09-05, VALID to 2025-09-05, TEST 2025-09-05 to
2026-09-05. Selected on VALID: `INDUSINDBK.NS`, `NTPC.NS`, `COALINDIA.NS`,
`ASIANPAINT.NS`, `SBIN.NS`, `AXISBANK.NS`, `ADANIPORTS.NS`, `SUNPHARMA.NS`,
`NESTLEIND.NS`, `ITC.NS`.

| Window | n_periods | Mean IC | IC std | t-stat | Hit rate |
|---|---:|---:|---:|---:|---:|
| VALID | 12 | -0.1073 | 0.1783 | -2.09 | 0.333 |
| TEST | 11 | 0.0369 | 0.1648 | 0.74 | 0.545 |

| Stream | CAGR | AnnVol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Black-Litterman | 7.28% | 12.09% | 0.641 | -9.94% |
| Equal-Weight (Selected) | 11.26% | 12.89% | 0.893 | -10.99% |
| Equal-Weight (Universe) | 3.31% | 12.24% | 0.327 | -12.42% |
| NIFTY 50 | -3.69% | 13.12% | -0.221 | -15.18% |

| Step | Sharpe diff | 95% CI | p | n |
|---|---:|---:|---:|---:|
| Universe EW vs NIFTY 50 | +0.572 | [+0.035, +1.166] | 0.047 | 246 |
| Selected EW vs Universe EW | +0.566 | [-0.322, +1.470] | 0.220 | 251 |
| BL vs Selected EW | -0.251 | [-0.547, +0.048] | 0.097 | 251 |
| BL vs NIFTY 50 | +0.902 | [-0.233, +2.079] | 0.136 | 246 |

The TEST IC t-stat of 0.74 is again indistinguishable from zero. The VALID IC
is *negative* at -2.09, meaning that over the window used to choose the stocks,
the model ranked them backwards.

**Why this table should not be trusted, and why walk-forward exists.** An
earlier run of this same single-window backtest -- before fundamentals were
removed from selection, and dated one day earlier -- reported a
Black-Litterman CAGR of -1.88% and a Sharpe of -0.115. Changing the selection
rule and moving the window by a single day swung the headline return by nine
percentage points. Nothing about the strategy changed; the sample was simply
too small to hold still. One allocation decision and eleven IC observations
cannot support a conclusion, which is precisely why every confidence interval
in that run straddled zero, and why the walk-forward result above is the one
worth reading.

## Known Limitations

- Yahoo Finance is a free, unofficial data source. Ticker availability, schemas, rate limits, and fundamentals can change without notice.
- `LTIM.NS` and `TATAMOTORS.NS` returned 404/no-timezone errors in the 2026-09-05 runs; the pipeline skipped them and continued with the remaining 47 tickers.
- **Fundamentals are excluded from selection, and that is a workaround rather than a fix.** `download_fundamentals` returns today's trailing P/E, D/E and market cap — a single current snapshot — and `SCORE_WEIGHTS` gives those three 60% of the selection score. Using a 2026 balance sheet to decide what to buy in 2022 is look-ahead bias, and walk-forward makes it worse because the same snapshot would drive all 18 rebalances. Doing it properly needs point-in-time fundamentals, which requires a paid data source, so `USE_FUNDAMENTALS_IN_SELECTION` defaults to `False` and selection ranks on the ML score alone. **If you set that flag to `True`, the selection layer is contaminated and the backtest stops being a measurement.**
- The ticker universe is today's NIFTY 50 constituents, so both backtests carry survivorship bias. This affects the universe benchmark too, and is the most likely reason equal weighting beats the index so cleanly: the 47 names are the ones that were still in the index in 2026.
- Market-implied equilibrium returns use an equal-weight proxy rather than float-adjusted market-cap weights.
- Transaction costs are modelled as basis points on turnover. Slippage, market impact, taxes, liquidity constraints and borrow are not modelled, and at ~76% quarterly turnover the true cost of the Black-Litterman portfolio is understated more than the benchmarks'.
- The walk-forward covers 2022-2026, a single mostly-rising regime for Indian equities. 18 rebalances is enough to say the IC is not distinguishable from zero; it is not enough to characterise behaviour across regimes.
- The models are refit quarterly but every hyperparameter is fixed across the whole period. Nothing is tuned per rebalance, which is deliberate — tuning inside the loop on VALID would need its own nested split — but it does mean the model is not adapting.
- This is a research and portfolio project, not production trading infrastructure.

## Roadmap

Potential future improvements:

- Source point-in-time fundamentals, so the fundamental blend can be re-enabled without contaminating selection.
- Build the universe from historical index membership to remove survivorship bias, which would also test whether the equal-weight premium above survives.
- Replace equal-weight prior weights with market-cap weights.
- Add sector exposure reporting and concentration constraints.
- Model slippage and market impact, not just a basis-point charge on turnover.
- Extend the walk-forward further back to cover more than one market regime.
- Store model artifacts and prediction diagnostics per run.

## License

MIT. See [LICENSE](LICENSE).
