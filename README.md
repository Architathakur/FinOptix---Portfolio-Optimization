# FinOptix - Portfolio Optimization

ML-assisted Black-Litterman portfolio optimization for NSE large-cap equities.

FinOptix is an end-to-end Python pipeline that combines technical-feature return forecasting, fundamental ranking, Black-Litterman posterior return estimation, and mean-variance optimization. It downloads market data from Yahoo Finance, trains one XGBoost return model per stock, selects a portfolio universe, builds ML-driven investor views, optimizes allocations, and backtests the result against equal weighting and the NIFTY 50 with transaction costs and a bootstrap significance test.

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
| Original features + same-day target + original XGB params | 0.394 |
| Original features + same-day target + **regularized** XGB params | 0.379 |
| Current features + 21-day forward target + purged split | 0.051 |

The middle row is the control that matters. Swapping in the current, heavily regularized hyperparameters barely moves the score, so the cause was the construction of the features and target, not model tuning. Per feature, the original set had `returns_20` correlating 0.226 with its own target on a random walk; the current set peaks at 0.080 (`dist_52w_high`), with none of its 14 features above 0.15.

This is a committed regression test, not a one-off script — see `tests/test_no_leakage.py`, which fails if the leaky construction is ever reintroduced.

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
- Backtest against equal weighting and the NIFTY 50, with transaction costs
- Bootstrapped Sharpe-difference tests with confidence intervals and p-values
- Synthetic-data unit tests that do not require network access, including a random-walk leakage guard

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
| `cumulative_returns.png` | BL vs. equal-weight vs. NIFTY 50 cumulative returns |
| `portfolio_weights.png` | Final portfolio allocation chart |

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

- `tests/test_no_leakage.py` — the regression guard. It feeds the feature pipeline a seeded random walk and asserts that no feature correlates with the future, that the target is exactly `close[t + 21] / close[t] - 1`, and that the configured model cannot predict an unpredictable series. It scores the original leaky feature set alongside the current one on the same data, so the assertion is a ratio rather than an absolute threshold.
- `tests/test_pipeline_smoke.py` — runs the entire pipeline end to end against synthetic generators with no network access, and asserts the purge gap between the end of training and `VALID_START`.

## Results

Generated from a live Yahoo Finance run on 2026-09-04. These are reported exactly as produced.

| Setting | Value |
|---|---|
| Training window | 2020-01-01 to 2024-09-04 (885 usable rows per ticker after purge) |
| Validation window | 2024-09-04 to 2025-09-04 |
| Test/backtest window | 2025-09-04 to 2026-09-04 |
| Universe | 47 of 49 tickers trained; `LTIM.NS` and `TATAMOTORS.NS` returned 404s |
| Entry cost | 20 bps of gross exposure, charged on day 0 |
| Selected stocks | `COALINDIA.NS`, `RELIANCE.NS`, `NTPC.NS`, `TCS.NS`, `ITC.NS`, `ONGC.NS`, `SUNPHARMA.NS`, `HINDALCO.NS`, `ADANIPORTS.NS`, `ASIANPAINT.NS` |

### Rank information coefficient

| Window | n_periods | Mean IC | IC std | t-stat | Hit rate |
|---|---:|---:|---:|---:|---:|
| VALID | 12 | -0.087 | 0.161 | -1.87 | 0.42 |
| TEST | 11 | 0.057 | 0.211 | 0.90 | 0.64 |

**The model has no demonstrated predictive skill on this data.** The test-window IC of 0.057 carries a t-stat of 0.90, which is not distinguishable from zero, and the pipeline logs a warning saying so at the end of every run. The validation IC is *negative*: over the window actually used to pick stocks, the model ranked them slightly backwards. Selection therefore ran on a signal that was, if anything, anti-predictive at the moment the choice was made.

This is the expected result. Cross-sectional equity returns are close to unpredictable from technical indicators alone, and a corrected pipeline should produce an IC near zero. The earlier 0.212-0.516 figures were not a better model; they were the same non-signal measured with a broken ruler.

### Backtest

| Portfolio | CAGR | Ann. Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Black-Litterman | -1.88% | 11.10% | -0.115 | -8.32% |
| Equal Weight | 2.15% | 11.66% | 0.241 | -8.48% |
| NIFTY 50 | -3.76% | 13.12% | -0.227 | -15.18% |

**The Black-Litterman portfolio lost money and lost to equal-weighting its own stock picks.** It finished ahead of the NIFTY 50, but see below before reading anything into that.

### Are the differences real?

Bootstrapped Sharpe differences, 5000 resamples, dates resampled jointly so contemporaneous correlation is preserved:

| Comparison | Sharpe difference | 95% CI | p-value | n |
|---|---:|---:|---:|---:|
| BL vs Equal Weight | -0.356 | [-0.779, 0.066] | 0.104 | 251 |
| BL vs NIFTY 50 | 0.133 | [-1.386, 1.589] | 0.862 | 246 |

Both confidence intervals straddle zero, so neither result is distinguishable from luck. The portfolio's loss to equal weighting is not statistically established, and neither is its win over the NIFTY 50 — a p-value of 0.862 on the latter means that comparison carries essentially no information. One year of daily data cannot resolve Sharpe differences of this size in either direction, which is itself worth knowing: a 12-month backtest is not enough evidence to conclude anything about a strategy, favourable or otherwise.

For context on why the per-ticker numbers are no longer the headline: across the 47 trained tickers, per-ticker time-series correlation between prediction and realized forward return averaged 0.048 on VALID and 0.086 on TEST, ranging from -0.43 to 0.62. Those correlations are computed on heavily overlapping 21-day windows, which inflates them, and they answer a question the strategy never asks — the strategy ranks stocks against each other on a given day, which is what the rank IC measures.

## Known Limitations

- Yahoo Finance is a free, unofficial data source. Ticker availability, schemas, rate limits, and fundamentals can change without notice.
- `LTIM.NS` and `TATAMOTORS.NS` returned 404/no-timezone errors in the 2026-09-04 run; the pipeline skipped them and continued with the remaining universe.
- Fundamentals from `yfinance` are current snapshots, not point-in-time historical fundamentals. This is a remaining source of look-ahead bias in the selection step, and it has not been fixed.
- The ticker universe is today's NIFTY 50 constituents, so the backtest carries survivorship bias.
- Market-implied equilibrium returns use an equal-weight proxy rather than float-adjusted market-cap weights.
- The backtest models a one-off entry cost only. It does not model slippage, market impact, taxes, liquidity constraints, or the cost of periodic rebalancing.
- The portfolio is formed once and held. There is no walk-forward re-estimation, so the reported result rests on a single allocation decision over a single 12-month window.
- A single test window cannot support a conclusion about strategy performance in either direction. See the p-values above.
- This is a research and portfolio project, not production trading infrastructure.

## Roadmap

Potential future improvements:

- Replace equal-weight prior weights with market-cap weights.
- Source point-in-time fundamentals to remove the remaining look-ahead bias in selection.
- Build the universe from historical index membership to remove survivorship bias.
- Add sector exposure reporting and concentration constraints.
- Add walk-forward rebalancing across many windows, which would give the IC and the Sharpe estimates enough observations to be meaningful.
- Model ongoing turnover costs rather than a single entry cost.
- Store model artifacts and prediction diagnostics per run.

## License

MIT. See [LICENSE](LICENSE).
