# FinOptix - Portfolio Optimization

ML-assisted Black-Litterman portfolio optimization for NSE large-cap equities.

Finoptix is an end-to-end Python pipeline that combines technical-feature return forecasting, fundamental ranking, Black-Litterman posterior return estimation, and mean-variance optimization. It downloads market data from Yahoo Finance, trains one XGBoost return model per stock, selects a top-ranked portfolio universe, builds ML-driven investor views, optimizes allocations, and backtests the result against an equal-weight benchmark.

> This project is for education and portfolio demonstration only. It is not financial advice, and historical backtests do not predict future performance.

## Highlights

- Live NSE price and fundamentals ingestion through `yfinance`
- Local data caching in `data_cache/` for repeatable development runs
- Per-ticker XGBoost models trained on technical indicators
- Composite stock ranking using ML expected returns, P/E, debt-to-equity, and market cap
- Black-Litterman posterior returns with absolute views derived from ML predictions
- Max-Sharpe portfolio optimization via PyPortfolioOpt
- Backtest metrics and plots saved to `outputs/`
- Synthetic-data unit tests that do not require network access

## Methodology

The pipeline follows this flow:

```text
NSE ticker universe
        |
        v
Download OHLCV prices and fundamentals
        |
        v
Engineer technical features
        |
        v
Train one XGBoost model per ticker
        |
        v
Blend ML expected returns with fundamental scores
        |
        v
Select top N stocks
        |
        v
Build Black-Litterman prior and ML-based absolute views
        |
        v
Optimize portfolio weights
        |
        v
Backtest vs. equal-weight benchmark
```

### ML-Driven Black-Litterman Views

A common weakness in portfolio notebooks is that ML predictions are used for screening, while Black-Litterman views are manually specified and disconnected from the model. Finoptix avoids that mismatch by using each selected stock's mean XGBoost-predicted return as an absolute Black-Litterman view.

View uncertainty is computed with the He-Litterman proportional rule and scaled by `VIEW_CONFIDENCE`. Lower confidence pulls the posterior closer to the market-implied prior; higher confidence gives more weight to the ML forecasts.

## Repository Structure

```text
finoptix/
├── main.py                  # Pipeline entry point and CLI
├── config.py                # Tickers, dates, model parameters, scoring weights
├── requirements.txt
├── src/
│   ├── data.py              # yfinance downloads, retries, cache handling
│   ├── features.py          # Technical indicator feature engineering
│   ├── ml_returns.py        # Per-ticker XGBoost training and prediction
│   ├── scoring.py           # ML + fundamentals composite ranking
│   ├── black_litterman.py   # Prior, views, Omega, posterior calculations
│   ├── optimizer.py         # Max-Sharpe optimization
│   └── backtest.py          # Performance metrics and comparisons
├── tests/                   # Unit tests using synthetic data
├── outputs/                 # Generated CSVs and plots
└── data_cache/              # Local market-data cache, generated at runtime
```

## Installation

Finoptix requires Python 3.11 or newer.

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
python main.py --tickers-file tickers.txt
python main.py --refresh-cache
```

Options:

| Flag | Description |
|---|---|
| `--top-n` | Number of ranked stocks passed into the optimizer |
| `--confidence` | Black-Litterman view confidence in `(0, 1]` |
| `--tickers-file` | Optional newline- or comma-separated ticker file |
| `--refresh-cache` | Re-download market data and update the local cache |

The pipeline writes these artifacts to `outputs/`:

| File | Description |
|---|---|
| `ml_model_metrics.csv` | Per-ticker prediction correlation and RMSE |
| `stock_scores.csv` | Composite score and intermediate scoring features |
| `portfolio_weights.csv` | Final optimized portfolio weights |
| `performance_stats.csv` | CAGR, annualized volatility, Sharpe, max drawdown |
| `cumulative_returns.png` | BL portfolio vs. equal-weight cumulative returns |
| `portfolio_weights.png` | Final portfolio allocation chart |

## Configuration

Most strategy parameters live in `config.py`.

| Parameter | Purpose |
|---|---|
| `TICKERS` | NSE ticker universe |
| `TRAIN_START`, `TRAIN_END` | Training window for ML models |
| `TEST_START`, `TEST_END` | Held-out window for evaluation and backtest |
| `FEATURE_COLUMNS` | Technical features used by XGBoost |
| `XGB_PARAMS` | XGBoost hyperparameters |
| `SCORE_WEIGHTS` | Blend of return, P/E, D/E, and market-cap scores |
| `TOP_N_STOCKS` | Default number of selected stocks |
| `RISK_AVERSION`, `TAU` | Black-Litterman model parameters |
| `VIEW_CONFIDENCE` | Weight assigned to ML-driven views |

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

The tests use synthetic data and do not require internet access. They cover Black-Litterman behavior, optimizer outputs, backtest statistics, CLI parsing, retry logic, and cache refresh behavior.

## Sample Results

The following sample was generated from a live Yahoo Finance run on July 4, 2026.

| Setting | Value |
|---|---|
| Training window | 2020-01-01 to 2025-07-04 |
| Test/backtest window | 2025-07-04 to 2026-07-04 |
| Selected stocks | `RELIANCE.NS`, `COALINDIA.NS`, `TCS.NS`, `HCLTECH.NS`, `INFY.NS`, `WIPRO.NS`, `ITC.NS`, `HEROMOTOCO.NS`, `ADANIPORTS.NS`, `POWERGRID.NS` |

Model validation summary:

- 47 tickers trained successfully.
- Prediction correlations were finite for all trained tickers.
- Correlation range: 0.212 to 0.516.
- No ticker had NaN or strongly negative prediction correlation.

Backtest comparison:

| Portfolio | CAGR | Ann. Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Black-Litterman | -7.24% | 13.73% | -0.479 | -10.92% |
| Equal Weight | -17.55% | 13.99% | -1.309 | -21.22% |

These results are included to demonstrate that the pipeline runs end to end on real data. They should not be interpreted as an investment recommendation or an expected future return.

## Known Limitations

- Yahoo Finance is a free, unofficial data source. Ticker availability, schemas, rate limits, and fundamentals can change without notice.
- `LTIM.NS` and `TATAMOTORS.NS` returned 404/no-timezone errors in the July 4, 2026 run; the pipeline skipped them and continued with the remaining universe.
- Fundamentals from `yfinance` are current snapshots, not point-in-time historical fundamentals, so historical backtests can contain look-ahead bias.
- Market-implied equilibrium returns currently use an equal-weight proxy rather than float-adjusted market-cap weights.
- The backtest does not model transaction costs, slippage, taxes, liquidity constraints, or periodic rebalancing.
- This is a research and portfolio project, not production trading infrastructure.

## Roadmap

Potential future improvements:

- Replace equal-weight prior weights with market-cap weights.
- Add sector exposure reporting and concentration constraints.
- Add walk-forward rebalancing rather than a single held-out allocation.
- Store model artifacts and prediction diagnostics per run.
- Add optional benchmark comparison against an NSE index ETF or index series.

## License

No license has been specified yet. Add a `LICENSE` file before distributing or reusing this project publicly.
