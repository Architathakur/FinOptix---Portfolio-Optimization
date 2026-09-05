# FinOptix: Project Walkthrough and Interview Guide

The project narrative and the questions you should expect. Read
`BEGINNER_CONCEPTS_AND_INTERVIEW_PREP.md` first for the underlying ideas.

Every figure here traces to a committed artifact in `outputs/`, to a test, or to
a named commit. Nothing is illustrative.

---

## 1. One sentence

An end-to-end pipeline that forecasts monthly returns for NSE large caps, builds
a Black-Litterman portfolio from those forecasts, and — the actual point —
evaluates the result rigorously enough to establish that the forecasts have no
predictive skill.

## 2. The honest summary

Lead with this. Do not bury it.

> I built an ML-driven portfolio optimiser. It reported strong results. The
> results were a look-ahead bug. I found it, fixed it, rebuilt the evaluation
> around it, and the corrected pipeline shows no predictive skill — which is the
> expected result for technical indicators on cross-sectional equity returns.
> The deliverable is the evaluation harness that can prove that.

Anyone can build a pipeline that reports a good number. The demonstrable skill
here is catching the reason it was wrong.

---

## 3. The arc

**Built it.** Download NSE prices, engineer technical features, train one
gradient-boosted model per stock, rank stocks, form Black-Litterman views,
optimise for maximum Sharpe, backtest against equal weighting.

**It reported prediction correlations of 0.212 to 0.516.** Published work on
daily-return models using technical indicators sits around 0.02–0.05. Being ten
times better than the field, with free data, is not a discovery.

**Found the bug.** The target was the return *ending* at day `t`, while
`ma_10`, `ma_50`, `volatility_20` and both Bollinger bands were rolling windows
containing `Close[t]`. The model was given today's closing price and asked what
today's move was.

**Proved it, rather than asserting it.** Fed the pipeline a seeded random walk,
where true predictability is exactly zero. The old construction scored 0.418;
the current one scores 0.044. Re-running the old features under the current
regularised settings still scored 0.404, which rules out overfitting as the
cause and pins it on the feature/target construction.

**Traced how it spread.** Selection ranked stocks by mean *predicted* return
over the test window, and those predictions echoed realised test-window
returns. The pipeline was choosing the stocks that had already done well over
the window it then scored. The old outperformance was circular.

**Rebuilt the evaluation.** Forward-looking target, scale-free trailing
features, three-way purged split, cross-sectional rank IC, real benchmarks,
transaction costs, bootstrap significance tests.

**Discovered the sample was too small to conclude anything.** Every confidence
interval straddled zero. So: quarterly walk-forward, 18 rebalances, IC sample
from 11 to 52.

**Added the benchmark that changed the interpretation.** Equal weighting of the
full universe, which isolates the selection layer. It showed the strategy's
apparent edge over the index was entirely the equal-weight premium.

---

## 4. Architecture

| Module | Responsibility |
|---|---|
| `src/data.py` | yfinance downloads, retries, local caching |
| `src/features.py` | Trailing-only, scale-free features; the forward-return target |
| `src/ml_returns.py` | Feature panel built once; purged training; multi-window prediction |
| `src/evaluation.py` | Cross-sectional rank IC; bootstrapped Sharpe-difference test |
| `src/scoring.py` | Ranking and top-N selection |
| `src/black_litterman.py` | Prior, views, Omega, posterior |
| `src/optimizer.py` | Max-Sharpe weights via PyPortfolioOpt |
| `src/backtest.py` | Simple-return buy-and-hold with drift and costs |
| `src/walkforward.py` | Quarterly walk-forward; its own entry point |
| `main.py` | Single-window pipeline entry point |

Two entry points: `python main.py` for the single held-out window, and
`python -m src.walkforward` for the walk-forward. The second is the one to read.

### Three structural decisions worth explaining

**Features are computed once over the full price history, then sliced.** The
original code built training features from one download and test features from a
second. Every rolling window then warmed up again inside the test slice — and
with a 252-day trailing-high feature on a one-year test window, nothing usable
survived. Compute on the unbroken series, slice afterwards, never the reverse.

**Every feature is scale-free.** The original set used raw rupee price levels
(`ma_10`, `upper_band`, `volatility_20`). Gradient-boosted trees split on
thresholds and cannot extrapolate past the range they trained on, so a stock
trading above its entire training range lands in a boundary leaf and stays
there. Ratios, z-scores and correlations mean the same thing at any price.

**Returns are simple, not logarithmic.** The original code fed log returns into
a weighted sum and then compounded the result as if it were simple. A weighted
sum of log returns is not the log return of the weighted portfolio — log is not
linear — so that expression described no portfolio at all. Compounding it as a
simple return was a second, opposite-signed error.

---

## 5. The results

Walk-forward, 18 quarterly rebalances, 2022-04-01 to 2026-09-05, 47 tickers,
20bps on turnover.

**Pooled rank IC: n = 52, mean 0.0096, std 0.1868, t = 0.37, hit rate 0.577.**

Zero. Per-quarter IC ranged −0.23 to +0.13, positive in 7 of 18.

| Stream | CAGR | AnnVol | Sharpe | Max DD | Turnover/qtr | Total cost |
|---|---:|---:|---:|---:|---:|---:|
| Black-Litterman | 11.75% | 14.69% | 0.830 | −22.13% | 0.759 | 2.73% |
| Equal-Weight (Selected) | 12.98% | 15.55% | 0.863 | −24.12% | 0.711 | 2.56% |
| Equal-Weight (Universe) | **13.00%** | 12.96% | **1.008** | **−17.40%** | 0.135 | 0.49% |
| NIFTY 50 | 7.44% | 13.18% | 0.611 | −15.77% | 0.056 | 0.20% |

| Decomposition step | Sharpe diff | 95% CI | p |
|---|---:|---:|---:|
| Universe EW vs NIFTY (equal- vs cap-weighting) | **+0.399** | [+0.102, +0.699] | **0.007** |
| Selected EW vs Universe EW (selection) | −0.145 | [−0.603, +0.314] | 0.536 |
| BL vs Selected EW (weighting) | −0.033 | [−0.296, +0.231] | 0.816 |
| BL vs NIFTY (everything) | +0.221 | [−0.417, +0.869] | 0.495 |

The strategy loses to a passive equal-weight portfolio of the same universe on
every measure, and pays 2.73% in costs against 0.49% to do it.

---

## 6. The questions you will be asked

### "So did it work?"

**No.** The model has no measurable ability to rank stocks. Pooled IC of 0.0096
with a t-statistic of 0.37 is indistinguishable from zero, across 52
non-overlapping observations. The portfolio underperformed a passive
equal-weight portfolio of the same universe on return, volatility, drawdown and
Sharpe, while paying five times the transaction costs.

The one effect that survives significance testing is equal-weighting versus
cap-weighting the index — the equal-weight premium, at p = 0.007. That is a
well-known passive effect and has nothing to do with the model.

Do not soften this. The interesting part of the answer is that the project can
*establish* it, with an effect size, a confidence interval and a p-value,
instead of shrugging.

### "Then why is a failing strategy still in the repository?"

Four reasons, and they are the substance of the project.

1. **It is the expected result.** Cross-sectional equity returns are close to
   unpredictable from technical indicators alone. A pipeline of this kind
   reporting a large edge is far more likely to have a bug than an insight — as
   this one did. Reproducing the field's actual finding correctly is a
   successful outcome, not a failed one.

2. **The harness is the deliverable, and it is reusable.** Purged splits,
   non-overlapping IC sampling, drift-aware turnover costs, bootstrap
   significance testing, walk-forward evaluation, and a regression test that
   fails if look-ahead bias is reintroduced. Point that at a signal that *does*
   work and it will tell you so, with error bars. That is the transferable part.

3. **Deleting it would destroy the evidence.** The value is the contrast: a
   pipeline that reported 0.212–0.516 and a pipeline that reports 0.0096, with a
   documented explanation and a committed test for why the first was wrong.
   A repository containing only the second is far less informative.

4. **Reporting a negative result honestly is the job.** The alternative was
   available: keep fundamentals in selection, drop the universe benchmark, quote
   the single-window window that happened to look good, and report "beat the
   NIFTY by 11 points." Every one of those choices is defensible in isolation
   and the combination would have been misleading. Choosing not to is the point.

### "How do you know it is not still leaking?"

Four independent lines, none of which relies on trusting the code by inspection.

1. **The random-walk regression test.** `tests/test_no_leakage.py` runs the
   current pipeline against seeded random walks where the truth is exactly zero,
   and asserts it scores below a quarter of the original leaky construction —
   which is kept as a live positive control and re-scored on every run. It also
   asserts the control still leaks under the current regularised settings, so
   the diagnosis cannot quietly become wrong. Reintroduce the leak and the test
   fails.

2. **The walk-forward split audit.** `tests/test_walkforward.py` materialises
   the actual TRAIN/VALID/HOLD date sets through the same slicing function the
   pipeline uses, then asserts the purge gap is exactly 21 trading days at both
   inner boundaries, that the forward-looking *label* of the last row of each
   window closes before the next window opens, that no hold-window date appears
   anywhere in that rebalance's training or validation data, and that the hold
   windows tile the period exactly once with nothing double-counted or skipped.

3. **A direct audit on real data.** 2,538 checks across 18 rebalances × 47
   tickers on the actual NSE calendar, holidays included: zero violations, purge
   gaps exactly 21 trading days at both boundaries.

4. **The result itself is consistent with no leakage.** Leakage produces
   implausibly good numbers. This produces an IC of 0.0096 — exactly what the
   field reports. The prior version failed that smell test badly; this one
   passes it. That is weak evidence alone, but it corroborates the other three.

The honest limit: these rule out the leaks that were found and the classes of
leak they were designed to catch. **No test proves the absence of all leakage.**
The remaining known one is documented rather than hidden — see the next answer.

### "Is anything still leaking?"

**Yes, one thing, and it is documented in the README limitations.**

`download_fundamentals` returns *today's* trailing P/E, debt-to-equity and
market cap — a single current snapshot. Those three carried 60% of the selection
score. Using a 2026 balance sheet to decide what to buy in 2022 is look-ahead
bias, and walk-forward makes it worse, since the same snapshot would drive all
18 rebalances.

Fixing it properly requires point-in-time fundamentals — what was actually
reported and known as of each date — which needs a paid data source. So the
workaround is `USE_FUNDAMENTALS_IN_SELECTION = False`: selection ranks on the ML
score alone, and the flag warns loudly if turned back on.

Calling that a fix would be dishonest. It is a known limitation with a
documented mitigation, and saying so is better than a silent contaminated
number.

### "Why cross-sectional rank IC instead of prediction accuracy?"

Because it matches the decision the strategy actually makes. The strategy does
not trade one stock against its own history; on a given day it ranks 47 stocks
and buys the top ten. Per-ticker time-series correlation can look strong while
being useless for that — a model that correctly says "everything rises in bull
markets" scores well and tells you nothing about *which* stock to buy. IC is
also the field standard, which makes the number comparable to published work.

### "Why did walk-forward matter if the answer did not change?"

Because "we cannot tell" and "there is no effect" are different claims, and only
the second is a finding.

The single-window backtest gave one allocation and 11 IC observations. Every
confidence interval straddled zero — not evidence of no effect, just an
experiment with no power. The sharpest illustration: an earlier single-window
run reported a CAGR of −1.88%; changing the selection rule and moving the window
one day gave +7.28%. Nine points, on a one-day shift.

Walk-forward produced 18 allocations and 52 IC observations. The verdict held,
and now it rests on a sample large enough to state.

### "What would you do differently?"

- Source point-in-time fundamentals, which closes the last known leak.
- Build the universe from historical index membership. The current 47 tickers
  are today's constituents, so the backtest has survivorship bias — and since
  equal weighting overweights smaller names, that bias probably inflates the one
  significant result. **+0.399 should be read as a ceiling, not an estimate.**
- Extend the walk-forward past a single 2022–2026 regime.
- Model slippage and market impact, not just basis points on turnover.
- Test signals with a real prior of working — earnings revisions, fundamental
  momentum — rather than technical indicators alone.

---

## 7. Design decisions

**Why one model per ticker?** Each stock gets its own return dynamics, and it
keeps the per-ticker diagnostics interpretable. The cost is that no cross-stock
structure is shared and 47 models are fit per rebalance. A single pooled model
with ticker features would be the natural comparison and was not tried.

**Why gradient-boosted trees?** They handle non-linearities and interactions
without feature scaling and are hard to beat on small tabular data. The
constraint they impose — no extrapolation beyond training splits — is exactly
why every feature had to be made scale-free.

**Why heavily regularised settings?** 300 trees at depth 3, `min_child_weight`
20, `reg_lambda` 5.0. Daily cross-sectional returns carry very little signal; a
depth-6, 500-tree model memorises noise. Note the random-walk control: the leaky
features scored 0.404 even under these settings, so regularisation was never
what fixed the bug.

**Why quarterly rebalancing?** It matches the 21-day prediction horizon at a
plausible trading frequency. At 76% turnover per quarter the strategy is already
expensive; monthly would be worse. This was fixed in advance and not tuned —
choosing a rebalance frequency by trying several and keeping the best is
exactly how a held-out window stops being held out.

**Why was nothing tuned after seeing the results?** Because the test window had
been observed. Any parameter changed in response to it — model settings, number
of stocks, view confidence, window lengths — makes the reported number
meaningless. The discipline only counts when the result is disappointing, which
is when it is hardest to keep.
