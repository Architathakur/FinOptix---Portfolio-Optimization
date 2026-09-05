# FinOptix: Concepts, In Plain Language

Background reading for the ideas the project turns on. It assumes no finance
and no machine learning. Read this first, then
`PROJECT_EXPLANATION_AND_INTERVIEW_GUIDE.md` for the project narrative and the
questions you should expect to be asked.

Every number quoted here comes from a committed artifact in `outputs/` or from
a test you can run. None of it is illustrative.

---

## 1. What the project actually is

FinOptix predicts which large Indian stocks will do well over the next month,
picks ten of them, decides how much to put in each, and measures the result
honestly.

The last word is the one that matters. **The deliverable is the measurement
harness, not a profitable strategy.** The strategy does not work — the final
numbers show no predictive skill at all — and the project's value is that it
can prove that, cleanly, instead of reporting a large number produced by a bug.

An earlier version of this code reported prediction correlations of 0.212 to
0.516 and a portfolio that comfortably beat its benchmark. Both were fake. The
work described here is how that was found, why it happened, and what replaced
it.

---

## 2. Look-ahead bias

**The idea.** Look-ahead bias is using information in a backtest that you could
not have had at the time. It is the single most common way a backtest lies.

**An analogy.** Imagine testing a system for predicting football results. You
run it over last season, and it is right 85% of the time. Then you notice that
one of the inputs you feed it is the final score. The system is not predicting
anything; it is reading the answer off the page. It will be right 85% of the
time in your test and useless on Saturday.

**What happened here.** The model was asked to predict this quantity:

```python
feat["returns"] = df["Close"].pct_change()   # the return ENDING today
```

That is today's price move — from yesterday's close to today's close. Among the
inputs it was given were these:

```python
feat["ma_10"] = df["Close"].rolling(10).mean()     # average of the last 10 closes
feat["volatility_20"] = df["Close"].rolling(20).std()
feat["upper_band"] = ma_20 + 2 * std_20
```

Every one of those rolling windows **includes today's close**. So the model was
handed today's closing price and asked what today's price move was. Today's
close and today's move are two views of the same number. The model was not
forecasting; it was doing arithmetic.

**Why it is easy to miss.** Nothing looks wrong. There is no `shift(-1)`
anywhere, no obvious peek at the future, no error message. `rolling(10).mean()`
is a completely normal thing to write. The bug lives in the *relationship*
between the target and the features, not in either one alone, so reading
either in isolation tells you nothing.

---

## 3. Why 0.21–0.52 was impossible on its face

This is the part worth internalising, because it is the skill that catches bugs
like this without reading any code: **know what a plausible number looks like
in your field.**

Predicting daily stock returns from price history is one of the most studied
problems in finance, attacked for decades by people with better data and more
resources than a personal project. Published results for daily-return models on
technical indicators land around a correlation of **0.02 to 0.05**. Even that is
hard to convert into money after costs.

The pipeline reported **0.212 to 0.516**. That is not "a good result." That is
four to twenty times the best published work in the field, produced by a
weekend project using free data. When a number is that far outside the range
the field lives in, the first hypothesis is not "I have discovered something."
It is "I have a bug." Investigating that instinct is what found the leak.

The general rule: **be more suspicious of good results than bad ones.** A bad
result has many boring causes. A result far above what the field achieves has
essentially one.

---

## 4. The random-walk control

**The problem.** How do you prove a feature pipeline leaks? You cannot just
look at real data, because on real data you never know what the true
predictability is. If the model scores 0.3, is that leakage or genuine signal?
There is no ground truth to compare against.

**The solution.** Build data where you know the answer in advance.

A **random walk** is a price series where every day's move is drawn
independently at random. Today's move tells you nothing about tomorrow's,
because that is how it was constructed. On such a series, **the true
predictability of the future is exactly zero** — not small, not hard to find.
Zero, by construction.

So: feed the pipeline a random walk. Anything it scores above zero cannot be
signal, because there is no signal. It can only be the model reading the
future.

**The result** (`tests/test_no_leakage.py`, mean of five seeds, 8000 days each):

| Setup | score on a random walk |
|---|---:|
| Original features + same-day target + original model settings | 0.418 |
| Original features + same-day target + **regularised** model settings | 0.404 |
| Current features + 21-day forward target + purged split | 0.044 |

**Why this is convincing, and not just suggestive:**

1. **The null is exact, not estimated.** On a random walk zero is the truth, not
   an approximation. There is no "well, maybe there was a little signal."
2. **The middle row rules out the obvious alternative explanation.** The first
   objection to "the old model scored 0.418" is "your old model was overfitting
   — it was too complex." So the same leaky features were re-run with the
   current, heavily regularised settings. Score: 0.404. Barely moved.
   Overfitting was not the cause; the feature/target construction was. Without
   that control row the diagnosis would be a guess.
3. **The control is live, not a remembered number.** The original buggy
   pipeline is committed as a test fixture (`old_calculate_features`) and is
   re-scored on every test run. If it ever stopped leaking, the test would say
   so instead of silently comparing against a stale figure.
4. **The test asserts a ratio, not a threshold.** It requires the current
   feature set to score below a quarter of the leaky one. A fixed threshold like
   "below 0.15" only holds at one data length — the current set's score is a
   noise floor, and noise floors rise as samples shrink. At 1650 days one seed
   reached 0.196, which is pure noise but would trip a fixed bar. The leaky
   control sits near 0.4 at *every* length, so dividing by it gives a number
   that stays stable. A test that fails when you shorten its input gets deleted
   rather than debugged.

---

## 5. The three-way split, and purging

### Why two splits are not enough

The classic setup is TRAIN and TEST: fit on one period, evaluate on another.
That is not enough as soon as you make *any* decision using the data — which
stocks to buy, how confident to be in the model, which risk model to use.

In the original pipeline, TEST was used to evaluate the model, **and** to pick
the stocks, **and** to estimate the covariance matrix. So the stocks were chosen
using knowledge of how they performed over the very window used to score the
portfolio. The reported outperformance was circular: it compared a benchmark
against a portfolio built with the answer already in hand.

### The fix

Three windows, each with exactly one job:

| Window | Job |
|---|---|
| **TRAIN** | Fit the models. Nothing else. |
| **VALID** | Make every decision: which stocks, what views, what covariance. |
| **TEST** | Score the result. Opened once, after the weights are frozen. |

The rule: **the moment any parameter is chosen by looking at TEST, TEST stops
being held out.** You get one look. If you tune anything after that look and
re-run, you are back to reporting fiction.

### Purging — the subtle part

Say TRAIN ends on 31 March and VALID begins 1 April. Clean split? No.

The model predicts the return over the **next 21 trading days**. So the label
attached to 31 March is "what happened between 1 and 30 April" — which is
VALID data. Training on the last day of TRAIN means training on the first month
of VALID.

The fix is a **purge**: drop the last 21 trading days of TRAIN, so the final
training label closes before VALID opens.

```
TRAIN ────────────────┐         ┌──────────── VALID
                      └─ purge ─┘
                       21 days, discarded
```

The walk-forward backtest purges at **both** inner boundaries — TRAIN/VALID
*and* VALID/hold — because the same argument applies at each. It counts the gap
in **trading days on the real exchange calendar**, not calendar days, so it does
not silently shrink across a holiday week.

**What breaks without it:** the model is trained on data from the window used to
evaluate it, so the evaluation is optimistic by an amount nobody can quantify
after the fact. It is a small leak compared to the original bug, which is
exactly what makes it dangerous — it does not produce an absurd number that
prompts investigation, just a slightly flattering one.

---

## 6. Cross-sectional rank IC

### The metric that was being used, and why it was wrong

The original pipeline measured, for each stock separately, the correlation
between predicted and actual returns over time. Call it per-ticker time-series
correlation.

**That answers a question the strategy never asks.** The strategy does not trade
one stock against its own history. On a given day it looks at all 47 stocks,
ranks them, and buys the top ten. What it needs to know is: **on a given day,
does the ranking come out in the right order?**

A model can score well on per-ticker correlation and still be useless for this.
Suppose it correctly predicts that every stock rises in bull markets and falls
in bear markets. Per-ticker correlation: excellent. Ability to tell you *which*
stock to buy today: none, because it says the same thing about all of them.

### What rank IC measures

The **information coefficient** is computed *per date, across stocks*:

1. On a given day, rank all 47 stocks by predicted return.
2. Rank the same 47 by what actually happened over the next 21 days.
3. Correlate the two rankings (Spearman correlation, which uses order, not size).
4. Repeat on many days and average.

An IC of 0 means the ranking is no better than shuffling. Professional equity
signals live around 0.02–0.05. It is the standard metric in the field, which
makes the number directly comparable to published work.

### The sampling detail that matters

The IC is measured every 21 trading days, not every day. This is not an
optimisation — it is required for the statistics to mean anything.

A 21-day forward return measured today and one measured tomorrow **share 20 of
their 21 days**. They are nearly the same number. Treating them as two
independent observations inflates your sample by roughly 21x, which shrinks the
standard error by about √21 ≈ 4.6 and inflates the t-statistic by the same
factor. A t-statistic built from overlapping windows is not a t-statistic.

Sampling every 21 days makes each observation cover a fresh, non-overlapping
stretch of time.

---

## 7. Walk-forward, and why it mattered

### The problem with one split

The single-window backtest produced **one** allocation decision and **11**
non-overlapping IC observations. That is far too little to conclude anything.
Every bootstrap confidence interval it produced straddled zero — not because
the strategy was proven neutral, but because the experiment was never powered
to detect an effect in either direction.

There is a sharper demonstration. An earlier single-window run reported a
Black-Litterman CAGR of **−1.88%**. After removing fundamentals from the
selection rule and shifting the window by a *single day*, the same backtest
reported **+7.28%**. Nine percentage points, on a one-day shift. Nothing about
the strategy changed. The sample was simply too small to hold still.

### The fix

Walk-forward re-runs the entire decision process quarterly:

```
At each rebalance date t:
  TRAIN   fixed start -> (t - 1 year - purge)     expanding window
  VALID   (t - 1 year) -> (t - purge)             makes every decision
  HOLD    t -> next rebalance                     strictly out of sample
```

Eighteen rebalances from 2022-04-01 to 2026-09-05. Each one refits the models,
re-picks the stocks, rebuilds the weights, and holds them for a quarter. Pooling
the hold windows takes the IC sample from **11 observations to 52**.

### The result

**The verdict did not change.** Pooled IC: mean 0.0096, t-statistic 0.37. Zero.

This is the outcome worth understanding. Walk-forward was not run in the hope
of a better number, and it did not produce one. What it produced was a
**conclusion that can be defended**. Before: "we cannot tell." After: "the
model does not rank stocks better than chance, and here are 52 observations
saying so." Individual quarters ranged from −0.23 to +0.13, positive in 7 of 18.

---

## 8. The benchmark decomposition — the strongest finding

### The trap

The obvious benchmark for "I picked ten stocks and weighted them cleverly" is
"I picked the same ten stocks and weighted them equally." That is a real
benchmark, but it only tests the *weighting*. It inherits the stock picks, so it
cannot tell you whether picking them was worth anything.

Against that benchmark and the NIFTY 50 index, the strategy looked respectable:
it beat the index by +0.22 Sharpe.

### The fix

Add a third benchmark: **equal weight across the entire tradeable universe** —
all 47 stocks, no selection at all. Now the result decomposes into a chain, each
step isolating one layer:

| Step | What it isolates | Sharpe diff | 95% CI | p |
|---|---|---:|---:|---:|
| Universe EW vs NIFTY 50 | equal- vs cap-weighting | **+0.399** | [+0.102, +0.699] | **0.007** |
| Selected EW vs Universe EW | the stock selection | −0.145 | [−0.603, +0.314] | 0.536 |
| BL vs Selected EW | the weighting model | −0.033 | [−0.296, +0.231] | 0.816 |
| BL vs NIFTY 50 | everything together | +0.221 | [−0.417, +0.869] | 0.495 |

The chain is additive and lands exactly on the observed Sharpe ratios:
`0.611 + 0.399 − 0.145 − 0.033 = 0.830`.

### What it says

**The only effect that is statistically distinguishable from noise is
equal-weighting versus cap-weighting the index** (p = 0.007). That is the
well-documented equal-weight premium. It is not a machine-learning result — it
is available to anyone willing to hold 47 stocks in equal proportion and
rebalance quarterly.

Both model layers **subtract**. Selection costs −0.145 Sharpe, weighting a
further −0.033. Neither is statistically distinguishable from zero, but neither
shows any sign of adding value.

So the headline "+0.221 versus the index" is entirely attributable to the
equal-weighting effect — more than entirely, since the model layers give some of
it back. **Without the universe benchmark, that +0.221 reads as the strategy
working.** One extra benchmark converted a flattering summary into an
attribution.

---

## 9. Costs and turnover

A backtest with no transaction costs is a backtest of a strategy nobody can
trade.

**Turnover** measures how much of the portfolio is bought and sold at each
rebalance. If you hold A and B at 50/50 and switch entirely to C and D, you have
sold 100% and bought 100%: turnover of 2.0.

**The detail that is easy to get wrong: weights drift.** Between rebalances you
are not trading. If one holding doubles and another halves, your 50/50 book
becomes roughly 67/33 on its own. When the next rebalance arrives, the position
you must trade *away from* is the drifted one, not the original target.
Computing turnover against the original targets invents a trade that never
happened and overstates costs.

The numbers:

| | Turnover per quarter | Total cost over the period |
|---|---:|---:|
| Black-Litterman | 0.759 | 2.73% |
| Equal-Weight (Universe) | 0.135 | 0.49% |

The model portfolio churns 76% of its holdings every quarter and pays 2.73% for
the privilege — against 0.49% for the passive universe portfolio that beat it on
return, volatility, drawdown and Sharpe. The activity is not just failing to add
value; it is actively expensive.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Look-ahead bias** | Using information in a backtest that was not available at the time. |
| **Purging** | Dropping rows at a split boundary whose forward-looking labels would reach across it. |
| **Random walk** | A series where each move is independent; true predictability is exactly zero. |
| **Rank IC** | Cross-sectional correlation between predicted and realised rankings, per date. |
| **Spearman correlation** | Correlation of ranks rather than values; cares about order, not size. |
| **t-statistic** | How many standard errors a result sits from zero. Below ~2, indistinguishable from chance. |
| **Bootstrap** | Re-sampling the data thousands of times to see how much a result moves by luck alone. |
| **Sharpe ratio** | Return per unit of volatility. The standard risk-adjusted comparison. |
| **Walk-forward** | Repeatedly refitting and re-deciding through time, instead of one fixed split. |
| **Turnover** | Fraction of the portfolio traded at a rebalance. |
| **Drift** | How weights change on their own between rebalances as holdings move. |
| **Black-Litterman** | A method for blending a market-implied baseline with your own views. |
| **Survivorship bias** | Studying only the names that survived, which flatters historical results. |
