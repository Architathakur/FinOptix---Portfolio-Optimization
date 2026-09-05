"""
Central configuration for the FinOptix pipeline.
Change values here rather than editing pipeline code.
"""

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
TICKERS = [
    "BHARTIARTL.NS", "LTIM.NS", "HDFCLIFE.NS", "NTPC.NS", "MARUTI.NS",
    "NESTLEIND.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "TATASTEEL.NS",
    "ONGC.NS", "BAJAJ-AUTO.NS", "LT.NS", "ITC.NS", "TCS.NS", "BRITANNIA.NS",
    "ADANIENT.NS", "CIPLA.NS", "WIPRO.NS", "INDUSINDBK.NS",
    "ULTRACEMCO.NS", "TATACONSUM.NS", "BAJAJFINSV.NS", "RELIANCE.NS",
    "HEROMOTOCO.NS", "COALINDIA.NS", "TITAN.NS", "HINDALCO.NS",
    "APOLLOHOSP.NS", "TECHM.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "BPCL.NS", "SBILIFE.NS", "GRASIM.NS", "JSWSTEEL.NS",
    "ASIANPAINT.NS", "POWERGRID.NS", "ADANIPORTS.NS", "M&M.NS",
    "TATAMOTORS.NS", "SUNPHARMA.NS", "AXISBANK.NS", "HCLTECH.NS",
    "HINDUNILVR.NS", "INFY.NS", "SBIN.NS", "ICICIBANK.NS", "HDFCBANK.NS",
]

# ---------------------------------------------------------------------------
# Date ranges: a strict three-way split.
#
#   TRAIN  fits the per-ticker models. Nothing else.
#   VALID  is the only window used to make decisions: it produces the
#          predictions that drive stock selection, the Black-Litterman views
#          (Q), and the covariance matrix. Every knob that gets turned is
#          turned here.
#   TEST   is backtest-only. Nothing reads it until the portfolio weights are
#          frozen. The moment a parameter is chosen by looking at TEST, TEST
#          stops being held out and the reported performance becomes fiction.
#
# The windows are half-open, [start, end), so a boundary date belongs to
# exactly one window. TRAIN is additionally purged (see PURGE_DAYS) because a
# forward-looking label near the end of TRAIN would otherwise peek into VALID.
# ---------------------------------------------------------------------------
# AS_OF_DATE is pinned rather than read from the clock. Every window below is
# derived from it, so a fresh clone produces exactly the TRAIN/VALID/TEST split
# that the committed artifacts in outputs/ -- and the numbers quoted in
# README.md -- were generated from. Leaving it as date.today() meant the split
# silently moved every day, so the published results could never be reproduced.
#
# Change it (or set it back to date.today()) to re-run against fresher market
# data. The windows shift with it, so the committed outputs/ CSVs and the
# README figures will no longer match and should be regenerated.
AS_OF_DATE = date(2026, 9, 5)

TRAIN_START = "2020-01-01"
TRAIN_END = (AS_OF_DATE - timedelta(days=730)).isoformat()
VALID_START = TRAIN_END
VALID_END = (AS_OF_DATE - timedelta(days=365)).isoformat()
TEST_START = VALID_END
TEST_END = AS_OF_DATE.isoformat()

# ---------------------------------------------------------------------------
# Prediction horizon
#
# The model predicts the return over the NEXT PRED_HORIZON trading days. The
# label at time t therefore spans (t, t + PRED_HORIZON], so the last
# PURGE_DAYS rows of any training slice carry labels that overlap the window
# that follows it. Those rows are dropped.
# ---------------------------------------------------------------------------
PRED_HORIZON = 21
PURGE_DAYS = PRED_HORIZON

# ---------------------------------------------------------------------------
# Feature engineering
#
# Every feature is scale-free (a ratio, a z-score or a correlation) and uses
# only information available at or before time t. Raw rupee price levels are
# deliberately absent: gradient-boosted trees cannot extrapolate past their
# training splits, so a stock trading outside its training price range lands
# in a boundary leaf and the model quietly stops working.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_21", "ret_63",
    "close_over_ma10", "close_over_ma50", "ma10_over_ma50",
    "vol_21", "vol_ratio", "band_pos", "rsi_14", "dist_52w_high",
    "volume_z", "corr_close_vol_20",
]

# ---------------------------------------------------------------------------
# XGBoost
#
# Heavily regularized on purpose. Daily cross-sectional equity returns carry
# very little signal; a depth-6, 500-tree model on this feature set memorizes
# noise and reports a training fit that does not survive contact with a
# held-out window.
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.02,
    min_child_weight=20,
    reg_lambda=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# ---------------------------------------------------------------------------
# Walk-forward backtesting (src/walkforward.py)
#
# A single train/valid/test split gives one allocation decision and ~11
# non-overlapping IC observations -- far too few to separate a real effect
# from noise in either direction, which is why every bootstrap CI straddled
# zero. Walk-forward re-fits quarterly and pools the results, turning one
# decision into ~18 and ~11 IC observations into ~55.
#
# At each rebalance date t:
#   TRAIN  WALKFORWARD_START -> (t - 1y - PURGE_DAYS)   expanding, fixed start
#   VALID  (t - 1y)          -> (t - PURGE_DAYS)        makes every decision
#   HOLD   t                 -> next rebalance date     strictly out of sample
#
# PURGE_DAYS of trading days are dropped at BOTH inner boundaries: a
# forward-looking label near the end of TRAIN would otherwise reach into
# VALID, and one near the end of VALID would reach past t into HOLD.
#
# WALKFORWARD_WARMUP_DAYS is downloaded BEFORE WALKFORWARD_START purely so the
# rolling features (the 252-day trailing high is the binding one) are already
# warm on the first training day. Without it the first ~252 trading days of
# TRAIN are unusable and the first viable rebalance slips by a year.
# ---------------------------------------------------------------------------
WALKFORWARD_START = "2020-01-01"
WALKFORWARD_VALID_DAYS = 365    # calendar days in the VALID window
WALKFORWARD_FREQ = "QS"         # quarterly rebalancing (pandas quarter-start)
WALKFORWARD_WARMUP_DAYS = 450   # calendar days of pre-history for feature warmup
FEATURE_WARMUP_ROWS = 252       # longest rolling window in FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Fundamentals in stock selection
#
# OFF by default, because it is a look-ahead leak that has no clean fix with
# free data. download_fundamentals returns TODAY's trailing P/E, D/E and market
# cap -- a single current snapshot -- and SCORE_WEIGHTS gives those three 60%
# of the selection score. Using a 2026 balance sheet to decide what to buy in
# 2022 is the future leaking in through a second door, and walk-forward makes
# it far worse: the same snapshot would drive every rebalance from 2022 on.
#
# Fixing it properly needs point-in-time fundamentals (what was actually
# reported and known as of each rebalance date), which requires a paid data
# source. Until then, selection ranks on the ML score alone.
#
# Setting this True restores the fundamental blend and CONTAMINATES selection
# in any backtest -- the resulting performance is not a measurement.
# ---------------------------------------------------------------------------
USE_FUNDAMENTALS_IN_SELECTION = False

# ---------------------------------------------------------------------------
# Fundamental scoring weights (must sum to 1.0)
# Only consulted when USE_FUNDAMENTALS_IN_SELECTION is True.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = dict(
    return_score=0.40,
    pe_score=0.20,
    de_score=0.20,
    mktcap_score=0.20,
)
TOP_N_STOCKS = 10

# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
RISK_AVERSION = 2.5   # delta
TAU = 0.025            # scales prior uncertainty
VIEW_CONFIDENCE = 0.5  # 0 = no confidence (ignore ML views), 1 = full confidence

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
BENCHMARK_TICKER = "^NSEI"   # NIFTY 50

# Stream labels, shared by the single-window and walk-forward reports so the
# two sets of artifacts can be compared row by row.
#
# "Equal-Weight (Selected)" equal-weights the model's own picks, which makes it
# a benchmark for the WEIGHTING layer only -- it cannot tell you whether the
# selection layer added anything, because it inherits the same picks.
# "Equal-Weight (Universe)" holds every tradeable ticker, which is what
# isolates selection.
STREAM_BL = "Black-Litterman"
STREAM_EW_SELECTED = "Equal-Weight (Selected)"
STREAM_EW_UNIVERSE = "Equal-Weight (Universe)"
STREAM_BENCHMARK = "NIFTY 50"
COST_BPS = 20.0              # basis points; charged on turnover at each rebalance
                             # (single-window mode has one rebalance, so it is an entry cost)
TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
