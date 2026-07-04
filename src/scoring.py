"""
Combine ML expected returns with fundamentals (P/E, D/E, market cap) into
a single score, and select the top-N stocks to feed into Black-Litterman.

Unlike the original notebook, weights come from config.SCORE_WEIGHTS
instead of blocking `input()` calls, so the pipeline can run unattended.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import SCORE_WEIGHTS, TOP_N_STOCKS


def score_and_select(expected_returns: pd.DataFrame, fundamentals: pd.DataFrame,
                     top_n: int = TOP_N_STOCKS, score_weights: dict | None = None):
    """
    Parameters
    ----------
    expected_returns : DataFrame (date x ticker) of ML-predicted returns
    fundamentals : DataFrame (ticker x [PE, DE, MktCap])

    Returns
    -------
    top_stocks : Series of tickers -> FinalScore, sorted descending, length top_n
    scores : full DataFrame of intermediate scores (for plotting/inspection)
    """
    score_weights = score_weights or SCORE_WEIGHTS
    weights_sum = sum(score_weights.values())
    if not (0.999 <= weights_sum <= 1.001):
        raise ValueError(f"SCORE_WEIGHTS must sum to 1.0, got {weights_sum}")

    scores = expected_returns.mean().to_frame(name="ExpectedReturn")
    scores = scores.join(fundamentals, how="inner").dropna()

    scaler = MinMaxScaler()
    scores["ReturnScore"] = scaler.fit_transform(scores[["ExpectedReturn"]])
    # Lower P/E and lower D/E are generally more attractive -> invert before scaling
    scores["PEScore"] = scaler.fit_transform(-scores[["PE"]])
    scores["DEScore"] = scaler.fit_transform(-scores[["DE"]])
    scores["MktCapScore"] = scaler.fit_transform(scores[["MktCap"]])

    scores["FinalScore"] = (
        score_weights["return_score"] * scores["ReturnScore"]
        + score_weights["pe_score"] * scores["PEScore"]
        + score_weights["de_score"] * scores["DEScore"]
        + score_weights["mktcap_score"] * scores["MktCapScore"]
    )

    if len(scores) < top_n:
        raise ValueError(
            f"Only {len(scores)} tickers have ML predictions and complete fundamentals; "
            f"need at least top_n={top_n}."
        )

    top_stocks = scores.sort_values("FinalScore", ascending=False).head(top_n)["FinalScore"]
    return top_stocks, scores
