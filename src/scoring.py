"""
Rank the universe and select the top-N stocks to feed into Black-Litterman.

By default the ranking uses the ML expected return alone. The fundamental
blend (P/E, D/E, market cap) is available behind
config.USE_FUNDAMENTALS_IN_SELECTION but is OFF, because `download_fundamentals`
returns a single CURRENT snapshot: using today's trailing P/E to decide what to
buy in 2022 is look-ahead bias, and it carries 60% of SCORE_WEIGHTS. See the
comment on that flag in config.py.

Scoring weights come from config.SCORE_WEIGHTS rather than interactive
prompts, so the pipeline runs unattended. They are consulted only when the
fundamentals path is enabled.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import SCORE_WEIGHTS, TOP_N_STOCKS, USE_FUNDAMENTALS_IN_SELECTION


def score_and_select(expected_returns: pd.DataFrame, fundamentals: pd.DataFrame | None = None,
                     top_n: int = TOP_N_STOCKS, score_weights: dict | None = None,
                     use_fundamentals: bool | None = None):
    """
    Parameters
    ----------
    expected_returns : DataFrame (date x ticker) of ML-predicted returns
    fundamentals : DataFrame (ticker x [PE, DE, MktCap]). Required only when
        fundamentals are in use; ignored otherwise.
    use_fundamentals : override for config.USE_FUNDAMENTALS_IN_SELECTION.

    Returns
    -------
    top_stocks : Series of tickers -> FinalScore, sorted descending, length top_n
    scores : full DataFrame of intermediate scores (for plotting/inspection)
    """
    if use_fundamentals is None:
        use_fundamentals = USE_FUNDAMENTALS_IN_SELECTION

    scores = expected_returns.mean().to_frame(name="ExpectedReturn").dropna()
    scaler = MinMaxScaler()

    if use_fundamentals:
        score_weights = score_weights or SCORE_WEIGHTS
        weights_sum = sum(score_weights.values())
        if not (0.999 <= weights_sum <= 1.001):
            raise ValueError(f"SCORE_WEIGHTS must sum to 1.0, got {weights_sum}")
        if fundamentals is None:
            raise ValueError(
                "use_fundamentals=True requires a fundamentals frame."
            )

        scores = scores.join(fundamentals, how="inner").dropna()
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
        shortfall = (
            f"Only {len(scores)} tickers have ML predictions and complete fundamentals; "
            f"need at least top_n={top_n}."
        )
    else:
        # ML score only. Nothing here knows anything the model did not know at
        # the time the prediction was made.
        scores["ReturnScore"] = scaler.fit_transform(scores[["ExpectedReturn"]])
        scores["FinalScore"] = scores["ReturnScore"]
        shortfall = (
            f"Only {len(scores)} tickers have ML predictions; "
            f"need at least top_n={top_n}."
        )

    if len(scores) < top_n:
        raise ValueError(shortfall)

    top_stocks = scores.sort_values("FinalScore", ascending=False).head(top_n)["FinalScore"]
    return top_stocks, scores
