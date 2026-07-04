"""
Data access layer: price history and fundamentals via yfinance.

All network calls live here so the rest of the pipeline can be tested
with synthetic data (see tests/) without hitting the network.
"""

import logging
import hashlib
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)
PRICE_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

try:
    yf.set_tz_cache_location(str(Path("data_cache") / "yfinance"))
except AttributeError:  # pragma: no cover - older yfinance versions
    pass


def _cache_key(prefix, tickers, start=None, end=None):
    tickers_part = "-".join(sorted(tickers)).replace("/", "_")
    if len(tickers_part) > 120:
        digest = hashlib.sha256(tickers_part.encode("utf-8")).hexdigest()[:16]
        tickers_part = f"{len(tickers)}tickers_{digest}"
    date_part = f"_{start}_{end}" if start and end else ""
    return f"{prefix}{date_part}_{tickers_part}.pkl"


def _normalize_price_columns(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data.empty:
        return data

    if not isinstance(data.columns, pd.MultiIndex):
        if len(tickers) != 1:
            raise ValueError("Expected MultiIndex columns for multi-ticker price data.")
        data = pd.concat({tickers[0]: data}, axis=1)

    level0 = set(map(str, data.columns.get_level_values(0)))
    level1 = set(map(str, data.columns.get_level_values(1)))
    if any(field in level0 for field in PRICE_FIELDS):
        normalized = data.copy()
    elif any(field in level1 for field in PRICE_FIELDS):
        normalized = data.swaplevel(axis=1)
    else:
        raise ValueError(f"Could not identify OHLCV fields in downloaded columns: {data.columns}")

    normalized = normalized.sort_index(axis=1)
    if "Close" not in normalized.columns.get_level_values(0):
        raise ValueError("Downloaded price data does not include Close prices.")
    return normalized


def _download_with_retries(tickers, start, end, retries=3, backoff=3):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                interval="1d",
                group_by="column",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if not data.empty:
                return data
            last_error = ValueError("empty response")
        except Exception as exc:  # pragma: no cover - depends on network failures
            last_error = exc
        sleep_seconds = backoff * attempt
        LOGGER.warning(
            "Price download attempt %s/%s failed for %s tickers: %s. Retrying in %ss.",
            attempt,
            retries,
            len(tickers),
            last_error,
            sleep_seconds,
        )
        time.sleep(sleep_seconds)
    raise ValueError(f"No price data returned after {retries} attempts: {last_error}")


def download_prices(tickers, start, end, cache_dir="data_cache", use_cache=True,
                    retries=3, backoff=3):
    """
    Download OHLCV data for a list of tickers.

    Returns a wide DataFrame with a MultiIndex column (field, ticker),
    e.g. data['Close']['INFY.NS'].
    """
    tickers = list(tickers)
    cache_path = Path(cache_dir) / _cache_key("prices", tickers, start, end)
    if use_cache and cache_path.exists():
        LOGGER.info("Loading cached prices: %s", cache_path)
        return pd.read_pickle(cache_path)

    data = _download_with_retries(tickers, start, end, retries=retries, backoff=backoff)
    data = _normalize_price_columns(data, tickers)
    if data.empty:
        raise ValueError("No price data returned - check tickers/date range/network.")

    close = data["Close"].dropna(axis=1, how="all")
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        LOGGER.warning("Skipping tickers with no close-price data: %s", ", ".join(missing))
    data = data.loc[:, data.columns.get_level_values(1).isin(close.columns)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(cache_path)
    LOGGER.info("Cached prices: %s", cache_path)
    return data


def download_fundamentals(tickers, cache_dir="data_cache", use_cache=True,
                          retries=2, backoff=2):
    """
    Pull trailing P/E, debt-to-equity, and market cap for each ticker.
    Tickers with missing data are dropped by the caller (see scoring.py).
    """
    tickers = list(tickers)
    cache_path = Path(cache_dir) / _cache_key("fundamentals", tickers)
    if use_cache and cache_path.exists():
        LOGGER.info("Loading cached fundamentals: %s", cache_path)
        return pd.read_pickle(cache_path)

    rows = {}
    for ticker in tickers:
        info = {}
        for attempt in range(1, retries + 1):
            try:
                info = yf.Ticker(ticker).info or {}
                break
            except Exception as exc:  # pragma: no cover - depends on network failures
                LOGGER.warning(
                    "Fundamental download attempt %s/%s failed for %s: %s",
                    attempt,
                    retries,
                    ticker,
                    exc,
                )
                time.sleep(backoff * attempt)
        rows[ticker] = {
            "PE": info.get("trailingPE"),
            "DE": info.get("debtToEquity"),
            "MktCap": info.get("marketCap"),
        }
    fundamentals = pd.DataFrame(rows).T
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fundamentals.to_pickle(cache_path)
    LOGGER.info("Cached fundamentals: %s", cache_path)
    return fundamentals
