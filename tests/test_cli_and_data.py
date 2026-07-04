import pandas as pd
import pytest

from main import load_tickers, parse_args
from src.data import download_prices
from src.scoring import score_and_select


def _sample_price_frame(tickers):
    dates = pd.bdate_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], tickers]
    )
    data = pd.DataFrame(1.0, index=dates, columns=columns)
    data[("Close", tickers[0])] = [100.0, 101.0, 102.0]
    return data


def test_parse_args_overrides_defaults():
    args = parse_args(["--top-n", "7", "--confidence", "0.8", "--refresh-cache"])
    assert args.top_n == 7
    assert args.confidence == pytest.approx(0.8)
    assert args.refresh_cache is True


def test_load_tickers_accepts_lines_and_commas(tmp_path):
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("INFY.NS, TCS.NS\n# comment\nRELIANCE.NS\n", encoding="utf-8")
    assert load_tickers(tickers_file) == ["INFY.NS", "TCS.NS", "RELIANCE.NS"]


def test_score_and_select_requires_enough_complete_tickers():
    expected_returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.03]})
    fundamentals = pd.DataFrame(
        {"PE": [20.0], "DE": [5.0], "MktCap": [1000.0]},
        index=["A"],
    )
    with pytest.raises(ValueError, match="Only 1 tickers"):
        score_and_select(expected_returns, fundamentals, top_n=2)


def test_download_prices_retries_and_caches(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary outage")
        return _sample_price_frame(["INFY.NS"])

    monkeypatch.setattr("src.data.yf.download", fake_download)
    monkeypatch.setattr("src.data.time.sleep", lambda seconds: None)

    first = download_prices(
        ["INFY.NS"],
        "2024-01-01",
        "2024-01-10",
        cache_dir=tmp_path,
        retries=2,
        backoff=0,
    )
    second = download_prices(
        ["INFY.NS"],
        "2024-01-01",
        "2024-01-10",
        cache_dir=tmp_path,
        retries=2,
        backoff=0,
    )

    assert calls["count"] == 2
    pd.testing.assert_frame_equal(first, second)


def test_download_prices_refresh_still_updates_cache(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_download(*args, **kwargs):
        calls["count"] += 1
        return _sample_price_frame(["INFY.NS"]) * calls["count"]

    monkeypatch.setattr("src.data.yf.download", fake_download)

    fresh = download_prices(
        ["INFY.NS"],
        "2024-01-01",
        "2024-01-10",
        cache_dir=tmp_path,
        use_cache=False,
    )
    cached = download_prices(
        ["INFY.NS"],
        "2024-01-01",
        "2024-01-10",
        cache_dir=tmp_path,
    )

    assert calls["count"] == 1
    pd.testing.assert_frame_equal(fresh, cached)
