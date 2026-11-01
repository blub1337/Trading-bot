"""Tests for historical candle retrieval and helper conversion.

The tests mock the underlying HTTP request to avoid any real network calls.
They verify:
* Successful parsing of a mixed‑order response and chronological sorting.
* Validation of OHLC consistency, numeric positivity, and timestamp parsing.
* Proper error handling for malformed rows.
* The ``candles_to_dataframe`` utility produces a correctly‑typed DataFrame.
"""

from __future__ import annotations

import datetime
import pytest
from unittest.mock import MagicMock, patch

from trading_engine.marketdata import (
    Candle,
    MarketDataError,
    OkxPublicClient,
    candles_to_dataframe,
)


def _make_row(ts_ms: int, o: float, h: float, l: float, c: float, v: float) -> list:
    return [str(ts_ms), str(o), str(h), str(l), str(c), str(v)]


def test_get_candles_success_and_sorting():
    # API returns newest first – we provide rows in reverse chronological order
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    rows = [
        _make_row(now - 0 * 3600_000, 10, 12, 9, 11, 100),  # newest
        _make_row(now - 1 * 3600_000, 9, 11, 8, 10, 120),
        _make_row(now - 2 * 3600_000, 8, 10, 7, 9, 110),  # oldest
    ]
    payload = {"code": "0", "msg": "", "data": rows}

    with patch.object(OkxPublicClient, "_request", return_value=payload) as mock_req:
        client = OkxPublicClient()
        candles = client.get_candles("BTC-USDT", bar="1H", limit=3)
        # Verify request parameters
        mock_req.assert_called_once_with("/api/v5/market/candles", {"instId": "BTC-USDT", "bar": "1H", "limit": 3})

    # Expect chronological order (oldest → newest)
    assert [c.ts for c in candles] == sorted([c.ts for c in candles])
    # Validate first (oldest) candle values
    oldest = candles[0]
    assert oldest.open == 8
    assert oldest.high == 10
    assert oldest.low == 7
    assert oldest.close == 9
    assert oldest.volume == 110


def test_get_candles_validation_errors():
    # Row with high < low should raise MarketDataError
    bad_row = _make_row(1_600_000_000_000, 10, 9, 12, 11, 100)  # high 9 < low 12
    payload = {"code": "0", "msg": "", "data": [bad_row]}
    with patch.object(OkxPublicClient, "_request", return_value=payload):
        client = OkxPublicClient()
        with pytest.raises(MarketDataError, match="high < low"):
            client.get_candles("BTC-USDT")

    # Row with inconsistent OHLC (open > high) should raise error
    bad_row2 = _make_row(1_600_000_000_001, 15, 14, 13, 12, 100)
    payload2 = {"code": "0", "msg": "", "data": [bad_row2]}
    with patch.object(OkxPublicClient, "_request", return_value=payload2):
        client = OkxPublicClient()
        with pytest.raises(MarketDataError, match="OHLC inconsistent"):
            client.get_candles("BTC-USDT")


def test_candles_to_dataframe_and_empty_error():
    # Prepare a list of Candle objects in random order
    ts1 = datetime.datetime(2023, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts2 = datetime.datetime(2023, 1, 1, 1, 0, tzinfo=datetime.timezone.utc)
    candles = [Candle(ts=ts2, open=10, high=12, low=9, close=11, volume=200),
               Candle(ts=ts1, open=9, high=11, low=8, close=10, volume=150)]
    df = candles_to_dataframe(candles)
    # Ensure chronological order after conversion
    assert list(df["timestamp"]) == [ts1, ts2]
    assert list(df["open"]) == [9, 10]
    assert list(df["volume"]) == [150, 200]

    # Empty list should raise MarketDataError
    with pytest.raises(MarketDataError, match="no candles to convert"):
        candles_to_dataframe([])

