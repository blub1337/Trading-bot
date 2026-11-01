"""Offline tests for the public, read-only OKX market data client.

All HTTP traffic is intercepted by httpx MockTransport. No test may contact
the real OKX API or require network access.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from trading_engine.marketdata import (
    Candle,
    MarketDataError,
    OkxPublicClient,
    TickerSnapshot,
)

VALID_TICKER = {
    "code": "0",
    "msg": "",
    "data": [
        {
            "instId": "BTC-USDT",
            "last": "50000.0",
            "bidPx": "49999.5",
            "askPx": "50000.5",
            "open24h": "49000.0",
            "high24h": "51000.0",
            "low24h": "48000.0",
            "ts": "1700000000000",
        }
    ],
}

VALID_CANDLES = {
    "code": "0",
    "msg": "",
    "data": [
        ["1700000000000", "50000.0", "50100.0", "49900.0", "50050.0", "123.45"],
        ["1700000060000", "50050.0", "50200.0", "50000.0", "50150.0", "234.56"],
    ],
}

VALID_INSTRUMENTS = {
    "code": "0",
    "msg": "",
    "data": [
        {
            "instId": "BTC-USDT",
            "instType": "SPOT",
            "baseCcy": "BTC",
            "quoteCcy": "USDT",
            "state": "live",
        }
    ],
}


def response(status_code: int, json_body=None, text_body: str | None = None):
    if json_body is not None:
        return httpx.Response(status_code, json=json_body)
    return httpx.Response(status_code, text=text_body or "")


def make_client(handler, follow_redirects: bool = False):
    transport = httpx.MockTransport(handler)
    inner = httpx.Client(
        timeout=2.0,
        transport=transport,
        base_url="https://www.okx.com",
        follow_redirects=follow_redirects,
    )
    return OkxPublicClient(client=inner), inner


def request_failed_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("network unreachable")


def test_ticker_valid_and_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url).startswith("https://www.okx.com/api/v5/market/ticker")
        assert request.url.params.get("instId") == "BTC-USDT"
        return response(200, VALID_TICKER)

    client, inner = make_client(handler)
    try:
        ticker = client.get_ticker("BTC-USDT")
    finally:
        client.close()

    assert isinstance(ticker, TickerSnapshot)
    assert ticker.inst_id == "BTC-USDT"
    assert ticker.last == 50000.0
    assert ticker.bid == 49999.5
    assert ticker.ask == 50000.5
    assert ticker.open_24h == 49000.0
    assert ticker.high_24h == 51000.0
    assert ticker.low_24h == 48000.0
    assert ticker.ts == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    assert ticker.received_at.tzinfo == timezone.utc
    assert ticker.is_stale(max_age_seconds=600) is False


def test_ticker_valid_none_bid_ask() -> None:
    payload = {
        "code": "0",
        "msg": "",
        "data": [
            {
                "instId": "BTC-USDT",
                "last": "50000.0",
                "bidPx": "",
                "askPx": None,
                "ts": "1700000000000",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return response(200, payload)

    client, inner = make_client(handler)
    try:
        ticker = client.get_ticker("BTC-USDT")
    finally:
        client.close()

    assert ticker.bid is None
    assert ticker.ask is None


def test_ticker_rejects_invalid_numeric_values() -> None:
    bad_values = [
        "0",
        "-1",
        "nan",
        "inf",
        "abc",
    ]
    for bad in bad_values:
        payload = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "instId": "BTC-USDT",
                    "last": bad,
                    "ts": "1700000000000",
                }
            ],
        }

        def handler(request: httpx.Request, body=payload) -> httpx.Response:
            return response(200, body)

        client, inner = make_client(handler)
        try:
            with pytest.raises(MarketDataError):
                client.get_ticker("BTC-USDT")
        finally:
            client.close()


def test_ticker_rejects_invalid_bid_ask() -> None:
    payloads = [
        {
            "code": "0",
            "msg": "",
            "data": [
                {"instId": "BTC-USDT", "last": "10", "bidPx": "0", "ts": "1700000000000"}
            ],
        },
        {
            "code": "0",
            "msg": "",
            "data": [
                {"instId": "BTC-USDT", "last": "10", "askPx": "nan", "ts": "1700000000000"}
            ],
        },
        {
            "code": "0",
            "msg": "",
            "data": [
                {"instId": "BTC-USDT", "last": "10", "bidPx": "abc", "ts": "1700000000000"}
            ],
        },
    ]

    for payload in payloads:
        def handler(request: httpx.Request, body=payload) -> httpx.Response:
            return response(200, body)

        client, inner = make_client(handler)
        try:
            with pytest.raises(MarketDataError):
                client.get_ticker("BTC-USDT")
        finally:
            client.close()


def test_ticker_rejects_invalid_timestamp() -> None:
    payload = {
        "code": "0",
        "msg": "",
        "data": [
            {"instId": "BTC-USDT", "last": "10", "ts": "not-a-number"}
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return response(200, payload)

    client, inner = make_client(handler)
    try:
        with pytest.raises(MarketDataError):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()


def test_ticker_rejects_inst_id_mismatch() -> None:
    payload = {
        "code": "0",
        "msg": "",
        "data": [
            {"instId": "ETH-USDT", "last": "10", "ts": "1700000000000"}
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return response(200, payload)

    client, inner = make_client(handler)
    try:
        with pytest.raises(MarketDataError, match="instId mismatch"):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()


def test_ticker_rejects_empty_payload_and_bad_inputs() -> None:
    empty = {"code": "0", "msg": "", "data": []}
    client, inner = make_client(lambda req: response(200, empty))
    try:
        with pytest.raises(MarketDataError):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()

    with OkxPublicClient() as outer:
        with pytest.raises(MarketDataError):
            outer.get_ticker("")
        with pytest.raises(MarketDataError):
            outer.get_ticker(None)  # type: ignore[arg-type]


def test_ticker_rejects_http_errors_okx_error_and_bad_json() -> None:
    client, inner = make_client(lambda req: response(500))
    try:
        with pytest.raises(MarketDataError, match="HTTP 500"):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()

    code_error = {"code": "1", "msg": "something wrong", "data": [{"instId": "BTC-USDT", "last": "10", "ts": "1700000000000"}]}
    client, inner = make_client(lambda req: response(200, code_error))
    try:
        with pytest.raises(MarketDataError, match="OKX error"):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()

    client, inner = make_client(lambda req: response(200, None, text_body="{not valid json"))
    try:
        with pytest.raises(MarketDataError, match="invalid JSON"):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()


def test_ticker_rejects_malformed_top_level_payload() -> None:
    client, inner = make_client(lambda req: response(200, ["array", "not", "dict"]))
    try:
        with pytest.raises(MarketDataError, match="payload is not object"):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()

    client2, inner2 = make_client(lambda req: response(200, 42))
    try:
        with pytest.raises(MarketDataError, match="payload is not object"):
            client2.get_ticker("BTC-USDT")
    finally:
        client2.close()


def test_request_transport_failures() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    client, inner = make_client(timeout_handler)
    try:
        with pytest.raises(MarketDataError, match="request timed out"):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()

    def http_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.HTTPError("boom")

    client, inner = make_client(http_handler)
    try:
        with pytest.raises(MarketDataError, match="request failed"):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()


def test_candles_valid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("https://www.okx.com/api/v5/market/candles")
        assert request.url.params.get("instId") == "BTC-USDT"
        assert request.url.params.get("bar") == "1H"
        assert request.url.params.get("limit") == "2"
        return response(200, VALID_CANDLES)

    client, inner = make_client(handler)
    try:
        candles = client.get_candles("BTC-USDT", bar="1H", limit=2)
    finally:
        client.close()

    assert len(candles) == 2
    assert isinstance(candles[0], Candle)
    assert candles[0].open == 50000.0
    assert candles[0].high == 50100.0
    assert candles[0].low == 49900.0
    assert candles[0].close == 50050.0
    assert candles[0].volume == 123.45


def test_candles_invalid_inputs_and_empty_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return response(200, {"code": "0", "msg": "", "data": []})

    client, inner = make_client(handler)
    try:
        with pytest.raises(MarketDataError):
            client.get_candles("BTC-USDT")
    finally:
        client.close()

    with OkxPublicClient() as owned:
        for kwargs in [
            {"inst_id": ""},
            {"inst_id": "BTC-USDT", "bar": ""},
            {"inst_id": "BTC-USDT", "limit": 0},
            {"inst_id": "BTC-USDT", "limit": 301},
            {"inst_id": "BTC-USDT", "limit": "2"},
            {"inst_id": 123},
        ]:
            with pytest.raises(MarketDataError):
                owned.get_candles(**kwargs)  # type: ignore[arg-type]


def test_candles_inconsistent_ohlc() -> None:
    payloads = [
        {  # high < low
            "code": "0",
            "msg": "",
            "data": [["1700000000000", "50000.0", "49000.0", "50100.0", "50050.0", "1"]],
        },
        {  # open outside [low, high]
            "code": "0",
            "msg": "",
            "data": [["1700000000000", "50200.0", "50100.0", "49900.0", "50050.0", "1"]],
        },
        {  # zero volume
            "code": "0",
            "msg": "",
            "data": [["1700000000000", "50000.0", "50100.0", "49900.0", "50050.0", "0"]],
        },
        {  # non-numeric
            "code": "0",
            "msg": "",
            "data": [["1700000000000", "50000.0", "nan", "49900.0", "50050.0", "1"]],
        },
        {  # malformed row
            "code": "0",
            "msg": "",
            "data": [["1700000000000", "50000.0"]],
        },
    ]

    for payload in payloads:
        def handler(request: httpx.Request, body=payload) -> httpx.Response:
            return response(200, body)

        client, inner = make_client(handler)
        try:
            with pytest.raises(MarketDataError):
                client.get_candles("BTC-USDT")
        finally:
            client.close()


def test_instruments_valid_and_invalid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("instType") == "SPOT"
        assert request.url.params.get("instId") == "BTC-USDT"
        return response(200, VALID_INSTRUMENTS)

    client, inner = make_client(handler)
    try:
        rows = client.get_instruments("SPOT", inst_id="BTC-USDT")
    finally:
        client.close()

    assert len(rows) == 1
    assert rows[0]["instType"] == "SPOT"

    bad = {"code": "0", "msg": "", "data": ["not", "dicts"]}
    client, inner = make_client(lambda req: response(200, bad))
    try:
        with pytest.raises(MarketDataError, match="invalid instrument row"):
            client.get_instruments("SPOT")
    finally:
        client.close()


def test_http_client_ownership_and_close() -> None:
    mock = httpx.MockTransport(lambda req: response(200, VALID_TICKER))

    # A client injected by the caller remains caller-owned after wrapper.close().
    external_inner = httpx.Client(
        timeout=2.0,
        transport=mock,
        base_url="https://www.okx.com",
    )
    wrapper = OkxPublicClient(client=external_inner)
    try:
        snapshot = wrapper.get_ticker("BTC-USDT")
        assert snapshot.last == 50000.0

        wrapper.close()
        assert not external_inner.is_closed

        # Still usable, but handled entirely by MockTransport (no network access).
        external_response = external_inner.get(
            "/api/v5/market/ticker", params={"instId": "BTC-USDT"}
        )
        assert external_response.status_code == 200
    finally:
        # The caller, not the wrapper, closes the externally supplied client.
        external_inner.close()
    assert external_inner.is_closed

    # A client created internally by the wrapper is wrapper-owned and closed.
    internally_managed = OkxPublicClient()
    internal_inner = internally_managed._client
    try:
        assert not internal_inner.is_closed
    finally:
        internally_managed.close()
    assert internal_inner.is_closed


def test_staleness_controlled_without_sleep() -> None:
    snapshot = TickerSnapshot(
        inst_id="BTC-USDT",
        last=50000.0,
        bid=None,
        ask=None,
        open_24h=None,
        high_24h=None,
        low_24h=None,
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        received_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )

    fresh_now = datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    assert snapshot.age_seconds(fresh_now) == 5.0
    assert snapshot.is_stale(max_age_seconds=10, now=fresh_now) is False

    stale_now = datetime(2024, 1, 1, 0, 0, 15, tzinfo=timezone.utc)
    assert snapshot.is_stale(max_age_seconds=10, now=stale_now) is True


def test_redirects_are_not_followed() -> None:
    mock = httpx.MockTransport(lambda req: response(302, None, text_body=""))
    inner = httpx.Client(
        timeout=2.0,
        transport=mock,
        base_url="https://www.okx.com",
        follow_redirects=False,
    )
    client = OkxPublicClient(client=inner)
    try:
        with pytest.raises(MarketDataError):
            client.get_ticker("BTC-USDT")
    finally:
        client.close()


def test_candle_constructor_fields() -> None:
    candle = Candle(
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
    )
    assert candle.ts.year == 2024
    assert candle.high >= candle.close >= candle.low
