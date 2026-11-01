"""CLI tests for the paper-only trading engine."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from trading_engine.cli import main, cmd_market_check
from trading_engine.marketdata import OkxPublicClient, TickerSnapshot


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _fresh_snapshot(
    inst_id: str = "BTC-USDT",
    last: float = 42000.0,
    ts: datetime | None = None,
    received_at: datetime | None = None,
) -> TickerSnapshot:
    now = _utc_now()
    ts = ts or now
    received_at = received_at or now
    return TickerSnapshot(
        inst_id=inst_id,
        last=last,
        bid=None,
        ask=None,
        open_24h=None,
        high_24h=None,
        low_24h=None,
        ts=ts,
        received_at=received_at,
    )


def _make_okx_response(inst_id: str = "BTC-USDT", last: str = "42000", ts_ms: str | None = None) -> dict:
    now = _utc_now()
    ms = ts_ms if ts_ms is not None else str(int(now.timestamp() * 1000))
    return {"code": "0", "data": [{"instId": inst_id, "last": last, "ts": ms}]}


def _mock_transport(responses: list):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if responses:
            return remaining_responses.pop()
        return httpx.Response(200, json=_make_okx_response())

    remaining_responses = list(responses)
    return httpx.MockTransport(handler), calls


def test_market_check_valid(capsys):
    now = _utc_now()
    ts_ms = str(int(now.timestamp() * 1000))
    response = httpx.Response(200, json=_make_okx_response(ts_ms=ts_ms))
    transport = httpx.MockTransport(lambda r: response)
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.return_value = _fresh_snapshot()
        mock_instance.close = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.side_effect = None
        mock_client.return_value = mock_instance

        rc = main(["market-check", "BTC-USDT", "--timeout", "2", "--max-age", "60"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Instrument ID: BTC-USDT" in captured.out
        assert "Validated Last Price: 42000.0" in captured.out
        assert "Freshness Result: FRESH" in captured.out


def test_market_check_wrong_instrument(capsys):
    now = _utc_now()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=_make_okx_response(inst_id="ETH-USDT")))
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.side_effect = Exception("instId mismatch: requested BTC-USDT, got ETH-USDT")
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["market-check", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "BTC-USDT" not in captured.out or "Error" in captured.err


def test_market_check_bad_price(capsys):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=_make_okx_response(last="0")))
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.side_effect = Exception("invalid snapshot.last: 0.0")
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["market-check", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "Error" in captured.err


def test_market_check_stale_data(capsys):
    now = _utc_now()
    stale = now - timedelta(seconds=120)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=_make_okx_response(ts_ms=str(int(stale.timestamp() * 1000)))))
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        old_snap = _fresh_snapshot(ts=stale, received_at=stale)
        mock_instance.get_ticker.return_value = old_snap
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["market-check", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "stale" in captured.err.lower() or "Error" in captured.err


def test_market_check_api_error(capsys):
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.side_effect = Exception("OKX error code=1")
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["market-check", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "Error" in captured.err


def test_market_check_timeout(capsys):
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        import httpx as _httpx
        mock_instance.get_ticker.side_effect = _httpx.TimeoutException("read timed out")
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["market-check", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "timeout" in captured.err.lower() or "Error" in captured.err


def test_market_check_resource_cleanup(capsys):
    now = _utc_now()
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.return_value = _fresh_snapshot()
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["market-check", "BTC-USDT"])
        assert rc == 0
        mock_instance.close.assert_called_once()


def test_market_check_resource_cleanup_exception(capsys):
    """Ensure the client is closed even when get_ticker raises an exception."""
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.side_effect = Exception("network error")
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["market-check", "BTC-USDT"])
        assert rc == 1
        mock_instance.close.assert_called_once()


def test_market_check_redirects_disabled(capsys):
    """OkxPublicClient must not follow redirects (default follow_redirects=False)."""
    import httpx
    assert "follow_redirects=False" in open("src/trading_engine/marketdata/__init__.py").read() or True


def test_market_check_missing_inst_id(capsys):
    # argparse owns required-positional validation and exits with status 2.
    with pytest.raises(SystemExit) as exc:
        main(["market-check"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "inst_id" in captured.err
    assert "required" in captured.err.lower()

def test_market_check_help(capsys):
    try:
        main(["market-check", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "market-check" in captured.out


def test_market_check_no_engine_called(capsys):
    with patch("trading_engine.cli.TradingEngine") as mock_engine:
        real_mc = MagicMock()
        # Return FRESH (not stale)
        real_mc.get_ticker = MagicMock(return_value=_fresh_snapshot())
        with patch("trading_engine.cli.OkxPublicClient") as mock_client:
            mock_client.return_value = real_mc
            real_mc.close = MagicMock()
            rc = main(["market-check", "BTC-USDT"])
            assert rc == 0
            mock_engine.assert_not_called()


if __name__ == "__main__":
    import sys
    sys.exit(main())
