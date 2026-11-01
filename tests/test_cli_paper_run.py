"""Tests for the ``paper‑run`` CLI sub‑command.

These tests mock ``OkxPublicClient`` so that no real network calls are made.
They verify three scenarios:

1. A fresh ticker results in an approved paper trade.
2. Stale data causes a rejection before any intent is created.
3. A risk‑gate rejection (simulated by mocking ``TradingEngine.process_signal``)
   yields a ``Paper‑run REJECTED`` outcome.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from trading_engine.cli import main
from trading_engine.marketdata import TickerSnapshot


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


def test_paper_run_success(capsys):
    """A fresh ticker leads to an approved paper trade."""
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.return_value = _fresh_snapshot()
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["--db", ":memory:", "paper-run", "--signal-id", "cli_signal_001", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Paper‑run APPROVED" in captured.out


def test_paper_run_stale(capsys):
    """Stale data should cause a rejection before any intent is created."""
    now = _utc_now()
    stale = now - timedelta(seconds=120)
    with patch("trading_engine.cli.OkxPublicClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_ticker.return_value = _fresh_snapshot(ts=stale, received_at=stale)
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance
        rc = main(["--db", ":memory:", "paper-run", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 1
        # Error message is printed to stderr for stale data.
        assert "Error" in captured.err or "stale" in captured.err.lower()


def test_paper_run_risk_reject(capsys):
    """If the risk gate rejects, the CLI reports ``Paper‑run REJECTED``."""
    with patch("trading_engine.cli.OkxPublicClient") as mock_client, \
         patch("trading_engine.cli.TradingEngine") as mock_engine:
        mock_instance = MagicMock()
        mock_instance.get_ticker.return_value = _fresh_snapshot()
        mock_instance.close = MagicMock()
        mock_client.return_value = mock_instance

        # Engine mock returns a rejected ExecutionResult.
        fake_result = MagicMock(status="REJECTED", fail_reason="risk reject")
        engine_obj = MagicMock()
        engine_obj.process_signal.return_value = fake_result
        engine_obj.shutdown.return_value = None
        mock_engine.return_value = engine_obj

        rc = main(["--db", ":memory:", "paper-run", "--signal-id", "cli_signal_002", "BTC-USDT"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "Paper‑run REJECTED" in captured.out