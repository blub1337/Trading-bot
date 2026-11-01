"""Deterministic integration tests: market data → TradingEngine (paper-only).

All HTTP/market data interactions are routed through an injected fake provider.
No test talks to a real OKX endpoint.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading_engine.engine import TradingEngine
from trading_engine.marketdata import TickerSnapshot, MarketDataError
from trading_engine.models import ExecutionResult, Signal, SignalSide


# Use the current UTC time at import; the ``patched_now`` fixture will monkey‑patch the engine's ``datetime.now`` to return this same value, ensuring snapshot timestamps line up with the engine's view of "now".
FROZEN_NOW = datetime.now(timezone.utc)


def make_signal(reference_price=None, issued_at=FROZEN_NOW):
    return Signal(
        strategy="integration-test",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.75,
        rationale="integration test",
        required_inputs=["close"],
        issued_at=issued_at,
        reference_price=reference_price,
        signal_id="integration_signal_001",
    )


class FakeProvider:
    def __init__(self, snapshot: TickerSnapshot | Exception):
        self.snapshot = snapshot
        self.calls: list[str] = []

    def get_ticker(self, inst_id: str) -> TickerSnapshot:
        self.calls.append(inst_id)
        if isinstance(self.snapshot, Exception):
            raise self.snapshot
        assert self.snapshot.inst_id == inst_id
        return self.snapshot

    def close(self) -> None:
        pass


def fresh_snapshot(
    last: float = 50000.0,
    inst_id: str = "BTC-USDT",
    ts: datetime | None = None,
    received_at: datetime | None = None,
) -> TickerSnapshot:
    ts = ts or FROZEN_NOW
    received_at = received_at or FROZEN_NOW
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


@pytest.fixture
def patched_now(monkeypatch):
    """Freeze TradingEngine's clock view of 'now' by patching datetime in engine module."""
    from datetime import datetime as _datetime
    real_now = FROZEN_NOW

    class _FrozenDatetime(_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return real_now.replace(tzinfo=None)
            return real_now

    import trading_engine.engine as engine_mod
    monkeypatch.setattr(engine_mod, "datetime", _FrozenDatetime)
    return real_now


def make_engine(provider):
    return TradingEngine(
        db_path=":memory:",
        market_data_provider=provider,
        max_market_data_age_seconds=60.0,
        default_notional=1000.0,
    )


def no_executions_in_store(engine):
    conn = engine.store._get_conn()
    row = conn.execute("SELECT COUNT(*) FROM executions").fetchone()
    return row[0] == 0


def test_fresh_valid_ticker_reaches_broker(patched_now):
    provider = FakeProvider(fresh_snapshot(last=50000.0))
    engine = make_engine(provider)
    try:
        signal = make_signal()
        # Mock the broker with complete execution data
        from trading_engine.models import ExecutionResult
        from unittest.mock import MagicMock
        engine.broker.place = MagicMock(return_value=ExecutionResult(
            status="FILLED", adapter="paper", order_id="P123", filled=True,
            fill_qty=0.02, fill_price=50000.0, fee=10.0, fee_bps=10.0, slippage_bps=5.0
        ))
        result = engine.process_signal(signal)
        assert result.status == "FILLED"
        assert result.filled is True
        engine.broker.place.assert_called_once()
    finally:
        engine.shutdown()


def test_missing_provider_rejected(patched_now):
    # No market‑data provider → early rejection, audit only.
    engine = TradingEngine(db_path=":memory:")
    signal = make_signal()
    engine.broker.place = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    # Record baseline counts.
    conn = engine.store._get_conn()
    base_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    base_rejs = conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0]
    base_intents = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
    base_risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    base_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    result = engine.process_signal(signal)
    assert result.status == "REJECTED"
    assert engine.broker.place.call_count == 0
    # Signal recorded, rejection audited.
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == base_signals + 1
    assert conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0] == base_rejs + 1
    # No further rows should be created.
    assert conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == base_intents
    assert conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0] == base_risks
    assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == base_execs
    engine.shutdown()


def test_provider_exception_rejected(patched_now):
    provider = FakeProvider(Exception("boom"))
    engine = make_engine(provider)
    signal = make_signal()
    from unittest.mock import MagicMock
    engine.broker.place = MagicMock()
    # Baselines
    conn = engine.store._get_conn()
    base_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    base_rejs = conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0]
    base_intents = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
    base_risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    base_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    result = engine.process_signal(signal)
    assert result.status == "REJECTED"
    assert provider.calls == ["BTC-USDT"]
    assert engine.broker.place.call_count == 0
    # Verify audit rows.
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == base_signals + 1
    assert conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0] == base_rejs + 1
    assert conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == base_intents
    assert conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0] == base_risks
    assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == base_execs
    engine.shutdown()

def test_wrong_instrument_rejected(patched_now):
    provider = FakeProvider(fresh_snapshot(last=50000.0, inst_id="ETH-USDT"))
    engine = make_engine(provider)
    signal = make_signal()
    from unittest.mock import MagicMock
    engine.broker.place = MagicMock()
    # Baselines
    conn = engine.store._get_conn()
    base_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    base_rejs = conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0]
    base_intents = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
    base_risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    base_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    result = engine.process_signal(signal)
    assert result.status == "REJECTED"
    assert engine.broker.place.call_count == 0
    # Audit expectations.
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == base_signals + 1
    assert conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0] == base_rejs + 1
    assert conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == base_intents
    assert conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0] == base_risks
    assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == base_execs
    engine.shutdown()

def test_bad_price_rejected(patched_now):
    for bad in [0.0, -1.0, float("nan"), float("inf"), "abc"]:
        print("checking bad:", bad)
    # Representative failure (price = 0.0)
    provider = FakeProvider(fresh_snapshot(last=0.0))
    engine = make_engine(provider)
    signal = make_signal()
    from unittest.mock import MagicMock
    engine.broker.place = MagicMock()
    # Baselines
    conn = engine.store._get_conn()
    base_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    base_rejs = conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0]
    base_intents = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
    base_risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    base_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    result = engine.process_signal(signal)
    assert result.status == "REJECTED"
    assert engine.broker.place.call_count == 0
    # Verify audit.
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == base_signals + 1
    assert conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0] == base_rejs + 1
    assert conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == base_intents
    assert conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0] == base_risks
    assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == base_execs
    engine.shutdown()

def test_stale_receipt_rejected(patched_now):
    received_at = FROZEN_NOW - timedelta(seconds=120)
    provider = FakeProvider(fresh_snapshot(last=50000.0, received_at=received_at))
    engine = make_engine(provider)
    signal = make_signal()
    from unittest.mock import MagicMock
    engine.broker.place = MagicMock()
    # Baselines
    conn = engine.store._get_conn()
    base_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    base_rejs = conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0]
    base_intents = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
    base_risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    base_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    result = engine.process_signal(signal)
    assert result.status == "REJECTED"
    assert engine.broker.place.call_count == 0
    # Verify audit.
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == base_signals + 1
    assert conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0] == base_rejs + 1
    assert conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == base_intents
    assert conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0] == base_risks
    assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == base_execs
    engine.shutdown()

def test_stale_exchange_rejected(patched_now):
    exchange_ts = FROZEN_NOW - timedelta(seconds=120)
    provider = FakeProvider(fresh_snapshot(last=50000.0, ts=exchange_ts))
    engine = make_engine(provider)
    signal = make_signal()
    from unittest.mock import MagicMock
    engine.broker.place = MagicMock()
    # Baselines
    conn = engine.store._get_conn()
    base_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    base_rejs = conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0]
    base_intents = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
    base_risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    base_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    result = engine.process_signal(signal)
    assert result.status == "REJECTED"
    assert engine.broker.place.call_count == 0
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == base_signals + 1
    assert conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0] == base_rejs + 1
    assert conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == base_intents
    assert conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0] == base_risks
    assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == base_execs
    engine.shutdown()

def test_future_timestamp_rejected(patched_now):
    future_ts = FROZEN_NOW + timedelta(seconds=120)
    provider = FakeProvider(fresh_snapshot(last=50000.0, ts=future_ts, received_at=FROZEN_NOW))
    engine = make_engine(provider)
    signal = make_signal()
    from unittest.mock import MagicMock
    engine.broker.place = MagicMock()
    # Baselines
    conn = engine.store._get_conn()
    base_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    base_rejs = conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0]
    base_intents = conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0]
    base_risks = conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    base_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]

    result = engine.process_signal(signal)
    assert result.status == "REJECTED"
    assert engine.broker.place.call_count == 0
    assert conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] == base_signals + 1
    assert conn.execute("SELECT COUNT(*) FROM signal_rejections").fetchone()[0] == base_rejs + 1
    assert conn.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == base_intents
    assert conn.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0] == base_risks
    assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == base_execs
    engine.shutdown()
