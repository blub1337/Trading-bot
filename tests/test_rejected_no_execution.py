"""Critical test: rejected trades must NEVER reach the execution adapter."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from trading_engine.models import Signal, SignalSide, OrderIntent
from trading_engine.models import ExecutionResult, RiskStatus
from trading_engine.persistence import SQLiteStore
from trading_engine.risk import RiskGates, RiskConfig
from trading_engine.engine import TradingEngine


@pytest.fixture
def engine():
    """Engine with in-memory DB."""
    e = TradingEngine(db_path=":memory:")
    yield e
    e.shutdown()


def test_rejected_trade_does_not_call_broker(engine):
    """A rejected trade must never invoke the paper broker."""
    # Mock the broker to track calls
    engine.broker.place = MagicMock()

    # Create a stale signal (will be rejected)
    signal = Signal(
        strategy="test",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.75,
        rationale="stale signal",
        required_inputs=["close"],
        issued_at=datetime.now(timezone.utc) - __import__("datetime").timedelta(seconds=120),
        signal_id="stale_signal_001",
    )

    result = engine.process_signal(signal)

    assert result.status == "REJECTED"
    assert result.filled is False
    # CRITICAL: broker.place must NEVER have been called
    engine.broker.place.assert_not_called()


def test_rejected_trade_no_execution_record(engine):
    """Rejected trade should still have a journal entry but status REJECTED.

    This test exercises the *post‑intent* risk‑gate rejection path (low confidence).
    We therefore provide a deterministic market‑data provider that returns a valid
    ``TickerSnapshot`` so that the engine creates an ``OrderIntent`` and reaches the
    risk‑gate evaluation, where the low ``conf`` value triggers a rejection.
    """
    from trading_engine.marketdata import TickerSnapshot
    from datetime import datetime, timezone

    # Deterministic valid snapshot (fresh timestamp, positive price)
    now = datetime.now(timezone.utc)
    snapshot = TickerSnapshot(
        inst_id="BTC-USDT",
        last=50000.0,
        bid=None,
        ask=None,
        open_24h=None,
        high_24h=None,
        low_24h=None,
        ts=now,
        received_at=now,
    )

    class SimpleProvider:
        def __init__(self, snap):
            self._snap = snap
            self.calls = []
        def get_ticker(self, inst_id: str):
            self.calls.append(inst_id)
            assert self._snap.inst_id == inst_id
            return self._snap
        def close(self):
            pass

    provider = SimpleProvider(snapshot)

    # Build an engine with the fake provider (in‑memory DB)
    engine = TradingEngine(db_path=":memory:", market_data_provider=provider)

    # Mock the broker to ensure it is never called (risk‑gate rejection).
    engine.broker.place = MagicMock()

    signal = Signal(
        strategy="test",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.2,  # Low confidence -> rejected by risk gates
        rationale="low conf",
        required_inputs=["close"],
        issued_at=datetime.now(timezone.utc),
        signal_id="low_conf_signal_002",
    )

    result = engine.process_signal(signal)

    assert result.status == "REJECTED"
    engine.broker.place.assert_not_called()

    # Verify execution record in store shows REJECTED (post‑intent)
    conn = engine.store._get_conn()
    exec_row = conn.execute("SELECT * FROM executions ORDER BY id DESC LIMIT 1").fetchone()
    assert exec_row is not None
    assert exec_row["status"] == "REJECTED"
    assert exec_row["filled"] == 0
    assert exec_row["adapter"] == "paper"


def test_approved_trade_reaches_broker(engine):
    """An approved trade MUST reach the paper broker (contrast test)."""
    engine.broker.place = MagicMock(return_value=ExecutionResult(
        status="FILLED", adapter="paper", order_id="P123", filled=True,
        fill_qty=0.02, fill_price=50000.0, fee=10.0, fee_bps=10.0, slippage_bps=5.0
    ))

    signal = Signal(
        strategy="test",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.75,
        rationale="valid",
        required_inputs=["close"],
        issued_at=datetime.now(timezone.utc),
        reference_price=50000.0,
        signal_id="approved_signal_003",
    )

    result = engine.process_signal(signal)

    # Broker should have been called
    engine.broker.place.assert_called_once()
    assert result.status == "FILLED"