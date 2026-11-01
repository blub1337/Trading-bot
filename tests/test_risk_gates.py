"""Tests for fail-closed risk gates."""
import pytest
from datetime import datetime, timezone

from trading_engine.models import Signal, SignalSide, OrderIntent
from trading_engine.persistence import SQLiteStore
from trading_engine.risk import RiskGates, RiskConfig


@pytest.fixture
def store():
    """In-memory SQLite store."""
    s = SQLiteStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def risk_gates(store):
    """Risk gates with daily loss limit DISABLED for basic tests."""
    base = RiskConfig.default()
    config = RiskConfig(
        max_notional_per_order=base.max_notional_per_order,
        max_position_notional=base.max_position_notional,
        max_daily_loss=0.0,  # DISABLED
        max_open_orders=base.max_open_orders,
        signal_staleness_seconds=base.signal_staleness_seconds,
        min_signal_confidence=base.min_signal_confidence,
        allowed_assets=base.allowed_assets,
        min_order_notional=base.min_order_notional,
    )
    return RiskGates(store, config)


@pytest.fixture
def risk_gates_with_daily_loss(store):
    """Risk gates with daily loss limit ENABLED (default 5000)."""
    return RiskGates(store, RiskConfig.default())


def make_signal(**overrides):
    """Create a test signal with overrides."""
    defaults = {
        "strategy": "test_strat",
        "asset": "BTC-USDT",
        "side": SignalSide.LONG,
        "conf": 0.75,
        "rationale": "test",
        "required_inputs": ["close"],
        "issued_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Signal(**defaults)


def make_intent(**overrides):
    """Create a test order intent with overrides."""
    defaults = {
        "client_order_id": "TEST_001",
        "asset": "BTC-USDT",
        "side": SignalSide.LONG,
        "notional": 1000.0,
        "qty": 0.02,
        "limit": None,
        "meta": {"test": True},
    }
    defaults.update(overrides)
    return OrderIntent(**defaults)


def test_approved_signal(risk_gates, store):
    """A valid signal and intent should be approved."""
    signal = make_signal()
    intent = make_intent()
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "APPROVED"


def test_rejected_low_confidence(risk_gates, store):
    """Low confidence signal should be rejected."""
    signal = make_signal(conf=0.2)
    intent = make_intent()
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "REJECTED"
    assert "signal_confidence" in decision.reason


def test_rejected_stale_signal(risk_gates, store):
    """Stale signal should be rejected."""
    from datetime import timedelta
    signal = make_signal(issued_at=datetime.now(timezone.utc) - timedelta(seconds=120))
    intent = make_intent()
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "REJECTED"


def test_rejected_invalid_asset(risk_gates, store):
    """Disallowed asset should be rejected."""
    signal = make_signal(asset="DOGE-USDT")
    intent = make_intent(asset="DOGE-USDT")
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "REJECTED"


def test_rejected_insufficient_cash(risk_gates, store):
    """Insufficient cash should be rejected."""
    signal = make_signal()
    intent = make_intent(notional=999999999.0)
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "REJECTED"


def test_rejected_asset_mismatch(risk_gates, store):
    """Asset mismatch between signal and intent should be rejected."""
    signal = make_signal(asset="BTC-USDT")
    intent = make_intent(asset="ETH-USDT")
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "REJECTED"


def test_rejected_side_mismatch(risk_gates, store):
    """Side mismatch between signal and intent should be rejected."""
    from trading_engine.models import SignalSide
    signal = make_signal(side=SignalSide.LONG)
    intent = make_intent(side=SignalSide.SHORT)
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "REJECTED"


def test_rejected_missing_client_order_id(risk_gates, store):
    """Missing client_order_id should be rejected."""
    signal = make_signal()
    intent = make_intent(client_order_id="")
    portfolio = store.get_portfolio()
    decision = risk_gates.evaluate(signal, intent, portfolio["cash"])
    assert decision.status.value == "REJECTED"