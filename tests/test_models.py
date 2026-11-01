"""Tests for core models."""
import pytest
from datetime import datetime, timezone
from trading_engine.models import (
    Signal, SignalSide, OrderIntent, RiskDecision, RiskStatus, ExecutionResult
)


def test_signal_creation():
    """Test Signal dataclass."""
    sig = Signal(
        strategy="test",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.8,
        rationale="test signal",
        required_inputs=["close"],
        issued_at=datetime.now(timezone.utc),
    )
    assert sig.asset == "BTC-USDT"
    assert sig.side == SignalSide.LONG


def test_risk_decision_to_dict():
    """Test RiskDecision serialization."""
    rd = RiskDecision(
        status=RiskStatus.APPROVED,
        reason="All checks passed",
        checks=[("test_check", True)],
        approved_at=datetime.now(timezone.utc),
    )
    d = rd.to_dict()
    assert d["status"] == "APPROVED"
    assert len(d["checks"]) == 1
    assert d["checks"][0]["ok"] is True


def test_execution_result_frozen():
    """Test ExecutionResult is frozen."""
    er = ExecutionResult(
        status="FILLED", adapter="paper", order_id="P123", filled=True
    )
    assert er.filled is True
    with pytest.raises(Exception):
        er.filled = False  # type: ignore