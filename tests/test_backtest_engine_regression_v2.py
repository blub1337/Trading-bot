"""Additional regression tests for the offline ``BacktestEngine``.

These tests target the specific behaviours outlined in the Phase 6.1
specification and supplement the existing suite without requiring any live
network access.
"""

from __future__ import annotations

import dataclasses
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from trading_engine.backtest import BacktestEngine, BacktestConfig
from trading_engine.models import Signal, SignalSide
from trading_engine.cli import cmd_backtest_csv
from trading_engine.risk import RiskConfig


def _make_df(prices, opens=None):
    """Utility to build a minimal OHLCV ``DataFrame``.

    ``prices`` defines the ``close`` column. ``opens`` may be supplied; otherwise
    the open price equals the close price. All other columns are given benign
    values that satisfy the CSV validation logic.
    """
    if opens is None:
        opens = prices
    ts = pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": [max(o, c) + 1 for o, c in zip(opens, prices)],
            "low": [min(o, c) - 1 for o, c in zip(opens, prices)],
            "close": prices,
            "volume": [100.0] * len(prices),
        }
    )


def test_no_lookahead_allowed():
    """The strategy should only receive data up to the current bar."""

    df = _make_df([10.0, 20.0, 30.0])

    def strategy(current_data: pd.DataFrame, bar_idx: int):  # pragma: no cover
        assert len(current_data) == bar_idx + 1
        with pytest.raises(IndexError):
            _ = current_data.iloc[bar_idx + 1]
        return None

    cfg = BacktestConfig(initial_cash=1_000.0, fee_bps=0.0, slippage_bps=0.0)
    engine = BacktestEngine(df, strategy, cfg)
    result = engine.run()
    assert result.total_trades == 0


def test_execution_timing_fill_at_next_open():
    """A signal generated on bar *i* is filled at the *open* of bar *i+1*."""

    df = _make_df([10.0, 20.0])

    def strategy(current_data: pd.DataFrame, bar_idx: int):  # pragma: no cover
        if bar_idx == 0:
            return Signal(
                strategy="test",
                asset="BTC-USDT",
                side=SignalSide.LONG,
                conf=1.0,
                rationale="test",
                required_inputs=["close"],
                issued_at=datetime.now(timezone.utc),
            )
        return None

    cfg = BacktestConfig(initial_cash=1_000.0, fee_bps=0.0, slippage_bps=0.0)
    engine = BacktestEngine(df, strategy, cfg)
    result = engine.run()

    assert result.total_trades == 1
    trade = result.trades[-1]
    assert trade["qty"] == pytest.approx(10.0)
    assert result.final_cash == pytest.approx(1_000.0)


def test_risk_rejection_prevents_trade():
    """A rejected trade is never filled."""

    df = _make_df([10.0, 10.0])

    def strategy(_: pd.DataFrame, __: int):
        return Signal(
            strategy="reject",
            asset="BTC-USDT",
            side=SignalSide.LONG,
            conf=1.0,
            rationale="reject",
            required_inputs=["close"],
            issued_at=datetime.now(timezone.utc),
        )

    cfg = BacktestConfig(initial_cash=1_000.0, fee_bps=0.0, slippage_bps=0.0)
    engine = BacktestEngine(df, strategy, cfg)
    low_cfg = dataclasses.replace(RiskConfig.default(), max_notional_per_order=1.0)
    engine.risk_gates = engine.risk_gates.__class__(engine.risk_gates.store, low_cfg)
    result = engine.run()

    assert result.total_trades == 0
    assert result.final_cash == pytest.approx(1_000.0)


def test_long_then_short_accounting():
    """Validate accounting after a long and a short trade."""

    df = _make_df([10.0, 20.0, 30.0])

    def strategy(current_data: pd.DataFrame, bar_idx: int):  # pragma: no cover
        if bar_idx == 0:
            return Signal(
                strategy="test",
                asset="BTC-USDT",
                side=SignalSide.LONG,
                conf=1.0,
                rationale="long",
                required_inputs=["close"],
                issued_at=datetime.now(timezone.utc),
            )
        if bar_idx == 1:
            return Signal(
                strategy="test",
                asset="BTC-USDT",
                side=SignalSide.SHORT,
                conf=1.0,
                rationale="short",
                required_inputs=["close"],
                issued_at=datetime.now(timezone.utc),
            )
        return None

    cfg = BacktestConfig(initial_cash=1_000.0, fee_bps=0.0, slippage_bps=0.0)
    engine = BacktestEngine(df, strategy, cfg)
    result = engine.run()

    assert result.total_trades == 2
    assert result.final_cash == pytest.approx(1_100.0)
    short_qty = result.trades[-1]["qty"]
    assert short_qty == pytest.approx(4.0)


def test_equity_curve_and_drawdown():
    """Exercise the equity-curve and max-draw-down calculation."""

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
            "open": [10.0, 20.0, 5.0, 30.0],
            "high": [11.0, 21.0, 6.0, 31.0],
            "low": [9.0, 19.0, 4.0, 29.0],
            "close": [10.0, 20.0, 5.0, 30.0],
            "volume": [100.0] * 4,
        }
    )

    def strategy(current_data: pd.DataFrame, bar_idx: int):  # pragma: no cover
        if bar_idx == 0:
            return Signal(
                strategy="test",
                asset="BTC-USDT",
                side=SignalSide.LONG,
                conf=1.0,
                rationale="long",
                required_inputs=["close"],
                issued_at=datetime.now(timezone.utc),
            )
        return None

    cfg = BacktestConfig(initial_cash=1_000.0, fee_bps=0.0, slippage_bps=0.0)
    engine = BacktestEngine(df, strategy, cfg)
    result = engine.run()

    notional = 100.0
    qty = notional / 10.0  # quantity is based on the strategy bar close (10.0)
    cash_after = 1_000.0 - qty * 20.0  # fill happens at bar 1 open (20.0)
    # Engine records the initial cash, then one mtm per bar.
    equity = [
        1_000.0,        # initial cash
        1_000.0,        # bar 0 mtm: cash 1000 + 0 * close 10 (signal generated, not yet filled)
        1_000.0,        # bar 1 mtm: cash 800 + 10 * close 20 (filled at bar 1 open 20)
        850.0,          # bar 2 mtm: cash 800 + 10 * close 5
        1_100.0,        # bar 3 mtm: cash 800 + 10 * close 30
    ]
    assert [e for _, e in result.equity_curve] == pytest.approx(equity)
    expected_dd = (equity[2] - equity[3]) / equity[2]
    assert result.max_drawdown == pytest.approx(expected_dd)


def test_sma_crossover_strategy_behavior(tmp_path: Path):
    """Validate the SMA-crossover strategy used by ``cmd_backtest_csv``."""

    prices = [100.0] * 25
    df = _make_df(prices)
    csv_path = tmp_path / "sma.csv"
    df.to_csv(csv_path, index=False)
    out_path = tmp_path / "out.json"

    args = Namespace(csv_path=str(csv_path), output=str(out_path))
    exit_code = cmd_backtest_csv(args)
    assert exit_code == 0
    assert out_path.exists()

    with out_path.open() as f:
        data = json.load(f)

    assert data["total_trades"] == 0


def test_cli_json_matches_engine(tmp_path: Path):
    """Cross-check the JSON file written by ``cmd_backtest_csv``."""

    df = _make_df([10.0, 15.0, 20.0])
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    out_path = tmp_path / "out.json"

    args = Namespace(csv_path=str(csv_path), output=str(out_path))
    assert cmd_backtest_csv(args) == 0
    assert out_path.exists()

    with out_path.open() as f:
        json_data = json.load(f)

    def sma_strategy(current_data: pd.DataFrame, bar_idx: int):
        if bar_idx < 20:
            return None
        sma = current_data["close"].rolling(20).mean().iloc[-1]
        cur = current_data["close"].iloc[-1]
        if cur > sma:
            return Signal(
                strategy="sma_crossover",
                asset="BTC-USDT",
                side=SignalSide.LONG,
                conf=0.6,
                rationale="",
                required_inputs=["close"],
                issued_at=current_data["timestamp"].iloc[-1],
            )
        return None

    cfg = BacktestConfig(initial_cash=100_000.0, fee_bps=10.0, slippage_bps=5.0)
    engine = BacktestEngine(df, sma_strategy, cfg)
    result = engine.run()

    assert json_data["initial_cash"] == pytest.approx(result.initial_cash)
    assert json_data["final_cash"] == pytest.approx(result.final_cash)
    assert json_data["total_trades"] == result.total_trades
    assert json_data["max_drawdown_pct"] == pytest.approx(result.max_drawdown * 100)


def test_reproducibility_identical_inputs():
    """Repeated runs with identical inputs must produce identical results."""

    df = _make_df([10.0, 15.0, 20.0])

    def strategy(current_data: pd.DataFrame, bar_idx: int):
        return None

    cfg = BacktestConfig(initial_cash=1_000.0, fee_bps=10.0, slippage_bps=5.0)
    engine_1 = BacktestEngine(df.copy(), strategy, cfg)
    engine_2 = BacktestEngine(df.copy(), strategy, cfg)
    result_1 = engine_1.run()
    result_2 = engine_2.run()

    assert result_1.initial_cash == result_2.initial_cash
    assert result_1.final_cash == pytest.approx(result_2.final_cash)
    assert result_1.total_trades == result_2.total_trades
    assert [e for _, e in result_1.equity_curve] == pytest.approx(
        [e for _, e in result_2.equity_curve]
    )
    assert result_1.max_drawdown == pytest.approx(result_2.max_drawdown)
