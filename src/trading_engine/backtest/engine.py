"""Backtest engine with deterministic, no-lookahead execution simulation.

All simulated fills are explicitly labeled. Fee and slippage
assumptions are placeholders and clearly marked.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from ..models import ExecutionResult, OrderIntent, Signal, SignalSide
from ..execution.paper_broker import PaperBroker
from ..persistence import SQLiteStore
from ..risk import RiskConfig, RiskGates


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest configuration. All parameters explicit.

    The defaults preserve the original baseline behavior while exposing the
    parameters that Phase 7 makes configurable.
    """

    # Core financial parameters
    initial_cash: float = 100000.0
    fee_bps: float = 10.0  # 10 bps = 0.1 % per side (placeholder)
    slippage_bps: float = 5.0  # 5 bps = 0.05 % per side (placeholder)

    # Strategy parameters (used by the SMA‑crossover demo strategy)
    sma_window: int = 20  # number of bars for the SMA calculation
    position_fraction: float = 0.10  # fraction of cash allocated per trade
    max_notional: float = 10000.0  # absolute cap on trade notional
    max_daily_loss: float = 5000.0  # maximum loss per day before trading halts

    # Data must have columns: timestamp, open, high, low, close, volume
    # Timestamp must be timezone‑aware UTC


@dataclass
class BacktestResult:
    """Backtest results. All P&L is simulated."""

    initial_cash: float
    final_cash: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    equity_curve: list[tuple[datetime, float]]
    trades: list[dict] = field(default_factory=list)
    # Explicitly labeled as simulated
    simulation: bool = True
    fee_assumption_bps: float = 10.0
    slippage_assumption_bps: float = 5.0


class BacktestEngine:
    """Deterministic event-driven backtest.

    Processes bars in chronological order. At each bar:
    1. Strategy generates signal (using only data up to current bar)
    2. Risk gates evaluate (fail-closed)
    3. If approved, paper broker simulates fill at NEXT bar's open
       (no lookahead - we don't know current bar's close yet)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, int], Optional[Signal]],
        config: Optional[BacktestConfig] = None,
        store: Optional[SQLiteStore] = None,
    ):
        """
        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
                  Must be sorted by timestamp ascending.
            strategy_fn: Function(current_data_up_to_bar, bar_index) -> Signal or None
            config: Backtest configuration
            store: Optional persistence store for journaling
        """
        self.data = data.sort_values("timestamp").reset_index(drop=True)
        self.strategy_fn = strategy_fn
        self.config = config or BacktestConfig()
        self.store = store or SQLiteStore(":memory:")

        # Validate data
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure timestamp is timezone-aware UTC
        if self.data["timestamp"].dt.tz is None:
            self.data["timestamp"] = self.data["timestamp"].dt.tz_localize(timezone.utc)
        elif self.data["timestamp"].dt.tz != timezone.utc:
            self.data["timestamp"] = self.data["timestamp"].dt.tz_convert(timezone.utc)

        _base_risk_cfg = RiskConfig.default()
        _backtest_allowed_assets = tuple(
            dict.fromkeys(_base_risk_cfg.allowed_assets + ("CSV_BACKTEST",))
        )
        _backtest_risk_cfg = dataclasses.replace(
            _base_risk_cfg,
            allowed_assets=_backtest_allowed_assets,
            max_notional_per_order=max(
                _base_risk_cfg.max_notional_per_order, self.config.max_notional
            ),
            min_order_notional=0.0,
        )
        self.risk_gates = RiskGates(self.store, _backtest_risk_cfg)
        self.broker = PaperBroker(paper=True)

        # State
        self.cash = self.config.initial_cash
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trades: list[dict] = []
        self.pending_order: Optional[OrderIntent] = None
        # Daily loss tracking
        self._daily_start_equity: Optional[float] = None
        self._last_date = None

    def run(self) -> BacktestResult:
        """Run backtest. Returns labeled simulation results."""
        self.equity_curve = [(self.data.iloc[0]["timestamp"], self.cash)]

        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]
            current_data = self.data.iloc[: i + 1]  # No lookahead

            # Daily loss tracking: reset at start of new day
            bar_ts = current_bar["timestamp"]
            if hasattr(bar_ts, "to_pydatetime"):
                bar_date = bar_ts.to_pydatetime().date()
            else:
                bar_date = bar_ts.date()
            if self._last_date is None or bar_date != self._last_date:
                # New day: reset daily start equity
                self._daily_start_equity = self.cash + self.position * float(current_bar["close"])
                self._last_date = bar_date

            # 1. Check for pending order fill (at current bar's open)
            if self.pending_order is not None:
                self._process_fill(current_bar, i)

            # 2. Generate signal using only data up to current bar
            signal = self.strategy_fn(current_data, i)
            if signal is not None:
                # 3. Create order intent
                intent = self._create_intent(signal, current_bar)

                # Evaluate against the bar's time, not wall-clock time, so
                # historical signals are not rejected as stale.
                evaluation_time = current_bar["timestamp"]
                if hasattr(evaluation_time, "to_pydatetime"):
                    evaluation_time = evaluation_time.to_pydatetime()
                self.risk_gates._evaluation_time = evaluation_time

                # 4. Risk evaluation
                current_equity = self.cash + self.position * float(current_bar["close"])
                decision = self.risk_gates.evaluate(signal, intent, self.cash, daily_start_equity=self._daily_start_equity, current_equity=current_equity)

                # 5. Record in persistence
                if self.store:
                    signal_id = self.store.record_signal(signal)
                    intent_id = self.store.record_order_intent(intent, signal_id)
                    risk_id = self.store.record_risk_decision(intent_id, decision)

                    if decision.status.value == "APPROVED":
                        # Order will fill at NEXT bar's open
                        self.pending_order = intent
                        self.pending_order_id = intent_id
                        self.pending_risk_id = risk_id
                    else:
                        # Rejected - record execution as rejected
                        result = ExecutionResult(
                            status="REJECTED",
                            adapter="paper",
                            order_id=None,
                            filled=False,
                            fail_reason=decision.reason,
                        )
                        self.store.record_execution(intent_id, risk_id, result)

            # 6. Update equity curve (mark-to-market at current bar's close)
            mtm = self.cash + self.position * current_bar["close"]
            self.equity_curve.append((current_bar["timestamp"], mtm))

        # Final equity
        final_mtm = self.cash + self.position * self.data.iloc[-1]["close"]

        # Calculate stats
        winning = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        losing = sum(1 for t in self.trades if t.get("pnl", 0) < 0)

        # Max drawdown
        peak = self.config.initial_cash
        max_dd = 0.0
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        return BacktestResult(
            initial_cash=self.config.initial_cash,
            final_cash=final_mtm,
            total_trades=len(self.trades),
            winning_trades=winning,
            losing_trades=losing,
            max_drawdown=max_dd,
            equity_curve=self.equity_curve,
            trades=self.trades,
            simulation=True,
            fee_assumption_bps=self.config.fee_bps,
            slippage_assumption_bps=self.config.slippage_bps,
        )

    def _create_intent(self, signal: Signal, bar: pd.Series) -> OrderIntent:
        """Create order intent from signal.

        Sizing now respects the configurable ``position_fraction`` and
        ``max_notional`` defined in :class:`BacktestConfig`. The price used for
        quantity conversion remains the close price of the current bar, which
        matches the original deterministic behaviour.
        """
        # Configurable notional‑based sizing (fallback to previous defaults)
        fraction = getattr(self.config, "position_fraction", 0.10)
        max_notional = getattr(self.config, "max_notional", 10000.0)
        notional = min(self.cash * fraction, max_notional)

        # Daily loss cap: if daily loss reaches max_daily_loss, notional = 0
        # Daily loss = start_of_day_equity - current_equity (cash + position * current_price)
        current_price = float(bar["close"])
        current_equity = self.cash + self.position * current_price
        if self._daily_start_equity is not None:
            daily_loss = self._daily_start_equity - current_equity
            daily_loss_remaining = max(0.0, self.config.max_daily_loss - daily_loss)
            notional = min(notional, daily_loss_remaining)

        price = float(bar["close"])
        qty = notional / price if price > 0 else 0.0

        return OrderIntent(
            client_order_id=f"BT_{signal.strategy}_{signal.asset}_{bar['timestamp'].isoformat()}",
            asset=signal.asset,
            side=signal.side,
            notional=notional,
            qty=qty,
            limit=None,  # Market order simulation
            meta={"backtest": True, "signal_conf": signal.conf},
        )

    def _process_fill(self, bar: pd.Series, bar_index: int) -> None:
        """Process pending order fill at bar's open (no lookahead)."""
        if self.pending_order is None:
            return

        # Simulate fill at open with slippage
        fill_price = float(bar["open"])
        slippage = fill_price * (self.config.slippage_bps / 10000.0)
        if self.pending_order.side == SignalSide.LONG:
            fill_price += slippage
        else:
            fill_price -= slippage

        # Fee
        fee = self.pending_order.notional * (self.config.fee_bps / 10000.0)

        # Execute
        qty = self.pending_order.qty or (self.pending_order.notional / fill_price)
        if self.pending_order.side == SignalSide.LONG:
            self.cash -= (qty * fill_price) + fee
            self.position += qty
        else:
            self.cash += (qty * fill_price) - fee
            self.position -= qty

        # Update avg entry price
        if self.position != 0:
            if self.pending_order.side == SignalSide.LONG:
                self.avg_entry_price = (
                    (self.avg_entry_price * (self.position - qty)) + (fill_price * qty)
                ) / self.position
            else:
                # For short, avg_entry_price tracks the short entry
                if self.position < 0:
                    self.avg_entry_price = fill_price

        # Record trade
        trade = {
            "timestamp": str(bar["timestamp"]),
            "asset": self.pending_order.asset,
            "side": self.pending_order.side.value,
            "qty": qty,
            "price": fill_price,
            "notional": self.pending_order.notional,
            "fee": fee,
            "simulated": True,
        }
        self.trades.append(trade)

        # Record execution in persistence
        if self.store:
            result = ExecutionResult(
                status="FILLED",
                adapter="paper",
                order_id=f"SIM_{self.pending_order.client_order_id[:12]}",
                filled=True,
                fail_reason=None,
            )
            self.store.record_execution(
                self.pending_order_id, self.pending_risk_id, result
            )

        self.pending_order = None
        self.pending_order_id = None
        self.pending_risk_id = None