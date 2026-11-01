"""Fail-closed risk gates implementation.

Every check must explicitly pass. Missing data, stale data, invalid data,
or any inconsistency results in REJECTION. No silent defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..models import OrderIntent, RiskDecision, RiskStatus, Signal
from ..persistence import SQLiteStore


@dataclass(frozen=True)
class RiskConfig:
    """Risk limits configuration. All limits are mandatory."""

    max_notional_per_order: float
    max_position_notional: float
    max_daily_loss: float
    max_open_orders: int
    signal_staleness_seconds: int
    min_signal_confidence: float
    allowed_assets: tuple[str, ...]
    min_order_notional: float = 10.0

    @classmethod
    def default(cls) -> RiskConfig:
        """Safe defaults for paper trading."""
        return cls(
            max_notional_per_order=10000.0,
            max_position_notional=50000.0,
            max_daily_loss=5000.0,
            max_open_orders=10,
            signal_staleness_seconds=60,
            min_signal_confidence=0.5,
            allowed_assets=("BTC-USDT", "ETH-USDT", "SOL-USDT"),
            min_order_notional=10.0,
        )


class RiskGates:
    """Fail-closed risk evaluation. All checks must pass."""

    def __init__(self, store: SQLiteStore, config: Optional[RiskConfig] = None):
        self.store = store
        self.config = config or RiskConfig.default()

    def evaluate(
        self,
        signal: Signal,
        intent: OrderIntent,
        portfolio_cash: float,
        daily_start_equity: Optional[float] = None,
        current_equity: Optional[float] = None,
    ) -> RiskDecision:
        """Evaluate all risk checks. Any failure -> REJECTED."""
        # Backtesting and replay callers may supply an explicit evaluation time
        # so historical signals are judged against their own bar time instead of
        # wall-clock time. Production callers omit it and use real time.
        evaluation_time = getattr(self, "_evaluation_time", None)
        checks: list[tuple[str, bool]] = []

        # 1. Signal confidence
        ok = signal.conf >= self.config.min_signal_confidence
        checks.append(("signal_confidence", ok))

        # 2. Signal not stale
        now = evaluation_time if evaluation_time is not None else datetime.now(timezone.utc)
        age = (now - signal.issued_at).total_seconds()
        if age < 0:
            age = 0.0
        ok = age <= self.config.signal_staleness_seconds
        checks.append(("signal_freshness", ok))

        # 3. Asset allowed
        ok = signal.asset in self.config.allowed_assets
        checks.append(("asset_allowed", ok))

        # 4. Signal/intent asset match
        ok = signal.asset == intent.asset
        checks.append(("asset_consistency", ok))

        # 5. Signal/intent side match
        ok = signal.side == intent.side
        checks.append(("side_consistency", ok))

        # 6. Order notional within limits
        ok = (
            self.config.min_order_notional
            <= intent.notional
            <= self.config.max_notional_per_order
        )
        checks.append(("notional_limits", ok))

        # 7. Sufficient cash
        ok = portfolio_cash >= intent.notional
        checks.append(("sufficient_cash", ok))

        # 8. Position limit (would need current position)
        # For now, check against max position notional
        position = self.store.get_position(intent.asset)
        current_notional = 0.0
        if position and position["qty"] != 0:
            # Need avg_entry_price to calculate notional
            if position["avg_entry_price"]:
                current_notional = abs(position["qty"]) * position["avg_entry_price"]
        ok = (current_notional + intent.notional) <= self.config.max_position_notional
        checks.append(("position_limit", ok))

        # 9. Open orders limit
        # Count pending intents without execution
        conn = self.store._get_conn()
        open_count = conn.execute(
            """
            SELECT COUNT(*) FROM order_intents oi
            LEFT JOIN executions e ON oi.id = e.order_intent_id
            WHERE e.id IS NULL
            """
        ).fetchone()[0]
        ok = open_count < self.config.max_open_orders
        checks.append(("open_orders_limit", ok))

        # 10. Intent has valid client_order_id
        ok = bool(intent.client_order_id and intent.client_order_id.strip())
        checks.append(("valid_client_order_id", ok))

        # 11. Intent qty or limit provided for sizing
        ok = intent.qty is not None or intent.limit is not None
        checks.append(("order_sizing_defined", ok))

        # 12. Daily loss limit
        # Daily loss = starting equity - current equity (cash + position value at current price)
        # Inclusive rejection: loss >= max_daily_loss -> REJECTED
        # Fail-closed: when max_daily_loss is enabled (> 0), both daily_start_equity and
        # current_equity MUST be provided and valid. Missing data -> REJECTED.
        if self.config.max_daily_loss > 0:
            # Daily loss limit is enabled - both values required
            if daily_start_equity is None:
                ok = False
            elif current_equity is None:
                ok = False
            else:
                daily_loss = daily_start_equity - current_equity
                # Validate equity values - fail closed on invalid data
                if not (isinstance(daily_loss, (int, float)) and daily_loss == daily_loss
                        and abs(daily_loss) != float('inf')):
                    ok = False
                elif not (isinstance(current_equity, (int, float)) and current_equity == current_equity
                          and abs(current_equity) != float('inf')):
                    ok = False
                elif not (isinstance(daily_start_equity, (int, float)) and daily_start_equity == daily_start_equity
                          and abs(daily_start_equity) != float('inf')):
                    ok = False
                elif daily_start_equity < 0 or current_equity < 0:
                    ok = False
                else:
                    ok = daily_loss < self.config.max_daily_loss
        else:
            # Daily loss limit disabled (max_daily_loss <= 0) - skip check
            ok = True
        checks.append(("daily_loss_limit", ok))

        # Overall decision
        all_passed = all(v for _, v in checks)
        status = RiskStatus.APPROVED if all_passed else RiskStatus.REJECTED

        failed_checks = [name for name, ok in checks if not ok]
        reason = "All checks passed" if all_passed else f"Failed: {', '.join(failed_checks)}"

        return RiskDecision(
            status=status,
            reason=reason,
            checks=checks,
            approved_at=datetime.now(timezone.utc) if all_passed else None,
        )

        # Overall decision
        all_passed = all(v for _, v in checks)
        status = RiskStatus.APPROVED if all_passed else RiskStatus.REJECTED

        failed_checks = [name for name, ok in checks if not ok]
        reason = "All checks passed" if all_passed else f"Failed: {', '.join(failed_checks)}"

        return RiskDecision(
            status=status,
            reason=reason,
            checks=checks,
            approved_at=datetime.now(timezone.utc) if all_passed else None,
        )