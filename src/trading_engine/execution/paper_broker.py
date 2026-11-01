"""Paper execution adapter — the ONLY execution path available in Phase 1.

This adapter records an order intent and returns a filled result.
It does NOT communicate with any remote broker or live endpoint.
"""
import hashlib
import time
from typing import Optional

from ..models import ExecutionResult, OrderIntent, SignalSide, PaperOrder, ExecutionStatus
from ..risk import RiskConfig


class PaperBroker:
    """Paper trading execution adapter with realistic fill simulation."""

    def __init__(self, paper: bool = True, store=None):
        if not paper:
            raise RuntimeError("PaperBroker refuses live mode unless paper=True")
        
        # Default execution parameters
        self.fee_bps = 10  # 10 bps = 0.1%
        self.slippage_bps = 5  # 5 bps = 0.05%
        self._store = store

    def place(self, intent: 'OrderIntent', reference_price: float) -> 'ExecutionResult':
        """
        Execute a paper trade with realistic fee and slippage modeling.
        
        Idempotent: if an order with the same client_order_id already exists,
        returns the existing result instead of executing again.
        
        Args:
            intent: The order intent to execute
            reference_price: The reference price for execution (e.g., current market price)
            
        Returns:
            ExecutionResult with fill details
        """
        from ..models import ExecutionResult, SignalSide, PaperOrder, ExecutionStatus
        import hashlib
        import time
        
        # Validate intent
        if intent.notional <= 0:
            return ExecutionResult(
                status="REJECTED",
                adapter="paper",
                order_id=None,
                filled=False,
                fail_reason="Notional must be positive",
                client_order_id=intent.client_order_id,
            )
        
        if intent.side not in (SignalSide.LONG, SignalSide.SHORT):
            return ExecutionResult(
                status="REJECTED",
                adapter="paper",
                order_id=None,
                filled=False,
                fail_reason=f"Invalid side: {intent.side}",
                client_order_id=intent.client_order_id,
            )
        
        # Validate reference price
        if reference_price is None or reference_price <= 0:
            return ExecutionResult(
                status="REJECTED",
                adapter="paper",
                order_id=None,
                filled=False,
                fail_reason="Invalid reference price for execution",
                client_order_id=intent.client_order_id,
            )
        
        # Atomic idempotency reservation: only one caller may own a client_order_id.
        # reserve_paper_order uses INSERT ... ON CONFLICT DO NOTHING under a transaction,
        # so concurrent duplicate calls cannot both proceed.
        if self._store is not None:
            reserved = self._store.reserve_paper_order(
                client_order_id=intent.client_order_id,
                asset=intent.asset,
                side=intent.side.value,
                notional=intent.notional,
                qty=intent.qty,
                limit_price=intent.limit,
                reference_price=reference_price,
                fee_bps=self.fee_bps,
                slippage_bps=self.slippage_bps,
                broker_order_id="",  # placeholder, updated after fill computation
            )
            if not reserved:
                # Another caller already owns this order - return existing result
                return self.get_execution_result(intent.client_order_id)
        
        # Calculate quantity from notional if not provided
        if intent.qty is not None and intent.qty > 0:
            qty = intent.qty
        elif intent.notional > 0:
            qty = intent.notional / reference_price
        else:
            return ExecutionResult(
                status="REJECTED",
                adapter="paper",
                order_id=None,
                filled=False,
                fail_reason="Cannot determine quantity from notional",
                client_order_id=intent.client_order_id,
            )
        
        # Handle limit orders
        if intent.limit is not None:
            # Limit order: check if limit price is reachable
            if intent.side == SignalSide.LONG and intent.limit < reference_price:
                # Buy limit below market - order sits on book, not filled immediately
                return ExecutionResult(
                    status="PENDING",
                    adapter="paper",
                    order_id=None,
                    filled=False,
                    fail_reason=f"Limit buy order at {intent.limit} below market {reference_price}, not filled",
                    client_order_id=intent.client_order_id,
                )
            elif intent.side == SignalSide.SHORT and intent.limit > reference_price:
                # Sell limit above market - order sits on book, not filled immediately
                return ExecutionResult(
                    status="PENDING",
                    adapter="paper",
                    order_id=None,
                    filled=False,
                    fail_reason=f"Limit sell order at {intent.limit} above market {reference_price}, not filled",
                    client_order_id=intent.client_order_id,
                )
            # Limit is favorable (buy limit >= market, sell limit <= market) - treat as market
            fill_price = reference_price
        else:
            # Market order - apply slippage
            if intent.side == SignalSide.LONG:
                # Buy at ask (higher price due to slippage)
                fill_price = reference_price * (1 + self.slippage_bps / 10000)
            else:
                # Sell at bid (lower price due to slippage)
                fill_price = reference_price * (1 - self.slippage_bps / 10000)
        
        # Calculate fee
        fee_bps = self.fee_bps
        notional = qty * fill_price
        fee = notional * (fee_bps / 10000)
        
        # Generate deterministic broker order ID (no time.time() dependency)
        broker_order_id = f"P{hashlib.md5(f'{intent.client_order_id}'.encode()).hexdigest()[:16]}"
        
        # Persist paper order with final status before returning result
        paper_order = PaperOrder(
            client_order_id=intent.client_order_id,
            asset=intent.asset,
            side=intent.side,
            notional=intent.notional,
            qty=qty,
            limit=intent.limit,
            reference_price=reference_price,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            status=ExecutionStatus.FILLED.value,
            fill_price=fill_price,
            fill_qty=qty,
            fee=notional * (fee_bps / 10000),
            broker_order_id=broker_order_id,
        )
        
        if self._store is not None:
            # Update the reserved PENDING row with final fill details
            self._store.update_paper_order_status(
                client_order_id=paper_order.client_order_id,
                status=paper_order.status,
                fill_price=paper_order.fill_price,
                fill_qty=paper_order.fill_qty,
                fee=paper_order.fee,
                fee_bps=paper_order.fee_bps,
                slippage_bps=paper_order.slippage_bps,
                broker_order_id=paper_order.broker_order_id,
            )
        
        return ExecutionResult(
            status="FILLED",
            adapter="paper",
            order_id=broker_order_id,
            filled=True,
            fail_reason=None,
            fill_price=fill_price,
            fill_qty=qty,
            fee=notional * (fee_bps / 10000),
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            broker_order_id=broker_order_id,
            client_order_id=intent.client_order_id,
        )

    def get_execution_result(self, client_order_id: str) -> ExecutionResult:
        """Retrieve an existing paper order's execution result by client_order_id.

        Used by crash recovery to restore the result of a previously
        persisted paper order without re-executing it.

        Args:
            client_order_id: The client order ID to look up.

        Returns:
            ExecutionResult reconstructed from the stored paper order.

        Raises:
            KeyError: If no paper order exists for the given client_order_id.
        """
        if self._store is None:
            raise KeyError(f"No store configured; cannot retrieve order {client_order_id}")

        existing_order = self._store.get_paper_order(client_order_id)
        if not existing_order:
            raise KeyError(f"No paper order found for client_order_id={client_order_id}")

        return ExecutionResult(
            status=existing_order["status"],
            adapter="paper",
            order_id=existing_order["broker_order_id"],
            filled=existing_order["status"] == ExecutionStatus.FILLED.value,
            fail_reason=existing_order["fail_reason"],
            fill_price=existing_order["fill_price"],
            fill_qty=existing_order["fill_qty"],
            fee=existing_order["fee"],
            fee_bps=existing_order["fee_bps"],
            slippage_bps=existing_order["slippage_bps"],
            broker_order_id=existing_order["broker_order_id"],
            client_order_id=existing_order["client_order_id"],
        )


# Ensure hashlib is imported at module level
import hashlib
from datetime import datetime, timezone
from ..models import PaperOrder, ExecutionStatus