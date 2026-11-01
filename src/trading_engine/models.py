# Core models (typed)
import dataclasses
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List

class SignalSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class RiskStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    PARTIAL = "PARTIAL"

@dataclass(frozen=True)
class Feature:
    ts: datetime
    value: float
    source: str
    stale: bool = False

@dataclass(frozen=True)
class Signal:
    strategy: str
    asset: str
    side: SignalSide
    conf: float
    rationale: str
    required_inputs: List[str]
    issued_at: datetime
    # Optional reference price for sizing validation. When present and
    # positive, quantity can be derived as notional / reference_price.
    reference_price: Optional[float] = None
    # Stable signal identity for idempotency. When provided, used as the
    # deduplication key instead of generating a new one.
    signal_id: Optional[str] = None

@dataclass
class OrderIntent:
    client_order_id: str
    asset: str
    side: SignalSide
    notional: float
    qty: Optional[float]
    limit: Optional[float]
    meta: dict
    # Stable intent identity linked to signal_id for idempotency
    signal_id: Optional[str] = None

@dataclass
class RiskDecision:
    status: RiskStatus
    reason: str
    checks: List[tuple[str, bool]]
    approved_at: Optional[datetime] = None

    def to_dict(self):
        return {
            "status": self.status.value,
            "reason": self.reason,
            "checks": [{"name": n, "ok": v} for (n, v) in self.checks],
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }

@dataclass(frozen=True)
class ExecutionResult:
    status: str  # FILLED, REJECTED, PENDING, ERROR, PARTIAL
    adapter: str
    order_id: Optional[str]
    filled: bool
    fail_reason: Optional[str] = None
    # Execution details for filled orders
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    fee: Optional[float] = None
    fee_bps: Optional[float] = None
    slippage_bps: Optional[float] = None
    # For reconciliation
    broker_order_id: Optional[str] = None  # ID returned by broker
    # For crash recovery - links to the original client order
    client_order_id: Optional[str] = None

@dataclass(frozen=True)
class PaperOrder:
    """Paper broker order persisted for crash recovery."""
    client_order_id: str
    asset: str
    side: SignalSide
    notional: float
    qty: Optional[float]
    limit: Optional[float]
    reference_price: float
    status: str  # PENDING, FILLED, REJECTED, ERROR
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    fee: Optional[float] = None
    fee_bps: Optional[int] = None
    slippage_bps: Optional[int] = None
    broker_order_id: Optional[str] = None
    fail_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        # Ensure status is valid
        valid_statuses = {"PENDING", "FILLED", "REJECTED", "ERROR", "PARTIAL"}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")