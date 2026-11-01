"""Trading Engine - Autonomous OKX Crypto Trading System (Paper Mode)."""
from .engine import TradingEngine
from .models import (
    ExecutionResult,
    Feature,
    OrderIntent,
    RiskDecision,
    RiskStatus,
    Signal,
    SignalSide,
)
from .persistence import SQLiteStore
from .risk import RiskConfig, RiskGates
from .backtest import BacktestConfig, BacktestEngine, BacktestResult
from .execution.paper_broker import PaperBroker

__version__ = "0.1.0"
__all__ = [
    "TradingEngine",
    "SQLiteStore",
    "RiskGates",
    "RiskConfig",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "PaperBroker",
    "Signal",
    "SignalSide",
    "OrderIntent",
    "RiskDecision",
    "RiskStatus",
    "ExecutionResult",
    "Feature",
]