"""Deterministic historical-data backtest skeleton.

No lookahead bias. Simulated fills explicitly labeled.
Fee and slippage assumptions marked as placeholders.
"""
from .engine import BacktestEngine, BacktestConfig, BacktestResult

__all__ = ["BacktestEngine", "BacktestConfig", "BacktestResult"]