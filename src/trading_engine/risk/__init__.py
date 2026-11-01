"""Fail-closed risk and authorization gates.

Missing, invalid, stale, or inconsistent data must reject a trade.
"""
from .gates import RiskGates, RiskConfig

__all__ = ["RiskGates", "RiskConfig"]