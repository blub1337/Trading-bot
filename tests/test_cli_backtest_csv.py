"""Tests for the new ``backtest-csv`` CLI sub‑command.

The command loads a user‑provided OHLCV CSV, validates it, runs a deterministic
SMA‑crossover strategy via :class:`BacktestEngine`, and optionally writes a JSON
summary. All tests run offline – no network calls are made.
"""

from __future__ import annotations

import json
import pathlib
import pytest
from datetime import datetime, timedelta, timezone

from trading_engine.cli import main


def _make_csv(path: pathlib.Path, rows: int = 30) -> pathlib.Path:
    """Create a minimal valid OHLCV CSV for testing.

    The data consists of a simple upward trend to keep the SMA‑crossover logic
    deterministic.
    """
    import pandas as pd

    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(rows)]
    close = [1000 + i * 10 for i in range(rows)]
    data = pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in timestamps],
        "open": [c - 1 for c in close],
        "high": [c + 5 for c in close],
        "low": [c - 5 for c in close],
        "close": close,
        "volume": [100 + i for i in range(rows)],
    })
    data.to_csv(path, index=False)
    return path


def test_backtest_csv_success(tmp_path, capsys):
    csv_path = _make_csv(tmp_path / "candles.csv")
    out_path = tmp_path / "result.json"
    rc = main(["backtest-csv", str(csv_path), "--output", str(out_path)])
    assert rc == 0
    captured = capsys.readouterr()
    # Verify printed summary contains expected markers
    assert "CSV Backtest" in captured.out
    assert "SMA‑crossover" in captured.out
    # Verify JSON output exists and contains required keys
    result = json.loads(out_path.read_text())
    for key in ["initial_cash", "final_cash", "total_return_pct", "total_trades", "max_drawdown_pct", "equity_curve", "trades"]:
        assert key in result


def test_backtest_csv_missing_column(tmp_path, capsys):
    # Create a CSV that lacks the required 'high' column
    import pandas as pd
    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()],
        "open": [1000],
        "low": [990],
        "close": [1005],
        "volume": [100],
    })
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    rc = main(["backtest-csv", str(csv_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "CSV missing columns" in captured.err
