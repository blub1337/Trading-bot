
"""Phase 7 execution-cost regression tests using offline synthetic candles."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_engine.cli import main


def _make_reversal_closes() -> list[float]:
    """Create deterministic prices with repeated rises and declines."""
    closes = [float(value) for value in range(116, 99, -1)]
    closes.extend(float(value) for value in range(101, 126))
    closes.extend(float(value) for value in range(124, 89, -1))
    closes.extend(float(value) for value in range(91, 126))
    closes.extend(float(value) for value in range(124, 89, -1))

    # Use exactly 120 candles for reproducible tests.
    return closes[:120]


REVERSAL_CLOSES = _make_reversal_closes()


def _write_candles(
    path: Path,
    closes: list[float] = REVERSAL_CLOSES,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )
        writer.writeheader()

        previous = closes[0]
        for index, close in enumerate(closes):
            open_price = previous if index else close

            writer.writerow(
                {
                    "timestamp": (
                        start + timedelta(hours=index)
                    ).isoformat(),
                    "open": open_price,
                    "high": max(open_price, close) + 1.0,
                    "low": min(open_price, close) - 1.0,
                    "close": close,
                    "volume": 100.0,
                }
            )
            previous = close

    return path


def _run_csv(
    tmp_path: Path,
    name: str,
    candles: Path,
    **overrides: object,
) -> dict:
    options = {
        "initial-cash": 1000.0,
        "fee-bps": 0.0,
        "slippage-bps": 0.0,
        "sma-window": 5,
        "position-fraction": 0.25,
        "max-notional": 5000.0,
    }
    options.update(
        {
            key.replace("_", "-"): value
            for key, value in overrides.items()
        }
    )

    output = tmp_path / f"{name}.json"
    argv = [
        "backtest-csv",
        str(candles),
        "--output",
        str(output),
    ]

    for key, value in options.items():
        argv.extend([f"--{key}", str(value)])

    rc = main(argv)

    assert rc == 0, f"backtest-csv returned {rc} for {name}"
    assert output.is_file(), (
        f"Backtest output was not created for {name}"
    )

    return json.loads(output.read_text(encoding="utf-8"))


def _final_equity(result: dict) -> float:
    for key in ("final_cash", "final_equity"):
        if key in result:
            return float(result[key])

    raise AssertionError(
        f"No final equity field found; keys={sorted(result)}"
    )


def _trade_count(result: dict) -> int:
    if "total_trades" in result:
        return int(result["total_trades"])

    if isinstance(result.get("trades"), list):
        return len(result["trades"])

    raise AssertionError(
        "Backtest JSON does not expose total_trades or a trades list"
    )


def _trade_price_signature(result: dict) -> tuple[float, ...]:
    """Extract trade prices if the JSON exposes them."""
    signature: list[float] = []
    trades = result.get("trades", [])

    if not isinstance(trades, list):
        return ()

    price_keys = (
        "fill_price",
        "entry_price",
        "exit_price",
        "price",
    )

    for trade in trades:
        if not isinstance(trade, dict):
            continue

        for key in price_keys:
            value = trade.get(key)
            if isinstance(value, (int, float)):
                signature.append(float(value))

    return tuple(signature)


@pytest.mark.phase7_asset
def test_fees_reduce_equity_when_trades_execute(
    tmp_path: Path,
) -> None:
    candles = _write_candles(
        tmp_path / "candles.csv",
        REVERSAL_CLOSES,
    )

    no_fee = _run_csv(
        tmp_path,
        "fees_zero",
        candles,
        fee_bps=0.0,
    )
    with_fee = _run_csv(
        tmp_path,
        "fees_100bps",
        candles,
        fee_bps=100.0,
    )

    assert _trade_count(no_fee) > 0
    # With corrected daily loss accounting, fees reduce equity which can
    # cause the daily loss limit to trigger at different points, so trade
    # counts may differ. The key assertion is that fees reduce final equity.
    assert with_fee["fee_assumption_bps"] == pytest.approx(100.0)
    assert _final_equity(with_fee) < _final_equity(no_fee)


@pytest.mark.phase7_asset
def test_slippage_changes_fill_or_net_accounting(
    tmp_path: Path,
) -> None:
    candles = _write_candles(
        tmp_path / "candles.csv",
        REVERSAL_CLOSES,
    )

    no_slippage = _run_csv(
        tmp_path,
        "slippage_zero",
        candles,
        slippage_bps=0.0,
    )
    with_slippage = _run_csv(
        tmp_path,
        "slippage_100bps",
        candles,
        slippage_bps=100.0,
    )

    assert _trade_count(no_slippage) > 0
    # With corrected daily loss accounting, slippage reduces equity which can
    # cause the daily loss limit to trigger at different points, so trade
    # counts may differ. The key assertion is that slippage changes accounting.
    assert (
        with_slippage["slippage_assumption_bps"]
        == pytest.approx(100.0)
    )

    changed_prices = (
        _trade_price_signature(no_slippage)
        != _trade_price_signature(with_slippage)
    )
    changed_accounting = (
        _final_equity(no_slippage)
        != _final_equity(with_slippage)
    )

    assert changed_prices or changed_accounting, (
        "Slippage must affect fills or resulting accounting"
    )


@pytest.mark.phase7_asset
def test_combined_fee_and_slippage_change_accounting_reproducibly(
    tmp_path: Path,
) -> None:
    candles = _write_candles(
        tmp_path / "candles.csv",
        REVERSAL_CLOSES,
    )

    baseline = _run_csv(
        tmp_path,
        "costs_base",
        candles,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    combined = _run_csv(
        tmp_path,
        "costs_combined",
        candles,
        fee_bps=25.0,
        slippage_bps=25.0,
    )

    repeated = _run_csv(
        tmp_path,
        "costs_combined_repeat",
        candles,
        fee_bps=25.0,
        slippage_bps=25.0,
    )

    assert _trade_count(baseline) > 0

    assert combined["fee_assumption_bps"] == pytest.approx(25.0)
    assert (
        combined["slippage_assumption_bps"]
        == pytest.approx(25.0)
    )

    assert _final_equity(combined) != _final_equity(baseline)
    assert _final_equity(repeated) == pytest.approx(
        _final_equity(combined)
    )
