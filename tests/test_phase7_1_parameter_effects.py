
"""Phase 7 regression tests for configurable backtest parameters.

These tests exercise the public CLI using deterministic, synthetic OHLCV CSV data.
They do not access the network or submit orders.
"""
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


BASE_CLOSES = _make_reversal_closes()


def _write_candles(
    path: Path,
    closes: list[float] = BASE_CLOSES,
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
        "position-fraction": 0.2,
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
        "Neither final_cash nor final_equity found in result keys: "
        f"{sorted(result)}"
    )


def _trade_count(result: dict) -> int:
    if "total_trades" in result:
        return int(result["total_trades"])

    if isinstance(result.get("trades"), list):
        return len(result["trades"])

    raise AssertionError(
        "Backtest JSON does not expose total_trades or a trades list"
    )


@pytest.mark.phase7_asset
def test_initial_cash_changes_actual_simulation_accounting(
    tmp_path: Path,
) -> None:
    candles = _write_candles(tmp_path / "candles.csv")

    low_capital = _run_csv(
        tmp_path,
        "capital_1000",
        candles,
        initial_cash=1000.0,
    )
    high_capital = _run_csv(
        tmp_path,
        "capital_2000",
        candles,
        initial_cash=2000.0,
    )

    assert low_capital["initial_cash"] == 1000.0
    assert high_capital["initial_cash"] == 2000.0

    assert _trade_count(low_capital) > 0
    assert _trade_count(high_capital) > 0

    assert _final_equity(low_capital) != _final_equity(high_capital)


@pytest.mark.phase7_asset
def test_sma_window_changes_observable_simulation_outcome(
    tmp_path: Path,
) -> None:
    candles = _write_candles(tmp_path / "candles.csv")

    two_period = _run_csv(
        tmp_path,
        "sma_2",
        candles,
        sma_window=2,
    )
    five_period = _run_csv(
        tmp_path,
        "sma_5",
        candles,
        sma_window=5,
    )

    assert two_period["sma_window"] == 2
    assert five_period["sma_window"] == 5

    assert _trade_count(two_period) > 0
    assert _trade_count(five_period) > 0

    assert _final_equity(two_period) != _final_equity(five_period)


@pytest.mark.phase7_asset
def test_position_fraction_changes_actual_simulation_outcome(
    tmp_path: Path,
) -> None:
    candles = _write_candles(tmp_path / "candles.csv")

    small = _run_csv(
        tmp_path,
        "fraction_010",
        candles,
        position_fraction=0.10,
    )
    large = _run_csv(
        tmp_path,
        "fraction_050",
        candles,
        position_fraction=0.50,
    )

    assert _trade_count(small) > 0
    assert _trade_count(large) > 0

    assert small["position_fraction"] == pytest.approx(0.10)
    assert large["position_fraction"] == pytest.approx(0.50)

    assert _final_equity(small) != _final_equity(large)


@pytest.mark.phase7_asset
def test_max_notional_changes_actual_simulation_outcome(
    tmp_path: Path,
) -> None:
    candles = _write_candles(tmp_path / "candles.csv")

    low_cap = _run_csv(
        tmp_path,
        "notional_200",
        candles,
        position_fraction=0.50,
        max_notional=200.0,
    )
    high_cap = _run_csv(
        tmp_path,
        "notional_800",
        candles,
        position_fraction=0.50,
        max_notional=800.0,
    )

    assert _trade_count(low_cap) > 0
    assert _trade_count(high_cap) > 0

    assert low_cap["max_notional"] == pytest.approx(200.0)
    assert high_cap["max_notional"] == pytest.approx(800.0)

    assert _final_equity(low_cap) != _final_equity(high_cap)


@pytest.mark.parametrize(
    ("option", "invalid_value"),
    [
        ("--initial-cash", "-100"),
        ("--initial-cash", "nan"),
        ("--sma-window", "0"),
        ("--position-fraction", "1.5"),
        ("--position-fraction", "nan"),
        ("--max-notional", "-10"),
        ("--max-notional", "inf"),
        ("--fee-bps", "-1"),
        ("--fee-bps", "nan"),
        ("--slippage-bps", "-1"),
        ("--slippage-bps", "inf"),
    ],
)
@pytest.mark.parametrize("command", ["backtest", "backtest-csv"])
def test_invalid_parameters_are_rejected_before_output_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    option: str,
    invalid_value: str,
) -> None:
    candles = _write_candles(tmp_path / "candles.csv")
    output = tmp_path / (
        f"invalid_{command}_{option.lstrip('-')}_{invalid_value}.json"
    )

    argv = [command]

    if command == "backtest-csv":
        argv.append(str(candles))

    argv.extend(
        [
            "--output",
            str(output),
            option,
            invalid_value,
        ]
    )

    rc = main(argv)
    captured = capsys.readouterr()

    assert rc != 0
    assert captured.err.strip(), (
        "Validation failure should write an error to stderr"
    )
    assert not output.exists(), (
        "Invalid parameters must not create the output JSON file"
    )
