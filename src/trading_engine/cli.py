"""Command-line interface for the trading engine."""
from __future__ import annotations

import argparse
from trading_engine.marketdata import MarketDataError, OkxPublicClient
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .engine import TradingEngine
from .models import Signal, SignalSide
from .backtest import BacktestEngine, BacktestConfig
from .persistence import SQLiteStore

try:
    from .marketdata import OkxPublicClient, MarketDataError
except Exception:  # pragma: no cover - keeps CLI importable if module absent
    OkxPublicClient = None
    MarketDataError = Exception


def demo_signal() -> Signal:
    """Create a demo signal for testing."""
    return Signal(
        strategy="demo_momentum",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.75,
        rationale="Demo signal for testing",
        required_inputs=["close", "volume"],
        issued_at=datetime.now(timezone.utc),
    )


def _validate_market_snapshot(snapshot: Any, max_age: float, now: datetime) -> timedelta:
    """Return the data age and raise MarketDataError if validation fails."""
    ts = getattr(snapshot, "ts", None)
    received_at = getattr(snapshot, "received_at", None)
    if not hasattr(ts, "isoformat"):
        raise MarketDataError(f"snapshot.ts must be a datetime, got {type(ts).__name__}")
    if not hasattr(received_at, "isoformat"):
        raise MarketDataError(f"snapshot.received_at must be a datetime, got {type(received_at).__name__}")

    ts_utc = ts if getattr(ts, "tzinfo", None) is not None else ts.replace(tzinfo=timezone.utc)
    received_utc = received_at if getattr(received_at, "tzinfo", None) is not None else received_at.replace(tzinfo=timezone.utc)

    future_skew = timedelta(seconds=10)
    if ts_utc > now + future_skew:
        raise MarketDataError(f"exchange timestamp too far in future: {ts.isoformat()}")
    if received_utc > now + future_skew:
        raise MarketDataError(f"receipt timestamp too far in future: {received_at.isoformat()}")
    if ts_utc < now - timedelta(days=3650):
        raise MarketDataError(f"exchange timestamp implausibly old: {ts.isoformat()}")

    exchange_age = now - ts_utc
    receipt_age = now - received_utc
    age = exchange_age if exchange_age >= receipt_age else receipt_age
    if age > timedelta(seconds=max_age):
        raise MarketDataError(
            f"market data stale: {age.total_seconds():.3f}s exceeds {max_age:.1f}s"
        )
    if ts_utc > now:
        age = timedelta(0)
    return age


def _format_market_check(snapshot: Any, age: timedelta, max_age: float) -> int:
    """Print a successful market-check report."""
    now = datetime.now(timezone.utc)
    ts = getattr(snapshot, "ts")
    received_at = getattr(snapshot, "received_at")
    ts_utc = ts if getattr(ts, "tzinfo", None) is not None else ts.replace(tzinfo=timezone.utc)
    received_utc = received_at if getattr(received_at, "tzinfo", None) is not None else received_at.replace(tzinfo=timezone.utc)

    fresh = age <= timedelta(seconds=max_age)
    print(f"Instrument ID: {snapshot.inst_id}")
    print(f"Validated Last Price: {snapshot.last}")
    print(f"Exchange Timestamp: {ts_utc.isoformat()}")
    print(f"Local Receipt Timestamp: {received_utc.isoformat()}")
    print(f"Data Age (s): {age.total_seconds():.3f}")
    print(f"Freshness Result: {'FRESH' if fresh else 'STALE'}")
    return 0


def _validate_backtest_params(config: BacktestConfig) -> bool:
    """Validate configurable back-test parameters before construction.

    Returns ``True`` when all values are acceptable. On failure it prints a
    specific error to stderr and returns ``False``. NaN and infinity values are
    rejected explicitly.
    """

    def _fail(message: str) -> bool:
        print(message, file=sys.stderr)
        return False

    if not (isinstance(config.initial_cash, (int, float)) and
            math.isfinite(config.initial_cash) and
            config.initial_cash > 0):
        return _fail("Error: --initial-cash must be a positive finite number")

    if not (isinstance(config.sma_window, int) and
            not isinstance(config.sma_window, bool) and
            config.sma_window > 0):
        return _fail("Error: --sma-window must be a positive integer")

    if not (isinstance(config.position_fraction, (int, float)) and
            math.isfinite(config.position_fraction) and
            0 < config.position_fraction <= 1):
        return _fail("Error: --position-fraction must be in (0, 1]")

    if not (isinstance(config.max_notional, (int, float)) and
            math.isfinite(config.max_notional) and
            config.max_notional > 0):
        return _fail("Error: --max-notional must be a positive finite number")

    if not (isinstance(config.fee_bps, (int, float)) and
            math.isfinite(config.fee_bps) and
            config.fee_bps >= 0):
        return _fail("Error: --fee-bps must be a non-negative finite number")

    if not (isinstance(config.slippage_bps, (int, float)) and
            math.isfinite(config.slippage_bps) and
            config.slippage_bps >= 0):
        return _fail("Error: --slippage-bps must be a non-negative finite number")

    return True


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a demo signal through the engine."""
    engine = TradingEngine(db_path=args.db)
    signal = demo_signal()
    result = engine.process_signal(signal)
    print(f"Result: {result.status} | Filled: {result.filled} | OrderID: {result.order_id}")
    if result.fail_reason:
        print(f"Fail reason: {result.fail_reason}")
    portfolio = engine.get_portfolio_summary()
    print(f"Portfolio: {portfolio}")
    engine.shutdown()
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a simple backtest on synthetic data."""
    import json

    import numpy as np
    import pandas as pd

    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_bars = 1000
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(n_bars) * 10,
        "high": close + np.abs(np.random.randn(n_bars) * 50),
        "low": close - np.abs(np.random.randn(n_bars) * 50),
        "close": close,
        "volume": np.random.rand(n_bars) * 1000 + 100,
    })

    # Simple strategy: buy when close > SMA(20), sell when close < SMA(20)
    def strategy_fn(current_data: pd.DataFrame, bar_idx: int) -> Signal | None:
        if bar_idx < 20:
            return None
        sma = current_data["close"].rolling(20).mean().iloc[-1]
        current_close = current_data["close"].iloc[-1]
        if current_close > sma and bar_idx % 50 == 0:  # Every 50 bars for demo
            return Signal(
                strategy="sma_crossover",
                asset="BTC-USDT",
                side=SignalSide.LONG,
                conf=0.6,
                rationale=f"Close {current_close:.2f} > SMA20 {sma:.2f}",
                required_inputs=["close"],
                issued_at=current_data["timestamp"].iloc[-1],
            )
        return None

    # Build BacktestConfig from optional CLI arguments, falling back to defaults.
    cfg_kwargs: dict = {}
    if getattr(args, "initial_cash", None) is not None:
        cfg_kwargs["initial_cash"] = args.initial_cash
    if getattr(args, "fee_bps", None) is not None:
        cfg_kwargs["fee_bps"] = args.fee_bps
    if getattr(args, "slippage_bps", None) is not None:
        cfg_kwargs["slippage_bps"] = args.slippage_bps
    if getattr(args, "sma_window", None) is not None:
        cfg_kwargs["sma_window"] = args.sma_window
    if getattr(args, "position_fraction", None) is not None:
        cfg_kwargs["position_fraction"] = args.position_fraction
    if getattr(args, "max_notional", None) is not None:
        cfg_kwargs["max_notional"] = args.max_notional

    config = BacktestConfig(**cfg_kwargs)
    if not _validate_backtest_params(config):
        return 1
    engine = BacktestEngine(data, strategy_fn, config)
    result = engine.run()

    print(f"Initial Cash: ${result.initial_cash:,.2f}")
    print(f"Final Equity: ${result.final_cash:,.2f}")
    print(f"Total Return: {((result.final_cash / result.initial_cash) - 1) * 100:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning: {result.winning_trades}, Losing: {result.losing_trades}")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"Fee Assumption: {result.fee_assumption_bps} bps")
    print(f"Slippage Assumption: {result.slippage_assumption_bps} bps")
    print("NOTE: All results are SIMULATED with placeholder fee/slippage assumptions.")

    # Save equity curve
    if args.output:
        output_data = {
            "initial_cash": result.initial_cash,
            "final_cash": result.final_cash,
            "total_return_pct": ((result.final_cash / result.initial_cash) - 1) * 100,
            "total_trades": result.total_trades,
            "max_drawdown_pct": result.max_drawdown * 100,
            "equity_curve": [(ts.isoformat(), float(eq)) for ts, eq in result.equity_curve],
            "trades": result.trades,
            "simulation": True,
            "fee_assumption_bps": result.fee_assumption_bps,
            "slippage_assumption_bps": result.slippage_assumption_bps,
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to {args.output}")

    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Show current portfolio state."""
    store = SQLiteStore(args.db)
    portfolio = store.get_portfolio()
    positions = store.get_all_positions()
    print(f"Cash: ${portfolio['cash']:,.2f}")
    print("Positions:")
    for pos in positions:
        print(f"  {pos['asset']}: qty={pos['qty']:.6f}, avg_entry={pos['avg_entry_price']}")
    store.close()
    return 0


def cmd_market_check(args: argparse.Namespace) -> int:
    """Read-only public OKX ticker validation.

    This command never processes a trading signal or calls the broker.
    """
    if OkxPublicClient is None:
        print("Error: OKX public market-data client is unavailable.", file=sys.stderr)
        return 1

    inst_id = getattr(args, "inst_id", None)
    if not isinstance(inst_id, str) or not inst_id.strip():
        print("Error: instrument ID must be a non-empty string.", file=sys.stderr)
        return 1
    inst_id = inst_id.strip()

    timeout = getattr(args, "timeout", 10.0)
    max_age = getattr(args, "max_age", 60.0)
    try:
        timeout = float(timeout)
        max_age = float(max_age)
    except (TypeError, ValueError):
        print("Error: timeout and max-age must be numeric.", file=sys.stderr)
        return 1
    if timeout <= 0 or not math.isfinite(timeout):
        print("Error: timeout must be a positive finite number of seconds.", file=sys.stderr)
        return 1
    if max_age <= 0 or not math.isfinite(max_age):
        print("Error: max-age must be a positive finite number of seconds.", file=sys.stderr)
        return 1

    injected_client = getattr(args, "market_data_client", None)
    client = injected_client
    owned = client is None

    # Create a client if we own it.
    if owned:
        client = OkxPublicClient(timeout=timeout)

    try:
        # Fetch ticker.
        try:
            snapshot = client.get_ticker(inst_id)
        except Exception as exc:  # httpx/MarketDataError are already fail-closed failures
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        # Validate snapshot.
        try:
            age = _validate_market_snapshot(snapshot, max_age, datetime.now(timezone.utc))
        except MarketDataError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        return _format_market_check(snapshot, age, max_age)
    finally:
        # Close client if we created it.
        if owned:
            close = getattr(client, "close", None)
            if callable(close):
                close()

# ---------------------------------------------------------------------------
# Offline CSV backtest support (no network calls)
# ---------------------------------------------------------------------------

def _validate_candle_row(row) -> None:
    """Validate a single OHLCV row.

    ``row`` may be a pandas ``Series`` or a mapping with keys ``open``, ``high``, ``low`` and ``close``.
    Raises ``ValueError`` with a clear description if validation fails.
    """
    low = row["low"]
    high = row["high"]
    open_ = row["open"]
    close = row["close"]
    if high < low:
        raise ValueError("high < low")
    if not (low <= open_ <= high and low <= close <= high):
        raise ValueError("OHLC inconsistency")


def _load_and_validate_csv(csv_path: str):
    """Load a CSV file and perform strict OHLCV validation.

    Expected columns: ``timestamp, open, high, low, close, volume``.
    * ``timestamp`` must be parseable as ISO‑8601 (or epoch) and will be converted to a UTC ``datetime``.
    * Numeric columns are coerced to ``float`` and must be positive.
    * Duplicate timestamps are rejected.
    * Row‑wise OHLC consistency is enforced via :func:`_validate_candle_row`.
    Returns a pandas ``DataFrame`` sorted by ``timestamp``.
    """
    import pandas as pd

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    df = pd.read_csv(csv_path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")

    # Parse timestamps to UTC
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception as exc:
        raise ValueError("cannot parse timestamp column") from exc

    # Convert numeric columns and ensure positivity
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if (df[col] <= 0).any():
            raise ValueError(f"{col} contains non‑positive values")

    # Duplicate timestamps check
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps found in CSV data")

    # Row‑wise OHLC validation
    for _, r in df.iterrows():
        _validate_candle_row(r)

    # Ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def cmd_backtest_csv(args: argparse.Namespace) -> int:
    """Run a backtest on a user‑provided OHLCV CSV file.

    The CSV must contain the columns ``timestamp, open, high, low, close, volume``.
    The function validates the file, then runs the existing :class:`BacktestEngine`
    with a deterministic SMA‑crossover strategy. Results are printed to stdout
    and optionally saved as JSON via ``--output``.
    """
    import json
    import pandas as pd

    try:
        data = _load_and_validate_csv(args.csv_path)
    except Exception as exc:
        print(f"Error loading CSV: {exc}", file=sys.stderr)
        return 1

    # Simple SMA‑crossover strategy: LONG when close > SMA(window), SHORT when close < SMA(window)
    def strategy_fn(current_data: pd.DataFrame, bar_idx: int):
        if bar_idx < config.sma_window:
            return None
        sma = current_data["close"].rolling(config.sma_window).mean().iloc[-1]
        cur_close = current_data["close"].iloc[-1]
        if cur_close > sma:
            return Signal(
                strategy="sma_crossover",
                asset="CSV_BACKTEST",
                side=SignalSide.LONG,
                conf=0.6,
                rationale=f"Close {cur_close:.2f} > SMA{config.sma_window} {sma:.2f}",
                required_inputs=["close"],
                issued_at=current_data["timestamp"].iloc[-1],
            )
        elif cur_close < sma:
            return Signal(
                strategy="sma_crossover",
                asset="CSV_BACKTEST",
                side=SignalSide.SHORT,
                conf=0.6,
                rationale=f"Close {cur_close:.2f} < SMA{config.sma_window} {sma:.2f}",
                required_inputs=["close"],
                issued_at=current_data["timestamp"].iloc[-1],
            )
        return None

    # Build BacktestConfig from optional CLI arguments, falling back to defaults.
    cfg_kwargs: dict = {}
    if getattr(args, "initial_cash", None) is not None:
        cfg_kwargs["initial_cash"] = args.initial_cash
    if getattr(args, "fee_bps", None) is not None:
        cfg_kwargs["fee_bps"] = args.fee_bps
    if getattr(args, "slippage_bps", None) is not None:
        cfg_kwargs["slippage_bps"] = args.slippage_bps
    if getattr(args, "sma_window", None) is not None:
        cfg_kwargs["sma_window"] = args.sma_window
    if getattr(args, "position_fraction", None) is not None:
        cfg_kwargs["position_fraction"] = args.position_fraction
    if getattr(args, "max_notional", None) is not None:
        cfg_kwargs["max_notional"] = args.max_notional

    config = BacktestConfig(**cfg_kwargs)
    if not _validate_backtest_params(config):
        return 1
    engine = BacktestEngine(data, strategy_fn, config)
    result = engine.run()

    print(f"CSV Backtest – {len(data)} candles – Strategy: SMA‑crossover")
    print(f"Initial Cash: ${result.initial_cash:,.2f}")
    print(f"Final Equity: ${result.final_cash:,.2f}")
    print(f"Total Return: {((result.final_cash / result.initial_cash) - 1) * 100:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
    # Show configurable parameters for transparency
    print(f"SMA Window: {config.sma_window}, Position Fraction: {config.position_fraction}, Max Notional: {config.max_notional}")

    if args.output:
        out = {
            "initial_cash": result.initial_cash,
            "final_cash": result.final_cash,
            "total_return_pct": ((result.final_cash / result.initial_cash) - 1) * 100,
            "total_trades": result.total_trades,
            "max_drawdown_pct": result.max_drawdown * 100,
            "equity_curve": [(ts.isoformat(), float(eq)) for ts, eq in result.equity_curve],
            "trades": result.trades,
            # Include configuration parameters for reproducibility
            "sma_window": config.sma_window,
            "position_fraction": config.position_fraction,
            "max_notional": config.max_notional,
            "fee_assumption_bps": config.fee_bps,
            "slippage_assumption_bps": config.slippage_bps,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Results saved to {args.output}")
    return 0

# ---------------------------------------------------------------------------
# Paper‑run sub‑command implementation
# ---------------------------------------------------------------------------

class _StaticProvider:
    """A minimal market‑data provider that always returns the same snapshot.

    The CLI fetches a ticker once, validates it, and then supplies this provider
    to :class:`TradingEngine` so the engine does not make a second network
    request.
    """

    def __init__(self, snapshot: Any):
        self._snapshot = snapshot

    def get_ticker(self, _: str) -> Any:  # pragma: no cover – exercised via tests
        return self._snapshot


def cmd_paper_run(args: argparse.Namespace) -> int:
    """Execute a single synthetic paper‑trade against a public ticker.

    The flow mirrors the normal engine processing but uses a synthetic signal
    that references the validated market price. The command prints a concise
    report of validation, sizing, risk decision and final simulated outcome.
    It exits ``0`` on an approved trade and ``1`` on any rejection (including
    validation failures).
    """
    if OkxPublicClient is None:
        print("Error: OKX public market-data client is unavailable.", file=sys.stderr)
        return 1

    inst_id = getattr(args, "inst_id", None)
    if not isinstance(inst_id, str) or not inst_id.strip():
        print("Error: instrument ID must be a non‑empty string.", file=sys.stderr)
        return 1
    inst_id = inst_id.strip()

    # Parse and validate numeric arguments (same rules as market‑check).
    timeout = getattr(args, "timeout", 10.0)
    max_age = getattr(args, "max_age", 60.0)
    try:
        timeout = float(timeout)
        max_age = float(max_age)
    except (TypeError, ValueError):
        print("Error: timeout and max‑age must be numeric.", file=sys.stderr)
        return 1
    if timeout <= 0 or not math.isfinite(timeout):
        print("Error: timeout must be a positive finite number of seconds.", file=sys.stderr)
        return 1
    if max_age <= 0 or not math.isfinite(max_age):
        print("Error: max‑age must be a positive finite number of seconds.", file=sys.stderr)
        return 1

    # Validate caller‑supplied signal ID for idempotent execution.
    signal_id = getattr(args, "signal_id", None)
    if not isinstance(signal_id, str) or not signal_id.strip():
        print("Error: --signal-id must be a non‑empty string for idempotent execution.", file=sys.stderr)
        return 1
    signal_id = signal_id.strip()

    # Fetch ticker and validate, ensuring the client is closed afterwards.
    client = OkxPublicClient(timeout=timeout)
    try:
        snapshot = client.get_ticker(inst_id)
    except Exception as exc:  # pragma: no cover – exercised via tests
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Close the client after the request (no external client ownership concerns).
        close = getattr(client, "close", None)
        if callable(close):
            close()

    # Validate snapshot.
    try:
        age = _validate_market_snapshot(snapshot, max_age, datetime.now(timezone.utc))
    except MarketDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover – defensive
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Show validation result.
    _format_market_check(snapshot, age, max_age)

    # Build synthetic signal that uses the validated price.
    signal = Signal(
        strategy="paper_run",
        asset=inst_id,
        side=SignalSide.LONG,
        conf=1.0,
        rationale="Synthetic paper‑run signal",
        required_inputs=[],
        issued_at=datetime.now(timezone.utc),
        reference_price=snapshot.last,
        signal_id=signal_id,
    )

    # Wire engine with static provider to avoid a second request.
    engine = TradingEngine(db_path=args.db, market_data_provider=_StaticProvider(snapshot))
    result = engine.process_signal(signal)

    if result.status == "REJECTED":
        print(f"Paper‑run REJECTED: {result.fail_reason}")
        exit_code = 1
    else:
        print(
            f"Paper‑run APPROVED – OrderID: {result.order_id}, "
            f"Filled: {result.filled}, Status: {result.status}"
        )
        exit_code = 0

    engine.shutdown()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trading Engine CLI (Paper Mode Only)")
    parser.add_argument("--db", default="data/trading_engine.db", help="SQLite database path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo signal through the engine")
    demo_parser.set_defaults(func=cmd_demo)

    # Backtest command (synthetic data)
    bt_parser = subparsers.add_parser("backtest", help="Run backtest on synthetic data (demo)")
    bt_parser.add_argument("--output", help="Output JSON file for results")
    # New configurable parameters – optional, default to BacktestConfig defaults
    bt_parser.add_argument("--initial-cash", type=float, help="Starting capital for the backtest")
    bt_parser.add_argument("--fee-bps", type=float, help="Fee in basis points per side")
    bt_parser.add_argument("--slippage-bps", type=float, help="Slippage in basis points per side")
    bt_parser.add_argument("--sma-window", type=int, help="SMA window size for the crossover strategy")
    bt_parser.add_argument("--position-fraction", type=float, help="Fraction of cash allocated per trade")
    bt_parser.add_argument("--max-notional", type=float, help="Maximum notional amount per trade")
    bt_parser.set_defaults(func=cmd_backtest)

    # Backtest command (offline CSV data)
    btcsv_parser = subparsers.add_parser(
        "backtest-csv",
        help="Run backtest on offline OHLCV CSV data. CSV must contain columns: timestamp, open, high, low, close, volume",
    )
    btcsv_parser.add_argument("csv_path", help="Path to the historical OHLCV CSV file")
    btcsv_parser.add_argument("--output", help="Output JSON file for results")
    # Configurable parameters – same set as the synthetic backtest
    btcsv_parser.add_argument("--initial-cash", type=float, help="Starting capital for the backtest")
    btcsv_parser.add_argument("--fee-bps", type=float, help="Fee in basis points per side")
    btcsv_parser.add_argument("--slippage-bps", type=float, help="Slippage in basis points per side")
    btcsv_parser.add_argument("--sma-window", type=int, help="SMA window size for the crossover strategy")
    btcsv_parser.add_argument("--position-fraction", type=float, help="Fraction of cash allocated per trade")
    btcsv_parser.add_argument("--max-notional", type=float, help="Maximum notional amount per trade")
    btcsv_parser.set_defaults(func=cmd_backtest_csv)

    # Portfolio command
    pf_parser = subparsers.add_parser("portfolio", help="Show current portfolio")
    pf_parser.set_defaults(func=cmd_portfolio)

    # Market check command
    mc_parser = subparsers.add_parser(
        "market-check",
        help="Read-only public OKX ticker check (validated, paper-only context)",
    )
    mc_parser.add_argument("inst_id", help="Instrument ID, e.g. BTC-USDT")
    mc_parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    mc_parser.add_argument("--max-age", dest="max_age", type=float, default=60.0, help="Freshness threshold in seconds")
    mc_parser.set_defaults(func=cmd_market_check)

    # Paper‑run command
    pr_parser = subparsers.add_parser(
        "paper-run",
        help="Execute a synthetic paper‑trade against a public ticker (validated, paper‑only)",
    )
    pr_parser.add_argument("inst_id", help="Instrument ID, e.g. BTC-USDT")
    pr_parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    pr_parser.add_argument("--max-age", dest="max_age", type=float, default=60.0, help="Freshness threshold in seconds")
    pr_parser.add_argument("--signal-id", type=str, default=None, help="Stable signal ID for idempotent execution across retries")
    pr_parser.set_defaults(func=cmd_paper_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
