# Architecture

## Scope

`trading-engine-okx` is a local trading-engine foundation with a paper-only execution adapter. Live trading is not implemented. Automated tests use deterministic fixtures and mocked HTTP responses; passing those tests does not prove that a real public OKX connection works in the target environment.

## Main components

- `src/trading_engine/models.py` defines the typed signal, intent, risk decision, and execution-result models.
- `src/trading_engine/engine.py` coordinates signal processing, market-data validation, sizing, risk evaluation, execution, and persistence.
- `src/trading_engine/marketdata/` contains `OkxPublicClient`, which reads public ticker, candle, and instrument metadata endpoints. It does not accept API credentials.
- `src/trading_engine/risk/` contains the fail-closed risk checks.
- `src/trading_engine/execution/` contains the paper broker; it simulates results and does not submit exchange orders.
- `src/trading_engine/persistence/` contains SQLite storage and the signal-rejection audit.
- `src/trading_engine/backtest/` contains the backtest implementation.
- `src/trading_engine/cli.py` defines CLI commands, including the explicit, read-only market check.

## Signal lifecycle

1. The engine records the incoming signal.
2. It obtains a ticker snapshot from the configured market-data provider and validates the instrument, price, timestamps, and data freshness.
3. If market data is missing or invalid, the engine returns `REJECTED` and records the signal and reason in `signal_rejections`. No `OrderIntent`, `RiskDecision`, or execution row is fabricated for this pre-intent rejection, and the broker is not called.
4. With valid market data, the engine creates an order intent and evaluates the risk gates.
5. A risk-gate rejection after a genuine intent is recorded through the normal risk-decision and rejected-execution path.
6. An approved intent reaches only the paper broker. No live exchange order path is implemented.

## CLI and network behavior

`demo`, `backtest`, and `portfolio` should remain local by default. `market-check` is an explicit opt-in command that may make a public, read-only OKX GET request. Consult the CLI's `--help` output for the exact supported arguments. Automated tests must use mocked HTTP and must not contact the network.

## Persistence

The SQLite store maintains signals and the lifecycle records used by the engine, including `signal_rejections`. Early market-data rejection is intentionally represented by a signal plus its rejection audit, rather than synthetic intent, risk, or execution records.