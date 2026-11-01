# Setup and Verification

## Requirements

- Python 3.12 or newer.
- Install project dependencies from `requirements.txt` or install the package in editable mode from the repository root.

## Windows PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

The `.venv/` directory should remain ignored by Git. No credentials or `.env` file are required for public market data.

## Test commands

```powershell
python -m pytest --collect-only -q
python -m pytest tests/test_cli_market_check.py -v
python -m pytest tests/ -v
```

The automated HTTP tests should remain offline and use mocked responses. Record the actual number of collected and passing tests; do not infer it from previous runs.

## CLI discovery

```powershell
python -m trading_engine.cli --help
python -m trading_engine.cli demo
python -m trading_engine.cli backtest --help
python -m trading_engine.cli portfolio --help
python -m trading_engine.cli market-check --help
```

Follow the actual argparse usage shown by `--help` for subcommands that require arguments. Do not assume the backtest or market-check argument syntax without checking the current CLI implementation.

The `demo`, `backtest`, and `portfolio` commands should remain offline unless the current code explicitly documents otherwise. `market-check` is opt-in and may contact only a public, read-only OKX market-data endpoint. It must not place orders or access private account endpoints.

## Current limitations

A passing mocked test suite does not verify real public OKX connectivity. Live trading is not implemented. Confirm these limitations against the current source when updating this document.