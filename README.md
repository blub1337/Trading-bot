# Autonomous OKX Crypto Trading System — Project Charter

**Repo owner:** blub1337 (to be repo: trading-engine-okx, TBD)
**Status:** Phase 0 — planning + read-only environment inspection
**Safety mode:** PAPER / SIMULATION ONLY — real-money execution disabled by default.

## Hard rules (from master directive)
- Do NOT modify VYDAS PRO repo (https://github.com/blub1337/vydaspro). Read/reuse concepts only.
- No real-money order execution until full validation chain passed.
- Never ask/note API secrets in chat; use local env config.
- Preserve risk limits; kill switch always armed; demo keys only until live approval.

## Phases (high-level)
A research → B scanner → C backtest → D acceptance → E decision pipeline →
F risk/gates → G dashboard → H paper/demo → I live (gated) → J journal/improvement →
K architecture/reliability/tests

Every phase must produce persistent artifacts and explicit acceptance criteria.
A failed stage blocks the next.

## First tasks
1. Environment inspection (tools/tools/dependencies/storage/market-data access)
2. Read OKX V5 docs + artificial instrument metadata
3. Research tradermonty/claude-trading-skills (MIT, reusable concepts only)
4. Charter risk policy, config templates, directory skeleton
5. (No code execution of orders yet.)

See `docs/ARCHITECTURE.md`, `docs/RISK.md`, `docs/ROADMAP.md`.
` :/
## CLI commands

Discover available commands and arguments with `python -m trading_engine.cli --help` or `trading-engine-okx --help` after installation. The `demo`, `backtest`, and `portfolio` commands should be checked against their own help output. `market-check` is an explicit, read-only public market-data check; inspect `python -m trading_engine.cli market-check --help` for its exact argument syntax. It may contact the public OKX service but must never submit orders. Automated tests remain offline.