# Risk and Safety Controls

## Execution boundary

The implemented execution adapter is paper-only. Do not describe this repository as supporting live trading: live order placement, account access, and private OKX endpoints are not implemented.

## Fail-closed behavior

- Missing market-data provider, provider errors, malformed responses, instrument mismatches, and invalid prices must reject the signal.
- Required prices must be numeric, finite, and strictly positive.
- Exchange timestamps and local receipt timestamps are checked against the configured freshness policy and future-time tolerance.
- Order sizing must use an explicit, validated reference price and the existing sizing contract. There is no fabricated production fallback price.
- Risk gates reject signals that fail confidence, cash, position, order, or other configured checks.
- A rejected trade must never call `broker.place()`.

## Rejection audit contract

When failure occurs before a valid intent exists, the engine records the signal and a reason in `signal_rejections`, then returns a `REJECTED` result. It does not create synthetic `OrderIntent`, `RiskDecision`, or execution rows. Rejections after a genuine intent has been created follow the normal risk-decision and rejected-execution audit path.

## Market-data connectivity

Automated client and CLI tests use mocks. A green mocked test suite proves the validation behavior under those test scenarios, not live availability or production reliability of the public OKX service. The `market-check` command is an explicit, read-only connectivity check and must not invoke any trading path.

## Credentials and endpoints

Public market-data access does not require an API key. Do not add credentials for this public-only stage. Private account endpoints and live order endpoints must not be introduced as part of this work.