"""Thin public, read-only OKX market data adapter.

Fail closed. No credentials, no private endpoints, no account access.
All reads are validated before they leave this module.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx

OKX_PUBLIC_BASE = "https://www.okx.com"
DEFAULT_TIMEOUT_SECONDS = 10.0


class MarketDataError(Exception):
    """Raised when market data cannot be validated."""


@dataclass(frozen=True)
class TickerSnapshot:
    inst_id: str
    last: float
    bid: Optional[float]
    ask: Optional[float]
    open_24h: Optional[float]
    high_24h: Optional[float]
    low_24h: Optional[float]
    ts: datetime
    received_at: datetime

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.received_at).total_seconds()

    def is_stale(self, max_age_seconds: float, now: Optional[datetime] = None) -> bool:
        return self.age_seconds(now) > max_age_seconds


@dataclass(frozen=True)
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _require_finite_positive(value, label: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketDataError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(f):
        raise MarketDataError(f"{label} is not finite: {value!r}")
    if f <= 0:
        raise MarketDataError(f"{label} must be strictly positive, got {f}")
    return f


def _parse_ms_timestamp(value) -> datetime:
    try:
        ms = int(value)
    except (TypeError, ValueError) as exc:
        raise MarketDataError(f"invalid timestamp: {value!r}") from exc
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except (ValueError, OSError, OverflowError) as exc:
        raise MarketDataError(f"timestamp out of range: {value!r}") from exc


def _parse_optional_numeric(value, label: str) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketDataError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(f):
        raise MarketDataError(f"{label} is not finite: {value!r}")
    if f <= 0:
        raise MarketDataError(f"{label} must be strictly positive, got {f}")
    return f


class OkxPublicClient:
    """Thin wrapper over OKX public HTTP API. Read-only, no credentials."""

    def __init__(self, base_url: str = OKX_PUBLIC_BASE, timeout: float = DEFAULT_TIMEOUT_SECONDS, client: Optional[httpx.Client] = None):
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout, base_url=base_url, follow_redirects=False)
        self.timeout = timeout

    def __enter__(self) -> "OkxPublicClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, path: str, params: Optional[dict] = None) -> dict:
        try:
            resp = self._client.get(path, params=params or {})
        except httpx.TimeoutException as exc:
            raise MarketDataError(f"request timed out: {path}") from exc
        except httpx.HTTPError as exc:
            raise MarketDataError(f"request failed: {path}: {exc}") from exc

        if resp.status_code != 200:
            raise MarketDataError(f"HTTP {resp.status_code} for {path}")

        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise MarketDataError(f"invalid JSON for {path}") from exc

        if not isinstance(payload, dict):
            raise MarketDataError(f"payload is not object for {path}")
        if payload.get("code") != "0":
            raise MarketDataError(f"OKX error code={payload.get('code')!r} msg={payload.get('msg')!r}")
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise MarketDataError(f"empty or invalid data for {path}")
        return payload

    def get_ticker(self, inst_id: str) -> TickerSnapshot:
        if not inst_id or not isinstance(inst_id, str):
            raise MarketDataError("inst_id must be a non-empty string")

        payload = self._request("/api/v5/market/ticker", {"instId": inst_id})
        rows = payload["data"]
        row = rows[0]
        if not isinstance(row, dict):
            raise MarketDataError("ticker row is not object")
        listed_inst = row.get("instId")
        if listed_inst != inst_id:
            raise MarketDataError(f"instId mismatch: requested {inst_id!r}, got {listed_inst!r}")

        last = _require_finite_positive(row.get("last"), "last")
        bid = _parse_optional_numeric(row.get("bidPx"), "bidPx")
        ask = _parse_optional_numeric(row.get("askPx"), "askPx")

        open_24h = _parse_optional_numeric(row.get("open24h"), "open24h")
        high_24h = _parse_optional_numeric(row.get("high24h"), "high24h")
        low_24h = _parse_optional_numeric(row.get("low24h"), "low24h")

        ts = _parse_ms_timestamp(row.get("ts"))
        received_at = datetime.now(timezone.utc)
        return TickerSnapshot(
            inst_id=inst_id,
            last=last,
            bid=bid,
            ask=ask,
            open_24h=open_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            ts=ts,
            received_at=received_at,
        )

    def get_candles(self, inst_id: str, bar: str = "1H", limit: int = 100) -> list[Candle]:
        """Retrieve up to *limit* historical OHLCV candles for *inst_id*.

        The OKX public endpoint returns candles in reverse‑chronological order
        (newest first). This method validates each row, normalises timestamps to
        UTC ``datetime`` objects, and returns the candles **chronologically**
        (oldest → newest) to simplify downstream back‑testing.

        Parameters
        ----------
        inst_id: str – OKX instrument identifier (e.g. ``"BTC-USDT"``).
        bar: str – Candle interval string accepted by the API (e.g. ``"1H"``).
        limit: int – Number of candles to request; the API caps at 300.

        Returns
        -------
        list[Candle]
            Validated candles sorted by timestamp ascending.
        """
        if not inst_id or not isinstance(inst_id, str):
            raise MarketDataError("inst_id must be a non-empty string")
        if not bar or not isinstance(bar, str):
            raise MarketDataError("bar must be a non-empty string")
        if not isinstance(limit, int) or limit <= 0 or limit > 300:
            raise MarketDataError("limit must be int in (0, 300]")

        payload = self._request("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit})
        rows = payload["data"]
        candles: list[Candle] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                raise MarketDataError(f"invalid candle row: {row!r}")
            ts = _parse_ms_timestamp(row[0])
            o = _require_finite_positive(row[1], "open")
            h = _require_finite_positive(row[2], "high")
            l = _require_finite_positive(row[3], "low")
            c = _require_finite_positive(row[4], "close")
            vol = _require_finite_positive(row[5], "volume")
            if h < l:
                raise MarketDataError(f"high < low in candle: {row!r}")
            if not (l <= o <= h and l <= c <= h):
                raise MarketDataError(f"OHLC inconsistent in candle: {row!r}")
            candles.append(Candle(ts=ts, open=o, high=h, low=l, close=c, volume=vol))
        if not candles:
            raise MarketDataError("no candles returned")
        # The API returns newest first; sort to oldest → newest for deterministic usage.
        candles.sort(key=lambda c: c.ts)
        return candles

    def get_instruments(self, inst_type: str, inst_id: Optional[str] = None) -> list[dict]:
        params = {"instType": inst_type}
        if inst_id:
            params["instId"] = inst_id
        payload = self._request("/api/v5/public/instruments", params)
        rows = payload["data"]
        out = []
        for row in rows:
            if not isinstance(row, dict):
                raise MarketDataError(f"invalid instrument row: {row!r}")
            out.append(row)
        return out


__all__ = ["OkxPublicClient", "TickerSnapshot", "Candle", "MarketDataError", "DEFAULT_TIMEOUT_SECONDS"]
# Helper utilities
def candles_to_dataframe(candles: list[Candle]):
    """Convert a list of :class:`Candle` objects to a ``pandas.DataFrame``.

    The returned DataFrame has columns ``timestamp, open, high, low, close, volume``
    and is sorted by ``timestamp`` ascending. ``timestamp`` is timezone‑aware UTC.
    """
    import pandas as pd

    if not candles:
        raise MarketDataError("no candles to convert")
    # Ensure chronological order (the client already sorts, but be safe)
    candles = sorted(candles, key=lambda c: c.ts)
    data = {
        "timestamp": [c.ts for c in candles],
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
    }
    df = pd.DataFrame(data)
    # Ensure dtype consistency
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    return df
