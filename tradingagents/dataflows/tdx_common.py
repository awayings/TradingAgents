"""Connection, symbol resolution and retry plumbing shared by the TDX vendor.

The 通达信 (TDX) protocol is a persistent TCP feed to a quote server, not a
stateless HTTP API, so this module owns the one thing every TDX call needs: a
live connection, addressed by an explicit ``(market, code)`` pair instead of a
suffixed ticker.

Two properties of the host framework shape the design:

* LangGraph runs the tool calls of a single assistant message through a thread
  pool (``ToolNode`` uses a ``ContextThreadPoolExecutor``), so a market analyst
  emitting several ``get_indicators`` calls in one message drives this module
  from several threads at once. One ``MacClient`` owns one socket whose request
  and response frames would interleave, so every call is serialized on a shared
  lock. That serialization is also the point of the swap: it replaces the burst
  of parallel HTTP requests that got the caller rate-limited.
* ``easy_tdx`` is imported lazily. A missing library is a vendor that cannot
  serve the call, which is exactly ``VendorNotConfiguredError`` — the router
  falls through to the next configured vendor instead of failing the run.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time

import pandas as pd

from .errors import NoMarketDataError, VendorNotConfiguredError
from .symbol_utils import resolve_a_share_symbol

logger = logging.getLogger(__name__)

# Dropped connections are usually the server's idle timeout, which a reconnect
# clears; a decode/command error is not retried because it repeats identically.
_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5

# Upper bound on the bars requested in one fetch, so a mistyped start date cannot
# ask a quote server for an unbounded history. ~24 years of trading days.
_MAX_BARS = 6000

# The shared client and the lock that guards both its creation and its use.
# Re-entrant because a retry path calls the reset helper while still holding it.
_client = None
_client_lock = threading.RLock()


def require_easy_tdx():
    """Import and return the ``easy_tdx`` module, or raise the "vendor unavailable" error.

    Lazy so that importing ``tradingagents.dataflows`` does not require the
    library: an environment without it keeps working on the other vendors.
    """
    try:
        import easy_tdx
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise VendorNotConfiguredError(
            "easy_tdx is not installed, so the 'tdx' data vendor cannot serve "
            "this call. Install it (e.g. `pip install easy-tdx`) or configure "
            "another vendor in `data_vendors`."
        ) from exc
    return easy_tdx


def _reset_client() -> None:
    """Drop the cached client so the next call dials a fresh connection."""
    global _client
    client, _client = _client, None
    if client is not None:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 - close() on a dead socket is best-effort
            logger.debug("Ignoring error while closing the TDX client: %s", exc)


atexit.register(_reset_client)


def _get_client():
    """The process-wide TDX client, dialed on first use."""
    global _client
    if _client is None:
        tdx = require_easy_tdx()
        logger.info("Connecting to a TDX quote server (best host by latency).")
        _client = tdx.MacClient.from_best_host()
        _client.connect()
    return _client


def call(method: str, *args, **kwargs):
    """Invoke a ``MacClient`` method on the shared connection, reconnecting on a drop.

    Serialized on the module lock: the client holds one socket, and concurrent
    writers would interleave protocol frames.
    """
    tdx = require_easy_tdx()
    last_exc: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            with _client_lock:
                return getattr(_get_client(), method)(*args, **kwargs)
        except tdx.TdxConnectionError as exc:
            last_exc = exc
            with _client_lock:
                _reset_client()
            if attempt + 1 < _MAX_ATTEMPTS:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "TDX connection error on %s (%s); reconnecting in %.1fs "
                    "(attempt %d/%d)", method, exc, delay, attempt + 1, _MAX_ATTEMPTS,
                )
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]  # loop only exits via return or this


def resolve(symbol: str) -> tuple[object, str, str]:
    """Resolve ``symbol`` to ``(market_enum, code, canonical)`` for the TDX client.

    ``canonical`` is the ``<code>.<MARKET>`` form used in cache filenames and
    report headers. Raises ``NoMarketDataError`` for a symbol that is not
    A-share shaped, so the router can try a vendor that does cover it.
    """
    resolved = resolve_a_share_symbol(symbol)
    if resolved is None:
        raise NoMarketDataError(
            symbol, symbol,
            "not an A-share symbol — the tdx vendor covers SH/SZ/BJ listed "
            "instruments only (e.g. 600519, 600519.SH, 000001.SZ)",
        )
    market_name, code = resolved
    market = require_easy_tdx().Market[market_name]
    return market, code, f"{code}.{market_name}"


def fetch_daily_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjust: int | None = None,
) -> pd.DataFrame:
    """Daily bars covering ``[start_date, end_date]``, oldest first.

    Returns a frame with ``Date``/``Open``/``High``/``Low``/``Close``/``Volume``
    (the shape every OHLCV consumer in this package expects), or raises
    ``NoMarketDataError`` when the window holds no bars. The requested range is
    inclusive of ``end_date``, unlike yfinance's exclusive ``end``.
    """
    tdx = require_easy_tdx()
    market, code, canonical = resolve(symbol)

    # The protocol pages backwards from the newest bar, so ask for enough bars
    # to reach start_date (~5 trading days per 7 calendar days) plus a buffer for
    # holiday weeks and for start_date itself landing on a non-trading day.
    today = pd.Timestamp.today().normalize()
    start_dt = pd.Timestamp(start_date).normalize()
    calendar_days = max((today - start_dt).days, 0)
    count = min(int(calendar_days * 5 / 7) + 40, _MAX_BARS)

    frame = call(
        "get_stock_kline",
        market, code, tdx.Period.DAILY, 0, count,
        adjust=tdx.Adjust.QFQ if adjust is None else adjust,
    )

    if frame is None or frame.empty:
        raise NoMarketDataError(
            symbol, canonical, f"quote server returned no daily bars (asked for {count})"
        )

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["datetime"]),
            "Open": frame["open"],
            "High": frame["high"],
            "Low": frame["low"],
            "Close": frame["close"],
            "Volume": frame["vol"],
        }
    )

    # Clip to the requested window; the fetch deliberately overshoots on both ends.
    start_dt = pd.Timestamp(start_date).normalize()
    end_dt = pd.Timestamp(end_date).normalize()
    data = data[(data["Date"] >= start_dt) & (data["Date"] <= end_dt)]

    if data.empty:
        raise NoMarketDataError(
            symbol, canonical, f"no daily bars between {start_date} and {end_date}"
        )
    return data.reset_index(drop=True)


def quote_snapshot(symbol: str) -> dict:
    """Latest quote row for ``symbol`` as a dict, or ``{}`` when unavailable.

    Used for identity and present-day profile fields (name, valuation) — see
    ``tdx_fundamentals``, which gates this behind the point-in-time guard.
    """
    market, code, _ = resolve(symbol)
    frame = call("get_stock_quotes", [(market, code)])
    if frame is None or frame.empty:
        return {}
    return frame.iloc[0].to_dict()


# ``get_belong_board`` tags each row with a board type. The named types in
# ``easy_tdx.BoardType`` are the concept/style/region boards (3/4/5); the
# industry levels this vendor uses are carried on type 12, which the enum does
# not name (verified: 600519 -> 酿酒/白酒, 000001 -> 全国性银行/股份制银行).
_INDUSTRY_BOARD_TYPE = 12


def industry_names(symbol: str) -> list[str]:
    """Industry classification names for ``symbol``, most general first.

    Best-effort: classification is optional context for the identity anchor, so
    any failure degrades to an empty list rather than blocking the lookup. Board
    membership is a present-day classification with no historical vintage, so
    callers must gate it behind the same point-in-time rule as the rest of the
    company profile.
    """
    market, code, _ = resolve(symbol)
    try:
        frame = call("get_belong_board", market, code)
    except Exception as exc:  # noqa: BLE001 - classification is optional context
        logger.debug("Could not resolve boards for %s: %s", symbol, exc)
        return []
    if frame is None or frame.empty or "board_name" not in frame.columns:
        return []
    industry = frame[frame["board_type"] == _INDUSTRY_BOARD_TYPE]
    return [str(name) for name in industry["board_name"].tolist() if str(name).strip()]


# Readable exchange labels for the identity context, keyed by the market the
# symbol resolved to. Agents reason better about "上海证券交易所" than "SH".
_EXCHANGE_LABELS = {
    "SH": "上海证券交易所 (SSE)",
    "SZ": "深圳证券交易所 (SZSE)",
    "BJ": "北京证券交易所 (BSE)",
}


def exchange_label(symbol: str) -> str:
    """The exchange ``symbol`` lists on, or "" when it is not A-share shaped."""
    resolved = resolve_a_share_symbol(symbol)
    return _EXCHANGE_LABELS.get(resolved[0], "") if resolved else ""
