import logging
import os
import time
from collections.abc import Callable
from typing import Annotated, NamedTuple

import pandas as pd
import yfinance as yf
from stockstats import wrap
from yfinance.exceptions import YFRateLimitError

from .config import get_config
from .symbol_utils import (
    NoMarketDataError,
    normalize_symbol,
    resolve_a_share_symbol,
)
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

# A vendor's latest OHLCV row this many calendar days before the requested date
# is treated as stale. Generous enough to span long holiday weekends, tight
# enough to catch the year-old frames yfinance occasionally returns (#1021).
MAX_OHLCV_STALE_DAYS = 10

# How long a same-day cache that does not yet reach the requested day may be
# reused before it is refetched (#1150). Short enough that an intraday run picks
# up today's close soon after it publishes, long enough that a day with no bar
# at all (weekend, holiday) cannot trigger a download on every call.
OHLCV_CACHE_TTL_SECONDS = 900


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _ensure_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the date column to ``Date``.

    Some yfinance builds leave the index unnamed (so ``reset_index()`` yields
    ``index``) or use ``Datetime`` for intraday data. Rename the first
    date-like column so indicators don't silently drop when it isn't ``Date``.
    """
    if "Date" in data.columns:
        return data
    for candidate in ("index", "Datetime", "date"):
        if candidate in data.columns:
            return data.rename(columns={candidate: "Date"})
    return data


def _local_midnight(value) -> pd.Timestamp:
    """A single timestamp as its naive, midnight-normalized local date (or NaT)."""
    if pd.isna(value):
        return pd.NaT
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)  # drop tz, keep the local wall-clock date
    return ts.normalize()


def _normalize_dates(dates) -> pd.Series:
    """Parse to naive, midnight-normalized dates so tz-aware or intraday
    timestamps compare correctly against the naive ``curr_date`` cutoff (#1201).

    Normalized per element: 5 years of yfinance bars span daylight-saving
    changes (and cache CSVs round-trip the offsets as strings), so the series can
    carry mixed UTC offsets that ``pd.to_datetime`` cannot unify without
    ``utc=True`` — which would shift non-US (positive-offset) markets to the
    previous day. Keeping each bar's own local date avoids both.
    """
    return pd.to_datetime(pd.Series(dates).map(_local_midnight))


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse/normalize dates and
    coerce prices to numeric (NaN where invalid). Dropping incomplete rows and
    filling gaps is left to ``_fill_price_gaps`` so the caller can first inspect
    the latest in-range bar (#1201)."""
    data = _ensure_date_column(data)
    data["Date"] = _normalize_dates(data["Date"])
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    return data


def _fill_price_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no close and forward/back-fill remaining price gaps so
    indicators compute on a continuous series."""
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    # copy() so a filtered (sliced) input is written to safely, not via a view.
    data = data.dropna(subset=["Close"]).copy()
    data[price_cols] = data[price_cols].ffill().bfill()
    return data


def _coerce_ohlcv_dates(data: pd.DataFrame) -> pd.Series:
    """Return parsed dates from an OHLCV frame, whether Date is a column or the index."""
    if "Date" in data.columns:
        return pd.to_datetime(data["Date"], errors="coerce").dropna()
    # yfinance keeps the dates in the index (a DatetimeIndex, sometimes unnamed).
    if isinstance(data.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(data.index, errors="coerce")).dropna()
    # Fallback: expose the index and look for any date-like column.
    df = data.reset_index()
    for col in ("Date", "Datetime", "date", "index"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                return parsed
    return pd.Series(dtype="datetime64[ns]")


def _assert_ohlcv_not_stale(
    data: pd.DataFrame,
    curr_date: str,
    symbol: str,
    canonical: str | None = None,
    *,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> None:
    """Reject OHLCV whose latest row is far older than curr_date.

    Raises NoMarketDataError (with a stale-specific detail) so the router treats
    it like any other "no usable data from this vendor" — try the next vendor,
    then emit one clear unavailable signal. Empty frames are left to the
    caller's existing no-data handling; this guards only the dangerous case of
    present-but-stale rows (a vendor returning a year-old frame that would
    otherwise feed wrong prices to the agent, #1021).
    """
    if data is None or data.empty:
        return
    requested = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(requested):
        return
    requested = requested.normalize()
    dates = _coerce_ohlcv_dates(data)
    if dates.empty:
        return
    latest = dates.max().normalize()
    stale_days = (requested - latest).days
    if stale_days > max_stale_days:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"latest row is {latest.date()}, {stale_days} days before the "
            f"requested {requested.date()} (stale) — refusing to use it",
        )


def _needs_same_day_refresh(data_file, curr_date_dt, today_date) -> bool:
    """Whether a cached frame must be refetched to reflect the requested day.

    The cache file is keyed per day, so without this a run started before the
    day's bar was final keeps serving that snapshot to every later run (#1150).
    Two distinct staleness cases exist for a current-day request: the bar may be
    missing entirely, or present but still in progress — Yahoo publishes a
    partial daily candle during market hours, whose ``Close`` is not the closing
    price. Row inspection cannot tell a partial bar from a final one, so the TTL
    governs every current-day cache. Historical requests always reuse the cache,
    since those rows are immutable.
    """
    if curr_date_dt.date() < today_date.date():
        return False
    return time.time() - os.path.getmtime(data_file) > OHLCV_CACHE_TTL_SECONDS


def _canonical_yfinance(symbol: str) -> str:
    """Resolve a broker/forex symbol (XAUUSD+ -> GC=F) to Yahoo's convention."""
    return normalize_symbol(symbol)


def _canonical_tdx(symbol: str) -> str:
    """Resolve an A-share symbol to this vendor's ``<code>.<MARKET>`` form.

    Raises ``NoMarketDataError`` for anything else, so a US ticker asked of an
    A-share-only vendor reports "no data" instead of being silently sent to a
    quote server that would answer about a different instrument.
    """
    resolved = resolve_a_share_symbol(symbol)
    if resolved is None:
        raise NoMarketDataError(
            symbol, symbol,
            "not an A-share symbol — the tdx vendor covers SH/SZ/BJ listed "
            "instruments only (e.g. 600519, 600519.SH, 000001.SZ)",
        )
    market_name, code = resolved
    return f"{code}.{market_name}"


def _fetch_ohlcv_yfinance(canonical: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Daily bars for ``canonical`` from Yahoo, dates as a ``Date`` column."""
    downloaded = yf_retry(lambda: yf.download(
        canonical,
        start=start_str,
        end=end_str,
        multi_level_index=False,
        progress=False,
        auto_adjust=True,
    ))
    return _ensure_date_column(downloaded.reset_index())


def _fetch_ohlcv_tdx(canonical: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Daily bars for ``canonical`` from a TDX quote server.

    Imported lazily so this module keeps working in an environment where
    ``easy_tdx`` is not installed and only the yfinance vendor is configured.
    """
    from .tdx_common import fetch_daily_bars

    return fetch_daily_bars(canonical, start_str, end_str)


class _OhlcvVendor(NamedTuple):
    """How to name, canonicalize and fetch the OHLCV cache for one vendor.

    ``tag`` is part of the cache filename, so two vendors never read each
    other's file — they disagree on adjusted prices (Yahoo's ``auto_adjust``
    vs TDX 前复权), and serving one's cache to the other would mix bases.
    """

    tag: str
    canonicalize: Callable[[str], str]
    fetch: Callable[[str, str, str], pd.DataFrame]


# Insertion order is the tie-breaker for the "default" sentinel (no explicit
# vendor configured) and mirrors VENDOR_METHODS order for get_indicators.
_OHLCV_VENDORS: dict[str, _OhlcvVendor] = {
    "tdx": _OhlcvVendor("TDX", _canonical_tdx, _fetch_ohlcv_tdx),
    "yfinance": _OhlcvVendor("YFin", _canonical_yfinance, _fetch_ohlcv_yfinance),
}


def resolve_ohlcv_vendor() -> str:
    """Which registered vendor backs OHLCV loads, from the ``technical_indicators`` chain.

    Indicators are computed here rather than by a vendor, so the price history
    they run on has to follow the same configured chain the router would use.
    Only vendors in ``_OHLCV_VENDORS`` can serve it; an unregistered name is
    skipped so a chain like ``"alpha_vantage"`` degrades to the default instead
    of raising.

    Note this single chain also feeds ``market_data_validator``, which verifies
    the prices the market analyst was shown. Configuring ``core_stock_apis`` and
    ``technical_indicators`` to *different* vendors therefore splits the prices
    the snapshot cross-checks from the ones ``get_stock_data`` reports; keep the
    two categories on the same vendor unless that divergence is intended.
    """
    chain = get_config().get("data_vendors", {}).get("technical_indicators", "default")
    for name in (v.strip() for v in str(chain).split(",")):
        if name in _OHLCV_VENDORS:
            return name
    return next(iter(_OHLCV_VENDORS))


def _fetch_range_yfinance(canonical: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Bars in ``[start_str, end_str)`` via the Ticker history endpoint."""
    frame = yf_retry(lambda: yf.Ticker(canonical).history(start=start_str, end=end_str))
    # Drop the tz so dates compare against a naive cutoff and print cleanly.
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return _ensure_date_column(frame.reset_index())


def _fetch_range_tdx(canonical: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Bars in ``[start_str, end_str)`` from a TDX quote server."""
    from .tdx_common import fetch_daily_bars

    # fetch_daily_bars takes an inclusive end; this registry's contract is exclusive.
    inclusive_end = (pd.Timestamp(end_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return fetch_daily_bars(canonical, start_str, inclusive_end)


# Range fetchers take an exclusive upper bound, matching yfinance's ``end``.
_RANGE_FETCHERS: dict[str, Callable[[str, str, str], pd.DataFrame]] = {
    "tdx": _fetch_range_tdx,
    "yfinance": _fetch_range_yfinance,
}


def fetch_ohlcv_range(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    vendor: str | None = None,
) -> pd.DataFrame:
    """Daily bars over the inclusive window ``[start_date, end_date]``, oldest first.

    Separate from ``load_ohlcv`` because it deliberately does the opposite thing:
    no cache and no look-ahead cutoff. Its caller is the realized-return
    settlement, which reads *forward* from a past trade date to see how the
    position actually resolved (#1251) — exactly the window ``load_ohlcv``'s
    ``curr_date`` filter would cut away.
    """
    vendor = vendor or resolve_ohlcv_vendor()
    spec = _OHLCV_VENDORS[vendor]
    canonical = spec.canonicalize(symbol)

    # Fetchers take an exclusive upper bound, so step one day past the inclusive
    # end_date to include that day's bar.
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return _RANGE_FETCHERS[vendor](canonical, start_date, end_exclusive)


def load_ohlcv(symbol: str, curr_date: str, *, vendor: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.

    ``vendor`` picks the price source; the vendor-specific implementations pass
    their own name so a multi-vendor fallback chain reaches the right backend
    when the router retries. ``None`` resolves it from configuration.
    """
    vendor = vendor or resolve_ohlcv_vendor()
    spec = _OHLCV_VENDORS[vendor]

    # Canonicalize for the chosen vendor, then reject values that would escape
    # the cache directory when interpolated into the cache filename (e.g.
    # ``../../tmp/x``).
    canonical = spec.canonicalize(symbol)
    safe_symbol = safe_ticker_component(canonical)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date).normalize()

    # Cache uses a fixed window (5y to today) so one file per symbol.
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    # yfinance ``end`` is EXCLUSIVE; request tomorrow so today's row is included
    # when curr_date is the current day (#986). Look-ahead is still prevented by
    # the curr_date filter below.
    end_str = (today_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-{spec.tag}-data-{start_str}-{end_str}.csv",
    )

    # A cached file may be empty if a prior fetch failed (unknown symbol,
    # transient rate limit). Treat an empty/columnless cache as a miss and
    # re-fetch rather than serving the poisoned file forever.
    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        # Serve the cache only when it is usable and not a stale snapshot of the
        # day being requested (#1150); otherwise fall through and refetch.
        if (
            not cached.empty
            and "Close" in cached.columns
            and not _needs_same_day_refresh(data_file, curr_date_dt, today_date)
        ):
            data = cached

    if data is None:
        downloaded = spec.fetch(canonical, start_str, end_str)
        # Only cache real data — never persist an empty frame.
        if downloaded.empty or "Close" not in downloaded.columns:
            raise NoMarketDataError(
                symbol, canonical, f"{vendor} returned no rows"
            )
        downloaded.to_csv(data_file, index=False, encoding="utf-8")
        data = downloaded

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting.
    data = data[data["Date"] <= curr_date_dt]

    # A closeless newest bar is an unsettled session, not a symbol without data.
    # _fill_price_gaps below drops it, here and mid-series alike, so the frame
    # ends at the last settled bar; only a range with no close anywhere is no
    # data (#1201, #1289).
    if not data.empty and pd.isna(data["Close"].iloc[-1]):
        settled = data["Close"].notna().to_numpy().nonzero()[0]
        if settled.size == 0:
            raise NoMarketDataError(
                symbol, canonical, "no bar in range has a closing price"
            )
        logger.warning(
            "%s: %d trailing bar(s) through %s have no closing price; using %s "
            "as the latest close.", canonical, len(data) - settled[-1] - 1,
            data["Date"].iloc[-1].date(), data["Date"].iloc[settled[-1]].date(),
        )

    data = _fill_price_gaps(data)

    # Reject a stale frame (latest row far older than curr_date) rather than
    # feeding year-old prices into indicators (#1021).
    _assert_ohlcv_not_stale(data, curr_date, symbol, canonical)

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        vendor: str | None = None,
    ):
        data = load_ohlcv(symbol, curr_date, vendor=vendor)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
