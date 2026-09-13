"""The tdx vendor's implementations of the core stock and indicator tools.

Mirrors ``y_finance``'s public surface for the tools that serve A-share price
data: same signatures, same report framing, so the router and the agents are
vendor-agnostic and only the numbers' provenance changes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from .indicators import get_indicators_window
from .stockstats_utils import _assert_ohlcv_not_stale
from .tdx_common import fetch_daily_bars, resolve

logger = logging.getLogger(__name__)


def get_tdx_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    """Daily OHLCV for an A-share over an inclusive date range, as a CSV report.

    Bars are 前复权 (forward-adjusted), so the series is continuous across
    dividends and splits — the basis every indicator below assumes.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Resolving here (rather than inside fetch_daily_bars) gets the canonical
    # label for the header before any network work happens.
    _, _, canonical = resolve(symbol)

    data = fetch_daily_bars(symbol, start_date, end_date)

    # Reject a stale frame before formatting it into the report. A suspended or
    # delisted symbol returns rows that stop well short of end_date, and feeding
    # those to the agent as current prices is the #1021 failure mode.
    _assert_ohlcv_not_stale(data, end_date, symbol, canonical)

    # Two decimals is the A-share quote convention; the raw floats carry binary
    # representation noise (1275.160034) that reads as false precision.
    for col in ("Open", "High", "Low", "Close"):
        data[col] = data[col].round(2)

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = f"# Stock data for {label} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + data.set_index("Date").to_csv()


def get_tdx_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Indicator window computed over TDX 前复权 bars.

    Pins the price source to this vendor so a multi-vendor fallback chain
    reaches TDX's bars even when another vendor is configured as the default.
    """
    return get_indicators_window(
        symbol, indicator, curr_date, look_back_days, vendor="tdx"
    )
