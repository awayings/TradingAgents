"""The tdx vendor's news tool, backed by 巨潮资讯网 (cninfo) announcements.

The TDX quote protocol carries no news feed, so this vendor's ticker news is the
official disclosure record instead: 上交所/深交所 filings as republished by
巨潮资讯网. That is a different genre from a newswire — it is what the company is
legally obliged to say, not what journalists wrote — but it is authoritative and
it is dated, which is what the look-ahead guard needs.

There is no market-wide equivalent: cninfo is queried per issuer, so the global
news tool reports that it is unavailable here rather than returning an empty
page the news analyst might read as a quiet market.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

import pandas as pd

from .config import get_config
from .date_window import in_window
from .errors import NoMarketDataError, VendorNotConfiguredError
from .tdx_common import require_easy_tdx, resolve

logger = logging.getLogger(__name__)

# cninfo serves at most 30 announcements per page, newest first, so covering a
# longer window means paging back. Four pages (~120 filings) reaches roughly a
# year for a large cap and bounds the request count for a quiet one.
_PAGE_SIZE = 30
_MAX_PAGES = 4


def _cninfo_client():
    """The 巨潮 announcement client, or a "vendor unavailable" error if it is missing."""
    require_easy_tdx()  # raises VendorNotConfiguredError when easy_tdx is absent
    try:
        # Imported by path: easy_tdx keeps this submodule out of its top-level
        # namespace (same reason as the 新浪 client in tdx_fundamentals).
        from easy_tdx.cninfo import CninfoClient
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise VendorNotConfiguredError(
            "easy_tdx.cninfo is unavailable, so the tdx vendor cannot serve news."
        ) from exc
    return CninfoClient()


def _announcements_in_window(
    code: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
) -> tuple[list[dict], int]:
    """Announcements inside ``[start_dt, end_dt]``, newest first.

    Returns ``(kept, seen)`` where ``seen`` counts every row the vendor returned
    regardless of the window, so the caller can tell "this issuer files nothing"
    from "nothing in this window".
    """
    client = _cninfo_client()
    kept: list[dict] = []
    seen = 0

    for page in range(1, _MAX_PAGES + 1):
        frame = client.get_announcements(code, count=_PAGE_SIZE, page=page)
        if frame is None or frame.empty:
            break
        seen += len(frame)

        for record in frame.to_dict("records"):
            published = pd.to_datetime(record.get("date"), errors="coerce")
            # An unparseable date is "undated": in_window keeps it only when the
            # window reaches the present, since a backtest cannot prove it is not
            # future content.
            pub_dt = None if pd.isna(published) else published.to_pydatetime()
            if in_window(pub_dt, start_dt, end_dt):
                kept.append(record)
                if len(kept) >= limit:
                    return kept, seen

        # Pages are newest-first, so once a page reaches back past the window
        # there is nothing older worth requesting.
        oldest = pd.to_datetime(frame["date"], errors="coerce").min()
        if pd.notna(oldest) and oldest.to_pydatetime() < start_dt:
            break

    return kept, seen


def get_tdx_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Company announcements for an A-share issuer over an inclusive window."""
    limit = get_config()["news_article_limit"]
    _, code, canonical = resolve(ticker)
    resolved = "" if canonical == ticker.upper() else f" (resolved to {canonical})"

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        kept, seen = _announcements_in_window(code, start_dt, end_dt, limit)

        if seen == 0:
            return f"No news found for {ticker}{resolved}"
        if not kept:
            return (
                f"No news found for {ticker}{resolved} between {start_date} "
                f"and {end_date} (the issuer filed {seen} announcement(s) outside "
                f"that window)"
            )

        news_str = ""
        for record in kept:
            type_label = str(record.get("type") or "").strip() or "公告"
            news_str += f"### {record.get('title', 'No title')} (source: 巨潮资讯网 {type_label})\n"
            if record.get("date"):
                news_str += f"Published: {record['date']}\n"
            if record.get("url"):
                news_str += f"Link: {record['url']}\n"
            news_str += "\n"

        return (
            f"## {ticker}{resolved} News, from {start_date} to {end_date}:\n\n{news_str}"
        )

    except NoMarketDataError:
        raise  # Not A-share shaped — let the router offer the symbol to another vendor
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


def get_tdx_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Market-wide news is not served by this vendor.

    cninfo is queried per issuer, so there is no cross-market feed to return.
    Explains the gap and points at the macro tool instead of returning an empty
    report, which the news analyst would otherwise read as an absence of events.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]

    start_dt = pd.Timestamp(curr_date) - pd.Timedelta(days=look_back_days)
    return (
        f"Global market news is not available from the tdx vendor for "
        f"{start_dt.strftime('%Y-%m-%d')} to {curr_date}: it sources ticker news "
        f"from 巨潮资讯网 filings, which are queried per issuer and have no "
        f"market-wide equivalent. Macro context for this window is available "
        f"from get_macro_indicators. Do not infer market-wide events from this "
        f"message."
    )
