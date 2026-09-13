"""The tdx vendor's news tools, backed by domestic Chinese sources.

The TDX quote protocol carries no news feed, so ticker news is the official
disclosure record instead: 上交所/深交所 filings as republished by 巨潮资讯网
(cninfo). That is a different genre from a newswire — it is what the company is
legally obliged to say, not what journalists wrote — but it is authoritative and
it is dated, which is what the look-ahead guard needs.

Market-wide news comes from the 东方财富 7x24 fast-news feed (东方财富网 7x24
快讯), a keyless, mainland-hosted headline stream — the domestic stand-in for
the Yahoo/Google news feeds this vendor replaces. It is queried cursor-wise,
newest first, and trimmed to the requested look-back window, so a backtest
never leaks headlines published after its trade date.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Annotated
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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

# 东方财富 7x24 fast news: one cursor-paged, keyless JSON feed of market-wide
# headlines. ``req_trace`` is a required request marker; any millisecond value
# serves. 100 items cover roughly half a day, so a week's window is a handful
# of pages — capped so a quiet or misbehaving feed cannot page forever.
_FASTNEWS_URL = "https://np-listapi.eastmoney.com/comm/web/getFastNewsList"
_FASTNEWS_PAGE_SIZE = 100
_FASTNEWS_MAX_PAGES = 12
_FASTNEWS_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


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


def _fast_news_page(sort_end: str, timeout: float) -> tuple[list[dict], str]:
    """One page of the 7x24 feed: ``(items, next_cursor)``.

    A failed fetch returns ``([], "")`` and logs — the caller turns an empty
    first page into an explicit unavailable message rather than an empty report
    the news analyst might read as a quiet market.
    """
    params = urlencode({
        "client": "web",
        "biz": "web_724",
        "fastColumn": "102",
        "sortEnd": sort_end,
        "pageSize": _FASTNEWS_PAGE_SIZE,
        "req_trace": int(time.time() * 1000),
    })
    req = Request(f"{_FASTNEWS_URL}?{params}", headers={"User-Agent": _FASTNEWS_UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as exc:  # noqa: BLE001 — degrade to unavailable, never raise
        logger.warning("Fast-news fetch failed: %s", exc)
        return [], ""

    data = payload.get("data") or {}
    items = data.get("fastNewsList") or []
    return items, data.get("sortEnd") or ""


def get_tdx_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Market-wide headlines from the 东方财富 7x24 fast-news feed.

    Paged newest-first and trimmed to the look-back window (look-ahead safe:
    a backtest window never sees headlines published after ``curr_date``).
    Degrades to an informative message on failure — this tool is one vendor in
    a routing chain, so it must not raise over a network blip.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - pd.Timedelta(days=look_back_days).to_pytimedelta()

    kept: list[dict] = []
    seen = 0
    cursor = ""
    for _ in range(_FASTNEWS_MAX_PAGES):
        items, cursor = _fast_news_page(cursor, timeout=10.0)
        if not items:
            break
        seen += len(items)

        for item in items:
            published = pd.to_datetime(item.get("showTime"), errors="coerce")
            pub_dt = None if pd.isna(published) else published.to_pydatetime()
            if in_window(pub_dt, start_dt, curr_dt):
                kept.append(item)
                if len(kept) >= limit:
                    break

        if len(kept) >= limit:
            break

        # Newest-first cursor feed: stop once a page reaches back past the
        # window — there is nothing older worth requesting.
        oldest = pd.to_datetime(
            [i.get("showTime") for i in items if i.get("showTime")], errors="coerce"
        ).min()
        if pd.notna(oldest) and oldest.to_pydatetime() < start_dt:
            break
        if not cursor:
            break

    start_label = start_dt.strftime("%Y-%m-%d")
    if seen == 0:
        return (
            f"Global market news is currently unavailable from the tdx vendor "
            f"(东方财富 7x24 fast-news feed could not be fetched). Do not infer "
            f"market-wide events from this message; macro context for this "
            f"window is available from get_macro_indicators."
        )
    if not kept:
        return (
            f"No global news found between {start_label} and {curr_date} "
            f"(the feed served {seen} headline(s) outside that window)"
        )

    news_str = ""
    for item in kept:
        title = str(item.get("title") or item.get("summary") or "No title").strip()
        news_str += f"### {title} (source: 东方财富网 7x24 快讯)\n"
        summary = str(item.get("summary") or "").strip()
        if summary and summary != title:
            news_str += f"{summary}\n"
        show_time = str(item.get("showTime") or "").strip()
        if show_time:
            news_str += f"Published: {show_time}\n"
        news_str += "\n"

    return f"## Global Market News, from {start_label} to {curr_date}:\n\n{news_str}"
