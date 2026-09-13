"""Eastmoney Guba (东方财富股吧) fetcher for A-share retail discussion.

Reddit and StockTwits are unreachable from mainland China without a proxy, and
even with one StockTwits is Cloudflare-challenged. Guba is the domestic
equivalent for A-shares: a per-stock investor forum whose list page is
server-rendered, so it needs no API key and no login. Each row carries the
post title, author, read count, reply count and last-update time — the same
engagement shape (read/reply ≈ score/comments) the sentiment analyst weights.

Scope is deliberately narrow: the forum covers A-share symbols only
(``list,<code>.html`` for SH/SZ/BJ codes). Anything else gets an explicit
``<unavailable>`` placeholder rather than an empty report, and a failed fetch
is reported as ``<unavailable>``, never as "no posts found" — the two are
different claims, and passing a network failure off as silence hands the
sentiment analyst a signal that was never observed (same contract as
``reddit.fetch_reddit_posts``, #1295).

Dates on the page are ``MM-DD HH:MM`` with no year, so the year is inferred
from the analysis window (previous year when the post would otherwise land in
the future). A backtest window therefore never leaks today's posts into a
historical run (#1220), and a live run still sees the forum's front page.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

from .date_window import in_window
from .symbol_utils import resolve_a_share_symbol

logger = logging.getLogger(__name__)

_LIST_URL = "https://guba.eastmoney.com/list,{code}{page}.html"
# Guba serves curl's default UA, but a browser UA keeps the server on its
# least-restrictive path as the site evolves.
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

_ROW_RE = re.compile(r'<tr class="listitem">(.*?)</tr>', re.S)
_READ_RE = re.compile(r'<div class="read">(\d+)</div>')
_REPLY_RE = re.compile(r'<div class="reply">(\d+)</div>')
_TITLE_RE = re.compile(
    r'<div class="title"><a[^>]*href="(?P<href>[^"]*)"[^>]*>(?P<title>.*?)</a>', re.S
)
_AUTHOR_RE = re.compile(r'<div class="author"><a[^>]*>(?P<author>.*?)</a>', re.S)
_UPDATE_RE = re.compile(r'<div class="update">(?P<time>\d\d-\d\d \d\d:\d\d)</div>')
_TAG_RE = re.compile(r"<[^>]+>")

# The front page alone covers roughly a day of posts on a busy stock; two more
# pages reach a quiet week without hammering the forum for content that the
# window filter would drop anyway. Parsing stops early once a page's oldest
# row predates the window.
_MAX_PAGES = 3


def _strip_tags(text: str) -> str:
    """Remove residual markup and unescape entities in a parsed text field."""
    return " ".join(html.unescape(_TAG_RE.sub(" ", text)).split())


def _parse_row(row_html: str) -> dict | None:
    """Extract one ``<tr class="listitem">`` block into a post dict.

    Returns ``None`` when the row lacks a title — the one field every post
    must have for the block to be useful.
    """
    title_m = _TITLE_RE.search(row_html)
    if not title_m:
        return None

    post: dict = {"title": _strip_tags(title_m.group("title"))}
    href = title_m.group("href")
    if href.startswith("//"):
        href = f"https:{href}"  # protocol-relative (e.g. //caifuhao.eastmoney.com/…)
    post["url"] = href if href.startswith("http") else f"https://guba.eastmoney.com{href}"

    read_m = _READ_RE.search(row_html)
    post["read_count"] = int(read_m.group(1)) if read_m else None
    reply_m = _REPLY_RE.search(row_html)
    post["reply_count"] = int(reply_m.group(1)) if reply_m else None

    author_m = _AUTHOR_RE.search(row_html)
    post["author"] = _strip_tags(author_m.group("author")) if author_m else None

    update_m = _UPDATE_RE.search(row_html)
    post["update"] = update_m.group("time") if update_m else None

    return post


def _post_time(update: str | None, end_dt: datetime) -> datetime | None:
    """Timestamp for a row's ``MM-DD HH:MM`` update label, or None.

    The label carries no year. Of the two candidate years (the window's year
    and the one before), pick whichever lands the post *closer* to the window
    end: a January window reading "12-30" gets the previous year's December,
    while a September backtest reading "09-06" keeps the same year instead of
    rolling a few-days-future row back eleven months.
    """
    if not update:
        return None
    try:
        base = datetime.strptime(update, "%m-%d %H:%M")
    except ValueError:
        return None
    this_year = base.replace(year=end_dt.year)
    last_year = base.replace(year=end_dt.year - 1)
    return this_year if abs(this_year - end_dt) <= abs(last_year - end_dt) else last_year


def _fetch_page(code: str, page: int, timeout: float) -> str | None:
    """Raw HTML of one list page, or None when the fetch fails."""
    page_suffix = "" if page == 1 else f"_{page}"
    url = _LIST_URL.format(code=code, page=page_suffix)
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read(512 * 1024 + 1)
            if len(body) > 512 * 1024:
                logger.warning("Guba page for %s exceeded cap; refusing to parse", code)
                return None
            return body.decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001 — degrade to unavailable, never raise
        logger.warning("Guba fetch failed for %s page %d: %s", code, page, exc)
        return None


def fetch_guba_posts(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Fetch recent Guba posts for an A-share ``ticker`` as a formatted block.

    ``limit`` caps the total posts returned. When ``start_date``/``end_date``
    (yyyy-mm-dd) are given, posts are trimmed to that window so a historical
    run does not leak current discussion into a backtest (#1220). Returns a
    clear ``<unavailable>`` placeholder for non-A-share symbols or failed
    fetches — never a fabricated "no posts" for a forum we could not read.
    """
    resolved = resolve_a_share_symbol(ticker)
    if resolved is None:
        return (
            f"<guba unavailable: 东方财富股吧 only covers A-share symbols; "
            f"{ticker} is not A-share shaped>"
        )
    code = resolved[1]

    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    start_dt = (
        datetime.strptime(start_date, "%Y-%m-%d") if start_date else end_dt - timedelta(days=7)
    )

    posts: list[dict] = []
    for page in range(1, _MAX_PAGES + 1):
        page_html = _fetch_page(code, page, timeout)
        if page_html is None:
            if not posts and page == 1:
                return (
                    f"<guba unavailable: fetch failed for {ticker}; this is not "
                    f"an absence of discussion>"
                )
            break

        page_posts = [p for p in (_parse_row(row) for row in _ROW_RE.findall(page_html)) if p]
        for post in page_posts:
            post["posted_dt"] = _post_time(post.get("update"), end_dt)
        if not page_posts:
            break

        kept_here = 0
        for post in page_posts:
            if in_window(post["posted_dt"], start_dt, end_dt):
                posts.append(post)
                kept_here += 1
                if len(posts) >= limit:
                    break
        if len(posts) >= limit:
            break

        # Newest-first: once a page reaches back past the window there is
        # nothing older worth requesting (a live front page reading an old
        # backtest window keeps paging through the newer-than-window rows
        # until it lands on the window itself).
        page_dts = [p["posted_dt"] for p in page_posts if p["posted_dt"] is not None]
        if page_dts and min(page_dts) < start_dt:
            break
        # Safety valve: a page that contributes nothing and carries no
        # parseable dates cannot locate the window — stop rather than page
        # through undated noise up to the cap.
        if not page_dts and kept_here == 0:
            break

    if not posts:
        return (
            f"<no Guba posts found for {ticker} between "
            f"{start_dt.strftime('%Y-%m-%d')} and {end_dt.strftime('%Y-%m-%d')}>"
        )

    lines = [
        f"guba.eastmoney.com (股吧) — {len(posts)} posts for {ticker} in window:"
    ]
    for post in posts:
        created = post["posted_dt"].strftime("%Y-%m-%d %H:%M") if post["posted_dt"] else "?"
        meta = created
        if post.get("read_count") is not None and post.get("reply_count") is not None:
            meta += f" · {post['read_count']:>4} reads · {post['reply_count']:>2} replies"
        lines.append(f"  [{meta}] {post['title']}")
        author = post.get("author")
        if author:
            lines.append(f"    author: {author}")
        if post.get("url"):
            lines.append(f"    link: {post['url']}")
    return "\n".join(lines)
