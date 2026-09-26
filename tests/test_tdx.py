"""The tdx (通达信) vendor: A-share symbol resolution, routing, and its look-ahead rules.

Everything here runs offline. The quote server, 新浪 and 巨潮 clients are all
replaced at their seam, so these assert the vendor's own logic — how a symbol
becomes a ``(market, code)`` pair, how the connection is shared across the
threads LangGraph runs tool calls on, and how a historical run is kept from
seeing a report that had not been published yet.
"""

from __future__ import annotations

import threading
import time
import types

import pandas as pd
import pytest

from tradingagents.dataflows import interface, stockstats_utils, tdx_common
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorNotConfiguredError
from tradingagents.dataflows.symbol_utils import is_a_share_symbol, resolve_a_share_symbol

# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Exchange inferred from the leading digits.
        ("600519", ("SH", "600519")),
        ("601088", ("SH", "601088")),
        ("688981", ("SH", "688981")),  # 科创板
        ("000001", ("SZ", "000001")),  # 平安银行, not the SSE Composite
        ("002176", ("SZ", "002176")),
        ("300750", ("SZ", "300750")),  # 创业板
        ("900901", ("SH", "900901")),  # B 股
        ("200011", ("SZ", "200011")),  # B 股
        ("430047", ("BJ", "430047")),  # 北交所
        ("830799", ("BJ", "830799")),
        # Explicit suffix wins.
        ("600519.SH", ("SH", "600519")),
        ("600519.SS", ("SH", "600519")),  # Yahoo's Shanghai suffix
        ("000001.SZ", ("SZ", "000001")),
        ("430047.BJ", ("BJ", "430047")),
        # Explicit prefix wins.
        ("sh600519", ("SH", "600519")),
        ("SZ000001", ("SZ", "000001")),
        # Whitespace and a broker CFD marker are tolerated.
        ("  600519  ", ("SH", "600519")),
        ("600519+", ("SH", "600519")),
    ],
)
def test_resolve_a_share_symbol(raw, expected):
    assert resolve_a_share_symbol(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "AAPL", "BTC-USD", "XAUUSD", "EURUSD", "7203.T", "0700.HK",
        "", "60051", "6005190", "abcdef", "60051X",
    ],
)
def test_non_a_share_symbols_are_not_claimed(raw):
    """Everything that is not A-share shaped belongs to another vendor."""
    assert resolve_a_share_symbol(raw) is None
    assert is_a_share_symbol(raw) is False


@pytest.mark.unit
def test_index_code_needs_a_qualifier_to_beat_the_stock():
    """``000001`` is 平安银行 on SZ; the SSE Composite needs ``.SH``."""
    assert resolve_a_share_symbol("000001") == ("SZ", "000001")
    assert resolve_a_share_symbol("000001.SH") == ("SH", "000001")


# ---------------------------------------------------------------------------
# Connection sharing
# ---------------------------------------------------------------------------


@pytest.fixture()
def reset_tdx_client():
    """Drop the module-level client around each test so state cannot leak."""
    tdx_common._reset_client()
    yield
    tdx_common._reset_client()


@pytest.mark.unit
def test_calls_are_serialized_across_threads(monkeypatch, reset_tdx_client):
    """One client owns one socket, so concurrent tool calls must not interleave.

    LangGraph runs the tool calls of a single assistant message on a thread pool,
    so this is the normal path, not an edge case.
    """
    inside: list[int] = []
    overlaps: list[int] = []

    class SlowClient:
        def work(self):
            inside.append(1)
            if len(inside) > 1:
                overlaps.append(1)
            time.sleep(0.02)
            inside.pop()
            return "done"

    monkeypatch.setattr(tdx_common, "_get_client", lambda: SlowClient())

    results: list[str] = []
    threads = [
        threading.Thread(target=lambda: results.append(tdx_common.call("work")))
        for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ["done"] * 5
    assert not overlaps, "two threads were inside the client at once"


@pytest.mark.unit
def test_dropped_connection_is_redialed(monkeypatch, reset_tdx_client):
    """A connection error reconnects and retries rather than failing the call."""
    dialed: list[int] = []
    attempts: list[int] = []

    class FlakyClient:
        def work(self):
            attempts.append(1)
            # Fail on the very first attempt regardless of which client instance
            # handles it: the point is that the retry lands on a fresh dial.
            if len(attempts) == 1:
                raise tdx_common.require_easy_tdx().TdxConnectionError("dropped")
            return "recovered"

    def _dial():
        dialed.append(1)
        return FlakyClient()

    monkeypatch.setattr(tdx_common, "_get_client", _dial)

    assert tdx_common.call("work") == "recovered"
    assert len(dialed) == 2, "the dead client should have been replaced"


@pytest.mark.unit
def test_missing_library_reports_vendor_not_configured(monkeypatch):
    """An absent easy_tdx is "this vendor cannot serve it", not a crash.

    That is the distinction the router needs to fall through to another vendor.
    """

    def _absent():
        raise VendorNotConfiguredError("easy_tdx is not installed")

    monkeypatch.setattr(tdx_common, "require_easy_tdx", _absent)
    with pytest.raises(VendorNotConfiguredError):
        tdx_common.resolve("600519")


@pytest.mark.unit
def test_resolve_rejects_a_non_a_share_symbol():
    """No data (not an error) so the router can offer the symbol elsewhere."""
    with pytest.raises(NoMarketDataError):
        tdx_common.resolve("AAPL")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "method",
    [
        "get_stock_data",
        "get_indicators",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_news",
        "get_global_news",
        "get_insider_transactions",
    ],
)
def test_tdx_is_registered_for_every_served_tool(method):
    """The router must have a tdx implementation for each tool the fork serves."""
    assert "tdx" in interface.VENDOR_METHODS[method]


@pytest.mark.unit
def test_router_sends_an_a_share_to_the_tdx_vendor(monkeypatch):
    """With the shipped default config, a tool call lands on the tdx implementation."""
    seen = {}

    def fake_stock_data(symbol, start, end):
        seen["symbol"] = symbol
        return "ok"

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_stock_data"], "tdx", fake_stock_data
    )
    set_config({"data_vendors": {"core_stock_apis": "tdx"}})

    assert interface.route_to_vendor("get_stock_data", "600519", "2026-01-01", "2026-01-10") == "ok"
    assert seen["symbol"] == "600519"


@pytest.mark.unit
def test_a_non_a_share_symbol_falls_through_to_the_next_vendor(monkeypatch):
    """tdx declining a US ticker must let a configured fallback serve it."""
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_stock_data"],
        "tdx",
        lambda *a, **k: (_ for _ in ()).throw(NoMarketDataError("AAPL", "AAPL", "not A-share")),
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_stock_data"], "yfinance", lambda *a, **k: "from-yahoo"
    )
    set_config({"data_vendors": {"core_stock_apis": "tdx,yfinance"}})

    assert interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10") == "from-yahoo"


# ---------------------------------------------------------------------------
# OHLCV vendor dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ohlcv_vendor_follows_the_technical_indicators_chain():
    set_config({"data_vendors": {"technical_indicators": "yfinance"}})
    assert stockstats_utils.resolve_ohlcv_vendor() == "yfinance"

    set_config({"data_vendors": {"technical_indicators": "tdx"}})
    assert stockstats_utils.resolve_ohlcv_vendor() == "tdx"


@pytest.mark.unit
def test_ohlcv_vendor_skips_a_chain_it_cannot_fetch(monkeypatch):
    """A vendor with no OHLCV fetcher (e.g. alpha_vantage) degrades to the default."""
    set_config({"data_vendors": {"technical_indicators": "alpha_vantage"}})
    assert stockstats_utils.resolve_ohlcv_vendor() in stockstats_utils._OHLCV_VENDORS


@pytest.mark.unit
def test_load_ohlcv_uses_the_tdx_fetcher_and_its_own_cache(tmp_path, monkeypatch, reset_tdx_client):
    """The tdx path fetches A-share bars and caches them under its own tag.

    The vendor tag matters: Yahoo's ``auto_adjust`` and TDX 前复权 are different
    price bases, so the two must never read each other's cache file.
    """
    set_config({"data_cache_dir": str(tmp_path)})

    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2026-05-07", "2026-05-08"]),
        "Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0],
        "Close": [100.5, 101.5], "Volume": [1_000_000, 1_000_000],
    })
    monkeypatch.setattr(tdx_common, "fetch_daily_bars", lambda *a, **k: frame)

    out = stockstats_utils.load_ohlcv("600519", "2026-05-08", vendor="tdx")

    assert len(out) == 2
    assert list(out["Close"]) == [100.5, 101.5]
    assert [p.name for p in tmp_path.iterdir()] == [
        next(p.name for p in tmp_path.iterdir() if p.name.startswith("600519.SH-TDX-data-"))
    ]


@pytest.mark.unit
def test_load_ohlcv_rejects_a_us_symbol_for_the_tdx_vendor():
    """A-share-only means a US ticker is "no data", not a mis-priced stock."""
    with pytest.raises(NoMarketDataError):
        stockstats_utils.load_ohlcv("AAPL", "2026-05-08", vendor="tdx")


@pytest.mark.unit
def test_fetch_ohlcv_range_reads_forward_from_the_trade_date(monkeypatch):
    """The realized-return window is the one ``load_ohlcv`` would cut away."""
    captured = {}

    def fake_fetch(canonical, start, end):
        captured.update(canonical=canonical, start=start, end=end)
        return pd.DataFrame({"Date": pd.to_datetime(["2026-05-01"]), "Close": [10.0]})

    monkeypatch.setitem(stockstats_utils._RANGE_FETCHERS, "tdx", fake_fetch)
    set_config({"data_vendors": {"technical_indicators": "tdx"}})

    stockstats_utils.fetch_ohlcv_range("600519", "2026-05-01", "2026-05-08")

    assert captured["canonical"] == "600519.SH"
    assert captured["start"] == "2026-05-01"
    # Exclusive upper bound: one day past the inclusive end_date.
    assert captured["end"] == "2026-05-09"


# ---------------------------------------------------------------------------
# Fundamentals: point-in-time rules
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_quote(monkeypatch):
    """A present-day quote snapshot, so the live profile path can be exercised."""

    def _install(rows: dict | None = None, *, industry=None, exchange="上海证券交易所 (SSE)"):
        import tradingagents.dataflows.tdx_fundamentals as tdf

        monkeypatch.setattr(
            tdf, "quote_snapshot",
            lambda symbol: {"name": "贵州茅台", "pe_dynamic": 17.9, "total_market_cap_ab": 1.59e12}
            if rows is None else rows,
        )
        monkeypatch.setattr(
            tdf, "industry_names", lambda symbol: ["酿酒", "白酒"] if industry is None else industry
        )
        monkeypatch.setattr(tdf, "exchange_label", lambda symbol: exchange)

    return _install


# The live-profile tests must mean the same thing forever, so they pin the wall
# clock. ``withhold_live_profile`` compares ``curr_date`` against the real today,
# so a hardcoded date exercises the live branch only until that date arrives —
# after which the same test silently switches to asserting the withheld branch.
_TODAY = "2026-09-13"


@pytest.fixture()
def pinned_today(monkeypatch):
    from tradingagents.dataflows import date_window

    monkeypatch.setattr(date_window, "get_current_date", lambda: _TODAY)


@pytest.mark.unit
def test_fundamentals_profile_is_withheld_for_a_past_date(monkeypatch, fake_quote):
    """A live snapshot has no historical vintage, so a past run must not see it."""
    import tradingagents.dataflows.tdx_fundamentals as tdf

    fake_quote()
    called = []
    monkeypatch.setattr(tdf, "quote_snapshot", lambda symbol: called.append(1) or {})

    out = tdf.get_tdx_fundamentals("600519", "2026-01-15")

    assert "withheld" in out
    assert "2026-01-15" in out
    assert not called, "the guard must run before the request, not after"


@pytest.mark.unit
def test_fundamentals_profile_serves_the_live_snapshot(pinned_today, fake_quote):
    from tradingagents.dataflows.tdx_fundamentals import get_tdx_fundamentals

    out = get_tdx_fundamentals("600519", _TODAY)

    assert "贵州茅台" in out
    assert "酿酒 / 白酒" in out
    assert "上海证券交易所 (SSE)" in out


@pytest.mark.unit
def test_fundamentals_raises_when_the_snapshot_has_no_usable_fields(
    monkeypatch, pinned_today, fake_quote
):
    from tradingagents.dataflows.tdx_fundamentals import get_tdx_fundamentals

    fake_quote(rows={"name": None, "pe_dynamic": float("nan")}, industry=[], exchange="")

    with pytest.raises(NoMarketDataError):
        get_tdx_fundamentals("600519", _TODAY)


@pytest.mark.unit
def test_statement_rows_are_cut_at_the_disclosure_deadline(monkeypatch):
    """A period is only usable once its report was due — not once it ended.

    茅台's H1 2026 figures (period end 30 Jun) were published on 15 Aug, so a
    1 July run could not have had them. The cut is the regulatory deadline (31
    Aug), which is the earliest date guaranteed to be no-earlier-than-actual.
    """
    import tradingagents.dataflows.tdx_fundamentals as tdf

    periods = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
    frame = pd.DataFrame({"报告期": periods, "营业总收入": [1.0, 2.0, 3.0, 4.0]})
    monkeypatch.setattr(
        tdf, "_sina_client",
        lambda: types.SimpleNamespace(get_financial_report=lambda *a, **k: frame),
    )

    def periods_for(curr_date):
        data = tdf._financial_report("600519", "income_statement", "quarterly", curr_date)
        return list(data["报告期"])

    assert periods_for("2026-09-13") == periods  # all published by now
    assert periods_for("2026-07-01") == ["2026-03-31", "2025-12-31", "2025-09-30"]
    assert periods_for("2026-01-15") == ["2025-09-30"]


@pytest.mark.unit
def test_annual_frequency_keeps_only_full_year_periods(monkeypatch):
    import tradingagents.dataflows.tdx_fundamentals as tdf

    frame = pd.DataFrame({
        "报告期": ["2025-12-31", "2025-09-30", "2024-12-31"],
        "营业总收入": [1.0, 2.0, 3.0],
    })
    monkeypatch.setattr(
        tdf, "_sina_client",
        lambda: types.SimpleNamespace(get_financial_report=lambda *a, **k: frame),
    )

    annual = tdf._financial_report("600519", "income_statement", "annual", None)
    assert list(annual["报告期"]) == ["2025-12-31", "2024-12-31"]


@pytest.mark.unit
def test_empty_statement_columns_are_dropped(monkeypatch):
    """新浪 returns the full statutory chart of accounts; empty ones are noise."""
    import tradingagents.dataflows.tdx_fundamentals as tdf

    frame = pd.DataFrame({
        "报告期": ["2025-12-31"],
        "营业总收入": [1.0],
        "退保金": [None],       # insurer-only line item
        "赔付支出净额": [None],
    })
    monkeypatch.setattr(
        tdf, "_sina_client",
        lambda: types.SimpleNamespace(get_financial_report=lambda *a, **k: frame),
    )

    out = tdf._financial_report("600519", "income_statement", "quarterly", None)
    assert list(out.columns) == ["报告期", "营业总收入"]


# ---------------------------------------------------------------------------
# News: window filtering over 巨潮 announcements
# ---------------------------------------------------------------------------


def _announcement_frame(rows):
    return pd.DataFrame(rows, columns=["title", "type", "date", "url"])


@pytest.fixture()
def fake_cninfo(monkeypatch):
    """Install a 巨潮 client returning ``pages`` (one frame per requested page)."""

    def _install(pages):
        import tradingagents.dataflows.tdx_news as tdn

        requested = []

        class FakeClient:
            def get_announcements(self, code, *, count, page):
                requested.append(page)
                return pages[page - 1] if page <= len(pages) else _announcement_frame([])

        monkeypatch.setattr(tdn, "_cninfo_client", FakeClient)
        return requested

    return _install


@pytest.mark.unit
def test_news_keeps_only_announcements_inside_the_window(fake_cninfo):
    fake_cninfo([_announcement_frame([
        {"title": "半年度报告", "type": "PDF", "date": "2026-08-15", "url": "u1"},
        {"title": "第一季度报告", "type": "PDF", "date": "2026-04-20", "url": "u2"},
    ])])
    from tradingagents.dataflows.tdx_news import get_tdx_news

    out = get_tdx_news("600519", "2026-08-01", "2026-09-13")

    assert "半年度报告" in out
    assert "第一季度报告" not in out  # before the window opens


@pytest.mark.unit
def test_news_reports_an_empty_window_without_claiming_silence(fake_cninfo):
    """Nothing in-window must not read as "this issuer files nothing"."""
    fake_cninfo([_announcement_frame([
        {"title": "旧公告", "type": "PDF", "date": "2026-01-05", "url": "u"},
    ])])
    from tradingagents.dataflows.tdx_news import get_tdx_news

    out = get_tdx_news("600519", "2026-10-01", "2026-10-31")

    assert "No news found" in out
    assert "outside that window" in out


@pytest.mark.unit
def test_news_reports_when_the_issuer_has_no_announcements(fake_cninfo):
    fake_cninfo([_announcement_frame([])])
    from tradingagents.dataflows.tdx_news import get_tdx_news

    assert "No news found" in get_tdx_news("600519", "2026-08-01", "2026-09-13")


@pytest.mark.unit
def test_news_pages_back_until_the_window_start(fake_cninfo):
    """Older windows live on later pages, so a single page is not enough."""
    page_one = _announcement_frame(
        [{"title": f"近况{i}", "type": "PDF", "date": "2026-09-01", "url": "u"} for i in range(3)]
    )
    page_two = _announcement_frame([
        {"title": "目标公告", "type": "PDF", "date": "2026-02-10", "url": "u"},
    ])
    requested = fake_cninfo([page_one, page_two])

    from tradingagents.dataflows.tdx_news import get_tdx_news

    out = get_tdx_news("600519", "2026-02-01", "2026-02-28")

    assert "目标公告" in out
    # Page 2 was reached, so a single page would not have found this window.
    assert requested[:2] == [1, 2]


@pytest.mark.unit
def test_news_declines_a_non_a_share_symbol():
    """Raise so the router can offer the symbol to a vendor that covers it."""
    from tradingagents.dataflows.tdx_news import get_tdx_news

    with pytest.raises(NoMarketDataError):
        get_tdx_news("AAPL", "2026-08-01", "2026-09-13")


@pytest.mark.unit
def test_global_news_explains_the_gap_instead_of_returning_nothing():
    """A failed fast-news fetch must degrade to an explicit unavailable
    message that points at the macro tool — never an empty report the news
    analyst could read as a quiet market."""
    from unittest.mock import patch

    from tradingagents.dataflows import tdx_news
    from tradingagents.dataflows.tdx_news import get_tdx_global_news

    with patch.object(tdx_news, "urlopen", side_effect=OSError("net down")):
        out = get_tdx_global_news("2026-09-13")

    assert "currently unavailable" in out
    assert "get_macro_indicators" in out  # points at what still works


@pytest.mark.unit
def test_insider_transactions_explain_the_gap():
    from tradingagents.dataflows.tdx_fundamentals import get_tdx_insider_transactions

    out = get_tdx_insider_transactions("600519")

    assert "not available" in out
    assert "600519.SH" in out
