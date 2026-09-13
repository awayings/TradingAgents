"""Tests for the Eastmoney Guba (股吧) A-share forum fetcher and the
东方财富 7x24 fast-news implementation of ``get_tdx_global_news``.

The Guba list page is server-rendered HTML; these tests feed a fixture page
shaped like the live markup and assert the parsing, the year-inference rule
for ``MM-DD HH:MM`` labels, the look-ahead window filter, and the honest
``<unavailable>`` / ``<no posts>`` distinction (#1295 contract).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.dataflows import guba, tdx_news

# Two rows copied from the live markup of guba.eastmoney.com/list,002594.html.
_SAMPLE_PAGE = """
<table>
<tbody class="listbody">
<tr class="listitem"><td><div class="read">2041</div></td><td><div class="reply">33</div></td><td><div class="title"><a data-postid="1772355550" data-posttype="1" href="/news,002594,1772355550.html">比亚迪2025年度奖金发放尚未明确</a></div></td><td><div class="author"><a href="//i.eastmoney.com/3258113638368582">比亚迪资讯</a></div></td><td><div class="update">09-13 04:42</div></td></tr>
<tr class="listitem"><td><div class="read">1651</div></td><td><div class="reply">15</div></td><td><div class="title"><a data-postid="1772333774" data-posttype="20" href="//caifuhao.eastmoney.com/news/20260913084517669457470">销量前十！定国际标准！</a></div></td><td><div class="author"><a href="//i.eastmoney.com/4898345794905954">南京通达</a></div></td><td><div class="update">09-08 08:45</div></td></tr>
</tbody>
</table>
"""


def _resp(read_fn):
    """A minimal context-manager response whose read() runs ``read_fn``."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner, size=-1):
            data = read_fn()
            return data if size is None or size < 0 else data[:size]
    return _Resp()


@pytest.mark.unit
class TestParseRow:
    def test_extracts_all_fields(self):
        rows = guba._ROW_RE.findall(_SAMPLE_PAGE)
        post = guba._parse_row(rows[0])
        assert post["title"] == "比亚迪2025年度奖金发放尚未明确"
        assert post["read_count"] == 2041
        assert post["reply_count"] == 33
        assert post["author"] == "比亚迪资讯"
        assert post["update"] == "09-13 04:42"
        assert post["url"].endswith("/news,002594,1772355550.html")

    def test_protocol_relative_caifuhao_link_is_normalized(self):
        post = guba._parse_row(guba._ROW_RE.findall(_SAMPLE_PAGE)[1])
        assert post["url"].startswith("https://caifuhao.eastmoney.com/")

    def test_row_without_title_is_dropped(self):
        assert guba._parse_row('<tr class="listitem"><td><div class="read">1</div></td></tr>') is None


@pytest.mark.unit
class TestPostTime:
    def test_year_inferred_from_window(self):
        end = guba.datetime.strptime("2026-09-13", "%Y-%m-%d")
        dt = guba._post_time("09-13 04:42", end)
        assert dt is not None and dt.year == 2026

    def test_december_label_in_january_window_steps_back_a_year(self):
        end = guba.datetime.strptime("2026-01-05", "%Y-%m-%d")
        dt = guba._post_time("12-30 10:00", end)
        assert dt is not None and dt.year == 2025

    def test_september_label_in_september_backtest_keeps_same_year(self):
        # A few days "future" beats eleven months "past" for closeness, so a
        # backtest window in September must not roll its rows back a year.
        end = guba.datetime.strptime("2026-09-02", "%Y-%m-%d")
        dt = guba._post_time("09-06 04:42", end)
        assert dt is not None and dt.year == 2026

    def test_garbage_label_returns_none(self):
        end = guba.datetime.strptime("2026-09-13", "%Y-%m-%d")
        assert guba._post_time("soon", end) is None


@pytest.mark.unit
class TestFetchGubaPosts:
    def test_non_a_share_symbol_gets_unavailable_placeholder(self):
        out = guba.fetch_guba_posts("AAPL", start_date="2026-09-06", end_date="2026-09-13")
        assert "<guba unavailable" in out

    def test_fetch_failure_reports_unavailable_not_silence(self):
        with patch.object(guba, "urlopen", side_effect=OSError("boom")):
            out = guba.fetch_guba_posts("002594.SZ", start_date="2026-09-06", end_date="2026-09-13")
        assert "<guba unavailable: fetch failed" in out

    def test_posts_are_window_filtered_and_formatted(self):
        with patch.object(guba, "urlopen", return_value=_resp(lambda: _SAMPLE_PAGE.encode("utf-8"))):
            out = guba.fetch_guba_posts("002594.SZ", start_date="2026-09-06", end_date="2026-09-13")
        # 09-13 is inside the window; 09-08 is inside too — both kept.
        assert "比亚迪2025年度奖金发放尚未明确" in out
        assert "销量前十！定国际标准！" in out
        assert "2041 reads" in out and "33 replies" in out

    def test_out_of_window_posts_reported_as_none_found(self):
        with patch.object(guba, "urlopen", return_value=_resp(lambda: _SAMPLE_PAGE.encode("utf-8"))):
            out = guba.fetch_guba_posts("002594.SZ", start_date="2026-09-01", end_date="2026-09-02")
        assert "<no Guba posts found" in out

    def test_pages_back_until_window_is_reached(self):
        # Page 1 holds only newer-than-window posts (a backtest reading the
        # forum's live front page); page 2 straddles the window and is where
        # paging stops once its oldest row predates it.
        newer = _SAMPLE_PAGE.replace("09-13 04:42", "09-06 04:42").replace("09-08 08:45", "09-05 08:45")
        in_window = _SAMPLE_PAGE.replace("09-13 04:42", "09-02 10:00").replace("09-08 08:45", "08-31 09:00")
        with patch.object(guba, "urlopen", side_effect=[
            _resp(lambda: newer.encode("utf-8")),
            _resp(lambda: in_window.encode("utf-8")),
        ]):
            out = guba.fetch_guba_posts("002594.SZ", start_date="2026-09-01", end_date="2026-09-02")
        # The 09-02 row is in the window; the 08-31 row is not, and paging
        # stops there rather than requesting a third page.
        assert "比亚迪2025年度奖金发放尚未明确" in out
        assert "销量前十！定国际标准！" not in out
        assert "<no Guba posts found" not in out


@pytest.mark.unit
class TestTdxGlobalNews:
    _FASTNEWS = (
        '{"code":"1","message":"success","data":{"sortEnd":"c2","fastNewsList":['
        '{"title":"标题一","summary":"摘要一","showTime":"2026-09-13 10:00:00"},'
        '{"title":"标题二","summary":"摘要二","showTime":"2026-09-01 10:00:00"}'
        ']}}'
    )

    def test_window_filter_and_format(self):
        with patch.object(tdx_news, "urlopen", side_effect=[
            _resp(lambda: self._FASTNEWS.encode("utf-8")),
            _resp(lambda: b'{"code":"1","message":"success","data":null}'),
        ]):
            out = tdx_news.get_tdx_global_news("2026-09-13", look_back_days=7, limit=10)
        assert "标题一" in out and "东方财富网 7x24 快讯" in out
        assert "标题二" not in out  # outside the 7-day window
        assert "Published: 2026-09-13 10:00:00" in out

    def test_empty_feed_reports_unavailable_not_silence(self):
        with patch.object(tdx_news, "urlopen", return_value=_resp(
            lambda: b'{"code":"1","message":"success","data":null}'
        )):
            out = tdx_news.get_tdx_global_news("2026-09-13", look_back_days=7, limit=10)
        assert "currently unavailable" in out

    def test_network_failure_reports_unavailable(self):
        with patch.object(tdx_news, "urlopen", side_effect=OSError("net down")):
            out = tdx_news.get_tdx_global_news("2026-09-13", look_back_days=7, limit=10)
        assert "currently unavailable" in out
