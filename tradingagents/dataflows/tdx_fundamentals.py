"""The tdx vendor's company profile and financial-statement tools.

Two upstream sources, both keyed by the bare 6-digit code and independent of the
quote server:

* the TDX quote snapshot plus board membership for the present-day profile
  (name, valuation, share count, price limits, industry), and
* 新浪财经 for the three statements, which carry their 报告期 so a historical
  run can be served point-in-time data.

The split matters for look-ahead: the profile is a present-day snapshot with no
historical vintage, so it is withheld for a past ``curr_date`` on the same rule
every other vendor follows (``date_window.withhold_live_profile``, #1300), while
the statements stay servable because each row is dated.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

import pandas as pd

from .date_window import withhold_live_profile
from .errors import NoMarketDataError, VendorNotConfiguredError
from .tdx_common import exchange_label, industry_names, quote_snapshot, require_easy_tdx, resolve

logger = logging.getLogger(__name__)

# Which 新浪 report backs each tool, and the header label the report carries.
_REPORT_TYPES = {
    "balance_sheet": ("fzb", "Balance Sheet"),
    "cashflow": ("llb", "Cash Flow"),
    "income_statement": ("lrb", "Income Statement"),
}

# Periods requested from 新浪. Twice the quarterly count an annual request needs,
# so filtering to 12-31 periods still leaves a multi-year annual series.
_REPORT_PERIODS = 24

# Quarterly periods reported after filtering; 12 quarters = 3 years of trend.
_QUARTERLY_PERIODS = 12


def _sina_client():
    """The 新浪 report client, or a "vendor unavailable" error if its deps are missing."""
    require_easy_tdx()  # raises VendorNotConfiguredError when easy_tdx is absent
    try:
        # easy_tdx deliberately keeps these optional-feeling submodules out of its
        # top-level namespace, so they are imported by path rather than attribute.
        from easy_tdx.sina import SinaClient
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise VendorNotConfiguredError(
            "easy_tdx.sina is unavailable, so the tdx vendor cannot serve "
            "financial statements."
        ) from exc
    return SinaClient()


# How long after a period ends its report may still be unpublished, from the
# exchange disclosure calendar (《上市规则》): Q1 and the annual report by 30
# April, the half-year report by 31 August, Q3 by 31 October. The deadline is
# used rather than an estimate of the real publication date because it is the
# only instant that is guaranteed to be a no-earlier-than-actual bound — see
# ``_filter_by_report_date``.
_DISCLOSURE_DEADLINE_MONTH_DAY = {3: (4, 30), 6: (8, 31), 9: (10, 31)}


def _disclosure_deadline(period_end: pd.Timestamp) -> pd.Timestamp:
    """Latest date ``period_end``'s report may still be unpublished."""
    if period_end.month == 12:
        # The annual report is due the following April.
        return pd.Timestamp(year=period_end.year + 1, month=4, day=30)
    month, day = _DISCLOSURE_DEADLINE_MONTH_DAY.get(period_end.month, (4, 30))
    return pd.Timestamp(year=period_end.year, month=month, day=day)


def _filter_by_report_date(data: pd.DataFrame, curr_date: str | None) -> pd.DataFrame:
    """Drop statement rows whose report was not yet published on ``curr_date``.

    Statements are dated per row (``报告期``), unlike yfinance's per-column fiscal
    periods, so the look-ahead cutoff is applied down the rows here.

    The cutoff is the period's regulatory *disclosure deadline*, not its end
    date. A period ends months before its report is published — 茅台's H1 2026
    figures (period end 30 Jun) were published on 15 Aug — so cutting on the
    period end would hand a July analysis figures it could not have had. This is
    strictly more conservative than the yfinance path, which cuts on the period
    end and can leak a report published after ``curr_date``; erring toward hiding
    a figure the analyst could have seen is the safe direction for a backtest.
    """
    if not curr_date or data.empty or "报告期" not in data.columns:
        return data
    periods = pd.to_datetime(data["报告期"], errors="coerce")
    published = periods.map(
        lambda p: pd.NaT if pd.isna(p) else _disclosure_deadline(p)
    )
    return data[published <= pd.Timestamp(curr_date)]


def _drop_empty_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drop line-item columns that are empty across every remaining period.

    新浪 returns the full statutory chart of accounts, so an issue that is not a
    bank or insurer carries ~35 permanently empty columns (利息收入, 退保金,
    赔付支出净额, …). They are pure prompt noise, and removing them drops no
    value: a column that is empty for every period the vendor returned carries no
    information about this issuer.
    """
    if data.empty:
        return data
    return data.loc[:, ~data.isna().all()]


def _financial_report(
    ticker: str,
    kind: str,
    freq: str,
    curr_date: str | None,
) -> pd.DataFrame:
    """One statement for ``ticker``, filtered to the as-of date and the requested frequency."""
    report_type, label = _REPORT_TYPES[kind]
    _, code, canonical = resolve(ticker)

    try:
        data = _sina_client().get_financial_report(
            code, report_type=report_type, num=_REPORT_PERIODS
        )
    except Exception as exc:
        raise NoMarketDataError(ticker, canonical, f"新浪 {label} request failed: {exc}") from exc

    if data is None or data.empty:
        raise NoMarketDataError(ticker, canonical, f"新浪 returned no {label} periods")

    data = _filter_by_report_date(data, curr_date)

    # 新浪 returns every period it holds (quarterly and half-yearly). An annual
    # request keeps only full-year periods, which is the frequency the
    # fundamentals analyst asks for when it wants a comparable series.
    annual = str(freq).lower() != "quarterly"
    if annual and "报告期" in data.columns:
        data = data[data["报告期"].astype(str).str.endswith("-12-31")]

    # Bound the quarterly output to 3 years. The annual series needs the full
    # fetch to leave a multi-year history after the 12-31 filter, but a quarterly
    # request is usually about the recent trend and 24 periods of a 100-column
    # statement is a lot of prompt for it.
    if not annual:
        data = data.head(_QUARTERLY_PERIODS)

    data = _drop_empty_columns(data)

    if data.empty:
        window = f" at or before {curr_date}" if curr_date else ""
        raise NoMarketDataError(
            ticker, canonical, f"no {label} periods{window} for freq={freq}"
        )
    return data


def _statement_report(
    ticker: str,
    freq: str,
    curr_date: str | None,
    kind: str,
) -> str:
    """Render one statement as the framed CSV report every statement tool returns."""
    _, _, canonical = resolve(ticker)
    data = _financial_report(ticker, kind, freq, curr_date)
    _, label = _REPORT_TYPES[kind]

    header = f"# {label} data for {canonical} ({freq})\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + data.to_csv(index=False)


def get_tdx_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None,
):
    """Company profile for an A-share from the TDX quote snapshot.

    Present-day by construction — market cap, valuation and share count all move
    with the current quote — so a past ``curr_date`` withholds it through the
    shared point-in-time guard, and the agent is pointed at the dated statement
    tools instead.
    """
    _, _, canonical = resolve(ticker)

    # Guard before the request: the response would only be discarded, and the
    # answer does not depend on it.
    withheld = withhold_live_profile(curr_date, canonical)
    if withheld:
        return withheld

    quote = quote_snapshot(ticker)
    if not quote:
        raise NoMarketDataError(ticker, canonical, "quote snapshot returned no rows")

    # Only fields whose meaning was verified against the data are emitted:
    # ``buy_price_limit``/``sell_price_limit`` reproduce pre_close * 1.1 / * 0.9
    # exactly, which confirms the field alignment for this block. The snapshot
    # also carries a ``dividend_yield`` that holds the same value as ``speed_pct``
    # (a price-velocity field), so it is deliberately not reported as a yield.
    fields = [
        ("Name", quote.get("name")),
        ("Exchange", exchange_label(ticker)),
        ("Industry", " / ".join(industry_names(ticker)) or None),
        ("Market Cap (CNY)", quote.get("total_market_cap_ab")),
        ("Shares Outstanding (10k)", quote.get("total_shares")),
        ("Float Shares (10k)", quote.get("float_shares")),
        ("PE Ratio (dynamic)", quote.get("pe_dynamic")),
        ("PE Ratio (TTM)", quote.get("pe_ttm")),
        ("PE Ratio (static)", quote.get("pe_static")),
        ("Turnover Rate (%)", quote.get("turnover")),
        ("Main Net Inflow (CNY)", quote.get("main_net_amount")),
        ("Limit Up", quote.get("buy_price_limit")),
        ("Limit Down", quote.get("sell_price_limit")),
        ("Previous Close", quote.get("pre_close")),
    ]

    def _usable(value) -> bool:
        # Two values must not reach the report: NaN passes a plain truthiness
        # test and would print as the literal "nan", and a blank string would
        # print as an empty field the agent may read as "reported as nothing".
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        return not (isinstance(value, str) and not value.strip())

    def _format(value):
        # The snapshot carries single-precision floats (17.903928756713867); two
        # decimals is the quote convention and keeps the report readable.
        return round(value, 2) if isinstance(value, float) else value

    lines = [f"{label}: {_format(value)}" for label, value in fields if _usable(value)]
    if not lines:
        raise NoMarketDataError(ticker, canonical, "no usable profile fields returned")

    header = f"# Company Fundamentals for {canonical}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    lines.append(
        "Note: Market Cap is in CNY and share counts are in 万股 (10k shares). "
        "Valuation fields are a present-day snapshot. Dated figures are on the "
        "balance sheet, income statement and cash flow tools."
    )
    return header + "\n".join(lines)


def get_tdx_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
):
    """Balance sheet (资产负债表) for an A-share, one row per report period."""
    return _statement_report(ticker, freq, curr_date, "balance_sheet")


def get_tdx_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
):
    """Cash flow statement (现金流量表) for an A-share, one row per report period."""
    return _statement_report(ticker, freq, curr_date, "cashflow")


def get_tdx_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
):
    """Income statement (利润表) for an A-share, one row per report period."""
    return _statement_report(ticker, freq, curr_date, "income_statement")


def get_tdx_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
):
    """Not served by this vendor.

    The TDX quote protocol carries no insider-filing feed, and 巨潮公告 exposes
    no structured transaction records. Returns an explanatory message rather than
    raising, because "no insider filings" is a normal, non-fatal answer on this
    path — and because raising would abort a core-category call for data the
    agents can and should reason without.
    """
    _, _, canonical = resolve(ticker)
    return (
        f"Insider transaction data is not available for {canonical}: the tdx "
        f"vendor does not provide an insider-filing feed. 巨潮公告 covers "
        f"company announcements only (use get_news). Do not infer insider "
        f"activity; report that it is unavailable for this symbol."
    )
