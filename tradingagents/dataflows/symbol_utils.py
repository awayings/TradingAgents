"""Symbol normalization and market-data error types for vendor calls.

Yahoo Finance (the default vendor) uses specific ticker conventions that
differ from the broker / TradingView / MT5 style symbols users often type:

    user types        Yahoo wants       why
    ---------------   ---------------   -----------------------------------
    XAUUSD, XAUUSD+   GC=F              gold has no forex pair on Yahoo;
                                        it is quoted as a COMEX future
    EURUSD            EURUSD=X          spot forex pairs take a ``=X`` suffix
    BTCUSD            BTC-USD           crypto pairs use a ``-`` separator
    SPX500, US500     ^GSPC             index CFDs map to Yahoo index symbols

Passing the raw broker symbol to Yahoo returns an empty result, which the
agents previously received as free text and could hallucinate a price
around (see issue #781). Centralizing the mapping here means every yfinance
entry point resolves symbols the same way, and new instruments are added by
appending a table row rather than editing call sites.
"""

from __future__ import annotations

import logging
import re

# NoMarketDataError lives in the vendor-error taxonomy (errors.py); re-exported
# here for the many call sites that import it alongside normalize_symbol.
from .errors import NoMarketDataError as NoMarketDataError

logger = logging.getLogger(__name__)


# ISO-4217 codes common enough to appear in retail forex pairs. A bare
# six-letter symbol whose halves are BOTH in this set is treated as a spot
# forex pair and given Yahoo's ``=X`` suffix.
_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

# Crypto bases that brokers quote against USD without a separator.
_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

# Explicit aliases for instruments whose broker symbol does not map to a
# Yahoo symbol by rule. Metals/energy resolve to their front-month future;
# index CFD names resolve to the underlying Yahoo index symbol. Extend by
# adding rows — no call site changes required.
_ALIASES = {
    # Precious metals (spot names -> COMEX/NYMEX futures)
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    # Energy
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    # Index CFDs -> Yahoo index symbols
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

# Yahoo symbols may contain letters, digits, and these structural characters.
_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")


# --- A-share (通达信 / TDX) symbols ------------------------------------------
#
# The TDX vendor addresses A-shares by an explicit (market, code) pair, not by
# a suffixed ticker, so a symbol has to be split before it can be queried. The
# exchange is recoverable from a bare 6-digit code's leading digits — that is
# what lets a user type ``600519`` with no qualifier. An explicit market — a
# ``.SH``/``.SS``/``.SZ``/``.BJ`` suffix or a ``SH``/``SZ``/``BJ`` prefix —
# always wins, which is the only way to reach the index codes that collide
# with a stock code (``000001`` is 平安银行 on SZ but 上证指数 on SH).
_A_SHARE_MARKET_BY_PREFIX = (
    ("6", "SH"),  # 600/601/603/605 主板, 688 科创板
    ("9", "SH"),  # 900xxx B 股
    ("4", "BJ"),  # 430xxx 北交所
    ("8", "BJ"),  # 830xxx/870xxx 北交所
    ("0", "SZ"),  # 000/001/002/003 主板
    ("2", "SZ"),  # 200xxx B 股
    ("3", "SZ"),  # 300/301 创业板
)

_A_SHARE_SUFFIXES = {".SH": "SH", ".SS": "SH", ".SZ": "SZ", ".BJ": "BJ"}
_A_SHARE_PREFIXES = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}

_A_SHARE_CODE = re.compile(r"^\d{6}$")


def resolve_a_share_symbol(raw: str) -> tuple[str, str] | None:
    """Split a user symbol into a ``(market, code)`` pair, or None if it isn't A-share shaped.

    Accepts the forms a user or another layer may hold::

        600519          -> ("SH", "600519")   exchange inferred from the code
        600519.SH       -> ("SH", "600519")
        600519.SS       -> ("SH", "600519")   Yahoo's Shanghai suffix
        000001.SZ       -> ("SZ", "000001")
        000001.SH       -> ("SH", "000001")   the SSE Composite, not 平安银行
        430047.BJ       -> ("BJ", "430047")
        sh600519        -> ("SH", "600519")   broker prefix form

    Market strings, not ``easy_tdx.Market`` enums, so this module stays purely
    syntactic and import-free — the TDX layer maps them. Returns None for
    anything else (US tickers, forex, crypto), leaving those to the Yahoo
    convention in ``normalize_symbol``.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper().rstrip("+")
    if not s:
        return None

    # Explicit market qualifier first: it is the only way to disambiguate an
    # index code from the same-numbered stock on the other exchange.
    for suffix, market in _A_SHARE_SUFFIXES.items():
        if s.endswith(suffix):
            code = s[: -len(suffix)]
            return (market, code) if _A_SHARE_CODE.match(code) else None
    if len(s) == 8 and s[:2] in _A_SHARE_PREFIXES:
        code = s[2:]
        return (_A_SHARE_PREFIXES[s[:2]], code) if _A_SHARE_CODE.match(code) else None

    # Bare code: infer the exchange from the leading digits.
    if _A_SHARE_CODE.match(s):
        for prefix, market in _A_SHARE_MARKET_BY_PREFIX:
            if s.startswith(prefix):
                return (market, s)
    return None


def is_a_share_symbol(raw: str) -> bool:
    """True when ``raw`` names an A-share (SH/SZ/BJ) symbol."""
    return resolve_a_share_symbol(raw) is not None


# Crypto quote currencies that all map to Yahoo's USD pair. Yahoo lists only
# ``<BASE>-USD`` (not the USDT/USDC stablecoin pairs), so a broker symbol quoted
# in any of these resolves to ``-USD`` (#982). Longest first so ``USDT``/``USDC``
# match before the ``USD`` substring.
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")


def crypto_base(raw: str) -> str | None:
    """Return the crypto base (e.g. ``BTC``) for a known USD/USDT/USDC-quoted
    crypto symbol in any form the pipeline may hold — ``BTC-USD``, ``BTCUSD``,
    ``BTC-USDT`` — or None for non-crypto symbols. Purely syntactic.
    """
    if not isinstance(raw, str):
        return None
    compact = raw.strip().upper().rstrip("+").replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            return base if base in _CRYPTO_BASES else None
    return None


def _normalize_crypto(s: str) -> str | None:
    """Return ``<BASE>-USD`` for a known USD/USDT/USDC-quoted crypto, else None."""
    base = crypto_base(s)
    return f"{base}-USD" if base else None


def normalize_symbol(raw: str) -> str:
    """Map a user/broker symbol to its canonical Yahoo Finance symbol.

    Resolution order (first match wins):
      1. Explicit alias table (metals, energy, index CFDs).
      2. Crypto rule: a known crypto base quoted in USD/USDT/USDC (dashed or
         not) -> ``BASE-USD``.
      3. Forex rule: six letters that are two ISO currency codes -> ``PAIR=X``.
      4. Otherwise the upper-cased symbol is returned unchanged (plain
         equities, ETFs, Yahoo-native symbols like ``GC=F`` or ``^GSPC``).

    A trailing ``+`` (broker CFD marker, e.g. ``XAUUSD+``) is stripped before
    matching. The function is purely syntactic — it performs no network
    calls — so it is safe to apply on every request.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper()
    # Broker CFD/qualifier suffixes Yahoo never uses.
    s = s.rstrip("+")

    crypto = _normalize_crypto(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    """True when ``symbol`` only contains characters Yahoo symbols use."""
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None
