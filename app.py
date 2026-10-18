"""Lowline Analytics Dashboard - app3 (two sources: MEXC + Moby).

Sources
-------
MEXC  : new listings from mexc.com; OHLCV from the public MEXC kline API.
Moby  : the app.moby.win Trending screener, ranked by Smart Money (1h),
        biggest net sellers first; OHLCV for these tokens comes from GMGN
        (with a GeckoTerminal fallback).

Why GMGN for the Moby section
-----------------------------
Moby's own chart API (web-api.mobyscreener.com/web/api_v2/tokens/v3/ohlcv)
is locked behind Cloudflare *and* a Privy bearer token, so it only answers
requests made from a logged-in browser tab. GMGN exposes the same pool candles
through a public JSON endpoint that works from a server:

    GET https://gmgn.ai/api/v1/token_kline/{chain}/{address}
        ?resolution=5m&from=<unix>&to=<unix>

    chain: sol | base | bsc | robinhood
    rows : open, close, high, low, time (ms), volume (quote/USD), amount (base)

Native resolutions are 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w. 3m is built from
1m and 2h from 1h by resampling. If GMGN is unreachable, the app falls back to
GeckoTerminal's free public OHLCV API (pool lookup + /ohlcv endpoint), which
covers solana/base/bsc but not robinhood.

The Moby universe itself cannot be fetched server-side (same Cloudflare +
bearer-token lock), so it ships as moby_snapshot.json - captured from the live
screener - and the sidebar can refresh it with a JSON export at any time.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from curl_cffi import requests as curl_requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Lowline Analytics Dashboard", layout="wide")

LISTING_API = "https://www.mexc.com/api/seo/coin/config/list"
KLINES_API = "https://api.mexc.com/api/v3/klines"
EXCHANGE_INFO_API = "https://api.mexc.com/api/v3/exchangeInfo"

# MEXC only accepts these interval strings.
TIMEFRAMES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "4h": 240,
    "1d": 1440,
    "1W": 10080,
    "1M": 43200,
}

CANDLE_LIMIT = 200
MAX_PARALLEL_REQUESTS = 8

# Strategy 1 thresholds (editable from the UI).
DEFAULT_PC_THRESHOLD = 5.0
DEFAULT_FISHER_THRESHOLD = -2.1
DEFAULT_CCI_THRESHOLD = -160.0

# Strategy 1 indicator periods.
PC_LOOKBACK = 100
FISHER_PERIOD = 70
TRIX_PERIOD = 15
CCI_PERIOD = 40

# Strategy 2 indicator periods.
PC130_LOOKBACK = 130
RSI_PERIOD = 14

# Strategy 3 defaults (must match pinecode.txt).
S3_LEFT_BARS = 8
S3_RIGHT_BARS = 3
S3_FALLBACK_LENGTH = 8
S3_BLACK_LEFT = 16
S3_BLACK_RIGHT = 2
S3_BLACK_OFFSET = -2
S3_STOP_LOSS_BUFFER = 0.985  # stop loss = support * 0.985

# --- Moby / GMGN ----------------------------------------------------------- #

SNAPSHOT_PATH = Path(__file__).with_name("moby_snapshot.json")

MOBY_TIMEFRAMES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}

# Timeframe -> (native GMGN resolution, resample factor).
MOBY_RESAMPLE: dict[str, tuple[str, int]] = {
    "1m": ("1m", 1),
    "3m": ("1m", 3),
    "5m": ("5m", 1),
    "15m": ("15m", 1),
    "30m": ("30m", 1),
    "1h": ("1h", 1),
    "2h": ("1h", 2),
    "4h": ("4h", 1),
    "12h": ("12h", 1),
    "1d": ("1d", 1),
    "1w": ("1w", 1),
}

GMGN_KLINE_API = "https://gmgn.ai/api/v1/token_kline/{chain}/{address}"
GMGN_CHAINS = {"solana": "sol", "base": "base", "bnb": "bsc", "robinhood": "robinhood"}
# GMGN's web client id. Override with the GMGN_CLIENT_ID env var if it rotates.
GMGN_CLIENT_ID = os.environ.get(
    "GMGN_CLIENT_ID", "gmgn_web_20261017-5594-69ec31fe"
)
MAX_MOBY_WORKERS = 6
MOBY_MAX_TOKENS = 150

GECKO_BASE = "https://api.geckoterminal.com/api/v2"
GECKO_NETWORKS = {"solana": "solana", "base": "base", "bnb": "bsc"}
# timeframe -> (ohlcv bucket, aggregate, resample factor)
GECKO_TF: dict[str, tuple[str, int, int]] = {
    "1m": ("minute", 1, 1),
    "3m": ("minute", 1, 3),
    "5m": ("minute", 5, 1),
    "15m": ("minute", 15, 1),
    "30m": ("minute", 15, 2),
    "1h": ("hour", 1, 1),
    "2h": ("hour", 1, 2),
    "4h": ("hour", 4, 1),
    "12h": ("hour", 12, 1),
    "1d": ("day", 1, 1),
    "1w": ("day", 7, 1),
}

MOBY_SORTS: dict[str, tuple[str | None, bool]] = {
    "Smart Money 1h (sellers first)": ("smart_money_1h_usd", False),
    "Smart Money 1h (buyers first)": ("smart_money_1h_usd", True),
    "Volume 24H (high to low)": ("volume_24h_usd", True),
    "Market cap (high to low)": ("market_cap_usd", True),
    "24H % (gainers first)": ("change_24h_pct", True),
    "Key holders (most first)": ("key_holders", True),
}

MOBY_EXPORT_SNIPPET = """(async () => {
  const raw = localStorage.getItem('privy:token') || '';
  let t = raw;
  try { const o = JSON.parse(raw); t = typeof o === 'string' ? o : (o.token || o.access_token); } catch (e) {}
  const url = 'https://web-api.mobyscreener.com/web/api_v2/tokens/screener/leaderboard/' +
    '?network=solana&networks=solana&networks=base&networks=bnb&networks=robinhood';
  const r = await fetch(url, { headers: { Authorization: 'Bearer ' + t } });
  const j = await r.json();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(j)], { type: 'application/json' }));
  a.download = 'moby_leaderboard.json';
  a.click();
})();"""


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

_thread_state = threading.local()


def _session() -> curl_requests.Session:
    """One browser-impersonating session per worker thread."""
    session = getattr(_thread_state, "session", None)
    if session is None:
        session = curl_requests.Session(impersonate="chrome120")
        session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        _thread_state.session = session
    return session


def _get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 20,
    tries: int = 3,
):
    """GET with retries for transient failures. Returns a response or None."""
    for attempt in range(tries):
        try:
            res = _session().get(
                url, params=params, headers=headers, timeout=timeout
            )
        except Exception:  # noqa: BLE001 - network issues are reported upstream
            time.sleep(0.4 * (attempt + 1))
            continue
        if res.status_code == 200:
            return res
        if res.status_code not in (429, 500, 502, 503, 504):
            return res
        time.sleep(0.6 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- #
# MEXC data layer
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=900, show_spinner=False)
def fetch_spot_pairs() -> dict[str, str]:
    """Map base asset -> tradable USDT spot pair using MEXC exchangeInfo.

    The SEO listing includes pre-market, delisted and futures-only coins, so
    resolving real pairs avoids hundreds of "Invalid symbol" responses.
    """
    res = _get(EXCHANGE_INFO_API, timeout=30)
    if res is None or res.status_code != 200:
        return {}
    try:
        payload = res.json() or {}
    except Exception:  # noqa: BLE001 - fall back to unvalidated symbols
        return {}

    pairs: dict[str, str] = {}
    for item in payload.get("symbols", []):
        if not isinstance(item, dict):
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        if str(item.get("status")) != "1":
            continue
        if not item.get("isSpotTradingAllowed", True):
            continue
        base = str(item.get("baseAsset") or "").strip().upper()
        pair = str(item.get("symbol") or "").strip().upper()
        if base and pair:
            pairs[base] = pair
    return pairs


@st.cache_data(ttl=300, show_spinner=False)
def fetch_listing_assets() -> list[dict]:
    """New listings from MEXC's SEO endpoint, resolved to real spot pairs."""
    response = _get(
        LISTING_API,
        headers={"Referer": "https://www.mexc.com/price/new-crypto"},
        timeout=20,
    )
    if response is None:
        raise RuntimeError("Could not load the MEXC asset list (network error).")
    if response.status_code != 200:
        raise RuntimeError(
            f"Could not load the MEXC asset list (HTTP {response.status_code})."
        )
    try:
        payload = response.json() or {}
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        raise RuntimeError(f"Could not parse the MEXC asset list ({exc}).") from exc

    data = payload.get("data") or []
    if isinstance(data, dict):
        data = data.get("result") or data.get("list") or []
    if not isinstance(data, list) or not data:
        raise RuntimeError("Could not load the MEXC asset list (empty response).")

    spot_pairs = fetch_spot_pairs()
    validate_pairs = bool(spot_pairs)
    rows: list[dict] = []
    for idx, item in enumerate(data[:250], 1):
        if not isinstance(item, dict):
            continue
        symbol_name = (
            item.get("symbolName") or item.get("currency") or item.get("symbol")
        )
        if not symbol_name:
            continue
        base = str(symbol_name).strip().upper()
        pair = spot_pairs.get(base)
        has_spot = pair is not None if validate_pairs else True
        if not pair:
            pair = base if base.endswith("USDT") else f"{base}USDT"
        full_name = item.get("symbolFullName") or base
        is_hidden = bool(
            item.get("hide", False)
            or item.get("state") == "HIDE"
            or not item.get("enableFetch", True)
        )
        rows.append(
            {
                "listing_rank": idx,
                "symbol": base,
                "pair": pair,
                "has_spot": has_spot,
                "full_name": full_name,
                "is_hidden": is_hidden,
                "chart_url": (
                    "https://www.mexc.com/exchange/" f"{quote(base)}_USDT"
                ),
            }
        )

    if not rows:
        raise RuntimeError("No valid symbols were found in the MEXC response.")
    return rows


@st.cache_data(ttl=60, show_spinner=False)
def get_candles(pair: str, interval: str) -> list[dict]:
    """Fetch MEXC OHLCV klines, including quote-asset volume (item[7]).

    Returns a list sorted by open time. Empty list means no usable data.
    """
    if interval not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {interval}")

    symbol = str(pair).strip().upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"

    res = _get(
        KLINES_API,
        params={"symbol": symbol, "interval": interval, "limit": CANDLE_LIMIT},
        timeout=12,
    )
    if res is None or res.status_code != 200:
        return []
    try:
        data = res.json()
    except Exception:  # noqa: BLE001 - one bad symbol must not break a scan
        return []

    if not isinstance(data, list) or not data:
        return []

    bars: list[dict] = []
    for item in data:
        try:
            close = float(item[4])
            base_volume = float(item[5])
            # item[7] is quote-asset volume for MEXC/Binance-compatible klines.
            quote_volume = (
                float(item[7]) if len(item) > 7 and item[7] is not None else 0.0
            )
            if quote_volume <= 0:
                quote_volume = base_volume * close
            bars.append(
                {
                    "t": int(item[0]),
                    "o": float(item[1]),
                    "h": float(item[2]),
                    "l": float(item[3]),
                    "c": close,
                    "v": base_volume,
                    "qv": quote_volume,
                }
            )
        except (TypeError, ValueError, IndexError):
            continue

    bars.sort(key=lambda bar: bar["t"])
    return bars


def estimate_24h_quote_volume(
    pair: str, interval: str, bars: list[dict]
) -> float:
    """Real 24h USDT volume, independent of the selected timeframe."""
    minutes = TIMEFRAMES[interval]
    if minutes <= 1440:
        bars_needed = max(1, int(round(1440 / minutes)))
        return float(sum(bar["qv"] for bar in bars[-bars_needed:]))

    daily = get_candles(pair, "1d")
    if not daily:
        return 0.0
    # The last daily candle is the best 24h approximation on weekly/monthly.
    return float(daily[-1]["qv"])


# --------------------------------------------------------------------------- #
# Moby data layer
# --------------------------------------------------------------------------- #

def _pick(mapping, key: str, default=0.0):
    """Read a nested timeframe dict, e.g. whale_net_flow_usd["h1"]."""
    if isinstance(mapping, dict):
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_moby_tokens(payload: dict) -> list[dict]:
    """Accept either the bundled snapshot or a raw leaderboard API export."""
    if not isinstance(payload, dict):
        return []

    raw = payload.get("tokens")
    if isinstance(raw, list):
        tokens = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("address"):
                continue
            tokens.append(
                {
                    "symbol": str(item.get("symbol") or "?"),
                    "network": str(item.get("network") or ""),
                    "address": str(item.get("address")),
                    "launchpad": str(item.get("launchpad") or ""),
                    "key_holders": _int(item.get("key_holders")),
                    "activity_24h": _int(item.get("activity_24h")),
                    "market_cap_usd": _num(item.get("market_cap_usd")),
                    "change_24h_pct": _num(item.get("change_24h_pct")),
                    "price_usd": _num(item.get("price_usd")),
                    "volume_24h_usd": _num(item.get("volume_24h_usd")),
                    "smart_money_1h_usd": _num(item.get("smart_money_1h_usd")),
                    "whale_buys_1h": _int(item.get("whale_buys_1h")),
                    "whale_sells_1h": _int(item.get("whale_sells_1h")),
                    "created": str(item.get("created") or ""),
                    "liquidity_usd": _num(item.get("liquidity_usd")),
                }
            )
        return tokens

    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []

    tokens = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("token_address"):
            continue
        launchpad = entry.get("launchpad")
        if isinstance(launchpad, dict):
            launchpad = launchpad.get("name") or ""
        tokens.append(
            {
                "symbol": str(entry.get("token_symbol") or "?"),
                "network": str(entry.get("network") or ""),
                "address": str(entry.get("token_address")),
                "launchpad": str(launchpad or ""),
                "key_holders": _int(entry.get("key_holder_count")),
                "activity_24h": _int(_pick(entry.get("whale_count"), "h24")),
                "market_cap_usd": _num(entry.get("market_cap_usd")),
                "change_24h_pct": _num(
                    _pick(entry.get("price_change_percent"), "h24")
                ),
                "price_usd": _num(entry.get("price_usd")),
                "volume_24h_usd": _num(_pick(entry.get("volume_usd"), "h24")),
                "smart_money_1h_usd": _num(
                    _pick(entry.get("whale_net_flow_usd"), "h1")
                ),
                "whale_buys_1h": _int(_pick(entry.get("whale_buys_count"), "h1")),
                "whale_sells_1h": _int(
                    _pick(entry.get("whale_sells_count"), "h1")
                ),
                "created": str(entry.get("token_created") or ""),
                "liquidity_usd": _num(entry.get("liquidity_usd")),
            }
        )
    tokens.sort(key=lambda item: item["smart_money_1h_usd"])
    return tokens


@st.cache_data(show_spinner=False)
def load_snapshot_file() -> dict:
    """Bundled Moby universe, captured from the live screener."""
    try:
        with open(SNAPSHOT_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001 - the UI reports an empty universe
        return {"captured_at": None, "tokens": []}
    return {
        "captured_at": payload.get("captured_at_utc"),
        "tokens": normalize_moby_tokens(payload),
    }


def fmt_age(created: str) -> str:
    """Token age label in Moby's style (2y, 3mo, 3w, 11h)."""
    if not created:
        return "-"
    try:
        moment = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except ValueError:
        return "-"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - moment).total_seconds()
    if seconds < 0:
        return "new"
    days = seconds / 86400.0
    if days >= 365:
        return f"{int(days // 365)}y"
    if days >= 30:
        return f"{int(days // 30)}mo"
    if days >= 7:
        return f"{int(days // 7)}w"
    if days >= 1:
        return f"{int(days)}d"
    return f"{int(seconds // 3600)}h"


def moby_token_url(token: dict) -> str:
    return (
        "https://app.moby.win/token"
        f"?address={quote(str(token.get('address') or ''))}"
        f"&symbol={quote(str(token.get('symbol') or ''))}"
        f"&network={quote(str(token.get('network') or ''))}"
    )


def resample_bars(bars: list[dict], factor: int, base_minutes: int) -> list[dict]:
    """Aggregate N consecutive candles into one (3m from 1m, 2h from 1h)."""
    if factor <= 1 or not bars:
        return bars
    bucket_ms = base_minutes * 60 * 1000 * factor
    out: list[dict] = []
    current: dict | None = None
    for bar in bars:
        bucket = bar["t"] - (bar["t"] % bucket_ms)
        if current is None or current["t"] != bucket:
            if current is not None:
                out.append(current)
            current = {
                "t": bucket,
                "o": bar["o"],
                "h": bar["h"],
                "l": bar["l"],
                "c": bar["c"],
                "v": bar["v"],
                "qv": bar["qv"],
            }
        else:
            current["h"] = max(current["h"], bar["h"])
            current["l"] = min(current["l"], bar["l"])
            current["c"] = bar["c"]
            current["v"] += bar["v"]
            current["qv"] += bar["qv"]
    if current is not None:
        out.append(current)
    return out


def _gmgn_headers() -> dict:
    return {
        "Accept": "application/json",
        "Origin": "https://gmgn.ai",
        "Referer": "https://gmgn.ai/",
        "client-id": GMGN_CLIENT_ID,
    }


@st.cache_data(ttl=60, show_spinner=False)
def gmgn_candles(
    address: str, network: str, resolution: str, bars_needed: int
) -> list[dict]:
    """GMGN OHLCV. Returns bars with quote volume in `qv` (USD)."""
    chain = GMGN_CHAINS.get(network)
    if not chain or resolution not in MOBY_RESAMPLE:
        return []

    minutes = MOBY_TIMEFRAMES[resolution]
    now = int(time.time())
    window = int(minutes * 60 * bars_needed * 1.25) + 3600
    window = min(window, 5 * 365 * 86400)

    res = _get(
        GMGN_KLINE_API.format(chain=chain, address=address),
        params={"resolution": resolution, "from": now - window, "to": now},
        headers=_gmgn_headers(),
        timeout=15,
    )
    if res is None or res.status_code != 200:
        return []
    try:
        payload = res.json() or {}
    except Exception:  # noqa: BLE001 - one bad token must not break a scan
        return []

    items = ((payload.get("data") or {}).get("list")) or []
    bars: list[dict] = []
    for item in items:
        try:
            quote_volume = _num(item.get("volume"))
            base_amount = _num(item.get("amount"))
            if quote_volume <= 0 and base_amount > 0:
                quote_volume = base_amount * _num(item.get("close"))
            bars.append(
                {
                    "t": int(_num(item.get("time"))),
                    "o": _num(item.get("open")),
                    "h": _num(item.get("high")),
                    "l": _num(item.get("low")),
                    "c": _num(item.get("close")),
                    "v": base_amount,
                    "qv": quote_volume,
                }
            )
        except (TypeError, ValueError):
            continue
    bars = [bar for bar in bars if bar["c"] > 0]
    bars.sort(key=lambda bar: bar["t"])
    return bars


@st.cache_data(ttl=1800, show_spinner=False)
def gecko_pool(address: str, network: str) -> tuple[str, str] | None:
    """Most liquid GeckoTerminal pool for a token -> (network, pool address)."""
    net = GECKO_NETWORKS.get(network)
    if not net:
        return None
    res = _get(
        f"{GECKO_BASE}/networks/{net}/tokens/{address}/pools",
        headers={"Accept": "application/json"},
        timeout=15,
    )
    if res is None or res.status_code != 200:
        return None
    try:
        pools = (res.json() or {}).get("data") or []
    except Exception:  # noqa: BLE001
        return None
    if not pools:
        return None
    best = max(
        pools,
        key=lambda pool: _num((pool.get("attributes") or {}).get("reserve_in_usd")),
    )
    pool_address = (best.get("attributes") or {}).get("address")
    if not pool_address:
        return None
    return net, str(pool_address)


@st.cache_data(ttl=60, show_spinner=False)
def gecko_candles(address: str, network: str, timeframe: str) -> list[dict]:
    """Fallback OHLCV from GeckoTerminal's free public API."""
    if timeframe not in GECKO_TF:
        return []
    pool = gecko_pool(address, network)
    if pool is None:
        return []
    net, pool_address = pool
    bucket, aggregate, factor = GECKO_TF[timeframe]
    limit = min(1000, CANDLE_LIMIT * factor + 40)

    res = _get(
        f"{GECKO_BASE}/networks/{net}/pools/{pool_address}/ohlcv/{bucket}",
        params={
            "aggregate": aggregate,
            "limit": limit,
            "currency": "usd",
            "token": "base",
        },
        headers={"Accept": "application/json"},
        timeout=15,
    )
    if res is None or res.status_code != 200:
        return []
    try:
        payload = res.json() or {}
    except Exception:  # noqa: BLE001
        return []

    rows = ((payload.get("data") or {}).get("attributes") or {}).get(
        "ohlcv_list"
    ) or []
    bars: list[dict] = []
    for row in reversed(rows):  # Gecko returns newest first
        try:
            bars.append(
                {
                    "t": int(row[0]) * 1000,
                    "o": float(row[1]),
                    "h": float(row[2]),
                    "l": float(row[3]),
                    "c": float(row[4]),
                    "v": 0.0,
                    "qv": float(row[5]),
                }
            )
        except (TypeError, ValueError, IndexError):
            continue
    if factor > 1:
        bars = resample_bars(bars, factor, MOBY_TIMEFRAMES[timeframe])
    return bars


@st.cache_data(ttl=60, show_spinner=False)
def get_moby_candles(address: str, network: str, timeframe: str) -> list[dict]:
    """OHLCV router for Moby tokens: GMGN first, GeckoTerminal as fallback."""
    if timeframe not in MOBY_RESAMPLE:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    native, factor = MOBY_RESAMPLE[timeframe]
    bars = gmgn_candles(
        address, network, native, CANDLE_LIMIT * factor + 40
    )
    if bars:
        return resample_bars(bars, factor, MOBY_TIMEFRAMES[native])
    return gecko_candles(address, network, timeframe)


def build_moby_universe(
    tokens: list[dict], limit: int, min_volume: float, sort_mode: str
) -> list[dict]:
    """Filter, sort and rank the Moby universe for the current run."""
    rows = [
        dict(token) for token in tokens if token["volume_24h_usd"] >= min_volume
    ]
    key, reverse = MOBY_SORTS[sort_mode]
    if key:
        rows.sort(key=lambda item: item[key], reverse=reverse)
    rows = rows[: int(limit)]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["age"] = fmt_age(row.get("created"))
        row["url"] = moby_token_url(row)
    return rows


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #

def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (vectorized seed, exact recursion)."""
    if values.size == 0:
        return values
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, values.size):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def fisher_transform(
    highs: np.ndarray, lows: np.ndarray, period: int = FISHER_PERIOD
) -> np.ndarray:
    """Recursive Fisher transform (Ehlers), clamped to +/-0.999."""
    n = highs.size
    fish = np.zeros(n, dtype=float)
    value = 0.0
    for i in range(n):
        start = max(0, i - period + 1)
        highest = float(np.max(highs[start : i + 1]))
        lowest = float(np.min(lows[start : i + 1]))
        spread = highest - lowest
        mid = (float(highs[i]) + float(lows[i])) / 2.0
        normalized = (mid - lowest) / spread if spread > 0 else 0.5
        value = 0.66 * (normalized - 0.5) + 0.67 * value
        value = max(min(value, 0.999), -0.999)
        fish[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
    return fish


def wilder_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float:
    """RSI with Wilder smoothing; NaN when there is not enough history."""
    if closes.size < period + 1:
        return float("nan")
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, deltas.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """Commodity Channel Index over CCI_PERIOD bars."""
    window = min(CCI_PERIOD, closes.size)
    tp = (highs[-window:] + lows[-window:] + closes[-window:]) / 3.0
    sma = float(np.mean(tp))
    mad = float(np.mean(np.abs(tp - sma)))
    if mad <= 0:
        return 0.0
    return (float(tp[-1]) - sma) / (0.015 * mad)


def trix(closes: np.ndarray, period: int = TRIX_PERIOD) -> float:
    """TRIX: 1-bar rate of change of a triple EMA, scaled by 10000."""
    if closes.size < 3:
        return 0.0
    triple = ema(ema(ema(closes, period), period), period)
    if triple[-2] == 0:
        return 0.0
    return (triple[-1] - triple[-2]) / triple[-2] * 10000.0


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def calc_strategy_1(
    bars: list[dict],
    weights: dict[str, float],
    pc_threshold: float = DEFAULT_PC_THRESHOLD,
    fisher_threshold: float = DEFAULT_FISHER_THRESHOLD,
    cci_threshold: float = DEFAULT_CCI_THRESHOLD,
) -> dict | None:
    """Weighted multi-factor score, always normalized to 0-100."""
    if len(bars) < 40:
        return None

    closes = np.array([b["c"] for b in bars], dtype=float)
    lows = np.array([b["l"] for b in bars], dtype=float)
    highs = np.array([b["h"] for b in bars], dtype=float)

    lookback = min(PC_LOOKBACK, lows.size)
    floor = float(np.min(lows[-lookback:]))
    pc_dist = (float(closes[-1]) / floor - 1.0) * 100.0 if floor > 0 else 999.0

    fisher_value = float(fisher_transform(highs, lows)[-1])
    trix_value = trix(closes)
    cci_value = cci(highs, lows, closes)

    active = {
        "pc": pc_dist <= pc_threshold,
        "fisher": fisher_value <= fisher_threshold,
        "trix": trix_value <= 0.0,
        "cci": cci_value <= cci_threshold,
    }
    total_weight = sum(max(0.0, float(w)) for w in weights.values())
    active_weight = sum(
        max(0.0, float(weights[key]))
        for key, is_active in active.items()
        if is_active
    )
    score = (active_weight / total_weight * 100.0) if total_weight > 0 else 0.0

    signals = [name for name, is_active in active.items() if is_active]
    return {
        "price": float(closes[-1]),
        "score": round(score, 1),
        "pc_dist": round(pc_dist, 2),
        "fisher": round(fisher_value, 2),
        "trix": float(round(trix_value, 2)),
        "cci": round(cci_value, 1),
        "signals": signals,
    }


def calc_strategy_2(
    bars: list[dict], pc_threshold: float, rsi_threshold: float
) -> dict | None:
    """PC130 proximity + adaptive wave bottom + Wilder RSI channel bottom."""
    if len(bars) < 40:
        return None

    closes = np.array([b["c"] for b in bars], dtype=float)
    lows = np.array([b["l"] for b in bars], dtype=float)
    highs = np.array([b["h"] for b in bars], dtype=float)

    lookback = min(PC130_LOOKBACK, lows.size)
    floor = float(np.min(lows[-lookback:]))
    pc130_dist = (float(closes[-1]) / floor - 1.0) * 100.0 if floor > 0 else 999.0

    if closes.size >= 2:
        true_range = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr_window = min(PC130_LOOKBACK, true_range.size)
        atr_value = float(np.mean(true_range[-atr_window:]))
    else:
        atr_value = 0.0
    wave_bottom = floor + atr_value * 0.5

    rsi_value = wilder_rsi(closes)

    signals = []
    if pc130_dist <= pc_threshold:
        signals.append("PC130")
    if float(closes[-1]) <= wave_bottom:
        signals.append("Wave")
    if np.isfinite(rsi_value) and rsi_value <= rsi_threshold:
        signals.append("RSI")

    return {
        "price": float(closes[-1]),
        "pc130_dist": round(pc130_dist, 2),
        "rsi": round(float(rsi_value), 1) if np.isfinite(rsi_value) else None,
        "wave_bottom": round(wave_bottom, 8),
        "signals": signals,
        "signal_count": len(signals),
        "status": f"TRIGGERED ({len(signals)}/3)" if signals else "NEUTRAL",
    }


def _is_pivot(
    values: np.ndarray, index: int, left: int, right: int, find_low: bool
) -> bool:
    """Strict pivot: unique extreme inside its window (no flat-bottom repeats)."""
    start = index - left
    end = index + right + 1
    if start < 0 or end > values.size:
        return False
    window = values[start:end]
    value = values[index]
    if np.count_nonzero(window == value) != 1:
        return False
    return value == (window.min() if find_low else window.max())


def calc_strategy_3(
    bars: list[dict],
    volume_24h: float,
    left_bars: int = S3_LEFT_BARS,
    right_bars: int = S3_RIGHT_BARS,
    fallback_length: int = S3_FALLBACK_LENGTH,
    black_left: int = S3_BLACK_LEFT,
    black_right: int = S3_BLACK_RIGHT,
    black_offset: int = S3_BLACK_OFFSET,
    min_volume_usdt: float = 0.0,
) -> dict | None:
    """Dynamic pivot channel + stop loss / take profit / risk-reward."""
    if len(bars) < 20:
        return None

    closes = np.array([b["c"] for b in bars], dtype=float)
    lows = np.array([b["l"] for b in bars], dtype=float)
    highs = np.array([b["h"] for b in bars], dtype=float)
    n = closes.size

    left_bars = max(1, int(left_bars))
    right_bars = max(1, int(right_bars))
    fallback_length = max(1, int(fallback_length))
    black_left = max(1, int(black_left))
    black_right = max(1, int(black_right))

    # --- main channel (confirmed pivots, step-held like Pine's var) ---
    sup_lines = np.full(n, np.nan)
    res_lines = np.full(n, np.nan)
    current_sup = np.nan
    current_res = np.nan
    for i in range(n):
        check_idx = i - right_bars
        if check_idx >= 0:
            if _is_pivot(lows, check_idx, left_bars, right_bars, True):
                current_sup = lows[check_idx]
            if _is_pivot(highs, check_idx, left_bars, right_bars, False):
                current_res = highs[check_idx]
        sup_lines[i] = current_sup
        res_lines[i] = current_res

    smart_sup = float(np.min(lows[-min(fallback_length, n) :]))
    smart_res = float(np.max(highs[-min(fallback_length, n) :]))
    final_sup = np.where(np.isnan(sup_lines), smart_sup, sup_lines)
    final_res = np.where(np.isnan(res_lines), smart_res, res_lines)

    # --- black structure line ---
    black_steps = np.full(n, np.nan)
    current_black = np.nan
    for i in range(n):
        check_idx = i - black_right
        if check_idx >= 0:
            if _is_pivot(lows, check_idx, black_left, black_right, True):
                current_black = lows[check_idx]
            elif _is_pivot(highs, check_idx, black_left, black_right, False):
                current_black = highs[check_idx]
        black_steps[i] = (
            final_sup[i] if np.isnan(current_black) else current_black
        )

    # Pine's plot(offset=...) only moves the drawing; it does not move signals.
    black_shifted = np.full(n, np.nan)
    for i in range(n):
        source = i - black_offset
        if 0 <= source < n:
            black_shifted[i] = black_steps[source]

    # --- evaluate on the latest candle ---
    last_close = float(closes[-1])
    last_sup = float(final_sup[-1])
    last_res = float(final_res[-1])
    finite_black = black_shifted[np.isfinite(black_shifted)]
    last_black = float(finite_black[-1]) if finite_black.size else None

    sup_dist = (
        (last_close / last_sup - 1.0) * 100.0 if last_sup > 0 else 999.0
    )

    if sup_dist <= 0.5:
        score = 100.0
    elif sup_dist <= 2.0:
        score = 90.0
    elif sup_dist <= 5.0:
        score = 70.0
    elif sup_dist <= 10.0:
        score = 40.0
    else:
        score = max(0.0, 100.0 - sup_dist * 3.0)

    if sup_dist <= 2.0:
        status = "BUY SETUP (TRUE TOUCH)"
    elif sup_dist <= 5.0:
        status = "NEAR SUPPORT"
    else:
        status = "NEUTRAL"

    stop_loss = last_sup * S3_STOP_LOSS_BUFFER
    take_profit = last_res
    risk = last_close - stop_loss
    reward = take_profit - last_close
    risk_pct = (risk / last_close * 100.0) if last_close > 0 else 0.0
    tp_pct = (reward / last_close * 100.0) if last_close > 0 else 0.0
    rr_ratio = (reward / risk) if (risk > 0 and reward > 0) else 0.0

    return {
        "price": last_close,
        "score": round(score, 1),
        "status": status,
        "support": round(last_sup, 8),
        "resistance": round(last_res, 8),
        "black_line": round(last_black, 8) if last_black is not None else None,
        "stop_loss": round(stop_loss, 8),
        "take_profit": round(take_profit, 8),
        "risk_pct": round(risk_pct, 2),
        "tp_pct": round(tp_pct, 2),
        "rr": round(rr_ratio, 2),
        "volume_24h": volume_24h,
        "vol_status": (
            "Healthy" if volume_24h >= min_volume_usdt else "Low liquidity"
        ),
    }


# --------------------------------------------------------------------------- #
# Scan plumbing
# --------------------------------------------------------------------------- #

def run_scan(worker, assets: list[dict], max_workers: int) -> tuple[list[dict], int, int]:
    """Run worker(item) in parallel. Returns (rows, skipped, failed).

    executor.map keeps the listing order, so the default view is deterministic
    even though requests finish out of order.
    """

    def guarded(item: dict):
        try:
            return "ok", worker(item)
        except Exception as exc:  # noqa: BLE001 - isolate per-asset failures
            return "error", str(exc)

    rows: list[dict] = []
    skipped = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for status, payload in executor.map(guarded, assets):
            if status == "error":
                failed += 1
                continue
            if payload is None:
                skipped += 1
                continue
            rows.append(payload)
    return rows, skipped, failed


def sort_rows(rows: list[dict], key: str, reverse: bool = True) -> list[dict]:
    """Stable sort with listing rank as the deterministic tie-breaker."""
    rows.sort(key=lambda row: row["#"])
    if key:
        rows.sort(key=lambda row: row[key], reverse=reverse)
    return rows


def strip_internal(rows: list[dict]) -> pd.DataFrame:
    """Drop helper keys and build the display DataFrame."""
    cleaned = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    return pd.DataFrame(cleaned)


def fmt_signals(signals: list[str]) -> str:
    pretty = {
        "pc": "PC",
        "fisher": "Fisher",
        "trix": "TRIX",
        "cci": "CCI",
        "PC130": "PC130",
        "Wave": "Wave",
        "RSI": "RSI",
    }
    return " + ".join(pretty.get(name, name) for name in signals) if signals else "-"


def pc_zone(pc_dist: float) -> str:
    if pc_dist <= 0.0:
        return "🟢 At/below floor"
    if pc_dist <= 5.0:
        return "🟠 Near floor"
    return "🔴 Far from floor"


def render_tab(
    rows: list[dict],
    total_assets: int,
    skipped: int,
    failed: int,
    column_config: dict,
    caption: str,
) -> None:
    if failed:
        st.warning(f"{failed} asset(s) could not be processed and were skipped.")
    st.success(
        f"Showing {len(rows)} of {total_assets} assets "
        f"({skipped} excluded by filters, {failed} failed)."
    )
    if not rows:
        st.info(
            "No asset passed the current filters. Lower the minimum 24H "
            "volume or increase the scan limit."
        )
        return
    st.dataframe(
        strip_internal(rows),
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )
    st.caption(caption)


# --- number formatting for the Moby tables --------------------------------- #

def fmt_price(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "-"
    if value >= 1000:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:,.4f}"
    if value >= 0.01:
        return f"${value:.4f}"
    if value >= 0.0001:
        return f"${value:.6f}"
    return f"${value:.9f}".rstrip("0")


def fmt_money(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{sign}${magnitude / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:,.2f}M"
    if magnitude >= 1e3:
        return f"{sign}${magnitude / 1e3:,.1f}K"
    return f"{sign}${magnitude:,.0f}"


def moby_stats(dataset: dict) -> dict:
    """Moby screener columns, appended after the strategy columns."""
    return {
        "Smart Money 1h": fmt_money(dataset["smart_money_1h_usd"]),
        "Vol 24H": fmt_money(dataset["volume_24h_usd"]),
        "MCap": fmt_money(dataset["market_cap_usd"]),
        "24H %": dataset["change_24h_pct"],
        "Price": fmt_price(dataset["price_usd"]),
        "Key Holders": dataset["key_holders"],
        "Activity 24H": dataset["activity_24h"],
        "Age": dataset["age"],
        "Moby": dataset["url"],
        "_address": dataset["address"],
        "_raw_price": dataset["price_usd"],
        "_raw_sm": dataset["smart_money_1h_usd"],
        "_raw_symbol": dataset["symbol"],
    }


MOBY_COLUMN_CONFIG = {
    "Moby": st.column_config.LinkColumn("Moby", display_text="Open"),
    "24H %": st.column_config.NumberColumn("24H %", format="%.2f%%"),
    "Key Holders": st.column_config.NumberColumn("Key Holders", format="%d"),
    "Activity 24H": st.column_config.NumberColumn("Activity 24H", format="%d"),
}


def _field(item: dict, *names: str) -> str:
    """Read a field from either a universe token or a display row."""
    for name in names:
        value = item.get(name)
        if value:
            return str(value)
    return ""


def moby_copy_box(items: list[dict], key: str) -> None:
    """One-click copy of a token contract address (Streamlit code block)."""
    items = [
        item for item in items if _field(item, "address", "_address", "Address")
    ]
    if not items:
        return
    labels = [
        " · ".join(
            part
            for part in (
                _field(item, "symbol", "Symbol"),
                _field(item, "network", "Network"),
                _field(item, "address", "_address", "Address"),
            )
            if part
        )
        for item in items
    ]
    choice = st.selectbox("Copy a token address", labels, key=key)
    st.code(
        _field(items[labels.index(choice)], "address", "_address", "Address"),
        language=None,
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.title("Lowline Analytics Dashboard")
st.caption(
    "Two sources: MEXC new listings and the Moby smart-money screener. "
    "Research tool only - not financial advice."
)

source = st.radio(
    "Data source",
    ["MEXC - new listings", "Moby - smart money sellers"],
    horizontal=True,
    key="data_source",
)
is_mexc = source.startswith("MEXC")

# --- Moby universe resolution --------------------------------------------- #

if "moby_upload" in st.session_state:
    moby_tokens = st.session_state["moby_upload"]["tokens"]
    moby_captured = st.session_state["moby_upload"]["captured_at"]
    moby_origin = "uploaded"
else:
    snapshot = load_snapshot_file()
    moby_tokens = snapshot["tokens"]
    moby_captured = snapshot["captured_at"]
    moby_origin = "bundled snapshot"

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Settings")

    if is_mexc:
        st.caption("Source: mexc.com public API")
        timeframe = st.selectbox(
            "Timeframe",
            options=list(TIMEFRAMES.keys()),
            index=list(TIMEFRAMES.keys()).index("60m"),
            help="Candle interval used for every indicator.",
        )
        max_scan_limit = st.slider(
            "Max new listings to scan",
            min_value=10,
            max_value=250,
            value=100,
            step=10,
        )
        min_vol_filter = st.number_input(
            "Minimum 24H volume (USDT)",
            min_value=0.0,
            value=50000.0,
            step=10000.0,
            help=(
                "Real quote volume over the last 24 hours. Assets below this "
                "value are hidden from every tab."
            ),
        )
        st.caption(
            "Volume is measured in USDT from quote-asset volume, not "
            "base-asset units."
        )
    else:
        st.caption("Source: app.moby.win screener + GMGN candles")
        moby_timeframe = st.selectbox(
            "Timeframe",
            options=list(MOBY_TIMEFRAMES.keys()),
            index=list(MOBY_TIMEFRAMES.keys()).index("1h"),
            help=(
                "GMGN native resolutions are 1m, 5m, 15m, 30m, 1h, 4h, 12h, "
                "1d, 1w; 3m is resampled from 1m and 2h from 1h."
            ),
        )
        max_moby_tokens = st.slider(
            "Tokens to scan",
            min_value=10,
            max_value=MOBY_MAX_TOKENS,
            value=MOBY_MAX_TOKENS,
            step=10,
            help="How many tokens of the ranked Moby list to keep.",
        )
        moby_min_volume = st.number_input(
            "Minimum 24H volume (USD)",
            min_value=0.0,
            value=10000.0,
            step=10000.0,
            help="Moby's 24H volume. Tokens below this are hidden everywhere.",
        )
        moby_sort = st.selectbox(
            "Rank the list by",
            options=list(MOBY_SORTS.keys()),
            index=0,
            help="Default: biggest smart-money net sellers in the last hour.",
        )
        st.caption(
            f"Moby data: {moby_origin} · {moby_captured or 'unknown time'} UTC "
            f"· {len(moby_tokens)} tokens"
        )
        with st.expander("Refresh Moby data"):
            st.caption(
                "Moby's API only answers a logged-in browser tab, so a fresh "
                "export can be uploaded at any time. Open app.moby.win, run "
                "this snippet in the browser console, then upload the "
                "downloaded JSON."
            )
            st.code(MOBY_EXPORT_SNIPPET, language="javascript")
            upload = st.file_uploader(
                "moby_leaderboard.json", type=["json"], key="moby_uploader"
            )
            if upload is not None:
                try:
                    parsed = normalize_moby_tokens(json.load(upload))
                except Exception as exc:  # noqa: BLE001 - shown to the user
                    st.error(f"Could not read that file ({exc}).")
                else:
                    if parsed:
                        st.session_state["moby_upload"] = {
                            "tokens": parsed,
                            "captured_at": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                        st.success(f"Loaded {len(parsed)} tokens.")
                        st.rerun()
                    else:
                        st.error("No tokens were found in that file.")
            if "moby_upload" in st.session_state and st.button(
                "Use the bundled snapshot", key="moby_reset"
            ):
                del st.session_state["moby_upload"]
                st.rerun()

# --------------------------------------------------------------------------- #
# MEXC section
# --------------------------------------------------------------------------- #

if is_mexc:
    tab1, tab2, tab3 = st.tabs(
        [
            "Strategy 1 (Quantitative Score)",
            "Strategy 2 (Pine Technical View)",
            "Strategy 3 (Dynamic Channel + Risk)",
        ]
    )

    # ------------------------------- TAB 1 -------------------------------- #

    with tab1:
        st.subheader("Strategy 1 - Quantitative Multi-Factor Score")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            w_pc = st.number_input("PC weight", value=40, min_value=0, key="s1_pc")
        with c2:
            w_fisher = st.number_input(
                "Fisher weight", value=30, min_value=0, key="s1_fish"
            )
        with c3:
            w_trix = st.number_input(
                "TRIX weight", value=20, min_value=0, key="s1_trix"
            )
        with c4:
            w_cci = st.number_input(
                "CCI weight", value=10, min_value=0, key="s1_cci"
            )

        with st.expander("Signal thresholds"):
            t1, t2, t3 = st.columns(3)
            with t1:
                pc_threshold = st.number_input(
                    "PC proximity threshold (%)",
                    value=DEFAULT_PC_THRESHOLD,
                    min_value=0.0,
                    key="s1_pc_thresh",
                )
            with t2:
                fisher_threshold = st.number_input(
                    "Fisher threshold",
                    value=DEFAULT_FISHER_THRESHOLD,
                    key="s1_fish_thresh",
                )
            with t3:
                cci_threshold = st.number_input(
                    "CCI threshold",
                    value=DEFAULT_CCI_THRESHOLD,
                    key="s1_cci_thresh",
                )

        sort_s1 = st.selectbox(
            "Sort results by",
            options=[
                "Listing sequence",
                "Score (high to low)",
                "PC distance (closest first)",
                "Price (high to low)",
                "Price (low to high)",
                "Symbol (A-Z)",
            ],
            key="sort_s1",
        )

        sort_keys_s1 = {
            "Listing sequence": (None, True),
            "Score (high to low)": ("Score", True),
            "PC distance (closest first)": ("_raw_pc", False),
            "Price (high to low)": ("_raw_price", True),
            "Price (low to high)": ("_raw_price", False),
            "Symbol (A-Z)": ("_raw_symbol", False),
        }

        if st.button("Run Strategy 1 Scan", type="primary", key="btn_s1"):
            try:
                with st.spinner("Loading assets..."):
                    assets = fetch_listing_assets()[:max_scan_limit]
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                weights = {
                    "pc": w_pc,
                    "fisher": w_fisher,
                    "trix": w_trix,
                    "cci": w_cci,
                }

                def worker_s1(item: dict) -> dict | None:
                    symbol = item["symbol"]
                    bars = (
                        []
                        if item["is_hidden"] or not item["has_spot"]
                        else get_candles(item["pair"], timeframe)
                    )
                    if not bars:
                        return None
                    volume = estimate_24h_quote_volume(
                        item["pair"], timeframe, bars
                    )
                    if volume < min_vol_filter:
                        return None
                    result = calc_strategy_1(
                        bars, weights, pc_threshold, fisher_threshold, cci_threshold
                    )
                    if result is None:
                        return None
                    return {
                        "#": item["listing_rank"],
                        "Symbol": f"{symbol} ({item['full_name']})",
                        "Chart": item["chart_url"],
                        "Price": result["price"],
                        "24H Vol (USDT)": volume,
                        "Score": result["score"],
                        "Zone": pc_zone(result["pc_dist"]),
                        "PC Dist %": result["pc_dist"],
                        "Fisher": result["fisher"],
                        "TRIX": result["trix"],
                        "CCI": result["cci"],
                        "Signals": fmt_signals(result["signals"]),
                        "_raw_price": result["price"],
                        "_raw_pc": result["pc_dist"],
                        "_raw_symbol": symbol,
                    }

                with st.spinner(
                    f"Scanning {len(assets)} assets on {timeframe}..."
                ):
                    rows, skipped, failed = run_scan(
                        worker_s1, assets, MAX_PARALLEL_REQUESTS
                    )

                st.session_state["s1_result"] = {
                    "rows": rows,
                    "total": len(assets),
                    "skipped": skipped,
                    "failed": failed,
                    "timeframe": timeframe,
                }

        stored_s1 = st.session_state.get("s1_result")
        if stored_s1:
            key, reverse = sort_keys_s1[sort_s1]
            sort_rows(stored_s1["rows"], key, reverse)
            render_tab(
                stored_s1["rows"],
                stored_s1["total"],
                stored_s1["skipped"],
                stored_s1["failed"],
                column_config={
                    "Chart": st.column_config.LinkColumn(
                        "Chart", display_text="Open chart"
                    ),
                    "Price": st.column_config.NumberColumn(
                        "Price (USDT)", format="$%.6f"
                    ),
                    "24H Vol (USDT)": st.column_config.NumberColumn(
                        "24H Vol (USDT)", format="$%.0f"
                    ),
                    "Score": st.column_config.ProgressColumn(
                        "Score / 100", min_value=0, max_value=100, format="%.1f"
                    ),
                    "PC Dist %": st.column_config.NumberColumn(
                        "PC Dist %", format="%.2f%%"
                    ),
                    "Fisher": st.column_config.NumberColumn(
                        "Fisher", format="%.2f"
                    ),
                    "TRIX": st.column_config.NumberColumn("TRIX", format="%.2f"),
                    "CCI": st.column_config.NumberColumn("CCI", format="%.1f"),
                },
                caption=(
                    "Score = active weights / total weights x 100. "
                    "Fisher is the recursive Ehlers transform; RSI on the next "
                    "tab uses Wilder smoothing."
                ),
            )
            st.caption(f"Last scan ran on the {stored_s1['timeframe']} timeframe.")

    # ------------------------------- TAB 2 -------------------------------- #

    with tab2:
        st.subheader(
            "Strategy 2 - PC130 Proximity + Adaptive Wave + RSI Channel Bottom"
        )
        c1, c2 = st.columns(2)
        with c1:
            pc_thresh = st.number_input(
                "PC130 proximity threshold (%)",
                value=3.0,
                min_value=0.0,
                key="s2_pc",
            )
        with c2:
            rsi_bottom = st.number_input(
                "RSI bottom threshold",
                value=30.0,
                min_value=0.0,
                max_value=100.0,
                key="s2_rsi",
            )

        sort_s2 = st.selectbox(
            "Sort results by",
            options=[
                "Listing sequence",
                "Signals (most first)",
                "PC130 distance (closest first)",
                "RSI (lowest first)",
                "Price (high to low)",
                "Price (low to high)",
                "Symbol (A-Z)",
            ],
            key="sort_s2",
        )

        sort_keys_s2 = {
            "Listing sequence": (None, True),
            "Signals (most first)": ("_raw_signals", True),
            "PC130 distance (closest first)": ("_raw_pc", False),
            "RSI (lowest first)": ("_raw_rsi", False),
            "Price (high to low)": ("_raw_price", True),
            "Price (low to high)": ("_raw_price", False),
            "Symbol (A-Z)": ("_raw_symbol", False),
        }

        if st.button("Run Strategy 2 Scan", type="primary", key="btn_s2"):
            try:
                with st.spinner("Loading assets..."):
                    assets = fetch_listing_assets()[:max_scan_limit]
            except RuntimeError as exc:
                st.error(str(exc))
            else:

                def worker_s2(item: dict) -> dict | None:
                    symbol = item["symbol"]
                    bars = (
                        []
                        if item["is_hidden"] or not item["has_spot"]
                        else get_candles(item["pair"], timeframe)
                    )
                    if not bars:
                        return None
                    volume = estimate_24h_quote_volume(
                        item["pair"], timeframe, bars
                    )
                    if volume < min_vol_filter:
                        return None
                    result = calc_strategy_2(bars, pc_thresh, rsi_bottom)
                    if result is None:
                        return None
                    return {
                        "#": item["listing_rank"],
                        "Symbol": f"{symbol} ({item['full_name']})",
                        "Chart": item["chart_url"],
                        "Price": result["price"],
                        "24H Vol (USDT)": volume,
                        "Status": result["status"],
                        "Signals": fmt_signals(result["signals"]),
                        "PC130 Dist %": result["pc130_dist"],
                        "RSI": result["rsi"],
                        "Wave Bottom": result["wave_bottom"],
                        "_raw_price": result["price"],
                        "_raw_signals": result["signal_count"],
                        "_raw_pc": result["pc130_dist"],
                        "_raw_rsi": (
                            result["rsi"] if result["rsi"] is not None else 999.0
                        ),
                        "_raw_symbol": symbol,
                    }

                with st.spinner(
                    f"Scanning {len(assets)} assets on {timeframe}..."
                ):
                    rows, skipped, failed = run_scan(
                        worker_s2, assets, MAX_PARALLEL_REQUESTS
                    )

                st.session_state["s2_result"] = {
                    "rows": rows,
                    "total": len(assets),
                    "skipped": skipped,
                    "failed": failed,
                    "timeframe": timeframe,
                }

        stored_s2 = st.session_state.get("s2_result")
        if stored_s2:
            key, reverse = sort_keys_s2[sort_s2]
            sort_rows(stored_s2["rows"], key, reverse)
            render_tab(
                stored_s2["rows"],
                stored_s2["total"],
                stored_s2["skipped"],
                stored_s2["failed"],
                column_config={
                    "Chart": st.column_config.LinkColumn(
                        "Chart", display_text="Open chart"
                    ),
                    "Price": st.column_config.NumberColumn(
                        "Price (USDT)", format="$%.6f"
                    ),
                    "24H Vol (USDT)": st.column_config.NumberColumn(
                        "24H Vol (USDT)", format="$%.0f"
                    ),
                    "PC130 Dist %": st.column_config.NumberColumn(
                        "PC130 Dist %", format="%.2f%%"
                    ),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "Wave Bottom": st.column_config.NumberColumn(
                        "Wave Bottom", format="$%.6f"
                    ),
                },
                caption=(
                    "Signals are evaluated on the latest candle. "
                    "Wave bottom = 130-bar low + 0.5 x ATR(130)."
                ),
            )
            st.caption(f"Last scan ran on the {stored_s2['timeframe']} timeframe.")

    # ------------------------------- TAB 3 -------------------------------- #

    with tab3:
        st.subheader(
            "Strategy 3 - Dynamic Pivot Channel + Risk Management (SL, TP, R:R)"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            s3_left = st.number_input(
                "Channel left bars", value=S3_LEFT_BARS, min_value=1, key="s3_l"
            )
            s3_black_left = st.number_input(
                "Black line left bars",
                value=S3_BLACK_LEFT,
                min_value=1,
                key="s3_bl",
            )
        with c2:
            s3_right = st.number_input(
                "Channel right bars", value=S3_RIGHT_BARS, min_value=1, key="s3_r"
            )
            s3_black_right = st.number_input(
                "Black line right bars",
                value=S3_BLACK_RIGHT,
                min_value=1,
                key="s3_br",
            )
        with c3:
            s3_fallback = st.number_input(
                "Fallback length",
                value=S3_FALLBACK_LENGTH,
                min_value=1,
                key="s3_fb",
            )
            s3_offset = st.number_input(
                "Black line shift (visual only)",
                value=S3_BLACK_OFFSET,
                min_value=-20,
                max_value=20,
                key="s3_off",
                help=(
                    "Matches Pine's plot(offset=...). It moves the black line "
                    "only; price, support, SL, TP and R:R are always computed "
                    "on the latest candle."
                ),
            )

        sort_s3 = st.selectbox(
            "Sort results by",
            options=[
                "Listing sequence",
                "Score (high to low)",
                "Risk/Reward (best first)",
                "Distance to support (closest first)",
                "Price (high to low)",
                "Price (low to high)",
                "Symbol (A-Z)",
            ],
            key="sort_s3",
        )

        sort_keys_s3 = {
            "Listing sequence": (None, True),
            "Score (high to low)": ("Score", True),
            "Risk/Reward (best first)": ("_raw_rr", True),
            "Distance to support (closest first)": ("_raw_sup_dist", False),
            "Price (high to low)": ("_raw_price", True),
            "Price (low to high)": ("_raw_price", False),
            "Symbol (A-Z)": ("_raw_symbol", False),
        }

        if st.button("Run Strategy 3 Scan", type="primary", key="btn_s3"):
            try:
                with st.spinner("Loading assets..."):
                    assets = fetch_listing_assets()[:max_scan_limit]
            except RuntimeError as exc:
                st.error(str(exc))
            else:

                def worker_s3(item: dict) -> dict | None:
                    symbol = item["symbol"]
                    bars = (
                        []
                        if item["is_hidden"] or not item["has_spot"]
                        else get_candles(item["pair"], timeframe)
                    )
                    if not bars:
                        return None
                    volume = estimate_24h_quote_volume(
                        item["pair"], timeframe, bars
                    )
                    if volume < min_vol_filter:
                        return None
                    result = calc_strategy_3(
                        bars,
                        volume,
                        int(s3_left),
                        int(s3_right),
                        int(s3_fallback),
                        int(s3_black_left),
                        int(s3_black_right),
                        int(s3_offset),
                        min_vol_filter,
                    )
                    if result is None:
                        return None
                    return {
                        "#": item["listing_rank"],
                        "Symbol": f"{symbol} ({item['full_name']})",
                        "Chart": item["chart_url"],
                        "Price": result["price"],
                        "24H Vol (USDT)": volume,
                        "Score": result["score"],
                        "Status": result["status"],
                        "Support": result["support"],
                        "Resistance": result["resistance"],
                        "Black Line": result["black_line"],
                        "Stop Loss": result["stop_loss"],
                        "Take Profit": result["take_profit"],
                        "Risk %": result["risk_pct"],
                        "TP %": result["tp_pct"],
                        "R:R": result["rr"],
                        "Liquidity": result["vol_status"],
                        "_raw_price": result["price"],
                        "_raw_rr": result["rr"],
                        "_raw_sup_dist": (
                            (result["price"] / result["support"] - 1.0) * 100.0
                            if result["support"]
                            else 999.0
                        ),
                        "_raw_symbol": symbol,
                    }

                with st.spinner(
                    f"Scanning {len(assets)} assets on {timeframe}..."
                ):
                    rows, skipped, failed = run_scan(
                        worker_s3, assets, MAX_PARALLEL_REQUESTS
                    )

                st.session_state["s3_result"] = {
                    "rows": rows,
                    "total": len(assets),
                    "skipped": skipped,
                    "failed": failed,
                    "timeframe": timeframe,
                }

        stored_s3 = st.session_state.get("s3_result")
        if stored_s3:
            key, reverse = sort_keys_s3[sort_s3]
            sort_rows(stored_s3["rows"], key, reverse)
            render_tab(
                stored_s3["rows"],
                stored_s3["total"],
                stored_s3["skipped"],
                stored_s3["failed"],
                column_config={
                    "Chart": st.column_config.LinkColumn(
                        "Chart", display_text="Open chart"
                    ),
                    "Price": st.column_config.NumberColumn(
                        "Price (USDT)", format="$%.6f"
                    ),
                    "24H Vol (USDT)": st.column_config.NumberColumn(
                        "24H Vol (USDT)", format="$%.0f"
                    ),
                    "Score": st.column_config.ProgressColumn(
                        "Score / 100", min_value=0, max_value=100, format="%.1f"
                    ),
                    "Support": st.column_config.NumberColumn(
                        "Support", format="$%.6f"
                    ),
                    "Resistance": st.column_config.NumberColumn(
                        "Resistance", format="$%.6f"
                    ),
                    "Black Line": st.column_config.NumberColumn(
                        "Black Line", format="$%.6f"
                    ),
                    "Stop Loss": st.column_config.NumberColumn(
                        "Stop Loss", format="$%.6f"
                    ),
                    "Take Profit": st.column_config.NumberColumn(
                        "Take Profit", format="$%.6f"
                    ),
                    "Risk %": st.column_config.NumberColumn(
                        "Risk %", format="%.2f%%"
                    ),
                    "TP %": st.column_config.NumberColumn(
                        "TP %", format="%.2f%%"
                    ),
                    "R:R": st.column_config.NumberColumn(
                        "Risk/Reward", format="%.2f"
                    ),
                },
                caption=(
                    "Stop loss = support x 0.985. Take profit = channel "
                    "resistance. R:R shows 0.00 when the target is below price "
                    "or the stop is above it, instead of a misleading number."
                ),
            )
            st.caption(f"Last scan ran on the {stored_s3['timeframe']} timeframe.")

    st.caption(
        "Data source: mexc.com public API. Signals are mechanical and can be "
        "wrong; always do your own research."
    )

# --------------------------------------------------------------------------- #
# Moby section
# --------------------------------------------------------------------------- #

else:
    if not moby_tokens:
        st.error(
            "No Moby universe loaded. Upload an export from the sidebar to "
            "start."
        )
        st.stop()

    universe = build_moby_universe(
        moby_tokens, max_moby_tokens, moby_min_volume, moby_sort
    )

    st.info(
        f"Ranked by **{moby_sort}** · {len(universe)} of {len(moby_tokens)} "
        f"tokens after the {fmt_money(moby_min_volume)} volume filter · "
        f"Moby snapshot {moby_captured or 'unknown'} UTC. "
        "Candles: GMGN (GeckoTerminal fallback)."
    )

    tab_list, mtab1, mtab2, mtab3 = st.tabs(
        [
            "Token List (copy address)",
            "Strategy 1 (Quantitative Score)",
            "Strategy 2 (Pine Technical View)",
            "Strategy 3 (Dynamic Channel + Risk)",
        ]
    )

    # ------------------------------ TOKEN LIST ---------------------------- #

    with tab_list:
        st.subheader("Moby ranked list")
        table_rows = []
        for token in universe:
            table_rows.append(
                {
                    "#": token["rank"],
                    "Symbol": token["symbol"],
                    "Network": token["network"],
                    "Smart Money 1h": fmt_money(token["smart_money_1h_usd"]),
                    "Vol 24H": fmt_money(token["volume_24h_usd"]),
                    "MCap": fmt_money(token["market_cap_usd"]),
                    "24H %": token["change_24h_pct"],
                    "Price": fmt_price(token["price_usd"]),
                    "Key Holders": token["key_holders"],
                    "Activity 24H": token["activity_24h"],
                    "Age": token["age"],
                    "Launchpad": token["launchpad"] or "-",
                    "Moby": token["url"],
                }
            )
        table = pd.DataFrame(table_rows)

        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Moby": st.column_config.LinkColumn("Moby", display_text="Open"),
                "24H %": st.column_config.NumberColumn(
                    "24H %", format="%.2f%%"
                ),
                "Key Holders": st.column_config.NumberColumn(
                    "Key Holders", format="%d"
                ),
                "Activity 24H": st.column_config.NumberColumn(
                    "Activity 24H", format="%d"
                ),
            },
            key="moby_list_table",
        )

        st.markdown("**One-click address copy**")
        moby_copy_box(universe, key="moby_copy_list")
        st.caption(
            "The copy icon appears at the top-right of the address block. "
            "Strategy tabs show the same box above their results."
        )

        st.download_button(
            "Download this list (CSV)",
            data=pd.DataFrame(
                [
                    {
                        "rank": token["rank"],
                        "symbol": token["symbol"],
                        "network": token["network"],
                        "address": token["address"],
                        "price_usd": token["price_usd"],
                        "market_cap_usd": token["market_cap_usd"],
                        "change_24h_pct": token["change_24h_pct"],
                        "volume_24h_usd": token["volume_24h_usd"],
                        "smart_money_1h_usd": token["smart_money_1h_usd"],
                        "key_holders": token["key_holders"],
                        "activity_24h": token["activity_24h"],
                        "age": token["age"],
                        "launchpad": token["launchpad"],
                    }
                    for token in universe
                ]
            ).to_csv(index=False),
            file_name="moby_ranked_list.csv",
            mime="text/csv",
        )

        st.caption(
            "Smart Money (1h) = net USD flow of Moby-tracked smart-money "
            "wallets in the last hour. Negative = net selling. Age is derived "
            "from the token creation date. Moby's numbers are a snapshot; "
            "candles and signals are live."
        )

    # ------------------------------ STRATEGY 1 ---------------------------- #

    def moby_scan_context() -> str:
        return (
            f"{len(universe)} Moby tokens on {moby_timeframe} "
            f"(snapshot {moby_captured or 'unknown'} UTC)"
        )

    with mtab1:
        st.subheader("Strategy 1 - Quantitative Multi-Factor Score")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            w_pc_m = st.number_input(
                "PC weight", value=40, min_value=0, key="m_s1_pc"
            )
        with c2:
            w_fisher_m = st.number_input(
                "Fisher weight", value=30, min_value=0, key="m_s1_fish"
            )
        with c3:
            w_trix_m = st.number_input(
                "TRIX weight", value=20, min_value=0, key="m_s1_trix"
            )
        with c4:
            w_cci_m = st.number_input(
                "CCI weight", value=10, min_value=0, key="m_s1_cci"
            )

        with st.expander("Signal thresholds"):
            t1, t2, t3 = st.columns(3)
            with t1:
                m_pc_threshold = st.number_input(
                    "PC proximity threshold (%)",
                    value=DEFAULT_PC_THRESHOLD,
                    min_value=0.0,
                    key="m_s1_pc_thresh",
                )
            with t2:
                m_fisher_threshold = st.number_input(
                    "Fisher threshold",
                    value=DEFAULT_FISHER_THRESHOLD,
                    key="m_s1_fish_thresh",
                )
            with t3:
                m_cci_threshold = st.number_input(
                    "CCI threshold",
                    value=DEFAULT_CCI_THRESHOLD,
                    key="m_s1_cci_thresh",
                )

        m_sort_s1 = st.selectbox(
            "Sort results by",
            options=[
                "Moby rank (sellers first)",
                "Score (high to low)",
                "PC distance (closest first)",
                "Smart Money 1h (most negative)",
                "Price (high to low)",
                "Symbol (A-Z)",
            ],
            key="m_sort_s1",
        )
        m_sort_keys_s1 = {
            "Moby rank (sellers first)": (None, True),
            "Score (high to low)": ("Score", True),
            "PC distance (closest first)": ("_raw_pc", False),
            "Smart Money 1h (most negative)": ("_raw_sm", False),
            "Price (high to low)": ("_raw_price", True),
            "Symbol (A-Z)": ("_raw_symbol", False),
        }

        if st.button("Run Moby Strategy 1", type="primary", key="m_btn_s1"):
            weights = {
                "pc": w_pc_m,
                "fisher": w_fisher_m,
                "trix": w_trix_m,
                "cci": w_cci_m,
            }

            def worker_m1(item: dict) -> dict | None:
                bars = get_moby_candles(
                    item["address"], item["network"], moby_timeframe
                )
                if not bars:
                    return None
                result = calc_strategy_1(
                    bars,
                    weights,
                    m_pc_threshold,
                    m_fisher_threshold,
                    m_cci_threshold,
                )
                if result is None:
                    return None
                row = {
                    "#": item["rank"],
                    "Symbol": item["symbol"],
                    "Network": item["network"],
                }
                row.update(
                    {
                        "Score": result["score"],
                        "Zone": pc_zone(result["pc_dist"]),
                        "PC Dist %": result["pc_dist"],
                        "Fisher": result["fisher"],
                        "TRIX": result["trix"],
                        "CCI": result["cci"],
                        "Signals": fmt_signals(result["signals"]),
                        "_raw_pc": result["pc_dist"],
                    }
                )
                row.update(moby_stats(item))
                return row

            with st.spinner(f"Scanning {moby_scan_context()}..."):
                rows, skipped, failed = run_scan(
                    worker_m1, universe, MAX_MOBY_WORKERS
                )
            st.session_state["m_s1_result"] = {
                "rows": rows,
                "total": len(universe),
                "skipped": skipped,
                "failed": failed,
                "timeframe": moby_timeframe,
                "captured": moby_captured,
            }

        stored = st.session_state.get("m_s1_result")
        if stored:
            key, reverse = m_sort_keys_s1[m_sort_s1]
            sort_rows(stored["rows"], key, reverse)
            moby_copy_box(stored["rows"], key="m_s1_copy")
            render_tab(
                stored["rows"],
                stored["total"],
                stored["skipped"],
                stored["failed"],
                column_config={
                    **MOBY_COLUMN_CONFIG,
                    "Score": st.column_config.ProgressColumn(
                        "Score / 100", min_value=0, max_value=100, format="%.1f"
                    ),
                    "PC Dist %": st.column_config.NumberColumn(
                        "PC Dist %", format="%.2f%%"
                    ),
                    "Fisher": st.column_config.NumberColumn(
                        "Fisher", format="%.2f"
                    ),
                    "TRIX": st.column_config.NumberColumn("TRIX", format="%.2f"),
                    "CCI": st.column_config.NumberColumn("CCI", format="%.1f"),
                },
                caption=(
                    "Moby stats come from the snapshot; candles and indicators "
                    "are live. Score = active weights / total weights x 100."
                ),
            )
            st.caption(
                f"Last scan: {stored['timeframe']} candles · Moby data "
                f"{stored['captured'] or 'unknown'} UTC."
            )

    # ------------------------------ STRATEGY 2 ---------------------------- #

    with mtab2:
        st.subheader(
            "Strategy 2 - PC130 Proximity + Adaptive Wave + RSI Channel Bottom"
        )
        c1, c2 = st.columns(2)
        with c1:
            m_pc_thresh = st.number_input(
                "PC130 proximity threshold (%)",
                value=3.0,
                min_value=0.0,
                key="m_s2_pc",
            )
        with c2:
            m_rsi_bottom = st.number_input(
                "RSI bottom threshold",
                value=30.0,
                min_value=0.0,
                max_value=100.0,
                key="m_s2_rsi",
            )

        m_sort_s2 = st.selectbox(
            "Sort results by",
            options=[
                "Moby rank (sellers first)",
                "Signals (most first)",
                "PC130 distance (closest first)",
                "RSI (lowest first)",
                "Smart Money 1h (most negative)",
                "Price (high to low)",
                "Symbol (A-Z)",
            ],
            key="m_sort_s2",
        )
        m_sort_keys_s2 = {
            "Moby rank (sellers first)": (None, True),
            "Signals (most first)": ("_raw_signals", True),
            "PC130 distance (closest first)": ("_raw_pc", False),
            "RSI (lowest first)": ("_raw_rsi", False),
            "Smart Money 1h (most negative)": ("_raw_sm", False),
            "Price (high to low)": ("_raw_price", True),
            "Symbol (A-Z)": ("_raw_symbol", False),
        }

        if st.button("Run Moby Strategy 2", type="primary", key="m_btn_s2"):

            def worker_m2(item: dict) -> dict | None:
                bars = get_moby_candles(
                    item["address"], item["network"], moby_timeframe
                )
                if not bars:
                    return None
                result = calc_strategy_2(bars, m_pc_thresh, m_rsi_bottom)
                if result is None:
                    return None
                row = {
                    "#": item["rank"],
                    "Symbol": item["symbol"],
                    "Network": item["network"],
                }
                row.update(
                    {
                        "Status": result["status"],
                        "Signals": fmt_signals(result["signals"]),
                        "PC130 Dist %": result["pc130_dist"],
                        "RSI": result["rsi"],
                        "Wave Bottom": fmt_price(result["wave_bottom"]),
                        "_raw_signals": result["signal_count"],
                        "_raw_pc": result["pc130_dist"],
                        "_raw_rsi": (
                            result["rsi"] if result["rsi"] is not None else 999.0
                        ),
                    }
                )
                row.update(moby_stats(item))
                return row

            with st.spinner(f"Scanning {moby_scan_context()}..."):
                rows, skipped, failed = run_scan(
                    worker_m2, universe, MAX_MOBY_WORKERS
                )
            st.session_state["m_s2_result"] = {
                "rows": rows,
                "total": len(universe),
                "skipped": skipped,
                "failed": failed,
                "timeframe": moby_timeframe,
                "captured": moby_captured,
            }

        stored = st.session_state.get("m_s2_result")
        if stored:
            key, reverse = m_sort_keys_s2[m_sort_s2]
            sort_rows(stored["rows"], key, reverse)
            moby_copy_box(stored["rows"], key="m_s2_copy")
            render_tab(
                stored["rows"],
                stored["total"],
                stored["skipped"],
                stored["failed"],
                column_config={
                    **MOBY_COLUMN_CONFIG,
                    "PC130 Dist %": st.column_config.NumberColumn(
                        "PC130 Dist %", format="%.2f%%"
                    ),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                },
                caption=(
                    "Signals are evaluated on the latest candle. "
                    "Wave bottom = 130-bar low + 0.5 x ATR(130)."
                ),
            )
            st.caption(
                f"Last scan: {stored['timeframe']} candles · Moby data "
                f"{stored['captured'] or 'unknown'} UTC."
            )

    # ------------------------------ STRATEGY 3 ---------------------------- #

    with mtab3:
        st.subheader(
            "Strategy 3 - Dynamic Pivot Channel + Risk Management (SL, TP, R:R)"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            m_s3_left = st.number_input(
                "Channel left bars",
                value=S3_LEFT_BARS,
                min_value=1,
                key="m_s3_l",
            )
            m_s3_black_left = st.number_input(
                "Black line left bars",
                value=S3_BLACK_LEFT,
                min_value=1,
                key="m_s3_bl",
            )
        with c2:
            m_s3_right = st.number_input(
                "Channel right bars",
                value=S3_RIGHT_BARS,
                min_value=1,
                key="m_s3_r",
            )
            m_s3_black_right = st.number_input(
                "Black line right bars",
                value=S3_BLACK_RIGHT,
                min_value=1,
                key="m_s3_br",
            )
        with c3:
            m_s3_fallback = st.number_input(
                "Fallback length",
                value=S3_FALLBACK_LENGTH,
                min_value=1,
                key="m_s3_fb",
            )
            m_s3_offset = st.number_input(
                "Black line shift (visual only)",
                value=S3_BLACK_OFFSET,
                min_value=-20,
                max_value=20,
                key="m_s3_off",
            )

        m_sort_s3 = st.selectbox(
            "Sort results by",
            options=[
                "Moby rank (sellers first)",
                "Score (high to low)",
                "Risk/Reward (best first)",
                "Distance to support (closest first)",
                "Smart Money 1h (most negative)",
                "Price (high to low)",
                "Symbol (A-Z)",
            ],
            key="m_sort_s3",
        )
        m_sort_keys_s3 = {
            "Moby rank (sellers first)": (None, True),
            "Score (high to low)": ("Score", True),
            "Risk/Reward (best first)": ("_raw_rr", True),
            "Distance to support (closest first)": ("_raw_sup_dist", False),
            "Smart Money 1h (most negative)": ("_raw_sm", False),
            "Price (high to low)": ("_raw_price", True),
            "Symbol (A-Z)": ("_raw_symbol", False),
        }

        if st.button("Run Moby Strategy 3", type="primary", key="m_btn_s3"):

            def worker_m3(item: dict) -> dict | None:
                bars = get_moby_candles(
                    item["address"], item["network"], moby_timeframe
                )
                if not bars:
                    return None
                result = calc_strategy_3(
                    bars,
                    item["volume_24h_usd"],
                    int(m_s3_left),
                    int(m_s3_right),
                    int(m_s3_fallback),
                    int(m_s3_black_left),
                    int(m_s3_black_right),
                    int(m_s3_offset),
                    moby_min_volume,
                )
                if result is None:
                    return None
                row = {
                    "#": item["rank"],
                    "Symbol": item["symbol"],
                    "Network": item["network"],
                }
                row.update(
                    {
                        "Score": result["score"],
                        "Status": result["status"],
                        "Support": result["support"],
                        "Resistance": result["resistance"],
                        "Black Line": result["black_line"],
                        "Stop Loss": result["stop_loss"],
                        "Take Profit": result["take_profit"],
                        "Risk %": result["risk_pct"],
                        "TP %": result["tp_pct"],
                        "R:R": result["rr"],
                        "Liquidity": result["vol_status"],
                        "_raw_rr": result["rr"],
                        "_raw_sup_dist": (
                            (result["price"] / result["support"] - 1.0) * 100.0
                            if result["support"]
                            else 999.0
                        ),
                    }
                )
                row.update(moby_stats(item))
                return row

            with st.spinner(f"Scanning {moby_scan_context()}..."):
                rows, skipped, failed = run_scan(
                    worker_m3, universe, MAX_MOBY_WORKERS
                )
            st.session_state["m_s3_result"] = {
                "rows": rows,
                "total": len(universe),
                "skipped": skipped,
                "failed": failed,
                "timeframe": moby_timeframe,
                "captured": moby_captured,
            }

        stored = st.session_state.get("m_s3_result")
        if stored:
            key, reverse = m_sort_keys_s3[m_sort_s3]
            sort_rows(stored["rows"], key, reverse)
            moby_copy_box(stored["rows"], key="m_s3_copy")
            render_tab(
                stored["rows"],
                stored["total"],
                stored["skipped"],
                stored["failed"],
                column_config={
                    **MOBY_COLUMN_CONFIG,
                    "Score": st.column_config.ProgressColumn(
                        "Score / 100", min_value=0, max_value=100, format="%.1f"
                    ),
                    "Risk %": st.column_config.NumberColumn(
                        "Risk %", format="%.2f%%"
                    ),
                    "TP %": st.column_config.NumberColumn(
                        "TP %", format="%.2f%%"
                    ),
                    "R:R": st.column_config.NumberColumn(
                        "Risk/Reward", format="%.2f"
                    ),
                },
                caption=(
                    "Stop loss = support x 0.985, take profit = channel "
                    "resistance. Prices are USD from GMGN candles."
                ),
            )
            st.caption(
                f"Last scan: {stored['timeframe']} candles · Moby data "
                f"{stored['captured'] or 'unknown'} UTC."
            )

    st.caption(
        "Moby list data: app.moby.win screener snapshot. Candles: GMGN public "
        "kline API with a GeckoTerminal fallback. Signals are mechanical and "
        "can be wrong; always do your own research."
    )
