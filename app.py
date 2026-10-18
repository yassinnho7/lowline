"""Lowline Analytics Dashboard - corrected build (app2).

Fixes compared to app.py
------------------------
1. 24H volume is now *real quote (USDT) volume* and is timeframe aware
   (previously it summed 96 candles of base-asset volume and labelled it USDT).
2. The sidebar volume filter actually removes assets from every tab.
3. Ordering is deterministic (executor.map, stable sorts, tie-break by rank).
4. Each tab has its own sorting options; Strategy 2 sorts numerically.
5. Signals are always evaluated on the latest candle. The black-line offset is
   visual only, exactly like Pine Script's plot(offset=...), and no longer
   shifts price / SL / TP back in time.
6. Risk/Reward is guarded against zero or negative risk and reward.
7. Timeframe is a select box, so unsupported values can no longer silently
   fall back to 60m.
8. Fisher transform is the recursive version and RSI uses Wilder smoothing.
9. HTTP calls are retried and cached; one failed asset cannot kill a scan.
10. Assets with no usable data are listed separately instead of being shown
    with fake zero prices/scores.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
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
    """Fetch OHLCV klines, including quote-asset volume (item[7]).

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


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.title("Lowline Analytics Dashboard")
st.caption(
    "Live data from MEXC public endpoints. Research tool only - "
    "not financial advice."
)

with st.sidebar:
    st.header("Global Settings")
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
        "Volume is measured in USDT from quote-asset volume, not base-asset "
        "units."
    )

tab1, tab2, tab3 = st.tabs(
    [
        "Strategy 1 (Quantitative Score)",
        "Strategy 2 (Pine Technical View)",
        "Strategy 3 (Dynamic Channel + Risk)",
    ]
)

# ------------------------------- TAB 1 ------------------------------------- #

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
        w_trix = st.number_input("TRIX weight", value=20, min_value=0, key="s1_trix")
    with c4:
        w_cci = st.number_input("CCI weight", value=10, min_value=0, key="s1_cci")

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

# ------------------------------- TAB 2 ------------------------------------- #

with tab2:
    st.subheader(
        "Strategy 2 - PC130 Proximity + Adaptive Wave + RSI Channel Bottom"
    )
    c1, c2 = st.columns(2)
    with c1:
        pc_thresh = st.number_input(
            "PC130 proximity threshold (%)", value=3.0, min_value=0.0, key="s2_pc"
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

# ------------------------------- TAB 3 ------------------------------------- #

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
            "Fallback length", value=S3_FALLBACK_LENGTH, min_value=1, key="s3_fb"
        )
        s3_offset = st.number_input(
            "Black line shift (visual only)",
            value=S3_BLACK_OFFSET,
            min_value=-20,
            max_value=20,
            key="s3_off",
            help=(
                "Matches Pine's plot(offset=...). It moves the black line only; "
                "price, support, SL, TP and R:R are always computed on the "
                "latest candle."
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
                "TP %": st.column_config.NumberColumn("TP %", format="%.2f%%"),
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
