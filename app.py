import concurrent.futures
from urllib.parse import quote
from curl_cffi import requests as async_requests
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Lowline Analytics Dashboard", layout="wide")

LISTING_API = "https://www.mexc.com/api/seo/coin/config/list"
KLINES_API = "https://api.mexc.com/api/v3/klines"


def fetch_seo_listing_assets():
    """Fetch top 250 assets directly from SEO endpoint following exact listing order."""
    try:
        res = async_requests.get(
            LISTING_API,
            impersonate="chrome120",
            headers={
                "Referer": "https://www.mexc.com/price/new-crypto",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=15,
        )

        if res.status_code != 200:
            st.error(f"HTTP Error {res.status_code} while fetching asset list.")
            return []

        res_json = res.json()
        data = res_json.get("data", [])

        if isinstance(data, dict):
            data = data.get("result") or data.get("list") or []

        rows = []
        for idx, item in enumerate(data[:250], 1):
            symbol_name = (
                item.get("symbolName") or item.get("currency") or item.get("symbol")
            )
            full_name = item.get("symbolFullName") or symbol_name
            is_hidden = (
                item.get("hide", False)
                or item.get("state") == "HIDE"
                or not item.get("enableFetch", True)
            )

            if symbol_name:
                sym_clean = str(symbol_name).strip().upper()
                chart_url = f"https://www.mexc.com/exchange/{quote(sym_clean)}_USDT"

                rows.append(
                    {
                        "listing_rank": idx,
                        "symbol": sym_clean,
                        "full_name": full_name,
                        "is_hidden": is_hidden,
                        "chart_url": chart_url,
                    }
                )
        return rows
    except Exception as e:
        st.error(f"Error fetching SEO list: {e}")
        return []


def get_candles(symbol, tf_str, target_bars=150):
    """Fetch OHLCV klines for specified timeframe."""
    params = {"symbol": f"{symbol}USDT", "interval": tf_str, "limit": target_bars}
    try:
        res = requests.get(KLINES_API, params=params, timeout=4)
        if res.status_code != 200:
            return None
        data = res.json()
        if not data or not isinstance(data, list):
            return None

        bars = []
        for item in data:
            bars.append(
                {
                    "t": item[0],
                    "o": float(item[1]),
                    "h": float(item[2]),
                    "l": float(item[3]),
                    "c": float(item[4]),
                    "v": float(item[5]),
                }
            )
        return bars
    except Exception:
        return None


# --- STRATEGY 1 CALCULATION ---
def calc_strategy_1(bars, w_pc, w_fisher, w_trix, w_cci):
    if not bars or len(bars) < 40:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    closes = np.array([b["c"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    highs = np.array([b["h"] for b in bars])
    volumes = np.array([b["v"] for b in bars])

    volume_24h = np.sum(volumes[-96:]) if len(volumes) >= 96 else np.sum(volumes)

    lookback_pc = min(100, len(lows))
    pc_lower = np.min(lows[-lookback_pc:])
    pc_dist = ((closes[-1] / pc_lower) - 1.0) * 100.0 if pc_lower > 0 else 999.0
    s_pc = w_pc if pc_dist <= 5.0 else 0.0

    lookback_fish = min(70, len(lows))
    hl2 = (highs[-lookback_fish:] + lows[-lookback_fish:]) / 2.0
    min_l = np.min(lows[-lookback_fish:])
    max_h = np.max(highs[-lookback_fish:])
    raw_val = (hl2[-1] - min_l) / (max_h - min_l) if (max_h - min_l) > 0 else 0.5
    val = max(min(0.66 * (raw_val - 0.5) + 0.5 * 0.5, 0.999), -0.999)
    fisher = 0.5 * np.log((1.0 + val) / (1.0 - val))
    s_fisher = w_fisher if fisher <= -2.1 else 0.0

    def ema(data, period):
        alpha = 2.0 / (period + 1.0)
        res = np.zeros_like(data)
        res[0] = data[0]
        for i in range(1, len(data)):
            res[i] = alpha * data[i] + (1.0 - alpha) * res[i - 1]
        return res

    e1 = ema(closes, 15)
    e2 = ema(e1, 15)
    e3 = ema(e2, 15)
    trix = ((e3[-1] - e3[-2]) / e3[-2]) * 10000.0 if len(e3) > 1 and e3[-2] != 0 else 0.0
    s_trix = w_trix if trix <= 0 else 0.0

    lookback_cci = min(40, len(closes))
    tp = (
        highs[-lookback_cci:] + lows[-lookback_cci:] + closes[-lookback_cci:]
    ) / 3.0
    sma_tp = np.mean(tp)
    mad = np.mean(np.abs(tp - sma_tp))
    cci = (tp[-1] - sma_tp) / (0.015 * mad) if mad > 0 else 0.0
    s_cci = w_cci if cci <= -160.0 else 0.0

    score = round(s_pc + s_fisher + s_trix + s_cci, 1)
    return (
        closes[-1],
        volume_24h,
        score,
        round(pc_dist, 2),
        round(fisher, 2),
        round(trix, 2),
        round(cci, 1),
    )


# --- STRATEGY 2 CALCULATION ---
def calc_strategy_2(bars, pc_threshold, rsi_bottom_thresh):
    if not bars or len(bars) < 40:
        return 0.0, 0.0, "N/A", 0.0, 0.0

    closes = np.array([b["c"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    highs = np.array([b["h"] for b in bars])
    volumes = np.array([b["v"] for b in bars])

    volume_24h = np.sum(volumes[-96:]) if len(volumes) >= 96 else np.sum(volumes)

    lookback = min(130, len(lows))
    min_130 = np.min(lows[-lookback:])
    pc130_dist = ((closes[-1] / min_130) - 1.0) * 100.0 if min_130 > 0 else 999.0
    pc_signal = pc130_dist <= pc_threshold

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    atr130 = np.mean(tr[-lookback:]) if len(tr) >= lookback else np.mean(tr)
    wave_bottom = min_130 + (atr130 * 0.5)
    wave_signal = closes[-1] <= wave_bottom

    rsi_lookback = min(85, len(closes) - 1)
    deltas = np.diff(closes[-rsi_lookback - 1 :])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_signal = rsi <= rsi_bottom_thresh

    signals = []
    if pc_signal:
        signals.append("PC130")
    if wave_signal:
        signals.append("Wave")
    if rsi_signal:
        signals.append("RSI Channel")

    status = f"TRIGGERED ({len(signals)})" if signals else "NEUTRAL"
    return closes[-1], volume_24h, status, round(pc130_dist, 2), round(rsi, 1)


# --- STRATEGY 3 CALCULATION (Dynamic Pivot Channel with Offset Support) ---
def calc_strategy_3(
    bars,
    left_bars=8,
    right_bars=3,
    fallback_length=8,
    black_left=16,
    black_right=2,
    black_offset=-2,
):
    if not bars or len(bars) < max(
        left_bars + right_bars, black_left + black_right, 40
    ):
        return 0.0, 0.0, 0.0, "N/A", 0.0, 0.0

    closes = np.array([b["c"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    highs = np.array([b["h"] for b in bars])
    volumes = np.array([b["v"] for b in bars])

    volume_24h = np.sum(volumes[-96:]) if len(volumes) >= 96 else np.sum(volumes)
    n = len(closes)

    sup_lines = np.full(n, np.nan)
    res_lines = np.full(n, np.nan)
    current_sup = np.nan
    current_res = np.nan

    for i in range(n):
        check_idx = i - right_bars
        if check_idx >= left_bars:
            w_lows = lows[check_idx - left_bars : check_idx + right_bars + 1]
            if len(w_lows) == left_bars + right_bars + 1 and lows[check_idx] == np.min(
                w_lows
            ):
                current_sup = lows[check_idx]

            w_highs = highs[check_idx - left_bars : check_idx + right_bars + 1]
            if len(w_highs) == left_bars + right_bars + 1 and highs[check_idx] == np.max(
                w_highs
            ):
                current_res = highs[check_idx]

        sup_lines[i] = current_sup
        res_lines[i] = current_res

    smart_sup = np.min(lows[-fallback_length:])
    smart_res = np.max(highs[-fallback_length:])

    final_sup = np.where(np.isnan(sup_lines), smart_sup, sup_lines)
    final_res = np.where(np.isnan(res_lines), smart_res, res_lines)

    black_steps = np.full(n, np.nan)
    current_black = np.nan
    for i in range(n):
        check_idx_b = i - black_right
        p_low_b = False
        p_high_b = False
        val_low = np.nan
        val_high = np.nan
        if check_idx_b >= black_left:
            w_lows = lows[check_idx_b - black_left : check_idx_b + black_right + 1]
            if len(w_lows) == black_left + black_right + 1 and lows[check_idx_b] == np.min(
                w_lows
            ):
                val_low = lows[check_idx_b]
                p_low_b = True

            w_highs = highs[check_idx_b - black_left : check_idx_b + black_right + 1]
            if len(w_highs) == black_left + black_right + 1 and highs[check_idx_b] == np.max(
                w_highs
            ):
                val_high = highs[check_idx_b]
                p_high_b = True

        if p_low_b:
            current_black = val_low
        elif p_high_b:
            current_black = val_high

        if np.isnan(current_black):
            current_black = final_sup[i]

        black_steps[i] = current_black

    # Apply offset evaluation index safely
    eval_idx = n - 1 + black_offset
    eval_idx = max(0, min(eval_idx, n - 1))

    last_close = closes[eval_idx]
    last_sup = final_sup[eval_idx]
    last_res = final_res[eval_idx]

    sup_dist = (
        ((last_close - last_sup) / last_sup) * 100.0 if last_sup > 0 else 999.0
    )

    score = 0.0
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

    status = (
        "BUY SETUP (TRUE TOUCH)"
        if sup_dist <= 2.0
        else ("NEAR SUPPORT" if sup_dist <= 5.0 else "NEUTRAL")
    )

    return (
        last_close,
        volume_24h,
        round(score, 1),
        status,
        round(last_sup, 6),
        round(last_res, 6),
    )


# --- STYLING HIGHLIGHT FUNCTION ---
def style_strategy_1(df):
    def highlight_cols(row):
        styles = [""] * len(row)
        pc = row["PC Dist %"]
        if pd.notnull(pc):
            if pc <= 0.0:
                styles[df.columns.get_loc("PC Dist %")] = (
                    "background-color: #1e4620; color: #a3f7a1; font-weight: bold;"
                )
            elif pc <= 5.0:
                styles[df.columns.get_loc("PC Dist %")] = (
                    "background-color: #5c3800; color: #ffca7a; font-weight: bold;"
                )
            else:
                styles[df.columns.get_loc("PC Dist %")] = (
                    "background-color: #4a1919; color: #f28b8b;"
                )

        fish = row["Fisher"]
        if pd.notnull(fish):
            if fish <= -4.2:
                styles[df.columns.get_loc("Fisher")] = (
                    "background-color: #1e4620; color: #a3f7a1; font-weight: bold;"
                )
            elif fish <= -2.1:
                styles[df.columns.get_loc("Fisher")] = (
                    "background-color: #5c3800; color: #ffca7a; font-weight: bold;"
                )
            else:
                styles[df.columns.get_loc("Fisher")] = (
                    "background-color: #4a1919; color: #f28b8b;"
                )

        trix = row["TRIX"]
        if pd.notnull(trix):
            if trix <= -220.0:
                styles[df.columns.get_loc("TRIX")] = (
                    "background-color
