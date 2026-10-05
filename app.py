import concurrent.futures
from urllib.parse import quote
import numpy as np
import requests
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Lowline Analytics Dashboard", layout="wide")

API_BASE = "https://api.mexc.com/api/v3"
PRIMARY_LISTING_API = "https://www.mexc.com/api/seo/coin/config/list"
BACKUP_LISTING_API = "https://api.mexc.com/api/v3/defaultSymbols"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.mexc.com",
    "Referer": "https://www.mexc.com/",
}


def fetch_top_250_assets():
    """Fetch top 250 assets with backup fallback to prevent empty returns."""
    rows = []
    try:
        res = requests.get(PRIMARY_LISTING_API, headers=HEADERS, timeout=8)
        if res.status_code == 200:
            res_json = res.json()
            data = res_json.get("data", [])
            if isinstance(data, dict):
                data = data.get("result", data.get("list", []))

            for item in data[:250]:
                symbol = item.get("symbolName") or item.get("currency") or item.get("symbol")
                is_hidden = (
                    item.get("hide", False)
                    or item.get("state") == "HIDE"
                    or not item.get("enableFetch", True)
                )
                if symbol:
                    sym_upper = symbol.upper().replace("_USDT", "").replace("USDT", "")
                    rows.append(
                        {
                            "symbol": sym_upper,
                            "is_hidden": is_hidden,
                            "chart": f"https://www.mexc.com/exchange/{quote(sym_upper)}_USDT",
                        }
                    )
    except Exception:
        pass

    # Fallback if primary API fails
    if not rows:
        try:
            res = requests.get(f"{API_BASE}/ticker/24hr", headers=HEADERS, timeout=8)
            if res.status_code == 200:
                data = res.json()
                usdt_pairs = [d for d in data if d.get("symbol", "").endswith("USDT")]
                usdt_pairs.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)

                for item in usdt_pairs[:250]:
                    sym_raw = item.get("symbol", "").replace("USDT", "")
                    rows.append(
                        {
                            "symbol": sym_raw,
                            "is_hidden": False,
                            "chart": f"https://www.mexc.com/exchange/{quote(sym_raw)}_USDT",
                        }
                    )
        except Exception:
            pass

    return rows


def get_candles(symbol, tf_str, target_bars=150):
    """Fetch klines for any user-defined timeframe."""
    url = f"{API_BASE}/klines"
    params = {"symbol": f"{symbol}USDT", "interval": tf_str, "limit": target_bars}

    try:
        res = requests.get(url, params=params, headers=HEADERS, timeout=4)
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
    if not bars or len(bars) < 100:
        return 0.0, "N/A"

    closes = np.array([b["c"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    highs = np.array([b["h"] for b in bars])

    # 1. PC Distance
    pc_lower = np.min(lows[-100:])
    pc_dist = ((closes[-1] / pc_lower) - 1.0) * 100.0 if pc_lower > 0 else 999.0
    s_pc = w_pc if pc_dist <= 5.0 else 0.0

    # 2. Fisher Transform (70)
    hl2 = (highs[-70:] + lows[-70:]) / 2.0
    min_l = np.min(lows[-70:])
    max_h = np.max(highs[-70:])
    raw_val = (hl2[-1] - min_l) / (max_h - min_l) if (max_h - min_l) > 0 else 0.5
    val = max(min(0.66 * (raw_val - 0.5) + 0.5 * 0.5, 0.999), -0.999)
    fisher = 0.5 * np.log((1.0 + val) / (1.0 - val))
    s_fisher = w_fisher if fisher <= -2.1 else 0.0

    # 3. TRIX (15)
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

    # 4. CCI (40)
    tp = (highs[-40:] + lows[-40:] + closes[-40:]) / 3.0
    sma_tp = np.mean(tp)
    mad = np.mean(np.abs(tp - sma_tp))
    cci = (tp[-1] - sma_tp) / (0.015 * mad) if mad > 0 else 0.0
    s_cci = w_cci if cci <= -160.0 else 0.0

    score = round(s_pc + s_fisher + s_trix + s_cci, 1)
    details = f"PC:{pc_dist:.1f}% | Fish:{fisher:.2f} | TRIX:{trix:.1f} | CCI:{cci:.0f}"
    return score, details


# --- STRATEGY 2 CALCULATION (Pine Technical View) ---
def calc_strategy_2(bars, pc_threshold, rsi_bottom_thresh):
    if not bars or len(bars) < 130:
        return "N/A", "Insufficient Data"

    closes = np.array([b["c"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    highs = np.array([b["h"] for b in bars])

    # 1. PC130 Proximity
    min_130 = np.min(lows[-130:])
    pc130_dist = ((closes[-1] / min_130) - 1.0) * 100.0 if min_130 > 0 else 999.0
    pc_signal = pc130_dist <= pc_threshold

    # 2. Adaptive Wave (ATR130 Channel)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    atr130 = np.mean(tr[-130:]) if len(tr) >= 130 else np.mean(tr)
    wave_bottom = min_130 + (atr130 * 0.5)
    wave_signal = closes[-1] <= wave_bottom

    # 3. SmartPulse RSI Channel (RSI 70 on 85-bar Channel)
    deltas = np.diff(closes[-85:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_signal = rsi <= rsi_bottom_thresh

    signals = []
    if pc_signal:
        signals.append("PC130 Proximity")
    if wave_signal:
        signals.append("Adaptive Wave")
    if rsi_signal:
        signals.append("RSI Channel Bottom")

    status = "TRIGGERED" if signals else "NEUTRAL"
    details = f"Signals: {', '.join(signals) if signals else 'None'} | PC130 Dist: {pc130_dist:.2f}% | RSI: {rsi:.1f}"
    return status, details


# --- INTERFACE DESIGN WITH TWO TABS ---
st.title("Lowline Analytics Dashboard")

# Global Timeframe input (Allows user to type anything)
st.sidebar.header("Global Configuration")
custom_tf = st.sidebar.text_input("Custom Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d):", value="15m")

tab1, tab2 = st.tabs(["Strategy 1 (Quantitative Score)", "Strategy 2 (Pine Technical View)"])

# ----------------- TAB 1: STRATEGY 1 -----------------
with tab1:
    st.subheader("Strategy 1 - Quantitative Multi-Factor Score")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        w_pc = st.number_input("PC Weight %", value=40, key="s1_pc")
    with col2:
        w_fisher = st.number_input("Fisher Weight %", value=30, key="s1_fish")
    with col3:
        w_trix = st.number_input("TRIX Weight %", value=20, key="s1_trix")
    with col4:
        w_cci = st.number_input("CCI Weight %", value=10, key="s1_cci")

    if st.button("Run Strategy 1 Scan", type="primary"):
        with st.spinner("Fetching top 250 assets and calculating Strategy 1..."):
            assets = fetch_top_250_assets()
            if not assets:
                st.error("Failed to retrieve asset list. Please retry.")
            else:
                results = []

                def worker_s1(item, idx):
                    symbol = item["symbol"]
                    is_hidden = item["is_hidden"]
                    bars = get_candles(symbol, custom_tf)

                    if is_hidden or not bars:
                        display_sym = f"⚠️ {symbol} [Hidden / No Data]"
                        price_str = "N/A"
                        score = 0.0
                        details = "Hidden asset / No data available"
                    else:
                        display_sym = symbol
                        price_str = f"${bars[-1]['c']:.6f}"
                        score, details = calc_strategy_1(bars, w_pc, w_fisher, w_trix, w_cci)

                    return {
                        "#": idx,
                        "Symbol": display_sym,
                        "Price": price_str,
                        "Setup Score / 100": score,
                        "Indicator Details": details,
                        "Chart": item["chart"],
                    }

                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    futures = [
                        executor.submit(worker_s1, item, idx)
                        for idx, item in enumerate(assets, 1)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())

                results.sort(key=lambda x: x["#"])
                st.success(f"Strategy 1 scan complete for {len(results)} assets!")
                st.dataframe(results, use_container_width=True)

# ----------------- TAB 2: STRATEGY 2 -----------------
with tab2:
    st.subheader("Strategy 2 - PC130 Proximity + Adaptive Wave + RSI Channel Bottom")
    col1, col2 = st.columns(2)
    with col1:
        pc_thresh = st.number_input(
            "PC130 Proximity Threshold (%)", value=3.0, key="s2_pc"
        )
    with col2:
        rsi_bottom = st.number_input(
            "RSI Channel Bottom Threshold", value=30.0, key="s2_rsi"
        )

    if st.button("Run Strategy 2 Scan", type="primary"):
        with st.spinner("Fetching top 250 assets and analyzing Technical Pine conditions..."):
            assets = fetch_top_250_assets()
            if not assets:
                st.error("Failed to retrieve asset list. Please retry.")
            else:
                results = []

                def worker_s2(item, idx):
                    symbol = item["symbol"]
                    is_hidden = item["is_hidden"]
                    bars = get_candles(symbol, custom_tf)

                    if is_hidden or not bars:
                        display_sym = f"⚠️ {symbol} [Hidden / No Data]"
                        price_str = "N/A"
                        status = "N/A"
                        details = "Hidden asset / No data available"
                    else:
                        display_sym = symbol
                        price_str = f"${bars[-1]['c']:.6f}"
                        status, details = calc_strategy_2(bars, pc_thresh, rsi_bottom)

                    return {
                        "#": idx,
                        "Symbol": display_sym,
                        "Price": price_str,
                        "Technical Status": status,
                        "Technical Breakdown": details,
                        "Chart": item["chart"],
                    }

                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    futures = [
                        executor.submit(worker_s2, item, idx)
                        for idx, item in enumerate(assets, 1)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())

                results.sort(key=lambda x: x["#"])
                st.success(f"Strategy 2 scan complete for {len(results)} assets!")
                st.dataframe(results, use_container_width=True)
