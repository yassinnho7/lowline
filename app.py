import concurrent.futures
from urllib.parse import quote
import numpy as np
import requests
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Lowline Dashboard", layout="wide")

API_BASE = "https://api.mexc.com/api/v3"
LISTING_API = "https://www.mexc.com/api/seo/coin/config/list"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

SUPPORTED_TIMEFRAMES = {
    "1m": (1, "1m"),
    "3m": (3, "1m"),
    "5m": (5, "5m"),
    "10m": (10, "5m"),
    "15m": (15, "15m"),
    "30m": (30, "30m"),
    "1h": (60, "60m"),
    "2h": (120, "60m"),
    "4h": (240, "4h"),
    "6h": (360, "4h"),
    "8h": (480, "4h"),
    "12h": (720, "4h"),
    "1d": (1440, "1d"),
    "2d": (2880, "1d"),
}


def fetch_top_250_assets():
    """Fetch top 250 symbols and flag hidden assets."""
    try:
        res = requests.get(LISTING_API, headers=HEADERS, timeout=10)
        res_json = res.json()
        data = res_json.get("data", [])
        if isinstance(data, dict):
            data = data.get("result", data.get("list", []))

        rows = []
        for item in data[:250]:
            symbol = item.get("symbolName") or item.get("currency") or item.get("symbol")
            is_hidden = (
                item.get("hide", False)
                or item.get("state") == "HIDE"
                or not item.get("enableFetch", True)
            )

            if symbol:
                sym_upper = symbol.upper()
                rows.append(
                    {
                        "symbol": sym_upper,
                        "is_hidden": is_hidden,
                        "chart": f"https://www.mexc.com/exchange/{quote(sym_upper)}_USDT",
                    }
                )
        return rows
    except Exception:
        return []


def get_candles(symbol, tf_str, target_bars=150):
    """Fetch OHLCV klines for specified timeframe."""
    _, interval = SUPPORTED_TIMEFRAMES.get(tf_str, (15, "15m"))
    url = f"{API_BASE}/klines"
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": target_bars}

    try:
        res = requests.get(url, params=params, timeout=5)
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


# --- STRATEGY 1: Setup Score (PC100 + Fisher70 + TRIX15 + CCI40) ---
def calc_strategy_1(bars, weights):
    if not bars or len(bars) < 100:
        return 0.0, "N/A"

    closes = np.array([b["c"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    highs = np.array([b["h"] for b in bars])

    # 1. PC100 Distance
    pc_lower = np.min(lows[-100:])
    pc_dist = ((closes[-1] / pc_lower) - 1.0) * 100.0 if pc_lower > 0 else 999.0
    s_pc = weights["pc"] if pc_dist <= 5.0 else 0.0

    # 2. Fisher Transform (70)
    hl2 = (highs[-70:] + lows[-70:]) / 2.0
    min_l = np.min(lows[-70:])
    max_h = np.max(highs[-70:])
    raw_val = (
        (hl2[-1] - min_l) / (max_h - min_l) if (max_h - min_l) > 0 else 0.5
    )
    val = max(min(0.66 * (raw_val - 0.5) + 0.5 * 0.5, 0.999), -0.999)
    fisher = 0.5 * np.log((1.0 + val) / (1.0 - val))
    s_fisher = weights["fisher"] if fisher <= -2.1 else 0.0

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
    trix = (
        ((e3[-1] - e3[-2]) / e3[-2]) * 10000.0 if len(e3) > 1 and e3[-2] != 0 else 0.0
    )
    s_trix = weights["trix"] if trix <= 0 else 0.0

    # 4. CCI (40)
    tp = (highs[-40:] + lows[-40:] + closes[-40:]) / 3.0
    sma_tp = np.mean(tp)
    mad = np.mean(np.abs(tp - sma_tp))
    cci = (tp[-1] - sma_tp) / (0.015 * mad) if mad > 0 else 0.0
    s_cci = weights["cci"] if cci <= -160.0 else 0.0

    score = round(s_pc + s_fisher + s_trix + s_cci, 1)
    details = f"PC:{pc_dist:.1f}%|Fish:{fisher:.2f}|CCI:{cci:.0f}"
    return score, details


# --- STRATEGY 2: Pine Technical View (PC130 Proximity + Adaptive Wave + RSI Channel) ---
def calc_strategy_2(bars):
    if not bars or len(bars) < 130:
        return "N/A"

    closes = np.array([b["c"] for b in bars])
    lows = np.array([b["l"] for b in bars])
    highs = np.array([b["h"] for b in bars])

    # 1. PC130 Proximity
    min_130 = np.min(lows[-130:])
    pc130_dist = ((closes[-1] / min_130) - 1.0) * 100.0 if min_130 > 0 else 999.0
    pc_signal = pc130_dist <= 3.0

    # 2. Adaptive Wave (ATR130 Channel)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])
        ),
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
    rsi_signal = rsi <= 30.0

    active_signals = []
    if pc_signal:
        active_signals.append("PC130")
    if wave_signal:
        active_signals.append("Wave")
    if rsi_signal:
        active_signals.append("RSI-Bot")

    return ", ".join(active_signals) if active_signals else "Neutral"


# --- STREAMLIT UI ---
st.title("Lowline Dashboard")

with st.sidebar:
    st.header("Settings")
    selected_tf = st.selectbox(
        "Timeframe:", options=list(SUPPORTED_TIMEFRAMES.keys()), index=4
    )

    st.subheader("Strategy 1 Weights (Total 100)")
    w_pc = st.number_input("PC Weight %", value=40)
    w_fisher = st.number_input("Fisher Weight %", value=30)
    w_trix = st.number_input("TRIX Weight %", value=20)
    w_cci = st.number_input("CCI Weight %", value=10)

weights = {"pc": w_pc, "fisher": w_fisher, "trix": w_trix, "cci": w_cci}

if st.button("Scan Market", type="primary"):
    with st.spinner("Processing market data..."):
        assets = fetch_top_250_assets()

        if not assets:
            st.warning("No assets retrieved.")
        else:
            results = []

            def analyze_asset(item, idx):
                symbol = item["symbol"]
                is_hidden = item["is_hidden"]

                bars = get_candles(symbol, selected_tf)

                if is_hidden or not bars:
                    display_symbol = f"⚠️ {symbol} [Hidden / No Data]"
                    price_str = "N/A"
                    score = 0.0
                    strat1_details = "N/A"
                    strat2_signals = "N/A"
                else:
                    display_symbol = symbol
                    price_str = f"${bars[-1]['c']:.6f}"
                    score, strat1_details = calc_strategy_1(bars, weights)
                    strat2_signals = calc_strategy_2(bars)

                return {
                    "#": idx,
                    "Symbol": display_symbol,
                    "Price": price_str,
                    "Strat 1 Score": score,
                    "Strat 1 Details": strat1_details,
                    "Strat 2 Signals": strat2_signals,
                    "Chart": item["chart"],
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(analyze_asset, item, idx)
                    for idx, item in enumerate(assets, 1)
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            results.sort(key=lambda x: x["#"])

            st.success(f"Scan complete. Displaying {len(results)} assets.")
            st.dataframe(results, use_container_width=True)
