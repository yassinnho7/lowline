import concurrent.futures
from urllib.parse import quote
import requests
import streamlit as st

# ضبط إعدادات الصفحة
st.set_page_config(page_title="Lowline - Live Market Dashboard", layout="wide")

API = "https://api.mexc.com/api/v3"
LISTING_API_URL = "https://www.mexc.com/api/seo/coin/config/list"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

# الأطر الزمنية المدعومة
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
    """جلب أحدث 250 عملة من رابط MEXC SEO API مع تمييز المخفية"""
    try:
        res = requests.get(LISTING_API_URL, headers=HEADERS, timeout=10)
        res_json = res.json()

        data = res_json.get("data", [])
        if isinstance(data, dict):
            data = data.get("result", data.get("list", []))

        rows = []
        # أخذ أحدث 250 عملة
        for item in data[:250]:
            symbol = item.get("symbolName") or item.get("currency") or item.get("symbol")

            # تحقق مما إذا كانت العملة مخفية أو غير نشطة
            is_hidden = (
                item.get("hide", False)
                or item.get("state") == "HIDE"
                or not item.get("enableFetch", True)
            )

            if symbol:
                symbol_clean = symbol.upper()
                rows.append(
                    {
                        "symbol": symbol_clean,
                        "name": item.get("symbolFullName", symbol_clean),
                        "is_hidden": is_hidden,
                        "chart": f"https://www.mexc.com/exchange/{quote(symbol_clean)}_USDT",
                    }
                )
        return rows
    except Exception as e:
        st.error(f"حدث خطأ أثناء جلب قائمة العملات: {e}")
        return []


def get_candles(symbol, tf_str, target_bars=100):
    """جلب بيانات الشموع للعملة"""
    tf_info = SUPPORTED_TIMEFRAMES.get(tf_str, (15, "15m"))
    _, interval = tf_info

    url = f"{API}/klines"
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


def calculate_strategy_1(bars, weights):
    """حساب مؤشرات الاستراتيجية الأولى"""
    if not bars or len(bars) < 40:
        return 0, "بيانات غير كافية"

    closes = [b["c"] for b in bars]
    lows = [b["l"] for b in bars]

    # PC100
    pc_lower = min(lows[-100:]) if len(lows) >= 100 else min(lows)
    pc_dist = ((closes[-1] / pc_lower) - 1) * 100 if pc_lower > 0 else 999
    pc_score = weights["pc"] if pc_dist <= 5 else 0

    # CCI40
    typical = [(b["h"] + b["l"] + b["c"]) / 3 for b in bars[-40:]]
    mean = sum(typical) / len(typical)
    mad = sum(abs(x - mean) for x in typical) / len(typical)
    cci = (typical[-1] - mean) / (0.015 * mad) if mad > 0 else 0
    cci_score = weights["cci"] if cci <= -160 else 0

    total_score = round(pc_score + cci_score, 1)
    return total_score, f"PC Dist: {pc_dist:.2f}% | CCI: {cci:.1f}"


# ------------------- واجهة المستخدم (Streamlit UI) -------------------

st.title("🚀 لوحة تحليل أسواق MEXC - Lowline")
st.markdown("جلب أحدث 250 عملة وتحليلها مع تمييز العملات المخفية.")

# شريط الإعدادات والتحكم
with st.sidebar:
    st.header("⚙️ إعدادات التحليل")
    selected_tf = st.selectbox(
        "⏱️ اختر الإطار الزمني (Timeframe):",
        options=list(SUPPORTED_TIMEFRAMES.keys()),
        index=4,  # الافتراضي 15m
    )

    st.subheader("أوزان الاستراتيجية (المجموع 100)")
    w_pc = st.number_input("وزن PC %", value=40)
    w_fisher = st.number_input("وزن Fisher %", value=30)
    w_trix = st.number_input("وزن TRIX %", value=20)
    w_cci = st.number_input("وزن CCI %", value=10)

weights = {"pc": w_pc, "fisher": w_fisher, "trix": w_trix, "cci": w_cci}

if st.button("🔍 فحص وتحليل العملات الان", type="primary"):
    with st.spinner("جاري جلب قائمة أحدث 250 عملة وتحليل البيانات..."):
        assets = fetch_top_250_assets()

        if not assets:
            st.warning("لم يتم العثور على عملات.")
        else:
            results = []

            def analyze_asset(item, idx):
                symbol = item["symbol"]
                is_hidden = item["is_hidden"]

                # محاولة جلب الشموع والسعر
                bars = get_candles(symbol, selected_tf)

                if is_hidden or not bars:
                    display_symbol = f"⚠️ {symbol} [Hidden / No Data]"
                    price_str = "N/A"
                    score = 0
                    details = "عملة مخفية / لا توجد بيانات سعر"
                else:
                    display_symbol = symbol
                    price_str = f"${bars[-1]['c']:.6f}"
                    score, details = calculate_strategy_1(bars, weights)

                return {
                    "#": idx,
                    "الرمز": display_symbol,
                    "السعر الحالي": price_str,
                    "النتيجة / 100": score,
                    "تفاصيل المؤشرات": details,
                    "الرابط": item["chart"],
                }

            # تسريع الفحص باستخدام التوازي (Multithreading)
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(analyze_asset, item, idx)
                    for idx, item in enumerate(assets, 1)
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            # ترتيب النتائج بنفس الترتيب الأصلي
            results.sort(key=lambda x: x["#"])

            st.success(f"تم فحص {len(results)} عملة بنجاح!")
            st.dataframe(results, use_container_width=True)
