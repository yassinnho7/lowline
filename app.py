import concurrent.futures
import math
import re
import time
from urllib.parse import quote
import gradio as gr
import requests

API = "https://api.mexc.com/api/v3"
LISTING_API_URL = "https://www.mexc.com/api/seo/coin/config/list"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

# 1. دعم التحويل للأطر الزمنية المختلفة
SUPPORTED_TIMEFRAMES = {
    "1m": (1, "1m"), "3m": (3, "1m"), "5m": (5, "5m"),
    "10m": (10, "5m"), "15m": (15, "15m"), "30m": (30, "30m"),
    "1h": (60, "60m"), "2h": (120, "60m"), "4h": (240, "4h"),
    "6h": (360, "4h"), "8h": (480, "4h"), "12h": (720, "4h"),
    "1d": (1440, "1d"), "2d": (2880, "1d")
}

def fetch_top_250_assets():
    """جلب قائمة أول 250 عملة من MEXC"""
    try:
        res = requests.get(LISTING_API_URL, headers=HEADERS, timeout=10)
        data = res.json().get("data", [])
        if isinstance(data, dict):
            data = data.get("result", data.get("list", []))
        
        rows = []
        for item in data[:250]:
            symbol = item.get("symbolName") or item.get("currency") or item.get("symbol")
            if symbol:
                rows.append({
                    "symbol": symbol.upper(),
                    "name": item.get("symbolFullName", symbol),
                    "chart": f"https://www.mexc.com/exchange/{quote(symbol)}_USDT"
                })
        return rows
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def get_candles(symbol, tf_str, target_bars=100):
    """جلب وبناء الشموع بحسب الإطار الزمني المحدد من قبل المستخدم"""
    tf_info = SUPPORTED_TIMEFRAMES.get(tf_str, (15, "15m"))
    minutes, interval = tf_info
    
    url = f"{API}/klines"
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": target_bars}
    res = requests.get(url, params=params, timeout=10)
    
    if res.status_code != 200:
        return []
    
    data = res.json()
    bars = []
    for item in data:
        bars.append({
            "t": item[0], "o": float(item[1]), "h": float(item[2]),
            "l": float(item[3]), "c": float(item[4]), "v": float(item[5])
        })
    return bars

def calculate_strategy_1(bars, weights):
    """حساب الاستراتيجية الأولى (النتيجة الإجمالية بناءً على الأوزان)"""
    if len(bars) < 40:
        return 0, "بيانات غير كافية"
    
    closes = [b["c"] for b in bars]
    lows = [b["l"] for b in bars]
    
    # 1. PC100 (أدنى سعر)
    pc_lower = min(lows[-100:]) if len(lows) >= 100 else min(lows)
    pc_dist = ((closes[-1] / pc_lower) - 1) * 100 if pc_lower > 0 else 999
    pc_score = weights["pc"] if pc_dist <= 5 else 0
    
    # 2. CCI (مؤشر قناة البضائع)
    typical = [(b["h"] + b["l"] + b["c"]) / 3 for b in bars[-40:]]
    mean = sum(typical) / len(typical)
    mad = sum(abs(x - mean) for x in typical) / len(typical)
    cci = (typical[-1] - mean) / (0.015 * mad) if mad > 0 else 0
    cci_score = weights["cci"] if cci <= -160 else 0
    
    # مجموع النتيجة الهيكلية
    total_score = round(pc_score + cci_score, 1)
    return total_score, f"PC Dist: {pc_dist:.2f}% | CCI: {cci:.1f}"

# بناء واجهة Gradio للتفاعل
with gr.Blocks(title="Lowline - Live Market Dashboard") as demo:
    gr.Markdown("# 🚀 لوحة تحليل أسواق MEXC - التحديث الجديد")
    
    with gr.Row():
        tf_input = gr.Dropdown(
            choices=list(SUPPORTED_TIMEFRAMES.keys()), 
            value="15m", 
            label="⏱️ اختر الإطار الزمني (Timeframe)"
        )
        w_pc = gr.Number(value=40, label="وزن PC %")
        w_fisher = gr.Number(value=30, label="وزن Fisher %")
        w_trix = gr.Number(value=20, label="وزن TRIX %")
        w_cci = gr.Number(value=10, label="وزن CCI %")
        
    btn_scan = gr.Button("🔍 فحص وعرض العملات", variant="primary")
    output_table = gr.Dataframe(headers=["#", "الرمز", "السعر الحالي", "النتيجة / 100", "تفاصيل الاستراتيجية"])
    
    def process_scan(tf, pc, fisher, trix, cci):
        weights = {"pc": pc, "fisher": fisher, "trix": trix, "cci": cci}
        assets = fetch_top_250_assets()
        
        results = []
        for idx, item in enumerate(assets[:30], 1):  # فحص عينة للتوضيح بسرعة
            bars = get_candles(item["symbol"], tf)
            if bars:
                price = bars[-1]["c"]
                score, details = calculate_strategy_1(bars, weights)
                results.append([idx, item["symbol"], price, score, details])
                
        return results

    btn_scan.click(process_scan, inputs=[tf_input, w_pc, w_fisher, w_trix, w_cci], outputs=[output_table])

if __name__ == "__main__":
    demo.launch()
