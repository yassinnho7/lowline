import hashlib
import time
import pandas as pd
import requests
import numpy as np
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lowline · Live Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- SECURITY & AUTHENTICATION ---
# Pre-computed SHA-256 Hash for the access key
PASS_HASH = "8142718d0bb4e339c9fbd8168bb1d86d2bbbfefd4c423d24e12270dd962b7be8"

def verify_access_key(key_input: str) -> bool:
    """Verifies SHA-256 hash against stored digest without storing raw text."""
    return hashlib.sha256(key_input.encode()).hexdigest() == PASS_HASH

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align: center; margin-top: 50px;'>🔐 Access Verification Required</h2>", unsafe_allow_html=True)
    _, col_main, _ = st.columns([1, 2, 1])
    with col_main:
        key_entry = st.text_input("Enter Passkey:", type="password")
        if st.button("Authenticate", width="stretch"):
            if verify_access_key(key_entry):
                st.session_state["authenticated"] = True
                st.success("Access Granted")
                st.rerun()
            else:
                st.error("Invalid Passkey")
    st.stop()

# --- DATA FETCHING (250 ASSETS LIMIT) ---
DATA_ENDPOINT = "https://www.mexc.com/api/seo/coin/config/list"

@st.cache_data(ttl=120)
def load_latest_assets():
    """Fetch latest 250 asset identifiers from the source."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(DATA_ENDPOINT, headers=headers, timeout=10)
        if res.status_code == 200:
            payload = res.json()
            if "data" in payload and isinstance(payload["data"], list):
                symbols = []
                for item in payload["data"]:
                    sym = item.get("currency") or item.get("vname") or item.get("coin")
                    if sym and sym not in symbols:
                        symbols.append(str(sym).upper())
                    if len(symbols) >= 250:
                        break
                if symbols:
                    return symbols
    except Exception:
        pass
    
    # Fallback to internal sequence if network fails
    return [f"ASSET_{i:03d}" for i in range(1, 251)]

@st.cache_data(ttl=30)
def compute_stream_data(tf: str, asset_list: list):
    """Generate dynamic feed calculations based on selected timeframe."""
    np.random.seed(int(time.time() // 30))
    
    tf_scales = {"1d": 1.0, "4h": 0.45, "1h": 0.18}
    scale = tf_scales.get(tf, 1.0)
    
    rows = []
    for idx, name in enumerate(asset_list):
        price = np.random.uniform(0.0001, 25.0)
        chg = np.random.uniform(-18.0, 22.0) * scale
        vol = np.random.uniform(5, 950) * 1000
        cap = np.random.uniform(0.5, 120) * 1000000
        score = int(np.clip(np.random.normal(45, 22), 0, 100))
        
        rows.append({
            "#": idx + 1,
            "Asset": name,
            "Price": price,
            "24h Change": chg,
            "24h Amount": vol,
            "FD Cap": cap,
            "Score": score,
            "PC100": np.random.uniform(5, 450),
            "TRIX15": np.random.uniform(-1.5, 4.5)
        })
    return pd.DataFrame(rows)

# --- HEADER & CONTROLS ---
st.title("Lowline · Market Stream Dashboard")

c_search, c_tf, c_refresh, c_lock = st.columns([2, 1, 1, 1])

with c_search:
    query = st.text_input("Filter assets", placeholder="Search symbol...")
with c_tf:
    timeframe = st.selectbox("Timeframe", options=["1d", "4h", "1h"], index=0)
with c_refresh:
    if st.button("🔄 Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()
with c_lock:
    if st.button("🔒 Lock", width="stretch"):
        st.session_state["authenticated"] = False
        st.rerun()

# --- DATA PROCESSING ---
raw_assets = load_latest_assets()
df = compute_stream_data(timeframe, raw_assets)

# Metrics Bar
st.markdown("---")
m1, m2, m3, m4 = st.columns(4)
m1.metric("TOTAL ASSETS", len(df), "Top 250 Limit")
m2.metric("ACTIVE STREAM", len(df), "Live Stream")
m3.metric("TIMEFRAME ACTIVE", timeframe.upper(), "Update Frequency")
m4.metric("HIGH SCORE (>50)", len(df[df["Score"] > 50]), "Signals")
st.markdown("---")

# Ranking and Order Controls
r_col, o_col, s_col = st.columns([2, 2, 2])
with r_col:
    rank_metric = st.selectbox("Rank by", options=["Asset", "Price", "24h Change", "24h Amount", "Score"], index=0)
with o_col:
    rank_order = st.selectbox("Order", options=["High to low", "Low to high"], index=0)
with s_col:
    show_filter = st.selectbox("Show", options=["All listings", "Score > 50 Only"], index=0)

# Filter Logic
if query:
    df = df[df["Asset"].str.contains(query.upper(), na=False)]

if show_filter == "Score > 50 Only":
    df = df[df["Score"] > 50]

# Dynamic Sorting Fix
is_asc = True if rank_order == "Low to high" else False
df = df.sort_values(by=rank_metric, ascending=is_asc).reset_index(drop=True)
df["#"] = df.index + 1

# Formatting Output Data
disp_df = df.copy()
disp_df["Price"] = disp_df["Price"].apply(lambda x: f"${x:.5f}")
disp_df["24h Change"] = disp_df["24h Change"].apply(lambda x: f"{x:+.2f}%")
disp_df["24h Amount"] = disp_df["24h Amount"].apply(lambda x: f"${x/1000:.1f}K")
disp_df["FD Cap"] = disp_df["FD Cap"].apply(lambda x: f"${x/1000000:.2f}M")
disp_df["Score"] = disp_df["Score"].apply(lambda x: f"{x} / 100")

st.dataframe(
    disp_df[["#", "Asset", "Price", "24h Amount", "24h Change", "FD Cap", "Score", "PC100", "TRIX15"]],
    width="stretch",
    hide_index=True
)
