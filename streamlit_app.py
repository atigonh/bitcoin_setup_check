import time
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime
from zoneinfo import ZoneInfo
def display(x):
    return None


# ===================== SETTINGS =====================

TICKER = "BTC-USD"

VWAP_WINDOW = 360          # 6-hour rolling VWAP
RANGE_WINDOW = 60          # local 60-minute high/low

ENTRY_BUFFER_PCT = 0.03

READY_DISTANCE = 0.10
NEAR_DISTANCE = 0.25
MAX_ENTRY_DISTANCE = 0.60
MAX_VWAP_DISTANCE = 2.50

MIN_R2_10 = 0.55
MIN_R2_15 = 0.60
MIN_R2_30 = 0.55

A_PLUS_SCORE = 80
A_SCORE = 68
B_PLUS_SCORE = 55

RUN_LOOP = True
LOOP_MINUTES = 1
MAX_RUNTIME_HOURS = 3


# ===================== OPTIONAL GOOGLE SHEETS =====================

USE_GOOGLE_SHEETS = False
GOOGLE_SHEET_URL = "PASTE_YOUR_GOOGLE_SHEET_URL_HERE"


# ===================== HELPERS =====================

def regression_stats(values):
    y = np.asarray(values, dtype=float)

    if len(y) < 2 or np.any(~np.isfinite(y)):
        return np.nan, np.nan

    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    slope_pct = slope / np.mean(y) * 100

    return float(r2), float(slope_pct)


def reg(close, bars):
    if len(close) < bars:
        return np.nan, np.nan

    return regression_stats(
        close.iloc[-bars:].values
    )


def ret(close, bars):
    if len(close) <= bars:
        return np.nan

    return (
        float(close.iloc[-1])
        / float(close.iloc[-bars - 1])
        - 1
    ) * 100


def rating(score, mandatory):
    if score >= A_PLUS_SCORE and mandatory:
        return "A+"

    if score >= A_SCORE:
        return "A"

    if score >= B_PLUS_SCORE:
        return "B+"

    return "WAIT"


def entry_status(distance, vwap_distance):
    if vwap_distance > MAX_VWAP_DISTANCE:
        return "TOO EXTENDED"

    if distance > MAX_ENTRY_DISTANCE:
        return "TOO FAR"

    if distance <= READY_DISTANCE:
        return "READY"

    if distance <= NEAR_DISTANCE:
        return "NEAR"

    return "WAIT"


# ===================== SCANNER =====================

def btc_scanner():
    print("\n" + "=" * 72)
    print("BTC-USD 24/7 A+ LONG / SHORT SCANNER")
    print("=" * 72)

    df = yf.download(
        TICKER, period="5d", interval="1m",
        auto_adjust=True, prepost=True,
        progress=False, threads=False
    )

    if df is None or df.empty:
        raise ValueError("BTC download returned no data.")

    if isinstance(df.columns, pd.MultiIndex):
        if TICKER in df.columns.get_level_values(0):
            df = df[TICKER].copy()
        elif TICKER in df.columns.get_level_values(-1):
            df = df.xs(TICKER, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    df = df[["Open","High","Low","Close","Volume"]].copy()
    df = df.dropna(subset=["Open","High","Low","Close"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if len(df) < VWAP_WINDOW + 10:
        raise ValueError("Not enough 1-minute data.")

    price = float(df["Close"].iloc[-1])

    # Rolling 6-hour VWAP. If Yahoo volume is unusable, use rolling typical-price mean.
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    roll_pv = pv.rolling(VWAP_WINDOW).sum()
    roll_vol = df["Volume"].rolling(VWAP_WINDOW).sum()
    df["VWAP"] = roll_pv / roll_vol.replace(0, np.nan)

    if pd.isna(df["VWAP"].iloc[-1]):
        df["VWAP"] = typical.rolling(VWAP_WINDOW).mean()

    vwap = float(df["VWAP"].iloc[-1])
    old_vwap = float(df["VWAP"].iloc[-6])
    vwap_slope_5m = (vwap / old_vwap - 1) * 100

    above_vwap = (price / vwap - 1) * 100
    below_vwap = (vwap / price - 1) * 100

    # Momentum + regressions
    r1, r3, r5, r15, r30 = [ret(df["Close"], n) for n in [1,3,5,15,30]]
    r2_5, slope_5 = reg(df["Close"], 5)
    r2_10, slope_10 = reg(df["Close"], 10)
    r2_15, slope_15 = reg(df["Close"], 15)
    r2_30, slope_30 = reg(df["Close"], 30)

    # 5-minute confirmation
    five = df.resample("5min").agg({
        "Open":"first","High":"max","Low":"min",
        "Close":"last","Volume":"sum"
    }).dropna(subset=["Close"])

    five_r2, five_slope = reg(five["Close"], min(6, len(five)))
    five_return = (
        (five["Close"].iloc[-1] / five["Close"].iloc[-3] - 1) * 100
        if len(five) >= 3 else np.nan
    )

    # Rolling volume ratio: current 5m vs median recent 5m volume
    current_5m_vol = float(df["Volume"].tail(5).sum())
    hist_5m = df["Volume"].iloc[-65:-5].rolling(5).sum().dropna()
    volume_ratio_5m = (
        current_5m_vol / hist_5m.median()
        if len(hist_5m) and hist_5m.median() > 0 else np.nan
    )

    # Local 60-minute high/low replaces HOD/LOD and NYSE ORB
    recent = df.tail(RANGE_WINDOW)
    range_high = float(recent["High"].max())
    range_low = float(recent["Low"].min())
    dist_high = (range_high / price - 1) * 100
    dist_low = (price / range_low - 1) * 100

    # LONG breakout / SHORT breakdown entries
    high3 = float(df["High"].tail(3).max())
    low3 = float(df["Low"].tail(3).min())

    long_entry = high3 * (1 + ENTRY_BUFFER_PCT / 100)
    short_entry = low3 * (1 - ENTRY_BUFFER_PCT / 100)

    long_dist = (long_entry / price - 1) * 100
    short_dist = (price / short_entry - 1) * 100

    long_entry_vwap = (long_entry / vwap - 1) * 100
    short_entry_vwap = (vwap / short_entry - 1) * 100

    long_status = entry_status(long_dist, long_entry_vwap)
    short_status = entry_status(short_dist, short_entry_vwap)

    # ===================== LONG SCORE =====================
    long_score = 0
    if price > vwap and vwap_slope_5m > 0: long_score += 20
    elif price > vwap: long_score += 10

    if r2_10 >= MIN_R2_10 and slope_10 > 0: long_score += 6
    if r2_15 >= MIN_R2_15 and slope_15 > 0: long_score += 8
    if r2_30 >= MIN_R2_30 and slope_30 > 0: long_score += 6

    if r3 > 0: long_score += 4
    if r5 > 0: long_score += 5
    if r15 > 0: long_score += 6

    if five_slope > 0 and five_return > 0: long_score += 15

    if not pd.isna(volume_ratio_5m):
        if volume_ratio_5m >= 2: long_score += 10
        elif volume_ratio_5m >= 1.5: long_score += 7
        elif volume_ratio_5m >= 1: long_score += 4

    if dist_high <= 0.25: long_score += 10
    elif dist_high <= 0.50: long_score += 7
    elif dist_high <= 1.00: long_score += 4

    if 0 < above_vwap <= 1.0: long_score += 10
    elif 1.0 < above_vwap <= MAX_VWAP_DISTANCE: long_score += 5

    long_mandatory = (
        price > vwap and vwap_slope_5m > 0
        and above_vwap <= MAX_VWAP_DISTANCE
        and slope_10 > 0 and slope_15 > 0
        and five_slope > 0 and r5 > 0
        and dist_high <= 1.0
        and (pd.isna(volume_ratio_5m) or volume_ratio_5m >= 1.0)
    )

    # ===================== SHORT SCORE =====================
    short_score = 0
    if price < vwap and vwap_slope_5m < 0: short_score += 20
    elif price < vwap: short_score += 10

    if r2_10 >= MIN_R2_10 and slope_10 < 0: short_score += 6
    if r2_15 >= MIN_R2_15 and slope_15 < 0: short_score += 8
    if r2_30 >= MIN_R2_30 and slope_30 < 0: short_score += 6

    if r3 < 0: short_score += 4
    if r5 < 0: short_score += 5
    if r15 < 0: short_score += 6

    if five_slope < 0 and five_return < 0: short_score += 15

    if not pd.isna(volume_ratio_5m):
        if volume_ratio_5m >= 2: short_score += 10
        elif volume_ratio_5m >= 1.5: short_score += 7
        elif volume_ratio_5m >= 1: short_score += 4

    if dist_low <= 0.25: short_score += 10
    elif dist_low <= 0.50: short_score += 7
    elif dist_low <= 1.00: short_score += 4

    if 0 < below_vwap <= 1.0: short_score += 10
    elif 1.0 < below_vwap <= MAX_VWAP_DISTANCE: short_score += 5

    short_mandatory = (
        price < vwap and vwap_slope_5m < 0
        and below_vwap <= MAX_VWAP_DISTANCE
        and slope_10 < 0 and slope_15 < 0
        and five_slope < 0 and r5 < 0
        and dist_low <= 1.0
        and (pd.isna(volume_ratio_5m) or volume_ratio_5m >= 1.0)
    )

    long_rating = rating(long_score, long_mandatory)
    short_rating = rating(short_score, short_mandatory)

    common = {
        "Ticker": TICKER, "Price": price, "VWAP": vwap,
        "VWAP_Slope_5m": vwap_slope_5m,
        "R2_5m": r2_5, "R2_10m": r2_10, "R2_15m": r2_15, "R2_30m": r2_30,
        "Slope_5m": slope_5, "Slope_10m": slope_10,
        "Slope_15m": slope_15, "Slope_30m": slope_30,
        "Return_1m%": r1, "Return_3m%": r3, "Return_5m%": r5,
        "Return_15m%": r15, "Return_30m%": r30,
        "5m_Trend_R2": five_r2, "5m_Trend_Slope": five_slope,
        "Volume_Ratio_5m": volume_ratio_5m,
        "Range_60m_High": range_high, "Range_60m_Low": range_low,
        "Time_UTC": df.index[-1],
    }

    rows = [
        {
            **common, "Direction":"LONG", "Rating":long_rating,
            "Score":long_score, "A+_Setup":long_rating=="A+",
            "Entry_Status":long_status, "Entry_Est":long_entry,
            "Entry_Distance%":long_dist, "VWAP_Distance%":above_vwap,
            "Entry_VWAP_Distance%":long_entry_vwap,
            "Distance_Range_Extreme%":dist_high,
        },
        {
            **common, "Direction":"SHORT", "Rating":short_rating,
            "Score":short_score, "A+_Setup":short_rating=="A+",
            "Entry_Status":short_status, "Entry_Est":short_entry,
            "Entry_Distance%":short_dist, "VWAP_Distance%":below_vwap,
            "Entry_VWAP_Distance%":short_entry_vwap,
            "Distance_Range_Extreme%":dist_low,
        }
    ]

    scan_df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    print(f"\nScan: {pt.strftime('%Y-%m-%d %I:%M:%S %p')} PT")
    print(f"BTC: ${price:,.2f} | Rolling VWAP: ${vwap:,.2f}")

    cols = [
        "Direction","Rating","Score","A+_Setup","Entry_Status",
        "Price","Entry_Est","Entry_Distance%","VWAP","VWAP_Distance%",
        "VWAP_Slope_5m","R2_10m","R2_15m","R2_30m",
        "Slope_10m","Slope_15m","Return_3m%","Return_5m%",
        "Return_15m%","Volume_Ratio_5m","Distance_Range_Extreme%","Time_UTC"
    ]
    display(scan_df[cols].round(4))

    ready = scan_df[(scan_df["Rating"]=="A+") & (scan_df["Entry_Status"]=="READY")]
    if ready.empty:
        print("\nNo A+ READY BTC setup right now.")
    else:
        print("\n*** A+ READY ***")
        display(ready[cols].round(4))

    return scan_df


# ===================== STREAMLIT GUI =====================
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="บิทคอยน้ายศ", page_icon="₿", layout="wide")

INTERVAL_SECONDS = 30
DURATION_MIN = 10
MAX_SCANS = (DURATION_MIN * 60) // INTERVAL_SECONDS
PT = ZoneInfo("America/Los_Angeles")

for k,v in {"running":False,"started":None,"next_scan":None,"count":0,"latest":None,"history":[]}.items():
    if k not in st.session_state:
        st.session_state[k]=v

st.title("₿ BTC A+ Live Scanner for น้ายศ")
st.caption("LONG + SHORT • refresh every 30 seconds • stops after 10 minutes")

c1,c2=st.columns(2)
if c1.button("▶ START 10-MIN SCANNER", type="primary", use_container_width=True, disabled=st.session_state.running):
    now=datetime.now(PT)
    st.session_state.running=True; st.session_state.started=now; st.session_state.next_scan=now
    st.session_state.count=0; st.session_state.latest=None; st.session_state.history=[]
    st.rerun()
if c2.button("■ STOP", use_container_width=True, disabled=not st.session_state.running):
    st.session_state.running=False; st.rerun()

now=datetime.now(PT)
if st.session_state.running:
    if now-st.session_state.started >= timedelta(minutes=DURATION_MIN) or st.session_state.count >= MAX_SCANS:
        st.session_state.running=False
    elif now >= st.session_state.next_scan:
        try:
            result=btc_scanner()
            st.session_state.latest=result
            st.session_state.count += 1
            for _,r in result.iterrows():
                st.session_state.history.append({
                    "Time_PT":now.strftime("%I:%M:%S %p"), "Direction":r["Direction"],
                    "Rating":r["Rating"], "Score":r["Score"], "Entry_Status":r["Entry_Status"],
                    "Price":r["Price"], "Entry_Est":r["Entry_Est"], "VWAP":r["VWAP"]})
            st.session_state.next_scan=now+timedelta(seconds=INTERVAL_SECONDS)
        except Exception as e:
            st.error(f"Scan error: {e}")
            st.session_state.next_scan=now+timedelta(seconds=INTERVAL_SECONDS)
        st.rerun()

if st.session_state.running:
    sec=max(0,int((st.session_state.next_scan-datetime.now(PT)).total_seconds()))
    a,b,c=st.columns(3)
    a.metric("Status","RUNNING")
    b.metric("Scans",f"{st.session_state.count} / {MAX_SCANS}")
    c.metric("Next scan",f"{sec//60:02d}:{sec%60:02d}")

    # Red countdown bar: full just after a scan, shrinking toward zero.
    interval_sec = INTERVAL_SECONDS
    progress_remaining = min(1.0, max(0.0, sec / interval_sec))
    st.markdown("""
        <style>
        div[data-testid="stProgress"] > div > div > div {
            background-color: #ff2b2b !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.progress(progress_remaining)
else:
    st.metric("Status","STOPPED")

if st.session_state.latest is not None:
    df=st.session_state.latest
    st.metric("BTC Price",f"${df.iloc[0]['Price']:,.2f}")
    left,right=st.columns(2)
    for col,direction in [(left,"LONG"),(right,"SHORT")]:
        r=df[df["Direction"]==direction].iloc[0]
        with col:
            # Bordered card makes LONG and SHORT visually distinct.
            with st.container(border=True):
                st.markdown(f"<h3 style='text-align:center; margin-top:0;'>{direction}</h3>", unsafe_allow_html=True)
                x,y=st.columns(2)
                with x:
                    st.caption("Rating")
                    st.markdown(f"### **{r['Rating']}**")
                y.metric("Score",int(r["Score"]))
                x,y=st.columns(2); x.metric("Entry Status",r["Entry_Status"]); y.metric("Entry",f"${r['Entry_Est']:,.2f}")
                st.metric("VWAP",f"${r['VWAP']:,.2f}")
                if r["Rating"]=="A+" and r["Entry_Status"]=="READY": st.success(f"A+ {direction} — READY")
else:
    st.info("Press START to run the first scan.")

if st.session_state.history:
    st.subheader("Session History")
    history_df = pd.DataFrame(st.session_state.history).iloc[::-1].round(4).reset_index(drop=True)

    # Each scan adds one LONG and one SHORT row, so the latest two sessions
    # are the newest four rows. Older sessions are faded gray.
    def fade_old_sessions(row):
        if row.name >= 4:
            return ["color: #9a9a9a; opacity: 0.55"] * len(row)
        return [""] * len(row)

    history_styled = history_df.style.apply(fade_old_sessions, axis=1)
    st.dataframe(history_styled,use_container_width=True,hide_index=True)

if st.session_state.running:
    time.sleep(1)
    st.rerun()
