# ============================================================
# JP Stock Market Model - COMPLETE LOCAL WEB DASHBOARD
# ============================================================
# Mobile + Laptop
# All NSE stocks
# BUY / HOLD / SELL
# Risk %
# Target / Stop Loss
# Holding Period
# Portfolio
# Past Predictions + Success Rate
# Sector Analysis
# Find Stock
# Live Market + Closing Market Analysis
# TradingView + Technical Charts
# ============================================================

import os
import re
import math
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

HISTORY_FILE = APP_DIR / "recommendation_history.csv"
PORTFOLIO_FILE = APP_DIR / "my_portfolio.csv"
RESULT_FILE = APP_DIR / "latest_results.csv"
TRADES_FILE = APP_DIR / "auto_trades_tracker.csv"
PAPER_FILE = APP_DIR / "paper_portfolio.csv"
LEARNING_FILE = APP_DIR / "model_learning.json"
STRATEGY_BT_FILE = APP_DIR / "strategy_backtest_results.csv"
STRATEGY_LIVE_FILE = APP_DIR / "strategy_live_signals.csv"
STRATEGY_PREV_FILE = APP_DIR / "strategy_live_signals_prev.csv"
STRATEGY_META_FILE = APP_DIR / "strategy_live_meta.json"
STRATEGY_ALL_MAP_FILE = APP_DIR / "strategy_all_stocks_map.csv"
STRATEGY_ALL_SUMMARY_FILE = APP_DIR / "strategy_all_stocks_summary.csv"
STRATEGY_ALL_META_FILE = APP_DIR / "strategy_all_stocks_meta.json"
SYNC_VERSION_FILE = APP_DIR / "realtime_sync_version.json"
MY_STRATEGY_PARAMS_FILE = APP_DIR / "my_strategy_params.json"
NSE_UNIVERSE_CACHE = APP_DIR / "nse_equity_universe.csv"
WATCHLIST_FILE = APP_DIR / "user_watchlist.json"
WATCHLIST_NOTES_FILE = APP_DIR / "user_watchlist_notes.json"
SAVED_QUERIES_FILE = APP_DIR / "saved_scanner_queries.json"
FUND_SCREEN_CACHE = APP_DIR / "fund_screen_cache.json"
INDEX_LIST_CACHE = APP_DIR / "index_constituents_cache.json"

MAX_SCAN_STOCKS = 10000
QUICK_SCAN_STOCKS = 120   # Cloud-friendly default (Nifty-like liquid set)
SCAN_CACHE_HOURS = 6     # reuse last scan instead of hitting NSE/Yahoo again
DEFAULT_HOLD_DAYS = 15
TOP_DEFAULT = 25

# Default auto-refresh interval (seconds). User can change in sidebar.
LIVE_REFRESH_SECONDS = 30  # Cloud-friendly: avoid hammering NSE/Yahoo

INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="JP Stock Market Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar fully hidden via CSS
)


# ============================================================
# CSS — desktop + strong mobile visibility
# ============================================================

st.markdown(
    """
    <style>
    /* Base — clean product UI */
    .block-container {
        padding-top: 5.25rem; /* room for fixed JP header */
        padding-bottom: 2.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        border: 1px solid #1e293b;
        background: #0f172a;
        color: #e2e8f0;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #38bdf8;
        color: #f0f9ff;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }
    .stock-card {
        border: 1px solid #dddddd;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 12px;
        background: white;
    }
    .small-text { font-size: 13px; color: #666666; }
    .buy-box { border-left: 6px solid #1a9b5f; padding-left: 12px; }
    .sell-box { border-left: 6px solid #d93025; padding-left: 12px; }
    .hold-box { border-left: 6px solid #e0a800; padding-left: 12px; }
    .watch-box { border-left: 6px solid #777777; padding-left: 12px; }
    .danger-box { border: 1px solid #d93025; border-radius: 10px; padding: 12px; }
    .success-box { border: 1px solid #1a9b5f; border-radius: 10px; padding: 12px; }

    /* ===== PHONE / SMALL TABLET ===== */
    @media (max-width: 900px) {
        html, body, [data-testid="stAppViewContainer"] {
            font-size: 16px !important;
            -webkit-text-size-adjust: 100% !important;
        }
        .block-container {
            padding: 0.4rem 0.55rem 5rem 0.55rem !important;
            max-width: 100vw !important;
        }
        /* Headings */
        h1 { font-size: 1.4rem !important; line-height: 1.2 !important; margin-bottom: 0.4rem !important; }
        h2 { font-size: 1.2rem !important; line-height: 1.25 !important; }
        h3, h4 { font-size: 1.05rem !important; }
        p, span, label, .stMarkdown, .stCaption, [data-testid="stCaption"] {
            font-size: 0.95rem !important;
            line-height: 1.4 !important;
        }
        /* Metrics — readable on small screens */
        [data-testid="stMetric"] {
            background: rgba(30,41,59,0.55);
            border-radius: 10px;
            padding: 8px 10px !important;
            margin-bottom: 6px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            opacity: 0.9;
        }
        div[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
        /* Full-width touch buttons */
        .stButton > button, .stDownloadButton > button {
            min-height: 3rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            width: 100% !important;
            border-radius: 10px !important;
            padding: 0.55rem 0.75rem !important;
        }
        /* Inputs */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            min-height: 2.75rem !important;
            font-size: 1rem !important;
        }
        /* Columns stack more cleanly */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.35rem !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            min-width: min(100%, 140px) !important;
            flex: 1 1 140px !important;
        }
        /* Tables — horizontal scroll, larger text */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] table,
        .stDataFrame {
            font-size: 0.85rem !important;
            width: 100% !important;
        }
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        /* Sidebar full width when open */
        section[data-testid="stSidebar"] {
            width: min(100vw, 320px) !important;
            min-width: min(100vw, 280px) !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            min-height: 2.85rem !important;
            margin-bottom: 0.25rem !important;
        }
        /* Expanders easier to tap */
        .streamlit-expanderHeader, [data-testid="stExpander"] summary {
            font-size: 1rem !important;
            min-height: 2.75rem !important;
            padding: 0.5rem 0.6rem !important;
        }
        /* Tabs */
        button[data-baseweb="tab"] {
            font-size: 0.9rem !important;
            padding: 0.6rem 0.75rem !important;
            min-height: 2.5rem !important;
        }
        /* Plotly charts */
        .js-plotly-plot, .plotly {
            max-width: 100% !important;
        }
        /* Reduce crowded element gaps */
        [data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
        /* Alert / info boxes */
        [data-testid="stAlert"] { font-size: 0.9rem !important; padding: 0.65rem !important; }
        /* Radio / checkbox */
        .stRadio label, .stCheckbox label { font-size: 0.95rem !important; }
        /* Hide excess top chrome spacing */
        header[data-testid="stHeader"] { background: transparent; }
    }

    /* Very small phones */
    @media (max-width: 400px) {
        h1 { font-size: 1.2rem !important; }
        div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        [data-testid="stHorizontalBlock"] > div {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }

    /* Smooth scroll + soft motion */
    html { scroll-behavior: smooth; }
    @media (prefers-reduced-motion: no-preference) {
        .block-container {
            animation: nseFadeIn 0.35s ease-out;
        }
        @keyframes nseFadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    }

    /* ===== Clean product typography & controls ===== */
    html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, label {
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    h1, h2, h3, h4, [data-testid="stMetricValue"] {
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;
        letter-spacing: -0.025em;
    }
    /* Select / dropdown */
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: #334155 !important;
        min-height: 2.6rem !important;
        background: #0f172a !important;
    }
    div[data-baseweb="select"] {
        font-size: 0.95rem !important;
    }
    /* Inputs */
    .stTextInput input, .stNumberInput input {
        border-radius: 10px !important;
        border-color: #334155 !important;
        background: #0f172a !important;
    }
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    /* Section headers */
    h1 {
        font-size: 1.75rem !important;
        margin-bottom: 0.35rem !important;
    }
    h2, h3 {
        margin-top: 0.75rem !important;
        margin-bottom: 0.35rem !important;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 10px 12px !important;
    }
    /* Expandable sections */
    [data-testid="stExpander"] {
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        background: #0f172a !important;
    }
    /* Sidebar menu buttons — clearer hierarchy */
    [data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.95rem !important;
        padding-left: 0.9rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Swipe-to-open sidebar disabled (table scroll was opening menu on mobile).


# ============================================================
# UTILITY
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    """Never raises on NaN / None / bad strings."""
    try:
        if value is None:
            return int(default)
        if isinstance(value, float) and value != value:
            return int(default)
        try:
            if pd.isna(value):
                return int(default)
        except Exception:
            pass
        return int(float(value))
    except Exception:
        try:
            return int(default)
        except Exception:
            return 0


def clean_symbol(symbol):
    symbol = str(symbol).upper().strip()

    if not symbol:
        return ""

    if symbol.endswith(".NS"):
        return symbol

    if symbol.startswith("^"):
        return symbol

    return symbol + ".NS"


def display_symbol(symbol):
    return str(symbol).upper().replace(".NS", "")


def normalize_columns(df):
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    if isinstance(x.columns, pd.MultiIndex):
        new_cols = []

        for c in x.columns:
            if isinstance(c, tuple):
                new_cols.append(str(c[0]))
            else:
                new_cols.append(str(c))

        x.columns = new_cols

    x.columns = [str(c).strip() for c in x.columns]

    return x


def ensure_result_columns(df):
    """Make saved/legacy scan results safe for every dashboard page.

    Older CSV files may not contain columns added in later versions.
    The dashboard should never crash with KeyError just because an older
    result file was loaded. Missing fields are filled with safe defaults.
    """
    if df is None:
        return pd.DataFrame()

    x = df.copy()

    # Legacy column names from earlier versions.
    aliases = {
        "signal": "Call",
        "call": "Call",
        "price": "Price",
        "risk": "Risk Level",
        "risk_level": "Risk Level",
        "risk_pct": "Risk %",
        "prediction": "Prediction",
        "target": "Target",
        "stop_loss": "Stop Loss",
        "hold_days": "Hold Days",
        "priority": "Priority",
        "reason": "Reason",
        "technical_reasons": "Technical Reasons",
        "patterns": "Patterns",
        "news_influence": "News Influence",
    }
    for old, new in aliases.items():
        if new not in x.columns and old in x.columns:
            x[new] = x[old]

    # Always derive Sector if it is absent or blank.
    if "Stock" not in x.columns and "Symbol" in x.columns:
        x["Stock"] = x["Symbol"].astype(str).str.replace(".NS", "", regex=False).str.upper()

    if "Stock" not in x.columns:
        x["Stock"] = "UNKNOWN"

    if "Sector" not in x.columns:
        x["Sector"] = x["Stock"].apply(sector_of)
    else:
        missing = x["Sector"].isna() | x["Sector"].astype(str).str.strip().isin(["", "nan", "None"])
        if missing.any():
            x.loc[missing, "Sector"] = x.loc[missing, "Stock"].apply(sector_of)

    defaults = {
        "Symbol": x["Stock"].astype(str).apply(clean_symbol),
        "Price": np.nan,
        "Call": "WATCH",
        "Prediction": 0.0,
        "Risk %": 0.0,
        "Risk Level": "UNKNOWN",
        "Target": np.nan,
        "Stop Loss": np.nan,
        "Hold Days": DEFAULT_HOLD_DAYS,
        "Priority": "LOW",
        "Patterns": "No major pattern detected",
        "News Influence": "No clear news influence assessed.",
        "Reason": "No explanation available.",
        "Technical Reasons": "No technical explanation available.",
    }

    for col, default in defaults.items():
        if col not in x.columns:
            if isinstance(default, pd.Series):
                x[col] = default
            else:
                x[col] = default

    # Numeric cleanup so sorting/filtering cannot fail on old text CSVs.
    for col in ["Price", "Prediction", "Risk %", "Target", "Stop Loss", "Hold Days"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x["Prediction"] = x["Prediction"].fillna(0)
    x["Risk %"] = x["Risk %"].fillna(0)
    x["Hold Days"] = x["Hold Days"].fillna(DEFAULT_HOLD_DAYS).astype(int)

    x["Call"] = x["Call"].astype(str).str.upper().replace({"NAN": "WATCH", "NONE": "WATCH"})
    x["Stock"] = x["Stock"].astype(str).str.replace(".NS", "", regex=False).str.upper()
    x["Symbol"] = x["Symbol"].astype(str).apply(clean_symbol)

    # Recreate Rank if a saved file did not have it.
    if "Rank" not in x.columns:
        x["Rank"] = np.arange(1, len(x) + 1)

    return x


# ============================================================
# MARKET STATUS
# ============================================================

def india_now():
    """
    Always India Standard Time (IST, UTC+5:30).
    Required on Streamlit Cloud / any UTC server so market hours are correct.
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        try:
            import pytz
            return datetime.now(pytz.timezone("Asia/Kolkata"))
        except Exception:
            # Fixed offset fallback UTC+5:30
            from datetime import timezone, timedelta
            return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def nse_market_open_now():
    """NSE cash market: Mon–Fri 09:15–15:30 IST (simple session; ignores special holidays)."""
    now = india_now()
    # weekday() on aware datetime is fine
    if now.weekday() >= 5:
        return False
    current = now.time()
    return (
        current >= dt_time(9, 15)
        and current <= dt_time(15, 30)
    )


def market_status_text():
    now = india_now()
    try:
        ist_label = now.strftime("%H:%M IST")
    except Exception:
        ist_label = ""
    if nse_market_open_now():
        return f"🟢 NSE MARKET OPEN · {ist_label}".strip()
    return f"🔴 NSE MARKET CLOSED · {ist_label}".strip()


# ============================================================
# FILE INITIALIZATION
# ============================================================

def ensure_files():

    if not HISTORY_FILE.exists():

        pd.DataFrame(
            columns=[
                "Prediction Date",
                "Stock",
                "Symbol",
                "Call",
                "Prediction",
                "Entry",
                "Target",
                "Stop Loss",
                "Risk %",
                "Risk Level",
                "Hold Days",
                "Expiry Date",
                "Status",
                "Evaluation Date",
                "Exit Price",
                "Return %",
                "Result",
                "Result Detail",
                "Days Taken",
                "Outcome Message",
                "Recommendation",
                "Reason",
            ]
        ).to_csv(
            HISTORY_FILE,
            index=False
        )
    else:
        # Ensure newer columns exist on older history files
        try:
            hist = pd.read_csv(HISTORY_FILE)
            extra_cols = {
                "Result Detail": "",
                "Days Taken": "",
                "Outcome Message": "",
                "Recommendation": "",
            }
            changed = False
            for col, default in extra_cols.items():
                if col not in hist.columns:
                    hist[col] = default
                    changed = True
            if changed:
                hist.to_csv(HISTORY_FILE, index=False)
        except Exception:
            pass

    if not PORTFOLIO_FILE.exists():

        pd.DataFrame(
            columns=[
                "Stock",
                "Shares",
                "Buy Price",
                "Purchase Date",
                "Maximum Holding Days",
            ]
        ).to_csv(
            PORTFOLIO_FILE,
            index=False
        )

    if not TRADES_FILE.exists():
        pd.DataFrame(
            columns=[
                "Trade ID",
                "Prediction Date",
                "Stock",
                "Call",
                "Entry",
                "Target",
                "Stop Loss",
                "Hold Days",
                "Status",
                "Result",
                "Current Price",
                "Unrealized %",
                "Realized %",
                "Exit Price",
                "Days Held",
                "Days Remaining",
                "Distance to Target %",
                "Distance to Stop %",
                "Last Checked",
                "Suggestion",
            ]
        ).to_csv(TRADES_FILE, index=False)

    if not PAPER_FILE.exists():
        pd.DataFrame(
            columns=[
                "Trade ID",
                "Open Date",
                "Stock",
                "Side",
                "Shares",
                "Entry",
                "Target",
                "Stop Loss",
                "Hold Days",
                "Status",
                "Result",
                "Exit Date",
                "Exit Price",
                "Return %",
                "PnL ₹",
                "Notes",
            ]
        ).to_csv(PAPER_FILE, index=False)


ensure_files()


# ============================================================
# NSE UNIVERSE
# ============================================================

def load_all_nse_stocks():

    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]

    for url in urls:

        try:

            u = pd.read_csv(url)

            col = next(
                (
                    c
                    for c in u.columns
                    if str(c).strip().upper() == "SYMBOL"
                ),
                None,
            )

            if col:

                symbols = []

                for value in u[col].dropna():

                    s = str(value).strip().upper()

                    if not s:
                        continue

                    if s == "NAN":
                        continue

                    symbols.append(
                        s + ".NS"
                    )

                symbols = sorted(
                    set(symbols)
                )

                if len(symbols) > 500:

                    pd.DataFrame(
                        {"Symbol": symbols}
                    ).to_csv(
                        NSE_UNIVERSE_CACHE,
                        index=False
                    )

                    return symbols

        except Exception:
            continue

    if NSE_UNIVERSE_CACHE.exists():

        try:

            return sorted(
                set(
                    pd.read_csv(
                        NSE_UNIVERSE_CACHE
                    )["Symbol"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
            )

        except Exception:
            pass

    # Built-in list — NO Excel needed. Used if NSE website CSV is unreachable.
    return [s + ".NS" for s in [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "BHARTIARTL",
        "LT", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "ASIANPAINT", "HCLTECH", "MARUTI",
        "SUNPHARMA", "TITAN", "NTPC", "TATAMOTORS", "POWERGRID", "ULTRACEMCO", "M&M",
        "WIPRO", "ONGC", "TATASTEEL", "COALINDIA", "JSWSTEEL", "ADANIENT", "TECHM",
        "HINDALCO", "BAJAJFINSV", "NESTLEIND", "INDUSINDBK", "CIPLA", "DRREDDY",
        "BPCL", "EICHERMOT", "APOLLOHOSP", "HEROMOTOCO", "DIVISLAB", "BRITANNIA",
        "TATACONSUM", "BAJAJ-AUTO", "HINDUNILVR", "SBILIFE", "HDFCLIFE", "BEL", "TRENT",
        "FEDERALBNK", "PNB", "BANKBARODA", "CANBK", "DMART", "ZOMATO", "PERSISTENT",
        "COFORGE", "LTIM", "PIDILITIND", "HAVELLS", "SIEMENS", "DLF", "IRCTC",
        "TATAPOWER", "RECLTD", "PFC", "TVSMOTOR", "ASHOKLEY", "VEDL", "GAIL", "IOC",
    ]]


NSE_STOCKS = load_all_nse_stocks()


# ============================================================
# SECTOR MAP
# ============================================================

SECTOR_MAP_CACHE = APP_DIR / "nse_sector_map.csv"

# Normalize NSE / Yahoo industry labels → dashboard sector buckets
_INDUSTRY_TO_SECTOR = [
    (("bank", "banking"), "Banking"),
    (("finance", "financial services", "nbfc", "housing finance", "capital markets", "insurance"), "Financial Services"),
    (("information technology", "it services", "software", "computers"), "IT"),
    (("pharma", "pharmaceutical", "healthcare", "hospital", "biotech", "drug"), "Pharma"),
    (("automobile", "auto component", "auto components", "tyre"), "Automobile"),
    (("metal", "steel", "mining", "mineral", "aluminium", "copper", "zinc"), "Metals"),
    (("oil", "gas", "petroleum", "refinery", "refineries", "consumable fuels"), "Energy"),
    (("power", "utilities", "electricity"), "Utilities"),
    (("fmcg", "fast moving consumer", "food product", "beverages", "tobacco"), "FMCG"),
    (("telecom", "telecommunication"), "Telecom"),
    (("cement",), "Cement"),
    (("construction", "infrastructure", "engineering", "capital goods", "industrial manufacturing"), "Infrastructure"),
    (("realty", "real estate"), "Realty"),
    (("consumer durable", "consumer goods", "textiles", "apparel", "jewellery", "paint"), "Consumer"),
    (("retail", "trading", "e-commerce"), "Retail"),
    (("media", "entertainment"), "Media"),
    (("chemical", "fertilizer", "fertilisers", "pesticide", "specialty chemical"), "Chemicals"),
    (("defence", "defense", "aerospace"), "Defence"),
    (("logistics", "transport", "shipping", "airline", "services"), "Services"),
    (("agriculture", "paper", "forest"), "Others"),
]


def _map_industry_label(label: str) -> str:
    text = str(label or "").strip().lower()
    if not text or text in {"nan", "none", "-"}:
        return "Other"
    for keys, sector in _INDUSTRY_TO_SECTOR:
        for k in keys:
            if k in text:
                return sector
    return "Other"


def _seed_sector_map() -> dict:
    """Hardcoded seeds for major names (used if NSE download fails)."""
    return {
        "RELIANCE": "Energy", "ONGC": "Energy", "IOC": "Energy", "BPCL": "Energy",
        "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
        "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
        "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
        "BANKBARODA": "Banking", "CANBK": "Banking", "PNB": "Banking",
        "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
        "MARUTI": "Automobile", "M&M": "Automobile", "TATAMOTORS": "Automobile",
        "EICHERMOT": "Automobile", "TATASTEEL": "Metals", "JSWSTEEL": "Metals",
        "HINDALCO": "Metals", "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG",
        "BHARTIARTL": "Telecom", "NTPC": "Utilities", "POWERGRID": "Utilities",
        "LT": "Infrastructure", "ADANIPORTS": "Infrastructure",
        "HAL": "Defence", "BEL": "Defence", "COFORGE": "IT", "PERSISTENT": "IT",
        "LTIM": "IT", "TRENT": "Retail", "TITAN": "Consumer", "ASIANPAINT": "Consumer",
    }


def load_sector_map() -> dict:
    """
    Build a large symbol → sector map from NSE index constituent files
    (Industry column) so sector filters show all stocks, not 2–3 names.
    """
    sector_map = _seed_sector_map()

    # Prefer cache if reasonably complete
    if SECTOR_MAP_CACHE.exists():
        try:
            cached = pd.read_csv(SECTOR_MAP_CACHE)
            if "Symbol" in cached.columns and "Sector" in cached.columns and len(cached) > 200:
                for _, row in cached.iterrows():
                    sym = str(row["Symbol"]).upper().strip().replace(".NS", "")
                    sec = str(row["Sector"]).strip()
                    if sym and sec and sec not in {"", "nan", "None"}:
                        sector_map[sym] = sec
                if len(sector_map) > 200:
                    return sector_map
        except Exception:
            pass

    # NSE index lists that include Industry / Company Name
    index_urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyautolist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymetallist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyenergylist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyinfralist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyrealtylist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyhealthcarelist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyconsumerdurableslist.csv",
        "https://archives.nseindia.com/content/indices/ind_niftyfinancelist.csv",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/csv,*/*",
    }

    for url in index_urls:
        try:
            df = pd.read_csv(url, storage_options={"User-Agent": headers["User-Agent"]})
        except Exception:
            try:
                import io
                import urllib.request
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    df = pd.read_csv(io.BytesIO(resp.read()))
            except Exception:
                continue

        if df is None or df.empty:
            continue

        cols = {str(c).strip().lower(): c for c in df.columns}
        sym_col = cols.get("symbol") or cols.get("symbol name")
        ind_col = (
            cols.get("industry")
            or cols.get("basic industry")
            or cols.get("sector")
            or cols.get("macro economic sector")
        )
        if not sym_col:
            continue

        for _, row in df.iterrows():
            sym = str(row[sym_col]).upper().strip()
            if not sym or sym in {"NAN", "NONE"}:
                continue
            industry = row[ind_col] if ind_col else ""
            sector = _map_industry_label(industry)
            # Keep a more specific assignment if we already have one that is not Other
            if sym not in sector_map or sector_map.get(sym) == "Other":
                if sector != "Other":
                    sector_map[sym] = sector
                elif sym not in sector_map:
                    sector_map[sym] = "Other"

    # Persist for next run
    try:
        pd.DataFrame(
            [{"Symbol": k, "Sector": v} for k, v in sorted(sector_map.items())]
        ).to_csv(SECTOR_MAP_CACHE, index=False)
    except Exception:
        pass

    return sector_map


SECTOR_MAP = load_sector_map()


def sector_of(symbol):
    key = display_symbol(symbol)
    return SECTOR_MAP.get(key, "Other")


def refresh_sector_for_results(results: pd.DataFrame) -> pd.DataFrame:
    """Re-apply sector map so sector pages are complete after scan."""
    if results is None or results.empty:
        return results
    x = results.copy()
    if "Stock" not in x.columns and "Symbol" in x.columns:
        x["Stock"] = x["Symbol"].astype(str).str.replace(".NS", "", regex=False).str.upper()
    x["Sector"] = x["Stock"].astype(str).str.upper().map(
        lambda s: SECTOR_MAP.get(s, "Other")
    )
    return x


# ============================================================
# INDEX MEMBERSHIP + MARKET CAP + LONG-TERM QUALITY
# ============================================================

INDEX_MEMBER_CACHE = APP_DIR / "nse_index_membership.json"

_INDEX_CSV = {
    "Nifty 50": [
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    ],
    "Nifty 100": [
        "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    ],
    "Nifty 200": [
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    ],
    "Nifty 500": [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    ],
    "Nifty Midcap 100": [
        "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
    ],
    "Nifty Smallcap 100": [
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
    ],
}


def _download_index_symbols(urls: list) -> set:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    for url in urls:
        try:
            df = pd.read_csv(url, storage_options={"User-Agent": headers["User-Agent"]})
        except Exception:
            try:
                import io
                import urllib.request
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    df = pd.read_csv(io.BytesIO(resp.read()))
            except Exception:
                continue
        if df is None or df.empty:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        sym_col = cols.get("symbol") or cols.get("symbol name")
        if not sym_col:
            continue
        out = set()
        for v in df[sym_col].dropna():
            s = str(v).upper().strip().replace(".NS", "")
            if s and s != "NAN":
                out.add(s)
        if out:
            return out
    return set()


@st.cache_data(ttl=86400, show_spinner=False)
def load_index_membership() -> dict:
    """
    symbol → list of index names (Nifty 50, 100, 500, Midcap, Smallcap…).
    Cached 24h; also written to disk.
    """
    membership = {}  # sym -> set of index labels

    if INDEX_MEMBER_CACHE.exists():
        try:
            with open(INDEX_MEMBER_CACHE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and len(raw) > 50:
                return {k: list(v) for k, v in raw.items()}
        except Exception:
            pass

    for name, urls in _INDEX_CSV.items():
        syms = _download_index_symbols(urls)
        for s in syms:
            membership.setdefault(s, set()).add(name)

    # Derive Large / Mid / Small style tags from index membership
    for s, idxs in list(membership.items()):
        idxs = set(idxs)
        if "Nifty 50" in idxs or "Nifty 100" in idxs:
            idxs.add("Large Cap (index)")
        if "Nifty Midcap 100" in idxs:
            idxs.add("Mid Cap (index)")
        if "Nifty Smallcap 100" in idxs:
            idxs.add("Small Cap (index)")
        membership[s] = idxs

    serializable = {k: sorted(list(v)) for k, v in membership.items()}
    try:
        with open(INDEX_MEMBER_CACHE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
    except Exception:
        pass
    return serializable


def index_tags_of(symbol: str) -> list:
    m = load_index_membership()
    return list(m.get(display_symbol(symbol), []) or [])


def market_cap_bucket(market_cap) -> str:
    """
    Approx India buckets using market cap in INR (Yahoo often returns absolute INR).
    Large ≥ ₹20,000 Cr | Mid ₹5,000–20,000 Cr | Small < ₹5,000 Cr
    """
    mc = safe_float(market_cap, 0)
    if mc <= 0:
        return "Unknown"
    # Cr = 10^7
    cr = mc / 1e7
    if cr >= 20000:
        return "Large Cap"
    if cr >= 5000:
        return "Mid Cap"
    if cr >= 500:
        return "Small Cap"
    return "Micro Cap"


def long_term_quality_from_fund(fund: dict) -> dict:
    """
    Experienced long-term desk score 0–100 + hold advice.
    If short-term SL hits, this says whether core holding still makes sense.
    """
    score = 50
    notes = []
    if not fund or fund.get("error"):
        return {
            "lt_score": 45,
            "lt_label": "Unknown / data thin",
            "lt_hold_if_sl": "Review manually — fundamentals not loaded",
            "notes": ["Fundamentals unavailable"],
            "market_cap_bucket": "Unknown",
        }

    pe = safe_float(fund.get("pe"))
    roe = safe_float(fund.get("roe"))
    if roe and roe < 1.5:
        roe = roe * 100  # sometimes fraction
    de = safe_float(fund.get("debt_to_equity"))
    pm = safe_float(fund.get("profit_margins"))
    if pm and pm < 1:
        pm = pm * 100
    rg = safe_float(fund.get("revenue_growth"))
    if rg and abs(rg) < 2:
        rg = rg * 100
    eg = safe_float(fund.get("earnings_growth"))
    if eg and abs(eg) < 2:
        eg = eg * 100
    dy = safe_float(fund.get("dividend_yield"))
    mcap_b = market_cap_bucket(fund.get("market_cap"))

    if roe >= 15:
        score += 12
        notes.append(f"ROE ~{roe:.1f}% — quality compounder zone")
    elif roe >= 10:
        score += 6
        notes.append(f"ROE ~{roe:.1f}% — acceptable")
    elif roe > 0:
        score -= 5
        notes.append(f"ROE ~{roe:.1f}% — weak for long-term core")

    if 0 < de <= 50:
        score += 8
        notes.append(f"Debt/Equity {de:.0f} — comfortable")
    elif 50 < de <= 100:
        score += 2
        notes.append(f"Debt/Equity {de:.0f} — watch leverage")
    elif de > 100:
        score -= 10
        notes.append(f"Debt/Equity {de:.0f} — high leverage risk")

    if pm >= 12:
        score += 8
        notes.append(f"Profit margin ~{pm:.1f}%")
    elif pm >= 6:
        score += 3
    elif pm > 0:
        score -= 4

    if rg >= 12:
        score += 8
        notes.append(f"Revenue growth ~{rg:.1f}%")
    elif rg >= 5:
        score += 3
    elif rg < 0:
        score -= 6
        notes.append("Revenue shrinking — long-term caution")

    if eg >= 12:
        score += 6
    elif eg < -10:
        score -= 5

    if 0 < pe <= 25:
        score += 5
        notes.append(f"PE ~{pe:.1f} — not extreme")
    elif pe > 50:
        score -= 6
        notes.append(f"PE ~{pe:.1f} — expensive; need growth to justify")
    elif pe > 35:
        score -= 2

    if dy >= 1.5:
        score += 3
        notes.append(f"Dividend yield ~{dy:.1f}%")

    if mcap_b == "Large Cap":
        score += 4
        notes.append("Large-cap stability bias")
    elif mcap_b == "Mid Cap":
        score += 2
        notes.append("Mid-cap — growth with volatility")
    elif mcap_b in ("Small Cap", "Micro Cap"):
        score -= 2
        notes.append("Small/micro — higher business risk; size positions smaller")

    score = int(max(5, min(95, score)))
    if score >= 72:
        label = "Strong long-term candidate"
        hold_if_sl = (
            "YES — if SL was only swing noise and business still intact, "
            "you may HOLD/accumulate for long-term instead of panic selling the whole position."
        )
    elif score >= 58:
        label = "Average long-term hold"
        hold_if_sl = (
            "PARTIAL — keep only a core size if thesis is intact; "
            "do not add aggressively after a technical SL."
        )
    else:
        label = "Weak / speculative long-term"
        hold_if_sl = (
            "NO core long-term — treat as trading book only. "
            "If swing SL hits, prefer exit; do not convert a bad trade into a long bag-hold."
        )

    return {
        "lt_score": score,
        "lt_label": label,
        "lt_hold_if_sl": hold_if_sl,
        "notes": notes[:8],
        "market_cap_bucket": mcap_b,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def in_crore(value) -> float:
    """Convert absolute INR amount to ₹ crore (1 Cr = 1e7)."""
    v = safe_float(value, 0)
    if v == 0:
        return 0.0
    # Yahoo marketCap/revenue usually full INR; if already small, leave as is
    if abs(v) >= 1e5:
        return round(v / 1e7, 2)
    return round(v, 2)


def fmt_crore(value, suffix=" Cr") -> str:
    cr = in_crore(value)
    if cr == 0 and safe_float(value, 0) == 0:
        return "—"
    return f"₹{cr:,.2f}{suffix}"


def enrich_stock_profile(symbol: str) -> dict:
    """Index tags + fundamentals + LT quality for one symbol."""
    sym = display_symbol(symbol)
    tags = index_tags_of(sym)
    fund = {}
    try:
        fund = fetch_fundamentals(sym)
    except Exception:
        fund = {}
    lt = long_term_quality_from_fund(fund)
    # Prefer index-based cap if Yahoo mcap missing
    if lt.get("market_cap_bucket") == "Unknown":
        if "Large Cap (index)" in tags or "Nifty 50" in tags or "Nifty 100" in tags:
            lt["market_cap_bucket"] = "Large Cap"
        elif "Mid Cap (index)" in tags or "Nifty Midcap 100" in tags:
            lt["market_cap_bucket"] = "Mid Cap"
        elif "Small Cap (index)" in tags or "Nifty Smallcap 100" in tags:
            lt["market_cap_bucket"] = "Small Cap"

    mcap = fund.get("market_cap")
    revenue = fund.get("revenue_latest") or fund.get("total_revenue")
    # totalRevenue from info if statement missing
    if not revenue and fund.get("error") == "":
        pass
    book = fund.get("book_value")
    lt_score = safe_float(lt.get("lt_score"), 45)

    # Long-term call paired with swing BUY/SELL
    if lt_score >= 72:
        lt_with_buy = "LT: ACCUMULATE / CORE HOLD — swing BUY can add; if SL hits keep small core"
        lt_with_sell = "LT: Still quality — swing SELL is for trade book; don't dump entire long-term core blindly"
    elif lt_score >= 58:
        lt_with_buy = "LT: HOLD small core only — swing BUY OK with tight risk"
        lt_with_sell = "LT: REDUCE on strength — swing SELL aligns with average quality"
    else:
        lt_with_buy = "LT: NO core — treat BUY as pure trade; exit fully if SL hits"
        lt_with_sell = "LT: EXIT / avoid — swing SELL and stay out until quality improves"

    return {
        "Stock": sym,
        "Index Tags": ", ".join(tags) if tags else "Outside major indices",
        "In Nifty 50": "Nifty 50" in tags,
        "In Nifty 100": "Nifty 100" in tags,
        "In Nifty 500": "Nifty 500" in tags,
        "Market Cap Bucket": lt.get("market_cap_bucket", "Unknown"),
        "Market Cap (₹ Cr)": in_crore(mcap) if mcap else None,
        "Market Cap Cr Text": fmt_crore(mcap) if mcap else "—",
        "Book Value": safe_float(book) if book else None,
        "Book Value Text": f"₹{safe_float(book):,.2f}" if book else "—",
        "Face Value": safe_float(fund.get("face_value")) if fund.get("face_value") else None,
        "Face Value Text": f"₹{safe_float(fund.get('face_value')):.2f}" if fund.get("face_value") else "—",
        "Revenue (₹ Cr)": in_crore(revenue) if revenue else None,
        "Revenue Cr Text": fmt_crore(revenue) if revenue else "—",
        "Sales (₹ Cr)": in_crore(revenue) if revenue else None,  # sales ≈ revenue for display
        "Sales Cr Text": fmt_crore(revenue) if revenue else "—",
        "Net Income (₹ Cr)": in_crore(fund.get("net_income_latest")) if fund.get("net_income_latest") else None,
        "Net Income Cr Text": fmt_crore(fund.get("net_income_latest")) if fund.get("net_income_latest") else "—",
        "LT Score": lt_score,
        "LT Label": lt.get("lt_label", ""),
        "LT Hold if SL hits": lt.get("lt_hold_if_sl", ""),
        "LT Notes": " | ".join(lt.get("notes") or []),
        "LT with BUY call": lt_with_buy,
        "LT with SELL call": lt_with_sell,
        "PE": fund.get("pe"),
        "PB": fund.get("pb"),
        "ROE": fund.get("roe"),
        "Debt/Equity": fund.get("debt_to_equity"),
        "Profit Margin": fund.get("profit_margins"),
        "Revenue Growth": fund.get("revenue_growth"),
        "Name": fund.get("name") or sym,
        "Industry": fund.get("industry") or "",
    }


def show_call_with_long_term(symbol: str, call: str, compact: bool = False):
    """Show swing call + long-term analysis + book value / revenue in Cr."""
    try:
        prof = enrich_stock_profile(symbol)
    except Exception as e:
        st.caption(f"Fundamentals unavailable: {e}")
        return
    call_u = str(call or "").upper()
    lt_line = prof.get("LT with BUY call") if "BUY" in call_u else (
        prof.get("LT with SELL call") if "SELL" in call_u else prof.get("LT Hold if SL hits")
    )
    if compact:
        st.caption(
            f"BV {prof.get('Book Value Text')} · Rev {prof.get('Revenue Cr Text')} · "
            f"Mcap {prof.get('Market Cap Cr Text')} · LT {prof.get('LT Score'):.0f} — {lt_line}"
        )
        return
    st.markdown(
        f"""
        <div class="hold-box" style="padding:10px;border-radius:8px;margin:6px 0;">
        <b>Long-term with this {call_u or 'CALL'}:</b> {lt_line}<br>
        <b>Book value:</b> {prof.get('Book Value Text')}
        &nbsp;|&nbsp; <b>Revenue / Sales:</b> {prof.get('Revenue Cr Text')}
        &nbsp;|&nbsp; <b>Net income:</b> {prof.get('Net Income Cr Text')}<br>
        <b>Market cap:</b> {prof.get('Market Cap Cr Text')} ({prof.get('Market Cap Bucket')})
        &nbsp;|&nbsp; <b>LT score:</b> {safe_float(prof.get('LT Score')):.0f} — {prof.get('LT Label')}<br>
        <b>PE / PB / ROE:</b>
        {safe_float(prof.get('PE')):.1f} /
        {safe_float(prof.get('PB')):.2f} /
        {safe_float(prof.get('ROE')):.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def attach_fundamentals_to_df(df: pd.DataFrame, max_n: int = 25) -> pd.DataFrame:
    """Add book value, revenue Cr, LT score columns for strategy tables."""
    if df is None or df.empty or "Stock" not in df.columns:
        return df
    x = df.copy()
    cols = {
        "Book Value": [],
        "Revenue (₹ Cr)": [],
        "Sales (₹ Cr)": [],
        "Market Cap (₹ Cr)": [],
        "LT Score": [],
        "LT Label": [],
        "LT with call": [],
    }
    for i, stock in enumerate(x["Stock"].astype(str).tolist()):
        if i >= max_n:
            for k in cols:
                cols[k].append(None if k != "LT Label" else "")
            continue
        try:
            p = enrich_stock_profile(stock)
            side = str(x.iloc[i].get("Side", x.iloc[i].get("Call", "BUY"))).upper()
            cols["Book Value"].append(p.get("Book Value"))
            cols["Revenue (₹ Cr)"].append(p.get("Revenue (₹ Cr)"))
            cols["Sales (₹ Cr)"].append(p.get("Sales (₹ Cr)"))
            cols["Market Cap (₹ Cr)"].append(p.get("Market Cap (₹ Cr)"))
            cols["LT Score"].append(p.get("LT Score"))
            cols["LT Label"].append(p.get("LT Label"))
            cols["LT with call"].append(
                p.get("LT with SELL call") if "SELL" in side else p.get("LT with BUY call")
            )
        except Exception:
            for k in cols:
                cols[k].append(None if k != "LT Label" else "")
    for k, v in cols.items():
        # pad if length mismatch
        while len(v) < len(x):
            v.append(None)
        x[k] = v[: len(x)]
    return x



def filterable_dataframe(
    df: pd.DataFrame,
    key: str,
    default_cols: list = None,
    height: int = 360,
):
    """Table with search + column filters for BUY/SELL/History visibility."""
    if df is None or df.empty:
        st.info("No rows to display.")
        return df
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    st.markdown(
        """
        <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;
                    padding:10px 12px;margin-bottom:8px;color:#94a3b8;font-size:0.9rem;">
        <b style="color:#e2e8f0;">Search + filters</b> · market cap / revenue in <b>₹ crore</b> when loaded.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("Search table", value="", key=f"{key}_q")
    with c2:
        cols_all = list(x.columns)
        default = default_cols if default_cols else cols_all[: min(12, len(cols_all))]
        default = [c for c in default if c in cols_all] or cols_all[:12]
        pick = st.multiselect("Columns", cols_all, default=default, key=f"{key}_cols")
    if not pick:
        pick = list(x.columns)[:12]
    filter_cols = [c for c in pick if c in x.columns and x[c].dtype == object][:4]
    if filter_cols:
        fcols = st.columns(max(len(filter_cols), 1))
        for i, col in enumerate(filter_cols):
            with fcols[i]:
                vals = sorted([str(v) for v in x[col].dropna().unique().tolist() if str(v).strip()][:40])
                if vals:
                    sel = st.multiselect(col, vals, default=[], key=f"{key}_f_{col}")
                    if sel:
                        x = x[x[col].astype(str).isin(sel)]
    if q and str(q).strip():
        qq = str(q).strip().lower()
        mask = pd.Series([False] * len(x), index=x.index)
        for c in pick:
            if c in x.columns:
                mask = mask | x[c].astype(str).str.lower().str.contains(qq, na=False)
        x = x[mask]
    st.caption(f"Showing **{len(x)}** rows")
    st.dataframe(x[pick], use_container_width=True, hide_index=True, height=height)
    return x


def render_call_stock_card(row, section_key: str = "card"):
    """Shared visual card for BUY / SELL / history."""
    stock = display_symbol(str(row.get("Stock", row.get("Symbol", ""))))
    if not stock:
        return
    call = str(row.get("Call", row.get("Side", "BUY"))).upper()
    border = "#16a34a" if "BUY" in call else ("#dc2626" if "SELL" in call else "#64748b")
    price = safe_float(row.get("Price", row.get("Entry", row.get("Current Price"))))
    tgt = safe_float(row.get("Target"))
    sl = safe_float(row.get("Stop Loss"))
    src = str(row.get("Call Source", row.get("Strategy", "")) or "")
    result = str(row.get("Result", "") or "")
    name = stock
    mcap_txt = cap_class = bv_txt = rev_txt = idx_txt = "—"
    lt_line = ""
    try:
        p = enrich_stock_profile(stock)
        name = p.get("Name") or stock
        mcap_txt = p.get("Market Cap Cr Text", "—")
        cap_class = p.get("Market Cap Bucket", "—")
        bv_txt = p.get("Book Value Text", "—")
        rev_txt = p.get("Revenue Cr Text", "—")
        idx_txt = p.get("Index Tags", "—")
        lt_line = p.get("LT with SELL call") if "SELL" in call else p.get("LT with BUY call", "")
    except Exception:
        pass
    cap_u = str(cap_class).upper()
    if "LARGE" in cap_u:
        cap_color = "#0d9488"
    elif "MID" in cap_u:
        cap_color = "#2563eb"
    elif "SMALL" in cap_u or "MICRO" in cap_u:
        cap_color = "#d97706"
    else:
        cap_color = "#64748b"
    # Risk / reward strip for trader
    try:
        _qc = trade_quality_check(price, tgt, sl, call if call in ("BUY", "SELL") else "BUY", min_rr=1.5)
        if _qc.get("ok"):
            _rr_html = (
                f"✅ <b>{_qc['label']}</b> · risk <b style='color:#f87171;'>₹{_qc['risk_rs']:,.0f}</b> · "
                f"reward <b style='color:#4ade80;'>₹{_qc['reward_rs']:,.0f}</b> · {_qc['shares']} sh"
            )
        else:
            _rr_html = "⛔ <b>" + (_qc.get("label") or "BLOCKED") + "</b> · " + " · ".join((_qc.get("reasons") or [])[:2])
    except Exception:
        _rr_html = "R:R —"
    st.markdown(
        f"""
        <div style="border:2px solid {border};border-radius:14px;padding:14px 16px;margin:10px 0;
                    background:linear-gradient(145deg,#0f172a 0%,#1e293b 100%);">
          <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;">
            <div>
              <div style="font-size:1.25rem;font-weight:700;color:#f8fafc;">{name}</div>
              <div style="color:#94a3b8;">{stock} · <b style="color:{border};">{call}</b>{(' · '+src) if src else ''}</div>
            </div>
            <div style="text-align:right;">
              <span style="background:{cap_color};color:#fff;padding:3px 10px;border-radius:16px;font-size:0.8rem;font-weight:600;">{cap_class}</span>
              <div style="color:#e2e8f0;margin-top:4px;"><b>Mcap</b> {mcap_txt}</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-top:10px;">
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">PRICE / ENTRY</div>
              <div style="color:#f8fafc;font-weight:600;">₹{price:,.2f}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">TARGET</div>
              <div style="color:#4ade80;font-weight:600;">₹{tgt:,.2f}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">STOP</div>
              <div style="color:#f87171;font-weight:600;">₹{sl:,.2f}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">BOOK VALUE</div>
              <div style="color:#f8fafc;font-weight:600;">{bv_txt}</div></div>
            <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.7rem;">REVENUE</div>
              <div style="color:#f8fafc;font-weight:600;">{rev_txt}</div></div>
          </div>
          <div style="margin-top:8px;color:#94a3b8;font-size:0.88rem;"><b>Index:</b> {idx_txt}{(' · <b>Result:</b> '+result) if result else ''}</div>
          <div style="margin-top:4px;color:#a5b4fc;font-size:0.88rem;"><b>LT:</b> {lt_line or '—'}</div>
          <div style="margin-top:8px;padding:8px 10px;border-radius:8px;background:#020617;border:1px solid #334155;">
            <span style="color:#e2e8f0;font-size:0.9rem;">{_rr_html}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Experienced trader — {stock}", expanded=False):
        show_expert_trader_box(stock, call_hint=call)
    render_active_trade_buttons(
        stock,
        side_hint=call,
        entry=price,
        target=tgt,
        stop=sl,
        key_prefix=f"{section_key}_{stock}",
    )


def aggregate_stocks_unique_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per stock. Merge all strategy names into a single list.
    Keeps best target/stop from first row; counts strategies.
    """
    if df is None or df.empty or "Stock" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["Stock"] = x["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()
    rows = []
    for stock, g in x.groupby("Stock", sort=False):
        strats = []
        for s in g.get("Strategy", pd.Series(dtype=str)).astype(str).tolist():
            if s and s.lower() not in ("nan", "none", ""):
                if s not in strats:
                    strats.append(s)
        # Also parse "Also matches" / "All matching strategies"
        for col in ("Also matches", "All matching strategies"):
            if col in g.columns:
                for cell in g[col].astype(str).tolist():
                    for part in str(cell).split(","):
                        part = part.strip()
                        if part and part.lower() not in ("nan", "none", "—", "-") and part not in strats:
                            strats.append(part)
        first = g.iloc[0]
        live = safe_float(first.get("Current / Live Price", first.get("Price")))
        tgt = safe_float(first.get("Target"))
        sl = safe_float(first.get("Stop Loss"))
        # Prefer row with most complete levels
        for _, rr in g.iterrows():
            if safe_float(rr.get("Target")) > 0 and safe_float(rr.get("Stop Loss")) > 0:
                live = safe_float(rr.get("Current / Live Price", rr.get("Price")), live)
                tgt = safe_float(rr.get("Target"), tgt)
                sl = safe_float(rr.get("Stop Loss"), sl)
                break
        side = str(first.get("Side", first.get("Call", "BUY"))).upper()
        rows.append({
            "Stock": stock,
            "Side": side if side in ("BUY", "SELL", "HOLD") else "BUY",
            "Strategies": " · ".join(strats) if strats else str(first.get("Strategy", "—")),
            "# Strategies": len(strats) if strats else 1,
            "Price": live,
            "Target": tgt,
            "Stop Loss": sl,
            "Hold Days": first.get("Hold Days", 15),
            "RSI": first.get("RSI"),
            "ADX": first.get("ADX"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("# Strategies", ascending=False)
    return out.reset_index(drop=True)


def render_unique_strategy_stock_cards(df: pd.DataFrame, max_cards: int = 12, key_prefix: str = "usc"):
    """
    Visual cards: company once, all strategies, mcap/cap class, BV, revenue Cr,
    experienced trader feedback.
    """
    if df is None or df.empty:
        st.info("No strategy stocks to show.")
        return
    uniq = aggregate_stocks_unique_strategies(df)
    st.success(f"**{len(uniq)} unique stocks** (duplicates merged — all strategies listed per name)")
    show_n = st.slider("Cards to show", 3, min(30, max(3, len(uniq))), min(max_cards, len(uniq)), key=f"{key_prefix}_n")
    load_fund = st.checkbox("Load company name · mcap · BV · revenue (₹ Cr) + desk feedback", value=True, key=f"{key_prefix}_fund")

    for i, row in uniq.head(show_n).iterrows():
        stock = str(row.get("Stock", ""))
        side = str(row.get("Side", "BUY"))
        strats = str(row.get("Strategies", "—"))
        n_s = int(safe_float(row.get("# Strategies"), 1))
        price = safe_float(row.get("Price"))
        tgt = safe_float(row.get("Target"))
        sl = safe_float(row.get("Stop Loss"))

        name = stock
        mcap_txt = "—"
        cap_class = "—"
        bv_txt = "—"
        rev_txt = "—"
        idx_txt = "—"
        face_txt = "—"
        profit_txt = "—"
        lt_line = ""
        if load_fund:
            try:
                p = enrich_stock_profile(stock)
                name = p.get("Name") or stock
                mcap_txt = p.get("Market Cap Cr Text", "—")
                cap_class = p.get("Market Cap Bucket", "—")
                bv_txt = p.get("Book Value Text", "—")
                rev_txt = p.get("Revenue Cr Text", "—")
                profit_txt = p.get("Net Income Cr Text", "—")
                face_txt = p.get("Face Value Text", "—")
                idx_txt = p.get("Index Tags", "—")
                lt_line = p.get("LT with SELL call") if side == "SELL" else p.get("LT with BUY call", "")
            except Exception:
                pass

        # Cap badge colour
        cap_u = str(cap_class).upper()
        if "LARGE" in cap_u:
            cap_color = "#0d9488"
        elif "MID" in cap_u:
            cap_color = "#2563eb"
        elif "SMALL" in cap_u or "MICRO" in cap_u:
            cap_color = "#d97706"
        else:
            cap_color = "#64748b"

        border = "#16a34a" if side == "BUY" else ("#dc2626" if side == "SELL" else "#64748b")
        st.markdown(
            f"""
            <div style="border:2px solid {border};border-radius:14px;padding:16px 18px;margin:12px 0;
                        background:linear-gradient(145deg,#0f172a 0%,#1e293b 100%);">
              <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;align-items:center;">
                <div>
                  <div style="font-size:1.35rem;font-weight:700;color:#f8fafc;">{name}</div>
                  <div style="color:#94a3b8;font-size:0.95rem;">{stock} · <b style="color:{border};">{side}</b></div>
                </div>
                <div style="text-align:right;">
                  <span style="background:{cap_color};color:#fff;padding:4px 10px;border-radius:20px;font-size:0.85rem;font-weight:600;">
                    {cap_class}
                  </span>
                  <div style="margin-top:6px;color:#e2e8f0;font-size:1.05rem;"><b>Mcap</b> {mcap_txt}</div>
                </div>
              </div>
              <div style="margin-top:12px;padding:10px;border-radius:8px;background:#020617;color:#cbd5e1;line-height:1.55;">
                <b style="color:#fbbf24;">Strategies ({n_s}):</b> {strats}
              </div>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:12px;">
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">LIVE</div>
                  <div style="color:#f8fafc;font-weight:600;">₹{price:,.2f}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">TARGET</div>
                  <div style="color:#4ade80;font-weight:600;">₹{tgt:,.2f}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">STOP</div>
                  <div style="color:#f87171;font-weight:600;">₹{sl:,.2f}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">BOOK VALUE</div>
                  <div style="color:#f8fafc;font-weight:600;">{bv_txt}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">REVENUE</div>
                  <div style="color:#f8fafc;font-weight:600;">{rev_txt}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">PROFIT</div>
                  <div style="color:#f8fafc;font-weight:600;">{profit_txt}</div></div>
                <div style="background:#020617;padding:8px;border-radius:8px;"><div style="color:#94a3b8;font-size:0.75rem;">FACE VALUE</div>
                  <div style="color:#f8fafc;font-weight:600;">{face_txt}</div></div>
              </div>
              <div style="margin-top:10px;color:#94a3b8;font-size:0.9rem;"><b>Index:</b> {idx_txt}</div>
              <div style="margin-top:6px;color:#a5b4fc;font-size:0.9rem;"><b>Long-term:</b> {lt_line or '—'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"🎓 Experienced trader feedback — {stock}", expanded=True):
            show_expert_trader_box(stock, call_hint=side)
        render_active_trade_buttons(
            stock,
            side_hint=side,
            entry=price,
            target=tgt,
            stop=sl,
            key_prefix=f"{key_prefix}_{stock}",
        )
        with st.expander(f"📈 Chart — {stock}", expanded=False):
            try:
                raw = stock_history(clean_symbol(stock), interval="1d")
                if raw is not None and not raw.empty:
                    fig = build_full_plotly_chart(
                        calculate_indicators(raw),
                        title=f"{name} ({stock})",
                        target=tgt,
                        stop_loss=sl,
                        height=480,
                        show_rsi=True,
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"Chart: {e}")


def enrich_results_profiles(results: pd.DataFrame, max_n: int = 40) -> pd.DataFrame:
    """Attach index/cap/LT columns for top rows (slow network — limited)."""
    if results is None or results.empty:
        return results
    x = results.copy()
    rows = []
    stocks = x["Stock"].astype(str).head(max_n).tolist()
    for s in stocks:
        try:
            rows.append(enrich_stock_profile(s))
        except Exception:
            rows.append({"Stock": display_symbol(s), "Market Cap Bucket": "Unknown", "LT Score": 45})
    if not rows:
        return x
    prof = pd.DataFrame(rows)
    # drop overlapping cols before merge
    for c in prof.columns:
        if c != "Stock" and c in x.columns:
            x = x.drop(columns=[c], errors="ignore")
    x["Stock_key"] = x["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False)
    prof["Stock_key"] = prof["Stock"].astype(str).str.upper()
    x = x.merge(prof.drop(columns=["Stock"], errors="ignore"), on="Stock_key", how="left")
    x = x.drop(columns=["Stock_key"], errors="ignore")
    return x


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def sma(series, period):

    return series.rolling(
        period
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return 100 - (
        100 /
        (1 + rs)
    )


def macd(series):

    fast = ema(
        series,
        12
    )

    slow = ema(
        series,
        26
    )

    line = fast - slow

    signal = ema(
        line,
        9
    )

    histogram = line - signal

    return line, signal, histogram


def atr(df, period=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    previous = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous
    ).abs()

    tr3 = (
        low - previous
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(
        period
    ).mean()


def adx(df, period=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (
            up_move > down_move
        )
        &
        (
            up_move > 0
        ),
        up_move,
        0
    )

    minus_dm = np.where(
        (
            down_move > up_move
        )
        &
        (
            down_move > 0
        ),
        down_move,
        0
    )

    tr = pd.concat(
        [
            high - low,
            (
                high - close.shift()
            ).abs(),
            (
                low - close.shift()
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr_value = tr.rolling(
        period
    ).mean()

    plus_di = (
        100
        *
        pd.Series(
            plus_dm,
            index=df.index
        ).rolling(period).mean()
        /
        atr_value.replace(
            0,
            np.nan
        )
    )

    minus_di = (
        100
        *
        pd.Series(
            minus_dm,
            index=df.index
        ).rolling(period).mean()
        /
        atr_value.replace(
            0,
            np.nan
        )
    )

    denominator = (
        plus_di +
        minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        (
            plus_di -
            minus_di
        ).abs()
        /
        denominator
    ) * 100

    return dx.rolling(
        period
    ).mean()


def stochastic(df, period=14):

    low_min = (
        df["Low"]
        .rolling(period)
        .min()
    )

    high_max = (
        df["High"]
        .rolling(period)
        .max()
    )

    denominator = (
        high_max -
        low_min
    ).replace(
        0,
        np.nan
    )

    k = (
        100
        *
        (
            df["Close"] -
            low_min
        )
        /
        denominator
    )

    d = k.rolling(
        3
    ).mean()

    return k, d


def bollinger(df, period=20):

    middle = (
        df["Close"]
        .rolling(period)
        .mean()
    )

    std = (
        df["Close"]
        .rolling(period)
        .std()
    )

    upper = middle + 2 * std

    lower = middle - 2 * std

    return (
        middle,
        upper,
        lower
    )


def vwap(df):

    typical = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    volume = (
        pd.to_numeric(
            df["Volume"],
            errors="coerce"
        )
        .fillna(0)
    )

    cumulative_volume = (
        volume.cumsum()
    )

    return (
        typical * volume
    ).cumsum() / cumulative_volume.replace(
        0,
        np.nan
    )


# ============================================================
# PATTERNS
# ============================================================

def detect_patterns(df):

    patterns = []

    if df is None or len(df) < 5:
        return patterns

    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]

    last = -1
    prev = -2

    body = abs(
        c.iloc[last] -
        o.iloc[last]
    )

    candle_range = (
        h.iloc[last] -
        l.iloc[last]
    )

    if candle_range > 0:

        upper_wick = (
            h.iloc[last] -
            max(
                c.iloc[last],
                o.iloc[last]
            )
        )

        lower_wick = (
            min(
                c.iloc[last],
                o.iloc[last]
            ) -
            l.iloc[last]
        )

        if (
            lower_wick > body * 2
            and
            upper_wick < body
        ):
            patterns.append(
                "Hammer"
            )

        if (
            upper_wick > body * 2
            and
            lower_wick < body
        ):
            patterns.append(
                "Shooting Star"
            )

    if (
        c.iloc[prev] < o.iloc[prev]
        and
        c.iloc[last] > o.iloc[last]
        and
        c.iloc[last] >= o.iloc[prev]
        and
        o.iloc[last] <= c.iloc[prev]
    ):
        patterns.append(
            "Bullish Engulfing"
        )

    if (
        c.iloc[prev] > o.iloc[prev]
        and
        c.iloc[last] < o.iloc[last]
        and
        o.iloc[last] >= c.iloc[prev]
        and
        c.iloc[last] <= o.iloc[prev]
    ):
        patterns.append(
            "Bearish Engulfing"
        )

    if (
        candle_range > 0
        and
        body / candle_range < 0.1
    ):
        patterns.append(
            "Doji"
        )

    if len(df) >= 20:

        recent = df.tail(10)

        if (
            recent["High"].iloc[-1]
            >
            recent["High"].iloc[0]
            and
            recent["Low"].iloc[-1]
            >
            recent["Low"].iloc[0]
        ):
            patterns.append(
                "Higher High / Higher Low"
            )

        if (
            recent["High"].iloc[-1]
            <
            recent["High"].iloc[0]
            and
            recent["Low"].iloc[-1]
            <
            recent["Low"].iloc[0]
        ):
            patterns.append(
                "Lower High / Lower Low"
            )

    # Extra structure / momentum patterns (for learning + teaching)
    if len(df) >= 5:
        # Marubozu-like strong close
        if candle_range > 0 and body / candle_range > 0.7:
            if c.iloc[last] > o.iloc[last]:
                patterns.append("Bullish Marubozu")
            else:
                patterns.append("Bearish Marubozu")
        # Inside bar (compression)
        if (
            h.iloc[last] <= h.iloc[prev]
            and l.iloc[last] >= l.iloc[prev]
        ):
            patterns.append("Inside Bar")
        # Outside bar (expansion)
        if (
            h.iloc[last] >= h.iloc[prev]
            and l.iloc[last] <= l.iloc[prev]
            and body > abs(c.iloc[prev] - o.iloc[prev])
        ):
            patterns.append("Outside Bar")

    if len(df) >= 15:
        # Simple swing breakout: close above prior 10-bar high
        prior_high = h.iloc[-11:-1].max()
        prior_low = l.iloc[-11:-1].min()
        if c.iloc[last] > prior_high:
            patterns.append("Breakout High")
        if c.iloc[last] < prior_low:
            patterns.append("Breakdown Low")

    # De-dupe preserve order
    seen = set()
    out = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# Default pattern importance (before learning adjusts weights)
PATTERN_IMPORTANCE = {
    "Hammer": {
        "bias": "bullish", "base_weight": 5,
        "why": "Rejection of lower prices; buyers stepped in after a selloff.",
        "use": "Best after decline or at support; confirm with next green close + volume.",
    },
    "Shooting Star": {
        "bias": "bearish", "base_weight": -5,
        "why": "Rejection of higher prices; supply appeared at the highs.",
        "use": "Best after rally or at resistance; avoid new longs until structure improves.",
    },
    "Bullish Engulfing": {
        "bias": "bullish", "base_weight": 6,
        "why": "Buyers fully overpowered the prior session’s sellers.",
        "use": "Stronger with volume > average and above EMA50.",
    },
    "Bearish Engulfing": {
        "bias": "bearish", "base_weight": -6,
        "why": "Sellers fully overpowered the prior session’s buyers.",
        "use": "Warning to tighten stops or skip fresh longs.",
    },
    "Doji": {
        "bias": "neutral", "base_weight": 0,
        "why": "Indecision; trend may pause or reverse.",
        "use": "Never trade alone — wait for next directional candle.",
    },
    "Higher High / Higher Low": {
        "bias": "bullish", "base_weight": 7,
        "why": "Uptrend structure — demand is stepping up.",
        "use": "Favour BUY/HOLD while structure holds; break of last HL is caution.",
    },
    "Lower High / Lower Low": {
        "bias": "bearish", "base_weight": -7,
        "why": "Downtrend structure — supply is in control.",
        "use": "Avoid fresh longs; prefer exit / wait for structure break up.",
    },
    "Bullish Marubozu": {
        "bias": "bullish", "base_weight": 4,
        "why": "Strong conviction buying with little wick rejection.",
        "use": "Momentum continuation signal; still respect stop below bar low.",
    },
    "Bearish Marubozu": {
        "bias": "bearish", "base_weight": -4,
        "why": "Strong conviction selling with little wick rejection.",
        "use": "Momentum downside; avoid catching falling knife.",
    },
    "Inside Bar": {
        "bias": "neutral", "base_weight": 1,
        "why": "Volatility compression — breakout often follows.",
        "use": "Trade the break of mother-bar high/low with volume.",
    },
    "Outside Bar": {
        "bias": "context", "base_weight": 2,
        "why": "Range expansion; can mark reversal or acceleration.",
        "use": "Combine with trend (EMA) to decide direction.",
    },
    "Breakout High": {
        "bias": "bullish", "base_weight": 5,
        "why": "Close above recent range high — demand broke supply.",
        "use": "Best with volume surge; false breakouts fail back inside range.",
    },
    "Breakdown Low": {
        "bias": "bearish", "base_weight": -5,
        "why": "Close below recent range low — supply broke demand.",
        "use": "Avoid longs until reclaim; trail shorts carefully.",
    },
}


# ============================================================
# CHART STRUCTURE PATTERN SCANNER (Double/Triple/H&S/ABCD/Flag…)
# ============================================================

CHART_STRUCTURE_PATTERNS = {
    # --- Harmonic / measured ---
    "ABCD": {"family": "Harmonic", "bias": "BOTH", "rules": "BC retraces 0.618–0.786 of AB; CD ≈ 1.0×AB (or 1.27/1.618). D is completion zone."},
    "Gartley": {"family": "Harmonic", "bias": "BOTH", "rules": "B=0.618 XA; D=0.786 XA; AB=CD symmetry. PRZ at D."},
    "Butterfly": {"family": "Harmonic", "bias": "BOTH", "rules": "B=0.786 XA; D=1.27–1.618 XA extension beyond X."},
    "Bat": {"family": "Harmonic", "bias": "BOTH", "rules": "B<0.618 XA (often 0.382–0.50); D=0.886 XA."},
    "Crab": {"family": "Harmonic", "bias": "BOTH", "rules": "B=0.382–0.618 XA; D≈1.618 XA (extreme PRZ)."},
    "Deep Crab": {"family": "Harmonic", "bias": "BOTH", "rules": "B≈0.886 XA; D≈1.618 XA."},
    "Shark": {"family": "Harmonic", "bias": "BOTH", "rules": "C extends 1.13–1.618 XA; D=0.886–1.13 XC."},
    "Cypher": {"family": "Harmonic", "bias": "BOTH", "rules": "B=0.382–0.618 XA; C=1.272–1.414 XA; D=0.786 XC."},
    # --- Reversal ---
    "Double Top": {"family": "Reversal", "bias": "BEARISH", "rules": "Two peaks ~same high; neckline = trough between; confirm close below neckline; target = height down."},
    "Double Bottom": {"family": "Reversal", "bias": "BULLISH", "rules": "Two troughs ~same low; neckline = peak between; confirm close above; target = height up."},
    "Triple Top": {"family": "Reversal", "bias": "BEARISH", "rules": "Three peaks at resistance; break below intervening lows."},
    "Triple Bottom": {"family": "Reversal", "bias": "BULLISH", "rules": "Three troughs at support; break above intervening highs."},
    "Head & Shoulders": {"family": "Reversal", "bias": "BEARISH", "rules": "LS, higher head, RS; neckline break down; target = head−neckline."},
    "Inverse H&S": {"family": "Reversal", "bias": "BULLISH", "rules": "LS, deeper head, RS; neckline break up; target = neckline−head up."},
    "Three Rising Valleys": {"family": "Reversal", "bias": "BULLISH", "rules": "Three higher lows (rising valleys) then break of recent high — bullish structure shift."},
    "Three Falling Peaks": {"family": "Reversal", "bias": "BEARISH", "rules": "Three lower highs (falling peaks) then break of recent low — bearish structure shift."},
    "Rounding Bottom": {"family": "Reversal", "bias": "BULLISH", "rules": "Gradual U-shaped base; buy strength above rim of the bowl."},
    "Rounding Top": {"family": "Reversal", "bias": "BEARISH", "rules": "Gradual inverted-U distribution; weakness below rim."},
    # --- Wedges / triangles ---
    "Rising Wedge": {"family": "Wedge/Triangle", "bias": "BEARISH", "rules": "Rising highs & lows, converging; usually bearish on break of lower line."},
    "Falling Wedge": {"family": "Wedge/Triangle", "bias": "BULLISH", "rules": "Falling highs & lows, converging; usually bullish on break of upper line."},
    "Symmetrical Triangle": {"family": "Wedge/Triangle", "bias": "BOTH", "rules": "Converging highs and lows; break either way; target ≈ triangle height."},
    "Ascending Triangle": {"family": "Wedge/Triangle", "bias": "BULLISH", "rules": "Flat resistance, rising lows; buy break of flat top."},
    "Descending Triangle": {"family": "Wedge/Triangle", "bias": "BEARISH", "rules": "Flat support, falling highs; sell break of flat bottom."},
    # --- Rectangle / broadening ---
    "Rectangle Top": {"family": "Rectangle", "bias": "BEARISH", "rules": "Sideways after up-move; break of support is bearish continuation/reversal."},
    "Rectangle Bottom": {"family": "Rectangle", "bias": "BULLISH", "rules": "Sideways after down-move; break of resistance is bullish."},
    "Broadening Top": {"family": "Broadening", "bias": "BEARISH", "rules": "Expanding swings (higher highs & lower lows) at tops — volatile, often resolves down."},
    "Broadening Bottom": {"family": "Broadening", "bias": "BULLISH", "rules": "Expanding swings at bottoms — volatile, often resolves up."},
    # --- Continuation ---
    "Flag": {"family": "Continuation", "bias": "BOTH", "rules": "Sharp pole + tight counter-trend channel; break in pole direction; target ≈ pole length."},
    "Pennant": {"family": "Continuation", "bias": "BOTH", "rules": "Sharp pole + small converging triangle; break with pole; target ≈ pole length."},
    "High & Tight Flag": {"family": "Continuation", "bias": "BULLISH", "rules": "Very strong advance (≥30–50% pole) then tight flag; high historical reliability when strict."},
    "Cup with Handle": {"family": "Continuation", "bias": "BULLISH", "rules": "Rounded cup + shallow handle; buy above rim; target ≈ cup depth."},
    # --- Momentum ---
    "Weak Rebound": {"family": "Momentum", "bias": "BEARISH", "rules": "After decline, bounce fails below prior swing / EMA — sellers still in control."},
    "Weak Pullback": {"family": "Momentum", "bias": "BULLISH", "rules": "After rally, shallow pullback holds above prior swing / EMA — buyers still in control."},
    # --- Trendline ---
    "Trendline S/R": {"family": "Trendline", "bias": "BOTH", "rules": "3+ touches on a support or resistance trendline; trade bounce or break with confirmation."},
}

# Aliases so old detector names still resolve
_PATTERN_ALIASES = {
    "Head and Shoulders": "Head & Shoulders",
    "Inverse Head and Shoulders": "Inverse H&S",
    "ABCD Bullish": "ABCD",
    "ABCD Bearish": "ABCD",
    "Bull Flag": "Flag",
    "Bear Flag": "Flag",
    "Cup and Handle": "Cup with Handle",
}


def _swing_points(df: pd.DataFrame, order: int = 3):
    """Return lists of (iloc, price, date) for swing highs and lows."""
    if df is None or len(df) < order * 2 + 5:
        return [], []
    h = df["High"].values
    l = df["Low"].values
    idx = df.index
    highs, lows = [], []
    n = len(df)
    for i in range(order, n - order):
        window_h = h[i - order : i + order + 1]
        window_l = l[i - order : i + order + 1]
        if h[i] >= window_h.max() and h[i] == window_h.max():
            highs.append((i, float(h[i]), idx[i]))
        if l[i] <= window_l.min() and l[i] == window_l.min():
            lows.append((i, float(l[i]), idx[i]))
    return highs, lows


def _pct_diff(a, b):
    if a is None or b is None or min(abs(a), abs(b)) < 1e-9:
        return 999.0
    return abs(a - b) / ((abs(a) + abs(b)) / 2.0) * 100.0


def _xabcd_pivots(highs, lows):
    piv = sorted(
        [(i, p, "H") for i, p, _ in highs] + [(i, p, "L") for i, p, _ in lows],
        key=lambda x: x[0],
    )
    out = []
    for t in range(len(piv) - 4):
        X, A, B, C, D = piv[t], piv[t + 1], piv[t + 2], piv[t + 3], piv[t + 4]
        # alternating types
        types = [X[2], A[2], B[2], C[2], D[2]]
        ok = all(types[i] != types[i + 1] for i in range(4))
        if not ok:
            continue
        out.append((X, A, B, C, D))
    return out


def _ratio_ok(val, lo, hi, tol=0.08):
    return (lo - tol) <= val <= (hi + tol)


def detect_structure_pattern(df: pd.DataFrame, pattern_name: str, swing_order: int = 3, min_fit: float = 65.0):
    """
    Detect one structure pattern on OHLC. Returns dict or None.
    Fit score 0–100 (higher = cleaner geometry).
    """
    if df is None or len(df) < 40:
        return None
    name = _PATTERN_ALIASES.get(pattern_name, pattern_name)
    highs, lows = _swing_points(df, order=max(2, int(swing_order)))
    close = float(df["Close"].iloc[-1])
    last_i = len(df) - 1

    def pack(direction, fit, points, start_i, end_i, entry, stop, target, explain, pname=None):
        if fit < min_fit:
            return None
        sessions = max(1, end_i - start_i)
        ago = max(0, last_i - end_i)
        try:
            d0 = pd.Timestamp(df.index[start_i]).strftime("%d-%b-%Y")
            d1 = pd.Timestamp(df.index[end_i]).strftime("%d-%b-%Y")
        except Exception:
            d0, d1 = "", ""
        pn = pname or name
        return {
            "pattern": pn,
            "direction": direction,
            "fit": round(fit, 0),
            "points": points,
            "start_i": start_i,
            "end_i": end_i,
            "sessions": sessions,
            "sessions_ago": ago,
            "date_range": f"{d0} → {d1}",
            "entry": round(entry, 2) if entry else None,
            "stop": round(stop, 2) if stop else None,
            "target": round(target, 2) if target else None,
            "explain": explain,
            "rules": CHART_STRUCTURE_PATTERNS.get(pn, CHART_STRUCTURE_PATTERNS.get(name, {})).get("rules", ""),
        }

    # --- Double Bottom ---
    if name == "Double Bottom" and len(lows) >= 2:
        for j in range(len(lows) - 1, 0, -1):
            i2, p2, _ = lows[j]
            i1, p1, _ = lows[j - 1]
            if i2 - i1 < 5 or i2 - i1 > 80:
                continue
            if _pct_diff(p1, p2) > 3.5:
                continue
            mid = df.iloc[i1 : i2 + 1]
            neck = float(mid["High"].max())
            if neck <= max(p1, p2):
                continue
            fit = 100 - _pct_diff(p1, p2) * 8
            if close > neck:
                fit = min(98, fit + 8)
            height = neck - min(p1, p2)
            return pack(
                "BULLISH", fit,
                [("A", i1, p1), ("B", int(mid["High"].values.argmax()) + i1, neck), ("C", i2, p2)],
                i1, i2, neck, min(p1, p2) * 0.99, neck + height,
                f"Double bottom at ~₹{min(p1,p2):.1f}. Neckline ₹{neck:.1f}. "
                f"{'Breakout above neckline confirmed.' if close > neck else 'Wait for close above neckline.'}",
            )

    # --- Double Top ---
    if name == "Double Top" and len(highs) >= 2:
        for j in range(len(highs) - 1, 0, -1):
            i2, p2, _ = highs[j]
            i1, p1, _ = highs[j - 1]
            if i2 - i1 < 5 or i2 - i1 > 80:
                continue
            if _pct_diff(p1, p2) > 3.5:
                continue
            mid = df.iloc[i1 : i2 + 1]
            neck = float(mid["Low"].min())
            if neck >= min(p1, p2):
                continue
            fit = 100 - _pct_diff(p1, p2) * 8
            if close < neck:
                fit = min(98, fit + 8)
            height = max(p1, p2) - neck
            return pack(
                "BEARISH", fit,
                [("A", i1, p1), ("B", int(mid["Low"].values.argmin()) + i1, neck), ("C", i2, p2)],
                i1, i2, neck, max(p1, p2) * 1.01, neck - height,
                f"Double top at ~₹{max(p1,p2):.1f}. Neckline ₹{neck:.1f}. "
                f"{'Breakdown below neckline confirmed.' if close < neck else 'Wait for close below neckline.'}",
            )

    # --- Triple Bottom / Top (3 swings) ---
    if name == "Triple Bottom" and len(lows) >= 3:
        a, b, c = lows[-3], lows[-2], lows[-1]
        if max(_pct_diff(a[1], b[1]), _pct_diff(b[1], c[1])) <= 4.0 and c[0] - a[0] >= 10:
            neck = float(df.iloc[a[0] : c[0] + 1]["High"].max())
            fit = 88 - max(_pct_diff(a[1], b[1]), _pct_diff(b[1], c[1])) * 5
            height = neck - min(a[1], b[1], c[1])
            return pack("BULLISH", fit, [("S1", a[0], a[1]), ("S2", b[0], b[1]), ("S3", c[0], c[1])],
                        a[0], c[0], neck, min(a[1], b[1], c[1]) * 0.99, neck + height,
                        f"Triple bottom support ~₹{min(a[1],b[1],c[1]):.1f}. Break ₹{neck:.1f} confirms.")

    if name == "Triple Top" and len(highs) >= 3:
        a, b, c = highs[-3], highs[-2], highs[-1]
        if max(_pct_diff(a[1], b[1]), _pct_diff(b[1], c[1])) <= 4.0 and c[0] - a[0] >= 10:
            neck = float(df.iloc[a[0] : c[0] + 1]["Low"].min())
            fit = 88 - max(_pct_diff(a[1], b[1]), _pct_diff(b[1], c[1])) * 5
            height = max(a[1], b[1], c[1]) - neck
            return pack("BEARISH", fit, [("P1", a[0], a[1]), ("P2", b[0], b[1]), ("P3", c[0], c[1])],
                        a[0], c[0], neck, max(a[1], b[1], c[1]) * 1.01, neck - height,
                        f"Triple top resistance ~₹{max(a[1],b[1],c[1]):.1f}. Break ₹{neck:.1f} confirms.")

    # --- Head & Shoulders ---
    if name == "Head & Shoulders" and len(highs) >= 3:
        for k in range(len(highs) - 1, 1, -1):
            ls, hd, rs = highs[k - 2], highs[k - 1], highs[k]
            if not (hd[1] > ls[1] and hd[1] > rs[1]):
                continue
            if _pct_diff(ls[1], rs[1]) > 6:
                continue
            if hd[0] - ls[0] < 3 or rs[0] - hd[0] < 3:
                continue
            seg = df.iloc[ls[0] : rs[0] + 1]
            neck = float(seg["Low"].min())
            fit = 90 - _pct_diff(ls[1], rs[1]) * 3
            height = hd[1] - neck
            return pack("BEARISH", fit,
                        [("LS", ls[0], ls[1]), ("H", hd[0], hd[1]), ("RS", rs[0], rs[1])],
                        ls[0], rs[0], neck, max(rs[1], ls[1]) * 1.01, neck - height,
                        f"H&S head ₹{hd[1]:.1f}, neckline ~₹{neck:.1f}. Bearish on break below neckline.")

    if name == "Inverse H&S" and len(lows) >= 3:
        for k in range(len(lows) - 1, 1, -1):
            ls, hd, rs = lows[k - 2], lows[k - 1], lows[k]
            if not (hd[1] < ls[1] and hd[1] < rs[1]):
                continue
            if _pct_diff(ls[1], rs[1]) > 6:
                continue
            if hd[0] - ls[0] < 3 or rs[0] - hd[0] < 3:
                continue
            seg = df.iloc[ls[0] : rs[0] + 1]
            neck = float(seg["High"].max())
            fit = 90 - _pct_diff(ls[1], rs[1]) * 3
            height = neck - hd[1]
            return pack("BULLISH", fit,
                        [("LS", ls[0], ls[1]), ("H", hd[0], hd[1]), ("RS", rs[0], rs[1])],
                        ls[0], rs[0], neck, min(rs[1], ls[1]) * 0.99, neck + height,
                        f"Inverse H&S head ₹{hd[1]:.1f}, neckline ~₹{neck:.1f}. Bullish on break above neckline.")

    # --- Three Rising Valleys / Falling Peaks ---
    if name == "Three Rising Valleys" and len(lows) >= 3:
        a, b, c = lows[-3], lows[-2], lows[-1]
        if a[1] < b[1] < c[1] and c[0] - a[0] >= 8:
            recent_high = float(df.iloc[a[0]:c[0] + 1]["High"].max())
            fit = 80 + min(15, (c[1] - a[1]) / max(a[1], 1) * 100)
            return pack("BULLISH", fit, [("V1", a[0], a[1]), ("V2", b[0], b[1]), ("V3", c[0], c[1])],
                        a[0], c[0], recent_high, a[1] * 0.99, recent_high + (c[1] - a[1]),
                        f"Three rising valleys ₹{a[1]:.1f}→₹{c[1]:.1f}. Break ₹{recent_high:.1f} confirms demand.")

    if name == "Three Falling Peaks" and len(highs) >= 3:
        a, b, c = highs[-3], highs[-2], highs[-1]
        if a[1] > b[1] > c[1] and c[0] - a[0] >= 8:
            recent_low = float(df.iloc[a[0]:c[0] + 1]["Low"].min())
            fit = 80 + min(15, (a[1] - c[1]) / max(a[1], 1) * 100)
            return pack("BEARISH", fit, [("P1", a[0], a[1]), ("P2", b[0], b[1]), ("P3", c[0], c[1])],
                        a[0], c[0], recent_low, a[1] * 1.01, recent_low - (a[1] - c[1]),
                        f"Three falling peaks ₹{a[1]:.1f}→₹{c[1]:.1f}. Break ₹{recent_low:.1f} confirms supply.")

    # --- Rounding Bottom / Top ---
    if name in ("Rounding Bottom", "Rounding Top") and len(df) >= 50:
        w = df.tail(50)
        left = float(w["Close"].iloc[:10].mean())
        mid = float(w["Close"].iloc[20:30].mean())
        right = float(w["Close"].iloc[-10:].mean())
        if name == "Rounding Bottom" and mid < left * 0.97 and right > mid * 1.03:
            fit = 72 + min(12, (right - mid) / mid * 50)
            rim = max(left, right)
            return pack("BULLISH", fit, [], len(df) - 50, len(df) - 1, rim, mid * 0.98, rim + (rim - mid),
                        f"Rounding bottom: mid ₹{mid:.1f}, recovery to ₹{right:.1f}. Strength above rim.")
        if name == "Rounding Top" and mid > left * 1.03 and right < mid * 0.97:
            fit = 72 + min(12, (mid - right) / mid * 50)
            rim = min(left, right)
            return pack("BEARISH", fit, [], len(df) - 50, len(df) - 1, rim, mid * 1.02, rim - (mid - rim),
                        f"Rounding top: mid ₹{mid:.1f}, fade to ₹{right:.1f}. Weakness below rim.")

    # --- Harmonics: ABCD + Gartley/Bat/Butterfly/Crab/Deep Crab/Shark/Cypher ---
    harmonic_names = {"ABCD", "Gartley", "Butterfly", "Bat", "Crab", "Deep Crab", "Shark", "Cypher"}
    if name in harmonic_names:
        for X, A, B, C, D in _xabcd_pivots(highs, lows):
            xa = abs(A[1] - X[1])
            if xa < 1e-6:
                continue
            ab = abs(B[1] - A[1])
            bc = abs(C[1] - B[1])
            cd = abs(D[1] - C[1])
            xb = abs(B[1] - X[1])
            xc = abs(C[1] - X[1])
            xd = abs(D[1] - X[1])
            ab_xa = ab / xa
            bc_ab = bc / ab if ab > 1e-6 else 0
            cd_ab = cd / ab if ab > 1e-6 else 0
            d_xa = xd / xa
            # direction of structure
            bullish = X[2] == "H"  # X high → A low → … ends at D low (buy D)
            # actually bullish Gartley: X low, A high, B low, C high, D low
            bullish = X[2] == "L" and D[2] == "L"
            bearish = X[2] == "H" and D[2] == "H"
            if not (bullish or bearish):
                continue
            direction = "BULLISH" if bullish else "BEARISH"
            ok = False
            fit = 70.0
            if name == "ABCD":
                # 4-point: use A-B-C-D only
                if not (0.5 <= bc_ab <= 0.9 and 0.85 <= cd_ab <= 1.7):
                    continue
                ok = True
                fit = 100 - abs(1.0 - cd_ab) * 25 - abs(0.70 - bc_ab) * 20
            elif name == "Gartley":
                ok = _ratio_ok(ab_xa, 0.55, 0.70) and _ratio_ok(d_xa, 0.72, 0.85)
                fit = 88 if ok else 0
            elif name == "Bat":
                ok = ab_xa < 0.62 and _ratio_ok(d_xa, 0.82, 0.95)
                fit = 88 if ok else 0
            elif name == "Butterfly":
                ok = _ratio_ok(ab_xa, 0.72, 0.85) and d_xa >= 1.20
                fit = 85 if ok else 0
            elif name == "Crab":
                ok = _ratio_ok(ab_xa, 0.35, 0.65) and d_xa >= 1.50
                fit = 85 if ok else 0
            elif name == "Deep Crab":
                ok = _ratio_ok(ab_xa, 0.82, 0.95) and d_xa >= 1.50
                fit = 84 if ok else 0
            elif name == "Shark":
                ok = _ratio_ok(xc / xa if xa else 0, 1.05, 1.70) and _ratio_ok(cd / xc if xc else 0, 0.80, 1.20)
                fit = 82 if ok else 0
            elif name == "Cypher":
                ok = _ratio_ok(ab_xa, 0.35, 0.65) and _ratio_ok(xc / xa if xa else 0, 1.20, 1.50)
                fit = 84 if ok else 0
            if not ok:
                continue
            fit = float(np.clip(fit, 50, 98))
            pts = [("X", X[0], X[1]), ("A", A[0], A[1]), ("B", B[0], B[1]), ("C", C[0], C[1]), ("D", D[0], D[1])]
            if direction == "BULLISH":
                entry, stop, tgt = D[1], D[1] * 0.97, D[1] + xa * 0.382
            else:
                entry, stop, tgt = D[1], D[1] * 1.03, D[1] - xa * 0.382
            return pack(direction, fit, pts, X[0], D[0], entry, stop, tgt,
                        f"{name} {direction} at D ₹{D[1]:.1f} (XA={xa:.1f}, B/XA={ab_xa:.2f}, D/XA={d_xa:.2f}).")

        # ABCD without X (4 pivots) fallback
        if name == "ABCD":
            piv = sorted([(i, p, t) for i, p, t in
                          [(i, p, "H") for i, p, _ in highs] + [(i, p, "L") for i, p, _ in lows]],
                         key=lambda x: x[0])
            for t in range(len(piv) - 3):
                A, B, C, D = piv[t], piv[t + 1], piv[t + 2], piv[t + 3]
                ab = abs(B[1] - A[1])
                if ab < 1e-6:
                    continue
                bc_r = abs(C[1] - B[1]) / ab
                cd_r = abs(D[1] - C[1]) / ab
                if not (0.45 <= bc_r <= 0.90 and 0.85 <= cd_r <= 1.75):
                    continue
                bullish = A[2] == "H" and B[2] == "L" and C[2] == "H" and D[2] == "L"
                bearish = A[2] == "L" and B[2] == "H" and C[2] == "L" and D[2] == "H"
                if not (bullish or bearish):
                    continue
                direction = "BULLISH" if bullish else "BEARISH"
                fit = float(np.clip(100 - abs(1.0 - cd_r) * 25 - abs(0.70 - bc_r) * 20, 55, 98))
                if direction == "BULLISH":
                    entry, stop, tgt = D[1], D[1] * 0.97, D[1] + ab * 0.5
                else:
                    entry, stop, tgt = D[1], D[1] * 1.03, D[1] - ab * 0.5
                return pack(direction, fit,
                            [("A", A[0], A[1]), ("B", B[0], B[1]), ("C", C[0], C[1]), ("D", D[0], D[1])],
                            A[0], D[0], entry, stop, tgt,
                            f"ABCD {direction}: BC={bc_r:.2f}×AB, CD={cd_r:.2f}×AB near D ₹{D[1]:.1f}.")

    # --- Wedges & triangles (use last 4–6 swings regression slope) ---
    if name in ("Rising Wedge", "Falling Wedge", "Symmetrical Triangle", "Ascending Triangle", "Descending Triangle") and len(highs) >= 3 and len(lows) >= 3:
        hh = highs[-4:] if len(highs) >= 4 else highs
        ll = lows[-4:] if len(lows) >= 4 else lows
        h_prices = [p for _, p, _ in hh]
        l_prices = [p for _, p, _ in ll]
        h_rising = h_prices[-1] > h_prices[0]
        h_falling = h_prices[-1] < h_prices[0]
        l_rising = l_prices[-1] > l_prices[0]
        l_falling = l_prices[-1] < l_prices[0]
        # range contraction
        early_range = abs(h_prices[0] - l_prices[0])
        late_range = abs(h_prices[-1] - l_prices[-1])
        contracting = late_range < early_range * 0.85
        expanding = late_range > early_range * 1.15
        start_i = min(hh[0][0], ll[0][0])
        end_i = max(hh[-1][0], ll[-1][0])
        if name == "Rising Wedge" and h_rising and l_rising and contracting:
            entry = l_prices[-1]
            return pack("BEARISH", 78, [], start_i, end_i, entry, h_prices[-1] * 1.01, entry - early_range,
                        "Rising wedge (higher highs & lows, converging) — bias break lower.")
        if name == "Falling Wedge" and h_falling and l_falling and contracting:
            entry = h_prices[-1]
            return pack("BULLISH", 78, [], start_i, end_i, entry, l_prices[-1] * 0.99, entry + early_range,
                        "Falling wedge — bias break higher.")
        if name == "Symmetrical Triangle" and contracting and ((h_falling and l_rising) or (h_rising and l_falling)):
            mid = (h_prices[-1] + l_prices[-1]) / 2
            return pack("BOTH" if False else ("BULLISH" if close > mid else "BEARISH"), 76, [], start_i, end_i,
                        h_prices[-1], l_prices[-1], mid + (early_range if close > mid else -early_range),
                        "Symmetrical triangle — wait for break of converging boundary.")
        if name == "Ascending Triangle" and _pct_diff(h_prices[0], h_prices[-1]) < 2.5 and l_rising:
            return pack("BULLISH", 80, [], start_i, end_i, h_prices[-1], l_prices[-1] * 0.99, h_prices[-1] + early_range,
                        f"Ascending triangle — flat top ~₹{h_prices[-1]:.1f}, rising lows. Buy break of top.")
        if name == "Descending Triangle" and _pct_diff(l_prices[0], l_prices[-1]) < 2.5 and h_falling:
            return pack("BEARISH", 80, [], start_i, end_i, l_prices[-1], h_prices[-1] * 1.01, l_prices[-1] - early_range,
                        f"Descending triangle — flat bottom ~₹{l_prices[-1]:.1f}, falling highs. Sell break of bottom.")

    # --- Rectangle / Broadening ---
    if name in ("Rectangle Top", "Rectangle Bottom", "Broadening Top", "Broadening Bottom") and len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-3:]
        ll = lows[-3:]
        h_avg = sum(p for _, p, _ in hh) / len(hh)
        l_avg = sum(p for _, p, _ in ll) / len(ll)
        h_spread = max(p for _, p, _ in hh) - min(p for _, p, _ in hh)
        l_spread = max(p for _, p, _ in ll) - min(p for _, p, _ in ll)
        height = h_avg - l_avg
        start_i = min(hh[0][0], ll[0][0])
        end_i = max(hh[-1][0], ll[-1][0])
        flat = h_spread / h_avg < 0.03 and l_spread / max(l_avg, 1) < 0.03 and height / h_avg > 0.02
        prior = float(df["Close"].iloc[max(0, start_i - 10)]) if start_i > 5 else close
        if name == "Rectangle Top" and flat and prior < h_avg:
            return pack("BEARISH", 74, [], start_i, end_i, l_avg, h_avg * 1.01, l_avg - height,
                        f"Rectangle after advance — support ₹{l_avg:.1f}, resistance ₹{h_avg:.1f}.")
        if name == "Rectangle Bottom" and flat and prior > l_avg:
            return pack("BULLISH", 74, [], start_i, end_i, h_avg, l_avg * 0.99, h_avg + height,
                        f"Rectangle after decline — resistance ₹{h_avg:.1f}. Buy break up.")
        expanding = h_spread / h_avg > 0.04 and l_spread / max(l_avg, 1) > 0.04
        if name == "Broadening Top" and expanding and prior < close:
            return pack("BEARISH", 72, [], start_i, end_i, l_avg, h_avg * 1.02, l_avg - height,
                        "Broadening top — expanding range, volatile; bias downside resolution.")
        if name == "Broadening Bottom" and expanding and prior > close:
            return pack("BULLISH", 72, [], start_i, end_i, h_avg, l_avg * 0.98, h_avg + height,
                        "Broadening bottom — expanding range; bias upside resolution.")

    # --- Flag / Pennant / High & Tight Flag ---
    if name in ("Flag", "Pennant", "High & Tight Flag") and len(df) >= 25:
        for pole_len in (5, 8, 12, 15):
            if len(df) < pole_len + 8:
                continue
            for end in range(len(df) - 5, max(pole_len + 5, len(df) - 45), -1):
                pole = df.iloc[end - pole_len - 6 : end - 6]
                flag = df.iloc[end - 6 : end + 1]
                if len(pole) < 3 or len(flag) < 4:
                    continue
                pole_move = float(pole["Close"].iloc[-1] - pole["Close"].iloc[0])
                flag_range = float(flag["High"].max() - flag["Low"].min())
                if flag_range < 1e-6 or abs(pole_move) / max(abs(pole["Close"].iloc[0]), 1) < 0.03:
                    continue
                tight = flag_range < abs(pole_move) * 0.55
                if not tight:
                    continue
                pole_pct = abs(pole_move) / max(abs(pole["Close"].iloc[0]), 1) * 100
                if name == "High & Tight Flag" and not (pole_move > 0 and pole_pct >= 25):
                    continue
                if name == "Pennant":
                    # prefer contracting flag highs/lows
                    if float(flag["High"].iloc[-1] - flag["Low"].iloc[-1]) > float(flag["High"].iloc[0] - flag["Low"].iloc[0]) * 0.95:
                        if name == "Pennant":
                            pass  # still allow
                fit = 74 + min(18, pole_pct * 0.4)
                if pole_move > 0:
                    entry = float(flag["High"].max())
                    stop = float(flag["Low"].min())
                    tgt = entry + abs(pole_move)
                    return pack("BULLISH", fit, [], end - pole_len - 6, end, entry, stop, tgt,
                                f"{name}: pole +{pole_pct:.1f}%, flag range ₹{flag_range:.1f}. Buy above ₹{entry:.1f}.")
                else:
                    entry = float(flag["Low"].min())
                    stop = float(flag["High"].max())
                    tgt = entry - abs(pole_move)
                    return pack("BEARISH", fit, [], end - pole_len - 6, end, entry, stop, tgt,
                                f"{name}: pole {pole_pct:.1f}%, flag range ₹{flag_range:.1f}. Sell below ₹{entry:.1f}.")

    # --- Cup with Handle ---
    if name == "Cup with Handle" and len(df) >= 50:
        window = df.tail(60)
        left = float(window["High"].iloc[:15].max())
        mid_low = float(window["Low"].iloc[15:40].min())
        right = float(window["High"].iloc[40:50].max())
        handle_low = float(window["Low"].iloc[50:].min()) if len(window) > 50 else mid_low
        if left > 0 and _pct_diff(left, right) < 5 and mid_low < left * 0.92:
            if handle_low >= mid_low * 0.98:
                rim = max(left, right)
                depth = rim - mid_low
                return pack("BULLISH", 78, [], len(df) - 60, len(df) - 1, rim, handle_low * 0.99, rim + depth,
                            f"Cup depth ₹{depth:.1f}, rim ₹{rim:.1f}. Buy close above rim; stop under handle.")

    # --- Weak Rebound / Weak Pullback ---
    if name in ("Weak Rebound", "Weak Pullback") and len(df) >= 30:
        recent = df.tail(20)
        prior = df.iloc[-40:-20] if len(df) >= 40 else df.iloc[:20]
        if name == "Weak Rebound":
            # prior down, bounce that fails under prior high
            if float(prior["Close"].iloc[-1]) < float(prior["Close"].iloc[0]) * 0.97:
                bounce_high = float(recent["High"].max())
                prior_high = float(prior["High"].max())
                if bounce_high < prior_high and close < bounce_high:
                    return pack("BEARISH", 73, [], len(df) - 40, len(df) - 1, bounce_high * 0.99, bounce_high * 1.02, bounce_high - (prior_high - bounce_high) * 0.5,
                                f"Weak rebound: bounce capped ₹{bounce_high:.1f} under prior high ₹{prior_high:.1f}.")
        if name == "Weak Pullback":
            if float(prior["Close"].iloc[-1]) > float(prior["Close"].iloc[0]) * 1.03:
                pull_low = float(recent["Low"].min())
                prior_low = float(prior["Low"].min())
                if pull_low > prior_low and close > pull_low:
                    return pack("BULLISH", 73, [], len(df) - 40, len(df) - 1, pull_low * 1.01, pull_low * 0.98, pull_low + (pull_low - prior_low) * 0.5,
                                f"Weak pullback: held ₹{pull_low:.1f} above prior low ₹{prior_low:.1f}.")

    # --- Trendline S/R (3+ touches) ---
    if name == "Trendline S/R" and (len(lows) >= 3 or len(highs) >= 3):
        if len(lows) >= 3:
            a, b, c = lows[-3], lows[-2], lows[-1]
            # roughly colinear
            if b[0] != a[0]:
                slope = (b[1] - a[1]) / (b[0] - a[0])
                pred_c = a[1] + slope * (c[0] - a[0])
                if abs(pred_c - c[1]) / max(c[1], 1) < 0.025:
                    return pack("BULLISH", 76, [("T1", a[0], a[1]), ("T2", b[0], b[1]), ("T3", c[0], c[1])],
                                a[0], c[0], c[1] * 1.01, c[1] * 0.98, c[1] + abs(c[1] - a[1]) * 0.5,
                                f"Support trendline 3 touches near ₹{c[1]:.1f}. Bounce or break decides.")
        if len(highs) >= 3:
            a, b, c = highs[-3], highs[-2], highs[-1]
            if b[0] != a[0]:
                slope = (b[1] - a[1]) / (b[0] - a[0])
                pred_c = a[1] + slope * (c[0] - a[0])
                if abs(pred_c - c[1]) / max(c[1], 1) < 0.025:
                    return pack("BEARISH", 76, [("T1", a[0], a[1]), ("T2", b[0], b[1]), ("T3", c[0], c[1])],
                                a[0], c[0], c[1] * 0.99, c[1] * 1.02, c[1] - abs(a[1] - c[1]) * 0.5,
                                f"Resistance trendline 3 touches near ₹{c[1]:.1f}. Rejection or break decides.")

    return None


def scan_chart_patterns(
    symbols,
    pattern_name: str,
    max_stocks: int = 80,
    history_bars: int = 180,
    swing_order: int = 3,
    min_fit: float = 65.0,
    completed_within: int = 15,
    interval: str = "1d",
    timeframe_label: str = "Daily",
) -> list:
    """Screen symbols for one structure pattern on a given timeframe."""
    hits = []
    symbols = list(symbols)[:max_stocks]
    min_bars = 30 if interval in ("1m", "5m", "15m", "30m", "1h", "60m") else 40
    for sym in symbols:
        try:
            df = stock_history(clean_symbol(sym), interval=interval)
            if df is None or len(df) < min_bars:
                continue
            df = df.tail(history_bars)
            hit = detect_structure_pattern(df, pattern_name, swing_order=swing_order, min_fit=min_fit)
            if not hit:
                continue
            if hit.get("sessions_ago", 99) > completed_within:
                continue
            hit["timeframe"] = timeframe_label
            hit["interval"] = interval
            # Live price + gap
            live = None
            gap_pct = None
            try:
                q = live_quote(sym)
                if q and q.get("price"):
                    live = safe_float(q["price"])
            except Exception:
                pass
            if live is None:
                live = safe_float(df["Close"].iloc[-1])
            try:
                prev = safe_float(df["Close"].iloc[-2]) if len(df) > 1 else live
                if prev:
                    gap_pct = (live - prev) / prev * 100.0
            except Exception:
                gap_pct = 0.0
            # Extra candle patterns
            extra = []
            try:
                extra = detect_patterns(df.tail(20))[:4]
            except Exception:
                pass
            # Chart window: pattern span + padding so all points stay on screen
            s_i = int(hit.get("start_i") or 0)
            e_i = int(hit.get("end_i") or (len(df) - 1))
            pad = 12
            left = max(0, s_i - pad)
            right = min(len(df), e_i + pad + 1)
            chart_df = df.iloc[left:right].copy()
            # Re-index points relative to chart_df
            rel_pts = []
            for p in (hit.get("points") or []):
                try:
                    lab, ii, price = p[0], int(p[1]), float(p[2])
                    rel = ii - left
                    if 0 <= rel < len(chart_df):
                        rel_pts.append((lab, rel, price))
                except Exception:
                    continue
            hit.update({
                "stock": display_symbol(sym),
                "live": round(live, 2) if live else None,
                "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
                "extra_patterns": extra,
                "df_tail": chart_df,
                "points": rel_pts if rel_pts else hit.get("points") or [],
                "_chart_left": left,
                "timeframe": hit.get("timeframe") or timeframe_label,
                "interval": hit.get("interval") or interval,
            })
            hits.append(hit)
        except Exception:
            continue
    hits.sort(key=lambda x: (x.get("sessions_ago", 99), -x.get("fit", 0)))
    return hits


def _pattern_mini_chart(df, hit: dict, height: int = 280):
    """
    Candles + ALL important pattern points marked (2, 3, 4, 5… as given).
    Labels A/B/C/D, LS/H/RS, V1/V2/V3, etc. + entry/stop/target lines.
    """
    if df is None or df.empty:
        return None
    n = len(df)
    xs_all = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=xs_all,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
    ))

    # Normalize points: list of (label, index, price)
    raw_pts = hit.get("points") or []
    pts = []
    for p in raw_pts:
        try:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                lab, ii, price = str(p[0]), int(p[1]), float(p[2])
            elif isinstance(p, dict):
                lab, ii, price = str(p.get("lab", "")), int(p.get("i", 0)), float(p.get("price", 0))
            else:
                continue
            if 0 <= ii < n and price > 0:
                pts.append((lab, ii, price))
        except Exception:
            continue

    # If detector gave no points, mark start / mid / end of pattern span
    if not pts:
        s = int(hit.get("start_i", 0) or 0)
        e = int(hit.get("end_i", n - 1) or (n - 1))
        # indices may be absolute from pre-slice; clamp to chart
        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))
        if e < s:
            s, e = e, s
        mid = (s + e) // 2
        try:
            pts = [
                ("Start", s, float(df["Low"].iloc[s])),
                ("Mid", mid, float(df["Close"].iloc[mid])),
                ("End", e, float(df["High"].iloc[e])),
            ]
        except Exception:
            pts = []

    # Sort by index so the connecting line follows time
    pts = sorted(pts, key=lambda t: t[1])

    if pts:
        px = [p[1] for p in pts]
        py = [p[2] for p in pts]
        plab = [p[0] for p in pts]

        # Connecting polyline through every point
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="lines",
            line=dict(color="#f97316", width=2.5, dash="solid"),
            name="Structure",
            hoverinfo="skip",
        ))

        # Each point as its own marker + label (all of them)
        colors = ["#fbbf24", "#38bdf8", "#a78bfa", "#f472b6", "#4ade80", "#fb7185", "#2dd4bf"]
        for k, (lab, xi, yi) in enumerate(pts):
            col = colors[k % len(colors)]
            fig.add_trace(go.Scatter(
                x=[xi], y=[yi],
                mode="markers+text",
                marker=dict(size=12, color=col, symbol="circle",
                            line=dict(width=2, color="#0f172a")),
                text=[str(lab)],
                textposition="top center" if k % 2 == 0 else "bottom center",
                textfont=dict(size=12, color=col, family="Arial Black"),
                name=str(lab),
                hovertemplate=f"<b>{lab}</b><br>Bar {xi}<br>₹{yi:,.2f}<extra></extra>",
            ))
            # Vertical guide at each key bar
            fig.add_vline(
                x=xi, line_width=1, line_dash="dot", line_color="rgba(148,163,184,0.35)"
            )

        # Caption under chart
        point_summary = " → ".join([f"{lab}(₹{yi:,.1f})" for lab, _, yi in pts])
    else:
        point_summary = ""

    # Entry / Stop / Target horizontal levels
    for key, color, dash in (
        ("entry", "#38bdf8", "dash"),
        ("stop", "#f87171", "dot"),
        ("target", "#4ade80", "dash"),
    ):
        val = hit.get(key)
        try:
            val = float(val) if val is not None else None
        except Exception:
            val = None
        if val and val > 0:
            fig.add_hline(
                y=val, line_width=1.5, line_dash=dash, line_color=color,
                annotation_text=f"{key.upper()} ₹{val:,.1f}",
                annotation_position="right",
                annotation_font_size=10,
                annotation_font_color=color,
            )

    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.95)",
        font=dict(color="#e2e8f0", size=10),
        showlegend=False,
        title=dict(
            text=point_summary[:80] + ("…" if len(point_summary) > 80 else ""),
            font=dict(size=11, color="#94a3b8"),
            x=0.01, xanchor="left",
        ),
    )
    fig.update_xaxes(showgrid=False, visible=True, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b", tickfont=dict(size=9))
    return fig


def show_chart_pattern_scanner(results: pd.DataFrame = None):
    """UI: pick pattern → screen NSE → cards with chart, live, gap, explanation."""
    st.markdown(
        """
        <div style="border-radius:16px;padding:18px 20px;margin-bottom:12px;
                    background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#0f766e 100%);
                    border:1px solid #334155;">
          <div style="color:#94a3b8;font-size:0.8rem;font-weight:600;letter-spacing:0.05em;">PATTERN SCREENER</div>
          <div style="font-size:1.55rem;font-weight:800;color:#f8fafc;">Chart Pattern Screener for NSE Stocks</div>
          <div style="color:#94a3b8;margin-top:6px;line-height:1.4;">
            Pick one pattern — check every stock — newest completed shapes first.
            Each hit is drawn with points marked, fit score, and <b>which timeframe</b> it formed on.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Universe
    if results is not None and not results.empty and "Stock" in results.columns:
        universe = results["Stock"].astype(str).str.replace(".NS", "", regex=False).str.upper().unique().tolist()
    else:
        universe = list(NSE_STOCKS) if NSE_STOCKS else []
    st.caption(f"Universe: **{len(universe):,}** symbols (from last scan when available).")

    with st.container():
        st.markdown("##### What to screen for")
        c1, c2 = st.columns(2)
        with c1:
            # Grouped pattern list (matches RG Tools style families)
            _pat_keys = list(CHART_STRUCTURE_PATTERNS.keys())
            pattern = st.selectbox(
                "PATTERN",
                _pat_keys,
                index=_pat_keys.index("Double Top") if "Double Top" in _pat_keys else 0,
                help="One structure pattern per screen. Families: Harmonic · Reversal · Wedge/Triangle · Rectangle · Continuation · Momentum · Trendline.",
                key="cps_pattern",
                format_func=lambda k: f"{CHART_STRUCTURE_PATTERNS[k]['family']} · {k}",
            )
            meta = CHART_STRUCTURE_PATTERNS[pattern]
            st.caption(f"**{meta['family']}** · bias **{meta['bias']}**")
            _tf_opts = [
                ("All timeframes (scan each)", "ALL"),
                ("Daily", "1d"),
                ("Weekly", "1wk"),
                ("Hourly", "1h"),
                ("30 minutes", "30m"),
                ("15 minutes", "15m"),
                ("5 minutes", "5m"),
            ]
            timeframe = st.selectbox(
                "TIMEFRAME",
                _tf_opts,
                index=1,
                format_func=lambda x: x[0],
                key="cps_tf",
                help="Pattern is detected on this chart interval. 'All timeframes' runs Daily+Weekly+Hourly+15m.",
            )
            min_fit = st.select_slider(
                "MINIMUM FIT",
                options=[55, 60, 65, 70, 75, 80],
                value=65,
                format_func=lambda x: f"{x} — {'loose' if x < 65 else 'good' if x < 75 else 'strict'} fits",
                key="cps_fit",
            )
        with c2:
            within = st.selectbox(
                "COMPLETED WITHIN",
                [("Last 5 bars", 5), ("Last 10 bars", 10), ("Last 15 bars", 15), ("Last 25 bars", 25)],
                format_func=lambda x: x[0],
                index=1,
                key="cps_within",
            )
            hist_bars = st.selectbox(
                "HISTORY READ PER STOCK",
                [90, 120, 180, 250],
                index=2,
                format_func=lambda x: f"Last {x} bars",
                key="cps_hist",
            )
            swing = st.select_slider(
                "SWING SIZE",
                options=[2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"{x} — {'tight' if x <= 2 else 'balanced' if x == 3 else 'wide'}",
                key="cps_swing",
            )
            max_chk = st.slider("Max stocks to check", 20, 200, 60, key="cps_max")

    with st.expander("📖 Pattern rules (reference)", expanded=False):
        st.markdown(f"**{pattern}** — {meta['rules']}")
        st.caption("Rules summarized from standard technical analysis (measured moves, Fib ratios for ABCD).")

    b1, b2 = st.columns([1, 1])
    run = b1.button("Screen the market", type="primary", use_container_width=True, key="cps_run")
    if b2.button("Start over / clear", use_container_width=True, key="cps_clear"):
        st.session_state.pop("cps_hits", None)
        st.rerun()

    if run:
        tf_label, tf_code = timeframe[0], timeframe[1]
        # Map label for single TF
        _label_map = {
            "1d": "Daily", "1wk": "Weekly", "1h": "Hourly",
            "30m": "30 min", "15m": "15 min", "5m": "5 min",
        }
        if tf_code == "ALL":
            tf_jobs = [
                ("Daily", "1d", max(hist_bars, 120)),
                ("Weekly", "1wk", 80),
                ("Hourly", "1h", 120),
                ("15 min", "15m", 96),
            ]
        else:
            tf_jobs = [(_label_map.get(tf_code, tf_label), tf_code, int(hist_bars))]

        all_hits = []
        with st.spinner(f"Screening {min(max_chk, len(universe))} stocks · {tf_label}…"):
            for lbl, iv, bars in tf_jobs:
                part = scan_chart_patterns(
                    universe,
                    pattern,
                    max_stocks=max_chk,
                    history_bars=bars,
                    swing_order=int(swing),
                    min_fit=float(min_fit),
                    completed_within=int(within[1]),
                    interval=iv,
                    timeframe_label=lbl,
                )
                all_hits.extend(part)
            # Prefer higher fit; keep multi-TF tags
            all_hits.sort(key=lambda x: (x.get("sessions_ago", 99), -x.get("fit", 0)))
            st.session_state["cps_hits"] = all_hits
            st.session_state["cps_hits_pattern"] = pattern
            st.session_state["cps_hits_tf"] = tf_label

    hits = st.session_state.get("cps_hits") or []
    if not hits:
        st.info("Pick a pattern, then **Screen the market**. Results appear as cards (most recent first).")
        return

    st.markdown(f"##### Most recently completed — **{len(hits)}** hits · {st.session_state.get('cps_hits_pattern', pattern)}")
    # Show top 12 cards in 2–4 columns
    n_show = min(12, len(hits))
    for row_start in range(0, n_show, 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= n_show:
                break
            hit = hits[idx]
            with col:
                direction = hit.get("direction", "")
                badge_bg = "#16a34a" if direction == "BULLISH" else "#dc2626"
                gap = hit.get("gap_pct")
                gap_txt = f"{gap:+.2f}%" if gap is not None else "—"
                gap_color = "#4ade80" if (gap or 0) >= 0 else "#f87171"
                st.markdown(
                    f"""
                    <div style="border:1px solid #334155;border-radius:14px;padding:12px 14px;margin-bottom:8px;
                                background:#0f172a;">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;">{hit.get('stock')}</div>
                        <div style="color:#94a3b8;font-size:0.85rem;">{hit.get('sessions_ago', '?')} sessions ago</div>
                      </div>
                      <div style="margin-top:6px;">
                        <span style="background:{badge_bg};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;">{direction}</span>
                        <span style="background:#1e3a5f;color:#7dd3fc;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;margin-left:6px;">⏱ {hit.get('timeframe') or 'Daily'}</span>
                        <span style="color:#e2e8f0;margin-left:8px;">Fit <b>{hit.get('fit', 0):.0f}</b></span>
                        <span style="color:#94a3b8;margin-left:8px;">{hit.get('date_range', '')}</span>
                      </div>
                      <div style="margin-top:8px;display:flex;gap:12px;flex-wrap:wrap;font-size:0.9rem;">
                        <span style="color:#f8fafc;">Live <b>₹{hit.get('live') or '—'}</b></span>
                        <span style="color:{gap_color};">Gap <b>{gap_txt}</b></span>
                        <span style="color:#94a3b8;">{hit.get('sessions')} sessions</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig = _pattern_mini_chart(hit.get("df_tail"), hit)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    f"**Timeframe:** `{hit.get('timeframe') or 'Daily'}` (`{hit.get('interval') or '1d'}`) · "
                    f"Entry ₹{hit.get('entry') or '—'} · "
                    f"Stop ₹{hit.get('stop') or '—'} · Target ₹{hit.get('target') or '—'}"
                )
                st.caption(f"Pattern formed on **{hit.get('timeframe') or 'Daily'}** chart. " + (hit.get("explain") or ""))
                with st.expander("Rules & more"):
                    st.write(hit.get("rules") or "")
                    extra = hit.get("extra_patterns") or []
                    if extra:
                        st.write("Also on chart (candles): " + ", ".join(extra))
                    render_active_trade_buttons(
                        hit.get("stock"),
                        side_hint="BUY" if direction == "BULLISH" else "SELL",
                        entry=safe_float(hit.get("live") or hit.get("entry")),
                        target=safe_float(hit.get("target")),
                        stop=safe_float(hit.get("stop")),
                        key_prefix=f"cps_{hit.get('stock')}_{idx}",
                    )


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Fold daily bars into weekly/monthly OHLC."""
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")
    x = x.dropna(how="all")
    ohlc = x.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum" if "Volume" in x.columns else "last",
    }).dropna(subset=["Close"])
    return ohlc


def _tf_candle_bias(df: pd.DataFrame, lookback: int = 3) -> dict:
    """
    Simple multi-bar bias for a timeframe: bullish / bearish / neutral.
    Uses structure (HH/HL vs LH/LL) + last candle direction — ignores pure doji.
    """
    if df is None or len(df) < 5:
        return {"bias": "NEUTRAL", "pattern": "Insufficient data", "detail": ""}
    tail = df.tail(max(lookback + 2, 8))
    c = tail["Close"]
    o = tail["Open"]
    h = tail["High"]
    l = tail["Low"]
    last_o, last_c = float(o.iloc[-1]), float(c.iloc[-1])
    body = abs(last_c - last_o)
    rng = float(h.iloc[-1] - l.iloc[-1]) or 1e-9
    # Doji / neutral single bar does not confirm direction
    if body / rng < 0.12:
        single = "Doji / indecision"
    elif last_c > last_o:
        single = "Bullish candle"
    else:
        single = "Bearish candle"

    hh = float(h.iloc[-1]) >= float(h.iloc[-3:-1].max())
    hl = float(l.iloc[-1]) >= float(l.iloc[-3:-1].min())
    lh = float(h.iloc[-1]) <= float(h.iloc[-3:-1].max())
    ll = float(l.iloc[-1]) <= float(l.iloc[-3:-1].min())

    if hh and hl and last_c > last_o:
        bias, pattern = "BULLISH", "Higher high / higher low + green"
    elif lh and ll and last_c < last_o:
        bias, pattern = "BEARISH", "Lower high / lower low + red"
    elif last_c > float(c.iloc[-5:].mean()) and last_c > last_o:
        bias, pattern = "BULLISH", "Above short mean + green"
    elif last_c < float(c.iloc[-5:].mean()) and last_c < last_o:
        bias, pattern = "BEARISH", "Below short mean + red"
    else:
        bias, pattern = "NEUTRAL", single

    # Candle patterns on this TF
    extra = []
    try:
        extra = detect_patterns(tail)[:3]
    except Exception:
        pass
    return {
        "bias": bias,
        "pattern": pattern,
        "candles": extra,
        "close": round(last_c, 2),
        "detail": f"Close ₹{last_c:.2f} · {pattern}",
    }


def analyze_multi_timeframe(symbol: str) -> dict:
    """Monthly + Weekly + Daily bias alignment for one NSE stock."""
    sym = display_symbol(symbol)
    daily = stock_history(clean_symbol(sym), interval="1d")
    if daily is None or len(daily) < 60:
        return {"ok": False, "msg": f"Not enough history for {sym}"}
    weekly = _resample_ohlc(daily, "W-FRI")
    monthly = _resample_ohlc(daily, "ME") if hasattr(pd, "offsets") else _resample_ohlc(daily, "M")
    try:
        monthly = _resample_ohlc(daily, "ME")
    except Exception:
        monthly = _resample_ohlc(daily, "M")

    m = _tf_candle_bias(monthly, 2)
    w = _tf_candle_bias(weekly, 3)
    d = _tf_candle_bias(daily, 5)

    live = None
    gap = None
    try:
        q = live_quote(sym)
        if q and q.get("price"):
            live = safe_float(q["price"])
    except Exception:
        pass
    if live is None:
        live = safe_float(daily["Close"].iloc[-1])
    try:
        prev = safe_float(daily["Close"].iloc[-2])
        gap = (live - prev) / prev * 100 if prev else 0
    except Exception:
        gap = 0

    biases = [m["bias"], w["bias"], d["bias"]]
    if biases.count("BULLISH") == 3:
        align = "BULLISH on all three"
        score = 95
    elif biases.count("BEARISH") == 3:
        align = "BEARISH on all three"
        score = 95
    elif biases.count("BULLISH") >= 2 and "BEARISH" not in biases:
        align = "Mostly BULLISH (no bearish TF)"
        score = 78
    elif biases.count("BEARISH") >= 2 and "BULLISH" not in biases:
        align = "Mostly BEARISH (no bullish TF)"
        score = 78
    else:
        align = "MIXED — timeframes disagree"
        score = 45

    return {
        "ok": True,
        "stock": sym,
        "live": round(live, 2),
        "gap_pct": round(gap, 2) if gap is not None else None,
        "monthly": m,
        "weekly": w,
        "daily": d,
        "align": align,
        "score": score,
        "daily_df": daily.tail(90),
    }


def _hub_card_grid(cards: list, key_prefix: str = "hub"):
    """
    cards = list of (title, description, page_key)
    Renders 2-column open-tool style cards on the main screen.
    """
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(cards):
                break
            title, desc, page = cards[i + j]
            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid #334155;border-radius:14px;padding:16px 18px;margin-bottom:8px;
                                background:linear-gradient(145deg,#0f172a 0%,#1e293b 100%);min-height:120px;">
                      <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;">{title}</div>
                      <div style="color:#94a3b8;font-size:0.88rem;margin-top:6px;line-height:1.4;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open → {title}", key=f"{key_prefix}_{page}", use_container_width=True):
                    nav_go(page)



def fetch_fii_dii_latest() -> dict:
    """Latest FII/DII cash from free Mr.Chartist API (NSE-sourced)."""
    out = {"ok": False, "error": None, "raw": None}
    try:
        import requests
        r = requests.get("https://fii-diidata.mrchartist.com/api/data", timeout=12)
        if r.status_code == 200:
            out["ok"] = True
            out["raw"] = r.json()
            return out
        out["error"] = f"HTTP {r.status_code}"
    except Exception as e:
        out["error"] = str(e)
    # Fallback: NSE-style via alternative mirror if any
    try:
        import requests
        r = requests.get("https://fii-diidata.mrchartist.com/api/history", timeout=12)
        if r.status_code == 200:
            data = r.json()
            rows = data if isinstance(data, list) else data.get("data") or data.get("history") or []
            if rows:
                out["ok"] = True
                out["raw"] = rows[0] if isinstance(rows[0], dict) else {"history": rows}
                out["history"] = rows
                return out
    except Exception as e2:
        out["error"] = (out.get("error") or "") + f" | {e2}"
    return out


def fetch_fii_dii_history(limit: int = 30) -> pd.DataFrame:
    """Recent FII/DII sessions as DataFrame (₹ Cr)."""
    try:
        import requests
        r = requests.get("https://fii-diidata.mrchartist.com/api/history", timeout=15)
        if r.status_code != 200:
            r = requests.get("https://fii-diidata.mrchartist.com/api/history-full", timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        rows = data if isinstance(data, list) else (
            data.get("data") or data.get("history") or data.get("rows") or []
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Normalize common keys
        rename = {
            "d": "Date", "date": "Date", "Date": "Date",
            "fb": "FII Buy", "fs": "FII Sell", "fn": "FII Net",
            "db": "DII Buy", "ds": "DII Sell", "dn": "DII Net",
            "fii_buy": "FII Buy", "fii_sell": "FII Sell", "fii_net": "FII Net",
            "dii_buy": "DII Buy", "dii_sell": "DII Sell", "dii_net": "DII Net",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date", ascending=False)
        return df.head(limit)
    except Exception:
        return pd.DataFrame()


def fetch_stock_institutional(symbol: str) -> dict:
    """
    Stock-level institutional picture via Yahoo (holders).
    NSE does not publish pure FII/DII per stock daily in free form;
    we show institutional + mutual fund holders + major holders.
    """
    out = {"symbol": display_symbol(symbol), "major": None, "inst": None, "mf": None, "error": None}
    try:
        t = yf.Ticker(clean_symbol(symbol))
        try:
            mh = t.major_holders
            if mh is not None and not mh.empty:
                out["major"] = mh
        except Exception:
            pass
        try:
            ih = t.institutional_holders
            if ih is not None and not ih.empty:
                out["inst"] = ih
        except Exception:
            pass
        try:
            mf = t.mutualfund_holders
            if mf is not None and not mf.empty:
                out["mf"] = mf
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)
    return out


def show_fii_dii_page():
    """
    FII / DII Data — Daily Cash Market Activity
    Format inspired by RG Tools (tools.ruchirgupta.in/tools/fii-dii/).
    Standalone page — not merged with other tools.
    """
    st.markdown(
        """
        <div style="border-radius:16px;padding:20px 22px;margin-bottom:16px;
                    background:linear-gradient(135deg,#0b1220 0%,#0f172a 50%,#14532d 100%);
                    border:1px solid #166534;">
          <div style="color:#86efac;font-size:0.8rem;font-weight:600;letter-spacing:0.06em;">FII / DII DATA</div>
          <div style="font-size:1.65rem;font-weight:800;color:#ecfdf5;margin-top:4px;">
            Daily Cash Market Activity
          </div>
          <div style="color:#a7f3d0;margin-top:8px;line-height:1.45;max-width:52rem;">
            Follow what the big money did. Foreign (FII/FPI) and domestic (DII) institutional
            <b>buying, selling and net flow</b> in the Indian cash market, date by date —
            with a chart for the trend and a table for the detail.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Controls — RG style: Load Range / Reload Latest
    r1, r2, r3 = st.columns([1.2, 1.2, 1])
    with r1:
        days = st.selectbox(
            "History window",
            [5, 10, 20, 30, 60, 90],
            index=2,
            format_func=lambda d: f"Last {d} sessions",
            key="fii_days_rg",
        )
    with r2:
        view_mode = st.radio(
            "View",
            ["Table View", "Chart View"],
            horizontal=True,
            key="fii_view_rg",
        )
    with r3:
        st.write("")
        st.write("")
        if st.button("↻ Reload Latest", type="primary", use_container_width=True, key="fii_reload_rg"):
            st.session_state.pop("_fii_cache", None)
            st.session_state.pop("_fii_hist", None)
            st.rerun()

    # Latest snapshot cards
    latest = st.session_state.get("_fii_cache")
    if not latest:
        with st.spinner("Loading latest FII / DII records…"):
            latest = fetch_fii_dii_latest()
            st.session_state["_fii_cache"] = latest

    if latest.get("ok") and latest.get("raw"):
        raw = latest["raw"]

        def g(*keys, default=None):
            if isinstance(raw, dict):
                for k in keys:
                    if k in raw and raw[k] is not None:
                        return raw[k]
                    for nest in ("cash", "data", "latest", "fii", "dii"):
                        if nest in raw and isinstance(raw[nest], dict) and k in raw[nest]:
                            return raw[nest][k]
            return default

        fii_buy = safe_float(g("fii_buy", "fb", "FII Buy", "fiiBuy"), 0)
        fii_sell = safe_float(g("fii_sell", "fs", "FII Sell", "fiiSell"), 0)
        fii_net = safe_float(g("fii_net", "fn", "FII Net", "fiiNet"), fii_buy - fii_sell)
        dii_buy = safe_float(g("dii_buy", "db", "DII Buy", "diiBuy"), 0)
        dii_sell = safe_float(g("dii_sell", "ds", "DII Sell", "diiSell"), 0)
        dii_net = safe_float(g("dii_net", "dn", "DII Net", "diiNet"), dii_buy - dii_sell)
        asof = g("date", "d", "as_of", "Date", "session") or "—"

        st.markdown("##### Latest session")
        st.caption(f"As of **{asof}** · values in ₹ Crore · cash market (provisional after close)")
        a, b, c = st.columns(3)
        with a:
            st.markdown(
                f"""
                <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;padding:14px;">
                  <div style="color:#7dd3fc;font-weight:700;">FII (Foreign)</div>
                  <div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">Buy &nbsp; <b style="color:#e2e8f0;">₹{fii_buy:,.0f} Cr</b></div>
                  <div style="color:#94a3b8;font-size:0.85rem;">Sell &nbsp; <b style="color:#e2e8f0;">₹{fii_sell:,.0f} Cr</b></div>
                  <div style="margin-top:8px;font-size:1.25rem;font-weight:800;color:{'#4ade80' if fii_net>=0 else '#f87171'};">
                    Net {fii_net:+,.0f} Cr
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b:
            st.markdown(
                f"""
                <div style="background:#0f172a;border:1px solid #14532d;border-radius:12px;padding:14px;">
                  <div style="color:#86efac;font-weight:700;">DII (Domestic)</div>
                  <div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">Buy &nbsp; <b style="color:#e2e8f0;">₹{dii_buy:,.0f} Cr</b></div>
                  <div style="color:#94a3b8;font-size:0.85rem;">Sell &nbsp; <b style="color:#e2e8f0;">₹{dii_sell:,.0f} Cr</b></div>
                  <div style="margin-top:8px;font-size:1.25rem;font-weight:800;color:{'#4ade80' if dii_net>=0 else '#f87171'};">
                    Net {dii_net:+,.0f} Cr
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c:
            bias = []
            if fii_net > 0:
                bias.append("FII net **buying**")
            elif fii_net < 0:
                bias.append("FII net **selling**")
            if dii_net > 0:
                bias.append("DII net **buying**")
            elif dii_net < 0:
                bias.append("DII net **selling**")
            st.markdown(
                f"""
                <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:14px;min-height:120px;">
                  <div style="color:#e2e8f0;font-weight:700;">Read</div>
                  <div style="color:#94a3b8;margin-top:8px;line-height:1.5;">{" · ".join(bias) if bias else "Flat institutional flow."}</div>
                  <div style="color:#64748b;font-size:0.8rem;margin-top:10px;">Cash market only — not F&amp;O participant data.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning(f"Could not load latest snapshot. {latest.get('error') or 'Try Reload Latest.'}")

    # History
    hist = st.session_state.get("_fii_hist")
    if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
        with st.spinner("Loading range…"):
            hist = fetch_fii_dii_history(limit=int(days))
            st.session_state["_fii_hist"] = hist
    elif isinstance(hist, pd.DataFrame) and len(hist) < int(days):
        hist = fetch_fii_dii_history(limit=int(days))
        st.session_state["_fii_hist"] = hist

    st.markdown("##### Date-wise activity")
    if hist is not None and not hist.empty:
        show_cols = [c for c in [
            "Date", "FII Buy", "FII Sell", "FII Net", "DII Buy", "DII Sell", "DII Net"
        ] if c in hist.columns]
        if view_mode.startswith("Table"):
            if show_cols:
                st.dataframe(hist[show_cols].head(int(days)), use_container_width=True, hide_index=True)
            else:
                st.dataframe(hist.head(int(days)), use_container_width=True, hide_index=True)
        else:
            try:
                plot_df = hist.copy()
                if "Date" in plot_df.columns:
                    plot_df = plot_df.sort_values("Date")
                fig = go.Figure()
                if "FII Net" in plot_df.columns:
                    fig.add_trace(go.Bar(
                        x=plot_df["Date"], y=plot_df["FII Net"], name="FII Net",
                        marker_color="#38bdf8",
                    ))
                if "DII Net" in plot_df.columns:
                    fig.add_trace(go.Bar(
                        x=plot_df["Date"], y=plot_df["DII Net"], name="DII Net",
                        marker_color="#4ade80",
                    ))
                fig.update_layout(
                    barmode="group", height=340, margin=dict(l=8, r=8, t=28, b=8),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.95)",
                    font=dict(color="#e2e8f0", size=11), legend=dict(orientation="h"),
                    yaxis_title="Net ₹ Crore", title="FII / DII net flow",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                st.caption(f"Chart unavailable: {e}")
                if show_cols:
                    st.dataframe(hist[show_cols].head(int(days)), use_container_width=True, hide_index=True)
    else:
        st.info("No history rows yet. Tap **Reload Latest**.")

    # Stock-level (separate block on same standalone page)
    st.markdown("---")
    st.markdown("##### Stock-level holders (optional)")
    st.caption(
        "NSE does not publish pure daily FII/DII **per stock** for free. "
        "Optional: institutional / mutual-fund holders for one symbol."
    )
    s1, s2 = st.columns([2, 1])
    with s1:
        sym = st.text_input(
            "NSE symbol",
            value=str(st.session_state.get("selected_stock", "RELIANCE") or "RELIANCE"),
            key="fii_stock_sym_rg",
        )
    with s2:
        st.write("")
        st.write("")
        load_stk = st.button("Load holders", use_container_width=True, key="fii_stock_btn_rg")
    if load_stk:
        with st.spinner(f"Loading holders for {sym}…"):
            info = fetch_stock_institutional(sym)
        if info.get("error"):
            st.warning(info["error"])
        if info.get("major") is not None:
            st.write("**Major holders**")
            st.dataframe(info["major"], use_container_width=True)
        if info.get("inst") is not None:
            st.write("**Institutional holders**")
            st.dataframe(info["inst"], use_container_width=True)
        if info.get("mf") is not None:
            st.write("**Mutual fund holders**")
            st.dataframe(info["mf"], use_container_width=True)
        if info.get("major") is None and info.get("inst") is None and info.get("mf") is None:
            st.info("No holder tables for this symbol.")

    with st.expander("How to use · FAQ (same ideas as RG Tools)", expanded=False):
        st.markdown(
            """
**How to use**
1. **Set the window** — last N sessions (or reload latest).
2. **Table view** — FII/DII buy, sell, net for every day in the window.
3. **Chart view** — net-flow trend instead of row by row.

**What do FII and DII mean?**  
FII/FPI = overseas funds. DII = Indian mutual funds, insurers, banks.  
**Net** = buy − sell (₹ Cr). Positive = net buyer that session.

**How often updated?**  
After NSE publishes evening provisional figures; confirmed next morning.

**Cash or derivatives?**  
Cash market only — equity institutional activity, not F&O OI.

*Educational / research only — not investment advice.*
            """
        )



def show_performance_summary():
    """
    Clear strategy performance desk: which source/strategy wins,
    combos, stop-loss behaviour, holding-period profit vs loss.
    """
    st.markdown(
        """
        <div style="border-radius:16px;padding:18px 20px;margin-bottom:14px;
                    background:linear-gradient(135deg,#0f172a 0%,#0e7490 100%);
                    border:1px solid #22d3ee;">
          <div style="font-size:1.55rem;font-weight:800;color:#ecfeff;">Call Performance Summary</div>
          <div style="color:#a5f3fc;margin-top:4px;">
            Which strategies work · stop-loss behaviour · holding-period outcomes · simple filters
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hist = normalize_history_df(load_history())
    if hist is None or hist.empty:
        st.info("No calls saved yet. Run a market scan or Strategy Lab, then open Past Predictions.")
        return

    hist = hist.copy()
    if "Call Source" not in hist.columns:
        hist["Call Source"] = "SCAN"
    hist["Call Source"] = hist["Call Source"].astype(str).str.upper().str.strip().replace(
        {"NAN": "SCAN", "NONE": "SCAN", "": "SCAN"}
    )
    if "Strategy" not in hist.columns:
        hist["Strategy"] = ""
    hist["Strategy"] = hist["Strategy"].astype(str).replace({"nan": "", "None": ""})
    hist["R:R"] = hist.apply(_row_reward_risk, axis=1)
    hist["_pnl"] = hist.apply(_pnl_from_row, axis=1)

    # ---- Simple filters ----
    st.markdown("##### Filters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        src_opts = ["ALL"] + sorted(hist["Call Source"].dropna().unique().tolist())
        f_src = st.selectbox("Source", src_opts, key="perf_src")
    with c2:
        f_call = st.selectbox("Call", ["ALL", "BUY", "SELL"], key="perf_call")
    with c3:
        f_res = st.selectbox(
            "Result",
            ["ALL", "TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED", "OPEN"],
            key="perf_res",
        )
    with c4:
        f_rr = st.selectbox("R:R", ["ALL", "≥ 1.5", "< 1.5"], key="perf_rr")

    view = hist
    if f_src != "ALL":
        view = view[view["Call Source"] == f_src]
    if f_call != "ALL" and "Call" in view.columns:
        view = view[view["Call"].astype(str).str.upper().str.contains(f_call, na=False)]
    ru = view["Result"].astype(str).str.upper() if "Result" in view.columns else pd.Series([""] * len(view))
    if f_res == "TARGET ACHIEVED":
        view = view[ru.str.contains("TARGET", na=False)]
    elif f_res == "STOP LOSS HIT":
        view = view[ru.str.contains("STOP", na=False)]
    elif f_res == "HOLDING PERIOD COMPLETED":
        view = view[ru.str.contains("HOLDING", na=False)]
    elif f_res == "OPEN":
        view = view[~ru.str.contains("TARGET|STOP|HOLDING|WIN|LOSS", na=False, regex=True)]
    if f_rr == "≥ 1.5":
        view = view[pd.to_numeric(view["R:R"], errors="coerce").fillna(0) >= 1.5]
    elif f_rr == "< 1.5":
        view = view[(pd.to_numeric(view["R:R"], errors="coerce").fillna(0) > 0) &
                    (pd.to_numeric(view["R:R"], errors="coerce").fillna(0) < 1.5)]

    s = _success_from_df(view)
    # Holding period split (recompute mask on filtered view)
    if not view.empty and "Result" in view.columns:
        hold_df = view[view["Result"].astype(str).str.upper().str.contains("HOLDING", na=False)]
    else:
        hold_df = pd.DataFrame()
    hold_profit = hold_loss = 0
    if not hold_df.empty:
        for _, r in hold_df.iterrows():
            if safe_float(r.get("_pnl")) >= 0:
                hold_profit += 1
            else:
                hold_loss += 1

    st.markdown("##### Snapshot (filtered)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Calls", len(view))
    k2.metric("🎯 Targets", s.get("wins", 0))
    k3.metric("🔴 Stops", s.get("losses", 0))
    k4.metric("Success %", f"{s.get('success', 0):.1f}%")
    k5.metric("⏰ Hold done", f"{hold_profit} profit / {hold_loss} loss")

    # ---- Stop-loss brief ----
    with st.expander("🛡️ Stop-loss — how to read it", expanded=True):
        st.markdown(
            """
**What stop-loss means here**

- **Locked at call time** — once a prediction is saved, Target and Stop do **not** change.
- **BUY stop** sits **below** entry (risk if price falls).
- **SELL stop** sits **above** entry (risk if price rises).
- **R:R** = reward to target ÷ risk to stop. Prefer **≥ 1.5** when possible; weaker R:R is still stored for learning.
- **Stop hit** = price touched the locked stop → counted as a loss in success rate.
- **Holding period over** = neither target nor stop hit within hold days → exit at close; we mark **toward profit** or **toward loss** vs entry.

**Practical tip:** Many small stops with rare large targets still lose money. Prefer strategies with **higher success %** and **avg win ≥ 1.5 × |avg loss|**.
            """
        )

    # ---- By source ----
    st.markdown("##### Performance by source")
    rows = []
    for src, g in hist.groupby(hist["Call Source"]):
        ss = _success_from_df(g)
        avg_rr = pd.to_numeric(g.get("R:R"), errors="coerce").replace(0, pd.NA).mean()
        rows.append({
            "Source": src,
            "Calls": len(g),
            "Targets": ss["wins"],
            "Stops": ss["losses"],
            "Success %": ss["success"],
            "Avg R:R": round(float(avg_rr), 2) if avg_rr == avg_rr else "—",
        })
    src_df = pd.DataFrame(rows).sort_values("Success %", ascending=False)
    if not src_df.empty:
        st.dataframe(src_df, use_container_width=True, hide_index=True)
        best = src_df.iloc[0]
        st.success(
            f"**Best source right now:** `{best['Source']}` — "
            f"**{best['Success %']}%** success ({int(best['Targets'])} targets / {int(best['Stops'])} stops)."
        )

    # ---- By strategy name ----
    st.markdown("##### Performance by strategy name")
    strat_rows = []
    for strat, g in hist.groupby(hist["Strategy"].replace("", "(none / scan)")):
        if len(g) < 1:
            continue
        ss = _success_from_df(g)
        if ss["decided"] < 1 and len(g) < 3:
            continue
        strat_rows.append({
            "Strategy": strat,
            "Calls": len(g),
            "Targets": ss["wins"],
            "Stops": ss["losses"],
            "Success %": ss["success"],
            "Open": ss.get("open", 0),
        })
    strat_df = pd.DataFrame(strat_rows)
    if not strat_df.empty:
        strat_df = strat_df.sort_values(["Success %", "Calls"], ascending=[False, False])
        st.dataframe(strat_df.head(25), use_container_width=True, hide_index=True)
        top = strat_df.iloc[0]
        st.info(
            f"**Top named strategy:** `{top['Strategy']}` · "
            f"{top['Success %']}% · {int(top['Targets'])}T / {int(top['Stops'])}S · {int(top['Calls'])} calls"
        )
    else:
        st.caption("No named strategies with outcomes yet — generate Strategy Lab signals and update Past Predictions.")

    # ---- Combos: source + strategy ----
    st.markdown("##### Combinations (source + strategy)")
    st.caption("When the same strategy appears under different sources, which pairing works better.")
    combo_rows = []
    hist["_combo"] = hist["Call Source"].astype(str) + " · " + hist["Strategy"].replace("", "—").astype(str)
    for combo, g in hist.groupby("_combo"):
        ss = _success_from_df(g)
        if ss["decided"] < 2:
            continue
        combo_rows.append({
            "Combo": combo,
            "Calls": len(g),
            "Targets": ss["wins"],
            "Stops": ss["losses"],
            "Success %": ss["success"],
        })
    combo_df = pd.DataFrame(combo_rows)
    if not combo_df.empty:
        combo_df = combo_df.sort_values("Success %", ascending=False)
        st.dataframe(combo_df.head(20), use_container_width=True, hide_index=True)
        cbest = combo_df.iloc[0]
        st.success(
            f"**Best pairing:** **{cbest['Combo']}** → **{cbest['Success %']}%** "
            f"({int(cbest['Targets'])} targets, {int(cbest['Stops'])} stops)."
        )
    else:
        st.caption("Need at least 2 closed outcomes per combo to rank pairings.")

    # ---- Holding period table ----
    st.markdown("##### Holding period over — profit vs loss")
    hold_all = hist[hist["Result"].astype(str).str.upper().str.contains("HOLDING", na=False)].copy()
    if hold_all.empty:
        st.caption("No holding-period exits yet.")
    else:
        hold_all["Toward"] = hold_all["_pnl"].apply(lambda x: "PROFIT" if safe_float(x) >= 0 else "LOSS")
        hold_all["PnL %"] = hold_all["_pnl"].apply(lambda x: round(safe_float(x), 2))
        cols = [c for c in ["Prediction Date", "Stock", "Call", "Call Source", "Entry", "Exit Price", "PnL %", "Toward", "Hold Days"] if c in hold_all.columns]
        st.dataframe(hold_all[cols].head(40), use_container_width=True, hide_index=True)
        n_p = int((hold_all["Toward"] == "PROFIT").sum())
        n_l = int((hold_all["Toward"] == "LOSS").sum())
        st.caption(f"Holding exits: **{n_p}** toward profit · **{n_l}** toward loss.")

    # ---- Knowledge brief ----
    st.markdown("##### How to use this page")
    st.markdown(
        """
1. **Filter** to the source you trade (e.g. STRATEGY / SURE).  
2. Check **Success %** and whether **stops** outnumber **targets**.  
3. Prefer **named strategies** with enough closed trades (≥5 decided).  
4. Use **Best pairing** when the same idea appears under multiple labels.  
5. Read **stop-loss brief** above — levels stay locked; holding period shows direction of the exit.  
6. Open **Past Predictions** for the full list; open **Paper** to simulate only the strong sources.
        """
    )


def show_swing_hub():
    st.markdown(
        """
        <div style="border-radius:14px;padding:16px 18px;margin-bottom:12px;
                    background:linear-gradient(135deg,#0f172a,#134e4a);border:1px solid #115e59;">
          <div style="font-size:1.4rem;font-weight:800;color:#f0fdfa;">Swing Trading Setup</div>
          <div style="color:#99f6e4;">Pick one card — simple path from idea → track → review</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**1 · See the market**")
    _hub_card_grid([
        ("📈 NIFTY 50", "Index backdrop before you pick stocks.", "Nifty Analysis"),
        ("🏦 BANK NIFTY", "Banking index chart and analysis.", "BankNifty Analysis"),
    ], key_prefix="swing_hub_m")
    st.markdown("**2 · Take calls**")
    _hub_card_grid([
        ("🟢 BUY Calls", "Long ideas with target & stop.", "BUY Calls"),
        ("🔴 SELL Calls", "Short / exit ideas.", "SELL Calls"),
        ("✅ Sure Call Desk", "Highest conviction only.", "Sure Calls"),
        ("🧪 Strategy Lab", "Named strategies & live list.", "Strategy Lab"),
    ], key_prefix="swing_hub_c")
    st.markdown("**3 · Track & learn**")
    _hub_card_grid([
        ("📊 Performance Summary", "Which strategy wins · stops · holding profit/loss.", "Performance Summary"),
        ("🕐 Past Predictions", "Every call · filters · success %.", "History"),
        ("🧪 Paper Trades", "Practice book with P&L.", "Paper Trading"),
        ("🤖 Trade Tracker", "Open calls with live prices.", "Trade Tracker"),
    ], key_prefix="swing_hub_t")
    st.markdown("**4 · Your book**")
    _hub_card_grid([
        ("📥 Holding Advisor", "Hold / sell / add on what you own.", "Holding Advisor"),
        ("💼 Portfolio", "Portfolio view.", "Portfolio"),
    ], key_prefix="swing_hub_p")



# ============================================================
# NSE TOOLS WORKSPACE (RG Tools–style grid + real data)
# ============================================================

def _yf_ticker(sym: str):
    return yf.Ticker(clean_symbol(sym))


def _option_expiries(sym: str) -> list:
    try:
        t = _yf_ticker(sym)
        exps = list(t.options or [])
        return exps
    except Exception:
        return []


def _option_chain(sym: str, expiry: str):
    """Returns (calls_df, puts_df) from Yahoo for NSE symbol."""
    try:
        t = _yf_ticker(sym)
        ch = t.option_chain(expiry)
        return ch.calls, ch.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _spot_price(sym: str) -> float:
    try:
        q = live_quote(sym)
        if q and q.get("price"):
            return safe_float(q["price"])
    except Exception:
        pass
    try:
        d = stock_history(clean_symbol(sym), interval="1d")
        if d is not None and not d.empty:
            return safe_float(d["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def black_scholes_price(S, K, T, r, sigma, option_type="CE"):
    """European BS price. T in years, r risk-free, sigma IV."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    try:
        from math import log, sqrt, exp
        try:
            from scipy.stats import norm
            N = norm.cdf
        except Exception:
            # fallback approx cdf
            def N(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if str(option_type).upper() in ("CE", "CALL", "C"):
            return S * N(d1) - K * exp(-r * T) * N(d2)
        return K * exp(-r * T) * N(-d2) - S * N(-d1)
    except Exception:
        return 0.0



# ---------- In-app + browser notifications (Target / Stop) ----------
NOTIFY_FILE = APP_DIR / "user_notifications.json"


def load_notify_prefs() -> dict:
    try:
        if NOTIFY_FILE.exists():
            return json.loads(NOTIFY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "enabled": False,
        "target": True,
        "stoploss": True,
        "browser": True,
        "src_strategy": True,
        "src_sure": True,
        "src_intraday": True,
        "src_scan": False,
        "src_paper": True,
        "seen_keys": [],
    }


def save_notify_prefs(prefs: dict):
    try:
        # keep seen_keys short
        keys = list(prefs.get("seen_keys") or [])[-500:]
        prefs["seen_keys"] = keys
        NOTIFY_FILE.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception:
        pass


def _notify_key(stock, result, date_s="") -> str:
    return f"{str(stock).upper()}|{str(result).upper()}|{date_s}"


def push_trade_notification(title: str, body: str, kind: str = "info"):
    """Queue toast + optional browser Notification API."""
    q = list(st.session_state.get("_notify_queue") or [])
    q.append({"title": title, "body": body, "kind": kind})
    st.session_state["_notify_queue"] = q[-20:]


def flush_notifications():
    """Show queued toasts and browser notifications once per run."""
    q = list(st.session_state.get("_notify_queue") or [])
    if not q:
        return
    prefs = load_notify_prefs()
    browser_on = prefs.get("browser", True) and prefs.get("enabled", False)
    for item in q:
        try:
            st.toast(f"{item['title']} — {item['body']}")
        except Exception:
            st.info(f"**{item['title']}** · {item['body']}")
    st.session_state["_notify_queue"] = []
    if browser_on and q:
        # Browser notification (user must allow permission once)
        payload = json.dumps([{"title": x["title"], "body": x["body"]} for x in q])
        components.html(
            f"""
            <script>
            (function() {{
              var items = {payload};
              function showAll() {{
                items.forEach(function(it) {{
                  try {{ new Notification(it.title, {{ body: it.body, icon: "📈" }}); }}
                  catch (e) {{}}
                }});
              }}
              if (!("Notification" in window)) return;
              if (Notification.permission === "granted") showAll();
              else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(p) {{
                  if (p === "granted") showAll();
                }});
              }}
            }})();
            </script>
            """,
            height=0,
            width=0,
        )


def scan_and_notify_outcomes(history: pd.DataFrame = None):
    """
    When Target Achieved or Stop Loss Hit appears as new vs seen_keys → notify.
    Call after evaluate_history / price refresh.
    """
    prefs = load_notify_prefs()
    if not prefs.get("enabled"):
        return
    if history is None:
        try:
            history = normalize_history_df(load_history())
        except Exception:
            return
    if history is None or history.empty or "Result" not in history.columns:
        return
    seen = set(prefs.get("seen_keys") or [])
    new_seen = list(seen)
    for _, row in history.iterrows():
        res = str(row.get("Result", "")).upper()
        is_tgt = "TARGET ACHIEVED" in res or res == "WIN"
        is_sl = "STOP LOSS" in res or res == "LOSS"
        if not is_tgt and not is_sl:
            continue
        if is_tgt and not prefs.get("target", True):
            continue
        if is_sl and not prefs.get("stoploss", True):
            continue
        # Source filter: Strategy / Sure / Scan / Intraday
        src = str(row.get("Call Source", "") or row.get("Strategy", "") or "").upper()
        allowed = False
        if prefs.get("src_strategy", True) and ("STRATEGY" in src or "MY_STRATEGY" in src):
            allowed = True
        if prefs.get("src_sure", True) and ("SURE" in src or "HIGH_CONV" in src):
            allowed = True
        if prefs.get("src_intraday", True) and ("INTRADAY" in src or "F&O" in src or "FO" in src):
            allowed = True
        if prefs.get("src_scan", False) and ("SCAN" in src or src == "" or src == "NAN"):
            allowed = True
        if not allowed and not any([
            prefs.get("src_strategy"), prefs.get("src_sure"),
            prefs.get("src_intraday"), prefs.get("src_scan"),
        ]):
            allowed = True  # if user disabled all sources, fall back to all
        if not allowed:
            # still allow if strategy name column present and strategy notify on
            if prefs.get("src_strategy", True) and str(row.get("Strategy", "")).strip():
                allowed = True
        if not allowed:
            continue
        stock = str(row.get("Stock", "")).upper()
        dt = str(row.get("Exit Date") or row.get("Prediction Date") or "")[:12]
        key = _notify_key(stock, res, dt)
        if key in seen:
            continue
        new_seen.append(key)
        if is_tgt:
            push_trade_notification(
                f"🎯 Target hit · {stock}",
                f"Result: {row.get('Result')} · Entry {row.get('Entry')} → Target {row.get('Target')}",
                "success",
            )
        else:
            push_trade_notification(
                f"🔴 Stop hit · {stock}",
                f"Result: {row.get('Result')} · Entry {row.get('Entry')} → SL {row.get('Stop Loss')}",
                "error",
            )
    prefs["seen_keys"] = new_seen[-500:]
    save_notify_prefs(prefs)
    # also paper book
    try:
        paper = load_paper_portfolio()
        if paper is not None and not paper.empty and "Result" in paper.columns:
            for _, row in paper.iterrows():
                res = str(row.get("Result", "")).upper()
                is_tgt = "TARGET" in res
                is_sl = "STOP" in res
                if not is_tgt and not is_sl:
                    continue
                if is_tgt and not prefs.get("target", True):
                    continue
                if is_sl and not prefs.get("stoploss", True):
                    continue
                if not prefs.get("src_paper", True):
                    continue
                stock = str(row.get("Stock", "")).upper()
                key = _notify_key(stock, "PAPER|" + res, str(row.get("Open Date", ""))[:12])
                if key in seen or key in prefs["seen_keys"]:
                    continue
                prefs["seen_keys"].append(key)
                push_trade_notification(
                    f"{'🎯' if is_tgt else '🔴'} Paper · {stock}",
                    f"{row.get('Result')} · PnL {row.get('PnL ₹', '—')}",
                    "success" if is_tgt else "error",
                )
            save_notify_prefs(prefs)
    except Exception:
        pass


def render_notification_controls(location_key: str = "main"):
    """Compact notification opt-in — place on pages / sidebar."""
    prefs = load_notify_prefs()
    with st.expander("🔔 Notifications (Target / Stop)", expanded=False):
        en = st.checkbox(
            "Enable notifications",
            value=bool(prefs.get("enabled")),
            key=f"notif_en_{location_key}",
            help="When a prediction or paper trade hits Target or Stop Loss, show toast + browser alert.",
        )
        t1, t2, t3 = st.columns(3)
        with t1:
            tg = st.checkbox("🎯 Target hit", value=bool(prefs.get("target", True)), key=f"notif_tg_{location_key}")
        with t2:
            sl = st.checkbox("🔴 Stop hit", value=bool(prefs.get("stoploss", True)), key=f"notif_sl_{location_key}")
        with t3:
            br = st.checkbox("Browser popup", value=bool(prefs.get("browser", True)), key=f"notif_br_{location_key}")
        st.caption("**Notify only from these sources** (untick to silence)")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            src_st = st.checkbox("Strategy", value=bool(prefs.get("src_strategy", True)), key=f"notif_src_st_{location_key}")
        with s2:
            src_su = st.checkbox("Sure Call", value=bool(prefs.get("src_sure", True)), key=f"notif_src_su_{location_key}")
        with s3:
            src_fo = st.checkbox("Intraday/F&O", value=bool(prefs.get("src_intraday", True)), key=f"notif_src_fo_{location_key}")
        with s4:
            src_sc = st.checkbox("Scan", value=bool(prefs.get("src_scan", False)), key=f"notif_src_sc_{location_key}")
        with s5:
            src_pa = st.checkbox("Paper", value=bool(prefs.get("src_paper", True)), key=f"notif_src_pa_{location_key}")
        prefs["enabled"] = en
        prefs["target"] = tg
        prefs["stoploss"] = sl
        prefs["browser"] = br
        prefs["src_strategy"] = src_st
        prefs["src_sure"] = src_su
        prefs["src_intraday"] = src_fo
        prefs["src_scan"] = src_sc
        prefs["src_paper"] = src_pa
        save_notify_prefs(prefs)
        if st.button("Test notification", key=f"notif_test_{location_key}"):
            push_trade_notification("JP Stock Market Model", "Notifications are working.", "info")
            flush_notifications()
        st.caption("Browser alerts need one-time **Allow** on the site. Works on phone if the tab is open.")


def show_tool_momentum_divergence():
    """Momentum divergence: sharp impulse vs weak recovery."""
    st.markdown("### Momentum Divergence Scanner")
    st.caption("Finds stocks where a sharp move was followed by a slower, smaller recovery (impulse vs limp recovery).")
    side = st.selectbox("Side", ["Both", "Bullish recovery after dump", "Bearish recovery after spike"], key="md_side")
    tf = st.selectbox("Timeframe", ["1 Day — daily bars"], key="md_tf")
    contrast = st.select_slider("Minimum contrast", options=[50, 55, 60, 65, 70, 75], value=65, key="md_c")
    within = st.selectbox("Still live within", [3, 5, 8, 12], index=1, key="md_w")
    max_n = st.slider("Max stocks", 20, 120, 50, key="md_n")
    if not st.button("Scan the market", type="primary", key="md_run"):
        with st.expander("How to use"):
            st.markdown(
                """
1. **Side** — both directions or only dump→weak bounce / spike→weak fade.  
2. **Contrast** — higher = only clear limp recoveries.  
3. **Still live within** — impulse must be recent.  
*Research aid — not a buy/sell tip.*
                """
            )
        return
    universe = list(NSE_STOCKS)[:max_n] if NSE_STOCKS else ["RELIANCE", "TCS", "INFY", "SBIN"]
    rows = []
    with st.spinner("Scanning…"):
        for sym in universe:
            try:
                df = stock_history(clean_symbol(sym), interval="1d")
                if df is None or len(df) < 40:
                    continue
                c = df["Close"].astype(float)
                # last impulse: max drop/rise in 5–8 bars, then recovery size
                window = c.tail(20)
                rets = window.pct_change()
                for look in range(5, 12):
                    if len(window) < look + within:
                        continue
                    seg = window.iloc[-(look + within): -within]
                    if len(seg) < 4:
                        continue
                    impulse = float(seg.iloc[-1] / seg.iloc[0] - 1)
                    rec = window.iloc[-within:]
                    recovery = float(rec.iloc[-1] / rec.iloc[0] - 1)
                    # dump then weak bounce
                    if impulse < -0.04 and 0 <= recovery < abs(impulse) * (1 - contrast / 100.0):
                        if side.startswith("Bearish"):
                            continue
                        score = min(99, int(abs(impulse) / (recovery + 0.001) * 20))
                        if score >= contrast:
                            rows.append({"Stock": display_symbol(sym), "Type": "Dump → weak bounce",
                                         "Impulse %": round(impulse * 100, 2), "Recovery %": round(recovery * 100, 2),
                                         "Score": score})
                            break
                    # spike then weak fade
                    if impulse > 0.04 and -abs(impulse) * (1 - contrast / 100.0) < recovery <= 0:
                        if side.startswith("Bullish"):
                            continue
                        score = min(99, int(abs(impulse) / (abs(recovery) + 0.001) * 20))
                        if score >= contrast:
                            rows.append({"Stock": display_symbol(sym), "Type": "Spike → weak fade",
                                         "Impulse %": round(impulse * 100, 2), "Recovery %": round(recovery * 100, 2),
                                         "Score": score})
                            break
            except Exception:
                continue
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No clear divergence cases in this sample — lower contrast or raise max stocks.")



def gann_square_of_9(price: float) -> dict:
    """
    Classic Square of 9 levels around a price (support / resistance / turns).
    Real price in → levels out.
    """
    if price <= 0:
        return {}
    root = math.sqrt(price)
    levels = {}
    for deg, name in [
        (45, "45°"), (90, "90°"), (135, "135°"), (180, "180°"),
        (225, "225°"), (270, "270°"), (315, "315°"), (360, "360°"),
    ]:
        step = deg / 360.0
        levels[f"Above {name}"] = round((root + step) ** 2, 2)
        levels[f"Below {name}"] = round(max(0.01, (root - step) ** 2), 2)
    levels["Root"] = round(root, 4)
    levels["Pivot"] = round(price, 2)
    # Cardinally used bands
    levels["Resistance 1"] = levels["Above 45°"]
    levels["Resistance 2"] = levels["Above 90°"]
    levels["Support 1"] = levels["Below 45°"]
    levels["Support 2"] = levels["Below 90°"]
    return levels





# ============================================================
# Index constituents + Watchlist + Scanner Query
# ============================================================

_FALLBACK_NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "BHARTIARTL",
    "LT", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "ASIANPAINT", "HCLTECH", "MARUTI",
    "SUNPHARMA", "TITAN", "NTPC", "TATAMOTORS", "POWERGRID", "ULTRACEMCO", "M&M",
    "NESTLEIND", "WIPRO", "ONGC", "ADANIENT", "JSWSTEEL", "TATASTEEL", "COALINDIA",
    "BAJAJFINSV", "ADANIPORTS", "TECHM", "HINDALCO", "GRASIM", "INDUSINDBK", "CIPLA",
    "DRREDDY", "BPCL", "EICHERMOT", "APOLLOHOSP", "HEROMOTOCO", "DIVISLAB", "BRITANNIA",
    "TATACONSUM", "BAJAJ-AUTO", "HINDUNILVR", "SBILIFE", "HDFCLIFE", "BEL", "TRENT",
]

_FALLBACK_BANKNIFTY = [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE",
    "BAJAJFINSV", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB", "PNB", "BANKBARODA",
]

_FALLBACK_NIFTY200_EXTRA = [
    "DMART", "PIDILITIND", "GODREJCP", "DABUR", "HAVELLS", "SIEMENS", "ABB",
    "AMBUJACEM", "SHREECEM", "DLF", "LODHA", "GODREJPROP", "IRCTC", "ZOMATO",
    "PAYTM", "POLICYBZR", "NYKAA", "PERSISTENT", "COFORGE", "LTIM", "MPHASIS",
    "INDIGO", "TATAPOWER", "ADANIGREEN", "ADANIPOWER", "VEDL", "HINDPETRO",
    "IOC", "GAIL", "RECLTD", "PFC", "IRFC", "TVSMOTOR", "ASHOKLEY", "BOSCHLTD",
]


def _fetch_nse_index_symbols(url: str) -> list:
    try:
        import io
        import requests as _req
        r = _req.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        df = pd.read_csv(io.StringIO(r.text))
        col = None
        for c in df.columns:
            if str(c).strip().lower() in ("symbol", "symbols"):
                col = c
                break
        if col is None:
            col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
        out = (
            df[col].astype(str).str.upper().str.replace(" ", "", regex=False)
            .str.replace("&", "&", regex=False).str.strip().tolist()
        )
        return [x for x in out if x and x not in ("NAN", "SYMBOL")]
    except Exception:
        return []


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def load_index_constituents() -> dict:
    """Nifty 50, Nifty 200, Bank Nifty symbol lists (NSE CSV + fallback)."""
    cache = {}
    try:
        if INDEX_LIST_CACHE.exists():
            cache = json.loads(INDEX_LIST_CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    nifty50 = _fetch_nse_index_symbols(
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    ) or _fetch_nse_index_symbols(
        "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
    ) or list(cache.get("nifty50") or []) or list(_FALLBACK_NIFTY50)

    bank = _fetch_nse_index_symbols(
        "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv"
    ) or _fetch_nse_index_symbols(
        "https://nsearchives.nseindia.com/content/indices/ind_niftybanklist.csv"
    ) or list(cache.get("banknifty") or []) or list(_FALLBACK_BANKNIFTY)

    nifty200 = _fetch_nse_index_symbols(
        "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
    ) or _fetch_nse_index_symbols(
        "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
    ) or list(cache.get("nifty200") or [])
    if not nifty200:
        # Nifty 200 ≈ Nifty 50 + liquid midcaps fallback
        nifty200 = sorted(set(list(nifty50) + list(_FALLBACK_NIFTY200_EXTRA) + list(bank)))

    out = {
        "nifty50": sorted(set(nifty50)),
        "nifty200": sorted(set(nifty200)),
        "banknifty": sorted(set(bank)),
    }
    try:
        INDEX_LIST_CACHE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        pass
    return out


def load_watchlist() -> list:
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def save_watchlist(items: list):
    try:
        WATCHLIST_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_watchlist_notes() -> dict:
    try:
        if WATCHLIST_NOTES_FILE.exists():
            return json.loads(WATCHLIST_NOTES_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_watchlist_notes(notes: dict):
    try:
        WATCHLIST_NOTES_FILE.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    except Exception:
        pass


def render_index_lists_home():
    """Home: Nifty + Bank Nifty prices + searchable lists."""
    st.markdown("### 📊 Indices & stock lists")
    c1, c2, c3 = st.columns(3)
    n_px = b_px = s_px = None
    try:
        n_px = _spot_price("^NSEI") or _spot_price("NIFTY")
    except Exception:
        pass
    try:
        b_px = _spot_price("^NSEBANK") or _spot_price("BANKNIFTY")
    except Exception:
        pass
    try:
        s_px = _spot_price("^BSESN")
    except Exception:
        pass
    with c1:
        st.metric("NIFTY 50", f"₹{n_px:,.2f}" if n_px else "—")
    with c2:
        st.metric("BANK NIFTY", f"₹{b_px:,.2f}" if b_px else "—")
    with c3:
        st.metric("SENSEX", f"₹{s_px:,.2f}" if s_px else "—")

    lists = load_index_constituents()
    t50, t200, tbn = st.tabs(["Nifty 50", "Nifty 200", "Bank Nifty stocks"])

    def _list_tab(label, symbols, key_prefix):
        st.caption(f"**{len(symbols)}** names · type in the box to search")
        pick = st.selectbox(
            f"Search / select · {label}",
            options=symbols,
            key=f"{key_prefix}_pick",
            help="Click and type to filter (e.g. rel → RELIANCE)",
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("📈 Analyse", key=f"{key_prefix}_an", use_container_width=True):
                st.session_state.selected_stock = pick
                st.session_state.page = "Stock Analysis"
                st.rerun()
        with col_b:
            if st.button("⭐ Add to watchlist", key=f"{key_prefix}_wl", use_container_width=True):
                wl = load_watchlist()
                if pick not in wl:
                    wl.append(pick)
                    save_watchlist(wl)
                    st.success(f"Added {pick}")
                else:
                    st.info("Already on watchlist")
        with col_c:
            if st.button("⭐ Watchlist only", key=f"{key_prefix}_sc", use_container_width=True):
                wl = load_watchlist()
                if pick not in wl:
                    wl.append(pick)
                    save_watchlist(wl)
                    st.success(f"Added {pick}")
                else:
                    st.info("Already on watchlist")
        with st.expander(f"Full {label} list", expanded=False):
            st.dataframe(pd.DataFrame({"Stock": symbols}), use_container_width=True, hide_index=True)

    with t50:
        _list_tab("Nifty 50", lists.get("nifty50") or _FALLBACK_NIFTY50, "n50")
    with t200:
        _list_tab("Nifty 200", lists.get("nifty200") or sorted(set(_FALLBACK_NIFTY50 + _FALLBACK_NIFTY200_EXTRA)), "n200")
    with tbn:
        _list_tab("Bank Nifty", lists.get("banknifty") or _FALLBACK_BANKNIFTY, "bnk")


def show_watchlist_page():
    """Dynamic watchlist: stocks, manual pattern/strategy notes, chart marks."""
    st.title("⭐ Dynamic Watchlist")
    st.caption("Add stocks, tag patterns & strategies, mark levels — saved on server for all devices.")

    try:
        universe = sorted({
            str(s).replace(".NS", "").upper().strip()
            for s in (NSE_STOCKS or [])
            if str(s).strip()
        })
    except Exception:
        universe = list(_FALLBACK_NIFTY50)
    if not universe:
        universe = list(_FALLBACK_NIFTY50)

    wl = load_watchlist()
    notes = load_watchlist_notes()

    a1, a2 = st.columns([2, 1])
    with a1:
        add_sym = st.selectbox("Add stock (type to search)", options=universe, key="wl_add_sym")
    with a2:
        st.write("")
        st.write("")
        if st.button("➕ Add", type="primary", key="wl_add_btn", use_container_width=True):
            if add_sym not in wl:
                wl.append(add_sym)
                save_watchlist(wl)
                st.success(f"Added {add_sym}")
                st.rerun()
            else:
                st.info("Already listed")

    if not wl:
        st.info("Watchlist empty — add a stock above.")
        return

    st.markdown(f"### Your list ({len(wl)})")
    for sym in list(wl):
        meta = notes.get(sym) or {}
        with st.expander(f"**{sym}** · {meta.get('strategy','—')} · {meta.get('pattern','—')}", expanded=False):
            px = None
            try:
                px = _spot_price(sym)
            except Exception:
                pass
            st.write(f"Live ≈ **₹{px:,.2f}**" if px else "Live price —")
            strategy = st.text_input(
                "Strategy / setup name",
                value=str(meta.get("strategy") or ""),
                key=f"wl_st_{sym}",
            )
            pattern = st.text_input(
                "Manual chart pattern",
                value=str(meta.get("pattern") or ""),
                key=f"wl_pt_{sym}",
            )
            marks = st.text_area(
                "Chart marks / levels (support, resistance, notes)",
                value=str(meta.get("marks") or ""),
                key=f"wl_mk_{sym}",
                height=80,
            )
            bias = st.selectbox(
                "Bias",
                ["WATCH", "BUY bias", "SELL bias", "HOLD"],
                index=["WATCH", "BUY bias", "SELL bias", "HOLD"].index(meta.get("bias", "WATCH"))
                if meta.get("bias") in ("WATCH", "BUY bias", "SELL bias", "HOLD") else 0,
                key=f"wl_bias_{sym}",
            )
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("💾 Save notes", key=f"wl_save_{sym}", use_container_width=True):
                    notes[sym] = {
                        "strategy": strategy,
                        "pattern": pattern,
                        "marks": marks,
                        "bias": bias,
                        "updated": india_now().strftime("%Y-%m-%d %H:%M") if "india_now" in dir() else "",
                    }
                    save_watchlist_notes(notes)
                    st.success("Saved")
            with b2:
                if st.button("📈 Chart / analyse", key=f"wl_an_{sym}", use_container_width=True):
                    st.session_state.selected_stock = sym
                    st.session_state.page = "Stock Analysis"
                    st.rerun()
            with b3:
                if st.button("🟢 Paper BUY", key=f"wl_buy_{sym}", use_container_width=True):
                    try:
                        render_active_trade_buttons(sym, side_hint="BUY", entry=float(px or 0), key_prefix=f"wltr_{sym}")
                    except Exception:
                        execute_paper_order(sym, "BUY", entry=float(px or 0), force=True, source="WATCHLIST")
                        st.success("Paper order sent")
            with b4:
                if st.button("🗑 Remove", key=f"wl_rm_{sym}", use_container_width=True):
                    wl = [x for x in wl if x != sym]
                    save_watchlist(wl)
                    notes.pop(sym, None)
                    save_watchlist_notes(notes)
                    st.rerun()
            # mini chart
            try:
                df = stock_history(clean_symbol(sym), interval="1d")
                if df is not None and not df.empty:
                    tail = df.tail(60)
                    fig = go.Figure(go.Candlestick(
                        x=list(range(len(tail))),
                        open=tail["Open"], high=tail["High"], low=tail["Low"], close=tail["Close"],
                    ))
                    fig.update_layout(
                        height=220, margin=dict(l=4, r=4, t=16, b=4),
                        xaxis_rangeslider_visible=False,
                        title=f"{sym} · marks: {(marks or '—')[:40]}",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.9)",
                        font=dict(color="#e2e8f0", size=10),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass


def load_saved_queries() -> dict:
    try:
        if SAVED_QUERIES_FILE.exists():
            data = json.loads(SAVED_QUERIES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_saved_queries(data: dict):
    try:
        SAVED_QUERIES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# --- Screener.in-style field map (normalized name -> internal key) ---
_SCREENER_FIELD_ALIASES = {
    "market capitalization": "market_cap_cr",
    "market capitalisation": "market_cap_cr",
    "market cap": "market_cap_cr",
    "mar cap": "market_cap_cr",
    "mcap": "market_cap_cr",
    "current price": "price",
    "cmp": "price",
    "price": "price",
    "price to earning": "pe",
    "price to earnings": "pe",
    "p/e": "pe",
    "pe": "pe",
    "stock p/e": "pe",
    "price to book value": "pb",
    "price to book": "pb",
    "p/b": "pb",
    "pb": "pb",
    "return on capital employed": "roce",
    "roce": "roce",
    "return on equity": "roe",
    "roe": "roe",
    "debt to equity": "debt_to_equity",
    "debt/equity": "debt_to_equity",
    "d/e": "debt_to_equity",
    "dividend yield": "div_yield",
    "div yld": "div_yield",
    "div yield": "div_yield",
    "book value": "book_value",
    "promoter holding": "promoter",
    "pledged percentage": "pledge",
    "prediction": "prediction",
    "pred": "prediction",
    "risk": "risk_pct",
    "risk %": "risk_pct",
    "r:r": "rr",
    "rr": "rr",
    "call": "call",
}


def _normalize_screener_field(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "").strip().lower())
    n = n.replace("%", "").strip()
    return _SCREENER_FIELD_ALIASES.get(n, n.replace(" ", "_"))


def parse_screener_query(query: str) -> list:
    """
    Parse Screener.in-style query into list of (field, op, value).
    Supports AND / OR between conditions (OR groups are simplified: all OR→union later).
    Example:
      Market capitalization > 500 AND
      Price to earning < 15 AND
      Return on capital employed > 22
    """
    if not query or not str(query).strip():
        return []
    q = str(query)
    # strip comments
    lines = []
    for ln in q.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # drop trailing AND/OR on line (Screener style)
        ln = re.sub(r"\s+(AND|OR)\s*$", "", ln, flags=re.I).strip()
        if ln:
            lines.append(ln)
    blob = " AND ".join(lines)
    # split by AND (OR kept as separate for future; treat as AND for simplicity first)
    parts = re.split(r"\s+AND\s+", blob, flags=re.I)
    conditions = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # also split OR into separate conditions tagged
        or_bits = re.split(r"\s+OR\s+", part, flags=re.I)
        for bit in or_bits:
            bit = bit.strip()
            m = re.match(
                r"^(.+?)\s*(>=|<=|!=|<>|>|<|=)\s*([-+]?[0-9]*\.?[0-9]+%?)\s*$",
                bit,
                flags=re.I,
            )
            if not m:
                # CALL = BUY style text
                m2 = re.match(r"^(.+?)\s*(=)\s*([A-Za-z/]+)\s*$", bit, flags=re.I)
                if not m2:
                    continue
                field, op, val = m2.group(1), m2.group(2), m2.group(3)
                conditions.append((_normalize_screener_field(field), op, val.upper()))
                continue
            field, op, val = m.group(1), m.group(2), m.group(3)
            val = str(val).replace("%", "").replace(",", "")
            try:
                val = float(val)
            except Exception:
                pass
            conditions.append((_normalize_screener_field(field), op, val))
    return conditions


def _cmp_op(left, op: str, right) -> bool:
    try:
        if left is None or (isinstance(left, float) and np.isnan(left)):
            return False
        if op == ">":
            return float(left) > float(right)
        if op == ">=":
            return float(left) >= float(right)
        if op == "<":
            return float(left) < float(right)
        if op == "<=":
            return float(left) <= float(right)
        if op in ("=", "=="):
            if isinstance(right, str):
                return str(left).upper() == str(right).upper()
            return float(left) == float(right)
        if op in ("!=", "<>"):
            return float(left) != float(right)
    except Exception:
        return False
    return False


def _load_fund_cache() -> dict:
    try:
        if FUND_SCREEN_CACHE.exists():
            return json.loads(FUND_SCREEN_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_fund_cache(cache: dict):
    try:
        # keep last 800 symbols
        keys = list(cache.keys())
        if len(keys) > 800:
            cache = {k: cache[k] for k in keys[-800:]}
        FUND_SCREEN_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _row_fundamentals(symbol: str, cache: dict, scan_row=None) -> dict:
    """Merge scan row + Yahoo fundamentals into Screener-like metrics (₹ Cr)."""
    sym = display_symbol(symbol)
    row = {
        "Stock": sym,
        "price": None,
        "pe": None,
        "pb": None,
        "market_cap_cr": None,
        "roe": None,
        "roce": None,
        "debt_to_equity": None,
        "div_yield": None,
        "book_value": None,
        "promoter": None,
        "pledge": None,
        "prediction": None,
        "risk_pct": None,
        "rr": None,
        "call": None,
        "sector": None,
    }
    if scan_row is not None:
        try:
            row["price"] = safe_float(scan_row.get("Price") or scan_row.get("Entry")) or None
            row["prediction"] = safe_float(scan_row.get("Prediction")) or None
            row["risk_pct"] = safe_float(scan_row.get("Risk %")) or None
            row["rr"] = safe_float(scan_row.get("R:R")) or None
            row["call"] = str(scan_row.get("Call", "") or "").upper() or None
            row["sector"] = str(scan_row.get("Sector", "") or "") or None
        except Exception:
            pass

    cached = cache.get(sym) or cache.get(str(sym).upper())
    if isinstance(cached, dict) and cached.get("_ok"):
        for k, v in cached.items():
            if k != "_ok" and v is not None:
                row[k] = v
        return row

    try:
        fund = fetch_fundamentals(sym)
        if fund and not fund.get("error"):
            mcap = safe_float(fund.get("market_cap"))
            if mcap and mcap > 1e5:  # full INR → Cr
                mcap = mcap / 1e7
            if mcap:
                row["market_cap_cr"] = mcap
            pe = safe_float(fund.get("pe"))
            if pe and pe > 0:
                row["pe"] = pe
            pb = safe_float(fund.get("pb") or fund.get("price_to_book"))
            if pb and pb > 0:
                row["pb"] = pb
            roe = safe_float(fund.get("roe"))
            if roe is not None:
                if abs(roe) <= 1.5:
                    roe = roe * 100
                row["roe"] = roe
            # ROCE often missing on Yahoo — approximate with ROA * 100 or ROE as fallback label
            roce = safe_float(fund.get("roce") or fund.get("roa"))
            if roce is not None:
                if abs(roce) <= 1.5:
                    roce = roce * 100
                row["roce"] = roce
            elif row["roe"] is not None:
                row["roce"] = row["roe"]  # fallback so ROCE filters are not empty
            de = safe_float(fund.get("debt_to_equity") or fund.get("debtToEquity"))
            if de is not None:
                if de > 10:  # Yahoo sometimes gives 50 for 0.5
                    de = de / 100.0
                row["debt_to_equity"] = de
            dy = safe_float(fund.get("dividend_yield"))
            if dy is not None:
                if abs(dy) <= 1:
                    dy = dy * 100
                row["div_yield"] = dy
            bv = safe_float(fund.get("book_value"))
            if bv is not None:
                row["book_value"] = bv
            cp = safe_float(fund.get("current_price") or fund.get("price"))
            if cp:
                row["price"] = cp
            cache[sym] = {k: row[k] for k in (
                "market_cap_cr", "pe", "pb", "roe", "roce", "debt_to_equity",
                "div_yield", "book_value", "price",
            )}
            cache[sym]["_ok"] = True
    except Exception:
        pass

    if not row["price"]:
        try:
            px = _spot_price(sym)
            if px:
                row["price"] = px
        except Exception:
            pass
    if not row["price"]:
        try:
            d = stock_history(clean_symbol(sym), interval="1d", period="5d")
            if d is not None and not d.empty and "Close" in d.columns:
                row["price"] = float(pd.to_numeric(d["Close"], errors="coerce").dropna().iloc[-1])
        except Exception:
            pass
    return row


def run_screener_style_query(
    query: str,
    results: pd.DataFrame = None,
    max_symbols: int = 100,
    skip_missing: bool = True,
) -> tuple:
    """
    Returns (table_df, meta_dict).
    skip_missing=True → if PE/ROCE missing, that condition is skipped for the stock
    (avoids 0 results when Yahoo omits fields). Set False for strict Screener behaviour.
    """
    conditions = parse_screener_query(query)
    symbols = []
    scan_map = {}
    if results is not None and not getattr(results, "empty", True) and "Stock" in results.columns:
        for _, r in results.iterrows():
            s = display_symbol(r["Stock"])
            if s and s not in scan_map:
                symbols.append(s)
                scan_map[s] = r
    # Always seed liquid names so queries work even without a scan
    try:
        lists = load_index_constituents()
        extra = (
            list(lists.get("nifty50") or [])
            + list(lists.get("banknifty") or [])
            + list(lists.get("nifty200") or [])[:60]
        )
    except Exception:
        extra = list(_FALLBACK_NIFTY50) + list(_FALLBACK_BANKNIFTY)
    for s in extra + list(_FALLBACK_NIFTY50) + list(_FALLBACK_BANKNIFTY):
        s = display_symbol(s)
        if s and s not in scan_map:
            symbols.append(s)
            scan_map[s] = None
    # unique preserve order
    seen = set()
    uniq = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    symbols = uniq[: max(30, int(max_symbols))]

    cache = _load_fund_cache()
    rows = []
    n_checked = 0
    n_with_pe = 0
    n_with_mcap = 0
    for sym in symbols:
        n_checked += 1
        row = _row_fundamentals(sym, cache, scan_map.get(sym))
        if row.get("pe"):
            n_with_pe += 1
        if row.get("market_cap_cr"):
            n_with_mcap += 1
        ok = True
        for field, op, val in conditions:
            left = row.get(field)
            if left is None or (isinstance(left, float) and (np.isnan(left) if left == left else True) is False and False):
                pass
            if left is None or (isinstance(left, float) and str(left) == "nan"):
                if skip_missing:
                    continue  # don't fail stock for missing field
                ok = False
                break
            if not _cmp_op(left, op, val):
                ok = False
                break
        if ok:
            rows.append(row)
    _save_fund_cache(cache)

    meta = {
        "checked": n_checked,
        "with_pe": n_with_pe,
        "with_mcap": n_with_mcap,
        "conditions": conditions,
        "matched": len(rows),
    }
    if not rows:
        return pd.DataFrame(), meta

    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "Company": df["Stock"],
        "CMP Rs.": [round(x, 2) if x is not None else None for x in df["price"]],
        "P/E": [round(x, 2) if x is not None else None for x in df["pe"]],
        "Mar Cap Rs.Cr.": [round(x, 2) if x is not None else None for x in df["market_cap_cr"]],
        "ROCE %": [round(x, 2) if x is not None else None for x in df["roce"]],
        "ROE %": [round(x, 2) if x is not None else None for x in df["roe"]],
        "Debt/Eq": [round(x, 2) if x is not None else None for x in df["debt_to_equity"]],
        "Div Yld %": [round(x, 2) if x is not None else None for x in df["div_yield"]],
        "Call": df["call"] if "call" in df.columns else None,
        "Pred %": [round(x, 1) if x is not None else None for x in df["prediction"]],
    })
    # Sort by mar cap desc when available
    try:
        out = out.sort_values("Mar Cap Rs.Cr.", ascending=False, na_position="last")
    except Exception:
        pass
    return out.reset_index(drop=True), meta



def show_scanner_query_page(results: pd.DataFrame = None):
    """
    Screener.in-style interface:
    - Query box with AND conditions
    - Run this Query
    - Results table (Company, CMP, P/E, Mar Cap, ROCE, ROE…)
    - Save screen by name
    """
    st.markdown(
        """
        <div style="margin-bottom:10px;">
          <div style="font-size:1.55rem;font-weight:800;color:#0f172a;">Create a Search Query</div>
          <div style="color:#64748b;font-size:0.92rem;margin-top:4px;">
            Same style as <b>Screener.in</b> — write conditions with <b>AND</b>, then
            <b>Run this Query</b>. Save screens by name to reuse.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📘 Screener-style rules (how to write)", expanded=False):
        st.markdown(
            """
### Syntax (same idea as [Screener.in](https://www.screener.in/))
```
Market capitalization > 500 AND
Price to earning < 15 AND
Return on capital employed > 22
```
- One condition per line (optional trailing `AND`)
- Operators: `>` `<` `>=` `<=` `=`
- Combine with **AND**

### Supported fields
| Screener name | Notes |
|---------------|--------|
| Market capitalization | ₹ **Cr** |
| Current price / CMP | Live / last price |
| Price to earning | P/E |
| Price to book value | P/B |
| Return on capital employed | ROCE % |
| Return on equity | ROE % |
| Debt to equity | D/E |
| Dividend yield | % |
| Book value | |
| Prediction | from **your scan** |
| Risk | from **your scan** |
| Call | `Call = BUY` (scan) |

### Examples
**Value (Screener classic)**
```
Market capitalization > 500 AND
Price to earning < 15 AND
Return on capital employed > 22
```
**Quality + our scan BUY**
```
Call = BUY AND
Prediction > 65 AND
Debt to equity < 1 AND
Return on equity > 15
```
            """
        )

    results = results if results is not None else st.session_state.get("results")
    if results is None:
        results = pd.DataFrame()
    try:
        if results is None or results.empty:
            load_scan_from_disk(force=True)
            results = st.session_state.get("results") or pd.DataFrame()
    except Exception:
        pass

    saved = load_saved_queries()
    names = sorted(saved.keys())

    # Saved screens row
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        pick = st.selectbox("Saved screens", ["— new query —"] + names, key="scr_saved")
    with c2:
        if st.button("📂 Load screen", use_container_width=True, key="scr_load"):
            if pick and pick != "— new query —":
                st.session_state["scr_query_text"] = saved.get(pick, "")
                st.rerun()
    with c3:
        if st.button("🗑 Delete screen", use_container_width=True, key="scr_del"):
            if pick and pick != "— new query —":
                saved.pop(pick, None)
                save_saved_queries(saved)
                st.rerun()

    default_q = st.session_state.get(
        "scr_query_text",
        "Market capitalization > 500 AND\n"
        "Price to earning < 40 AND\n"
        "Return on equity > 10\n",
    )
    st.markdown("##### Search Query")
    st.caption("You can customize the query below:")
    query_text = st.text_area(
        "Query",
        value=default_q,
        height=150,
        key="scr_query_box",
        label_visibility="collapsed",
    )
    st.session_state["scr_query_text"] = query_text

    st.caption(
        "Custom query example:  "
        "`Market capitalization > 500 AND Price to earning < 15 AND Return on capital employed > 22%`"
    )

    b1, b2, b3 = st.columns([1.2, 1.2, 2])
    with b1:
        run = st.button("▶ Run this Query", type="primary", use_container_width=True, key="scr_run")
    with b2:
        max_n = st.number_input("Universe size", 40, 250, 100, 10, key="scr_maxn")
    with b3:
        save_name = st.text_input("Save screen as", placeholder="e.g. value_roce", key="scr_save_name")
        if st.button("💾 Save screen", use_container_width=True, key="scr_save"):
            nm = str(save_name or "").strip()
            if not nm:
                st.error("Enter a screen name")
            else:
                saved[nm] = query_text
                save_saved_queries(saved)
                st.success(f"Saved **{nm}**")

    skip_missing = st.checkbox(
        "Skip missing fields (recommended)",
        value=True,
        key="scr_skip_miss",
        help="If Yahoo has no PE/ROCE for a stock, ignore that condition instead of excluding the stock. "
             "Turn off for strict Screener-like behaviour.",
    )

    if run:
        with st.spinner("Running Screener-style query (Nifty universe + fundamentals)…"):
            try:
                table, meta = run_screener_style_query(
                    query_text, results, max_symbols=int(max_n), skip_missing=skip_missing
                )
                st.session_state["_scr_last_table"] = table
                st.session_state["_scr_last_meta"] = meta
                st.session_state["_scr_last_q"] = query_text
            except Exception as e:
                st.error(f"Query error: {e}")
                table, meta = pd.DataFrame(), {}
    else:
        table = st.session_state.get("_scr_last_table")
        meta = st.session_state.get("_scr_last_meta") or {}

    if table is None:
        st.info("Write a query and click **Run this Query**.")
        st.caption(
            "Tip: start with  "
            "`Market capitalization > 500 AND Price to earning < 40 AND Return on equity > 10`"
        )
        return

    if meta:
        st.caption(
            f"Checked **{meta.get('checked', 0)}** stocks · "
            f"with P/E data: **{meta.get('with_pe', 0)}** · "
            f"with Mar Cap: **{meta.get('with_mcap', 0)}** · "
            f"parsed conditions: **{len(meta.get('conditions') or [])}**"
        )
        if meta.get("conditions"):
            st.caption(
                "Active: "
                + " AND ".join(f"{f} {o} {v}" for f, o, v in meta["conditions"])
            )

    if table is None or (hasattr(table, "empty") and table.empty):
        st.warning(
            "0 results found.\n\n"
            "**Common causes**\n"
            "- `Price to earning < 2` is extremely rare (try `< 25` or `< 40`)\n"
            "- `Return on capital employed > 50` may be too high (try `> 12`)\n"
            "- Missing Yahoo data → keep **Skip missing fields** ON\n"
            "- Raise **Universe size** or run header **Scan** once\n"
        )
        return

    st.markdown(f"**{len(table)} results found**")
    # S.No column
    show = table.copy()
    show.insert(0, "S.No.", range(1, len(show) + 1))
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("##### Open company")
    companies = show["Company"].astype(str).tolist()
    pick_co = st.selectbox("Company", companies, key="scr_co")
    x1, x2, x3 = st.columns(3)
    with x1:
        if st.button("📈 Analyse", key="scr_an", use_container_width=True):
            st.session_state.selected_stock = pick_co
            st.session_state.page = "Stock Analysis"
            st.rerun()
    with x2:
        if st.button("⭐ Watchlist", key="scr_wl", use_container_width=True):
            wl = load_watchlist()
            if pick_co not in wl:
                wl.append(pick_co)
                save_watchlist(wl)
            st.success("Added")
    with x3:
        if st.button("🟢 Paper BUY", key="scr_buy", use_container_width=True):
            execute_paper_order(pick_co, "BUY", force=True, source="SCREENER_QUERY")
            st.success("Paper order placed")



def show_intraday_fo_desk():
    """
    Intraday + Futures & Options desk:
    NIFTY 50 · BANK NIFTY · stocks — CE/PE buy/sell, expiry or 1–2 day trades,
    chart patterns + strategy explanations.
    """
    st.markdown(
        """
        <div style="border-radius:16px;padding:18px 20px;margin-bottom:14px;
                    background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#0e7490 100%);
                    border:1px solid #334155;">
          <div style="font-size:1.45rem;font-weight:800;color:#f8fafc;">Intraday &amp; F&amp;O Desk</div>
          <div style="color:#94a3b8;margin-top:6px;line-height:1.45;">
            Index &amp; stock options · Call / Put · Buy / Sell · Expiry or 1–2 day holds ·
            Chart patterns + strategy rules (educational).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_live, tab_idx, tab_stk, tab_strat, tab_bt = st.tabs([
        "📡 Live CE/PE calls",
        "📈 Index desk",
        "📊 Stock options",
        "📘 Strategies + rank",
        "🧪 Backtest",
    ])

    with tab_live:
        st.markdown("### Live Call & Put recommendations (strategy + patterns)")
        st.caption("Like swing desk: strategy name · chart pattern · entry · target · stop · paper BUY.")
        und_l = st.selectbox("Underlying", ["NIFTY 50", "BANK NIFTY"], key="ifo_live_und")
        yf_l = "^NSEI" if und_l.startswith("NIFTY") else "^NSEBANK"
        disp_l = "NIFTY" if und_l.startswith("NIFTY") else "BANKNIFTY"
        spot_l = _spot_price(yf_l) or 0
        st.metric(f"{und_l} live spot", f"₹{spot_l:,.2f}" if spot_l else "—")

        df_l = None
        try:
            df_l = stock_history(yf_l, interval="5m")
            if df_l is None or df_l.empty:
                df_l = stock_history(yf_l, interval="15m")
            if df_l is None or df_l.empty:
                df_l = stock_history(yf_l, interval="1d")
        except Exception:
            df_l = None

        pats_l = []
        if df_l is not None and not df_l.empty and len(df_l) >= 20:
            try:
                pats_l = detect_patterns(df_l.tail(60)) or []
            except Exception:
                pats_l = []
            try:
                # structure patterns if available
                for pname in list(CHART_STRUCTURE_PATTERNS.keys())[:8] if "CHART_STRUCTURE_PATTERNS" in dir() else []:
                    pass
            except Exception:
                pass

        # Bias from structure
        bias = "NEUTRAL"
        rsi_v = 50.0
        ema_bias = 0
        if df_l is not None and not df_l.empty and "Close" in df_l.columns:
            c = pd.to_numeric(df_l["Close"], errors="coerce").dropna()
            if len(c) >= 20:
                ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
                ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
                price = float(c.iloc[-1])
                delta = c.diff()
                up = delta.clip(lower=0).rolling(14).mean()
                down = (-delta.clip(upper=0)).rolling(14).mean()
                rs = up / down.replace(0, np.nan)
                rsi_v = float((100 - (100 / (1 + rs))).iloc[-1] or 50)
                if price > ema9 > ema21 and rsi_v >= 52:
                    bias = "BULLISH"
                    ema_bias = 1
                elif price < ema9 < ema21 and rsi_v <= 48:
                    bias = "BEARISH"
                    ema_bias = -1

        step = 50 if disp_l == "NIFTY" else 100
        atm = round(spot_l / step) * step if spot_l else 0
        # Premiums
        ce_prem = max(15.0, spot_l * 0.0048) if spot_l else 50
        pe_prem = max(14.0, spot_l * 0.0042) if spot_l else 48
        try:
            exps = _option_expiries(yf_l) or []
            if exps and atm:
                calls, puts = _option_chain(yf_l, exps[0])
                if calls is not None and not calls.empty and "strike" in calls.columns:
                    cn = calls.iloc[(calls["strike"] - atm).abs().argsort()[:1]]
                    if not cn.empty:
                        ce_prem = safe_float(cn["ask"].iloc[0] if "ask" in cn.columns else cn.get("lastPrice", pd.Series([ce_prem])).iloc[0]) or ce_prem
                if puts is not None and not puts.empty and "strike" in puts.columns:
                    pn = puts.iloc[(puts["strike"] - atm).abs().argsort()[:1]]
                    if not pn.empty:
                        pe_prem = safe_float(pn["ask"].iloc[0] if "ask" in pn.columns else pn.get("lastPrice", pd.Series([pe_prem])).iloc[0]) or pe_prem
        except Exception:
            pass

        pat_txt = ", ".join(pats_l[:5]) if pats_l else "No strong candle pattern on last bars"
        st.info(f"**Bias:** {bias} · RSI {rsi_v:.0f} · Patterns: {pat_txt}")

        # Clear CE / PE / HOLD decision from market structure
        st.markdown("#### Market decision — Buy Call, Buy Put, or Hold")
        if bias == "BULLISH" and rsi_v < 72:
            st.success(
                f"**BUY CALL (CE)** preferred on **{disp_l}**. "
                "Structure supports upside. Avoid buying PE unless hedging. "
                "**Do not** buy CE+PE together unless a big event is due (see Long Straddle below)."
            )
            decision = "BUY CE"
        elif bias == "BEARISH" and rsi_v > 28:
            st.error(
                f"**BUY PUT (PE)** preferred on **{disp_l}**. "
                "Structure supports downside. Avoid buying CE for direction. "
                "Short CE only via spreads if experienced."
            )
            decision = "BUY PE"
        else:
            st.warning(
                f"**HOLD / wait** on directional options. Bias is **{bias}** (RSI {rsi_v:.0f}). "
                "Choppy tape → prefer no trade, or defined-risk spreads only. "
                "Long Straddle only if a **known event** (result/policy) can force a big move."
            )
            decision = "HOLD"

        with st.expander("📘 Long Straddle — buy Call + Put together (when & how)", expanded=False):
            st.markdown(
                """
### Name: **Long Straddle** (also: ATM straddle)

**What you buy:** **ATM Call (CE) + ATM Put (PE)** of the **same** expiry and underlying, usually in **same quantity**.

**Why it exists:** You do **not** need to pick direction. You profit if the **underlying moves a lot** either way (volatility expansion). One side’s gain can exceed the other’s loss after a big move.

**When to use**
- **Before a known event** that can gap the index/stock (RBI, Budget, election, major result) **if** you accept that IV is already high.
- When **implied volatility is relatively cheap** vs recent history and you expect a **range expansion**.
- **Not** ideal on a quiet afternoon with no catalyst — **theta** (time decay) hurts both CE and PE every day.

**When NOT to use**
- Sideways, low-event days (both premiums bleed).
- Right after IV has already exploded (you overpay).
- If you only have a **directional** view → buy **only CE** or **only PE**, not both.

**Targets / stops (premium terms)**
- Typical educational guide: exit when **combined premium** is **+30% to +50%**, or cut if **−25% to −30%** and no move.
- Prefer **weekly ATM**; avoid far OTM “lottery” pairs.

**Related names**
- **Long Strangle** = OTM CE + OTM PE (cheaper, needs even bigger move).
- **Iron Condor** = opposite idea (profit if market stays in a range).
                """
            )

        # Build recommendation cards
        recs = []
        if bias == "BULLISH" or ema_bias >= 0:
            recs.append({
                "side": "BUY CE",
                "strike": int(atm),
                "entry": round(ce_prem, 1),
                "target": round(ce_prem * 1.40, 1),
                "stop": round(ce_prem * 0.70, 1),
                "strategy": "Trend / ORB long · Call side only",
                "pattern": pat_txt,
                "why": "Bullish structure — prefer CE, not PE.",
            })
            recs.append({
                "side": "SELL PE (credit)",
                "strike": int(atm - step),
                "entry": round(pe_prem * 0.75, 1),
                "target": round(pe_prem * 0.75 * 0.50, 1),  # keep 50% credit
                "stop": round(pe_prem * 0.75 * 1.80, 1),
                "strategy": "Bullish premium decay on OTM PE (defined risk preferred)",
                "pattern": pat_txt,
                "why": "Only if you accept short-option risk; prefer put credit spread.",
            })
        if bias == "BEARISH" or ema_bias <= 0:
            recs.append({
                "side": "BUY PE",
                "strike": int(atm),
                "entry": round(pe_prem, 1),
                "target": round(pe_prem * 1.40, 1),
                "stop": round(pe_prem * 0.70, 1),
                "strategy": "Trend / ORB short · Put side only",
                "pattern": pat_txt,
                "why": "Bearish structure — prefer PE, not CE.",
            })
            recs.append({
                "side": "SELL CE (credit)",
                "strike": int(atm + step),
                "entry": round(ce_prem * 0.75, 1),
                "target": round(ce_prem * 0.75 * 0.50, 1),
                "stop": round(ce_prem * 0.75 * 1.80, 1),
                "strategy": "Bearish premium decay on OTM CE (defined risk preferred)",
                "pattern": pat_txt,
                "why": "Prefer call credit spread over naked short CE.",
            })
        # Always show both pure directional cards for clarity
        if not any(r["side"] == "BUY CE" for r in recs) and spot_l:
            recs.insert(0, {
                "side": "BUY CE", "strike": int(atm), "entry": round(ce_prem, 1),
                "target": round(ce_prem * 1.35, 1), "stop": round(ce_prem * 0.72, 1),
                "strategy": "Watchlist CE", "pattern": pat_txt, "why": "Neutral–mild: CE only if breakout confirms.",
            })
        if not any(r["side"] == "BUY PE" for r in recs) and spot_l:
            recs.append({
                "side": "BUY PE", "strike": int(atm), "entry": round(pe_prem, 1),
                "target": round(pe_prem * 1.35, 1), "stop": round(pe_prem * 0.72, 1),
                "strategy": "Watchlist PE", "pattern": pat_txt, "why": "Neutral–mild: PE only if breakdown confirms.",
            })

        for i, r in enumerate(recs[:4]):
            with st.container():
                st.markdown(
                    f"""
                    <div style="border:1px solid #334155;border-radius:14px;padding:14px 16px;margin-bottom:10px;
                                background:linear-gradient(135deg,#0f172a,#1e293b);">
                      <div style="font-weight:800;font-size:1.05rem;color:#f8fafc;">{r['side']} · {disp_l} {r['strike']}</div>
                      <div style="color:#94a3b8;margin-top:4px;">Strategy: {r['strategy']}</div>
                      <div style="color:#a5b4fc;font-size:0.85rem;">Pattern: {r['pattern']}</div>
                      <div style="margin-top:8px;color:#e2e8f0;">
                        Entry <b>₹{r['entry']}</b> · Target <b>₹{r['target']}</b> · Stop <b>₹{r['stop']}</b>
                      </div>
                      <div style="color:#64748b;font-size:0.8rem;margin-top:4px;">{r['why']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Paper trade: use underlying as symbol + note in source
                paper_sym = f"{disp_l}"
                side_paper = "BUY" if r["side"].startswith("BUY") else "SELL"
                try:
                    render_active_trade_buttons(
                        paper_sym,
                        side_hint=side_paper,
                        entry=float(r["entry"]),
                        target=float(r["target"]),
                        stop=float(r["stop"]),
                        key_prefix=f"ifo_rec_{i}_{r['side'].replace(' ', '_')}",
                    )
                except Exception as _e:
                    if st.button(f"🟢 Paper {r['side']}", key=f"ifo_pb_{i}"):
                        try:
                            execute_paper_order(
                                paper_sym, side_paper,
                                entry=float(r["entry"]), target=float(r["target"]), stop=float(r["stop"]),
                                source=f"INTRADAY|{r['side']}|{r['strategy']}", force=True, hold_days=1,
                            )
                            st.success("Added to paper book")
                        except Exception as e2:
                            st.error(str(e2))
                st.divider()

    with tab_idx:
        und = st.selectbox(
            "Underlying",
            ["NIFTY 50", "BANK NIFTY"],
            key="ifo_und",
        )
        yf_sym = "^NSEI" if und.startswith("NIFTY") else "^NSEBANK"
        display = "NIFTY" if und.startswith("NIFTY") else "BANKNIFTY"
        spot = _spot_price(yf_sym)
        st.metric(f"{und} spot", f"₹{spot:,.2f}" if spot else "—")

        horizon = st.radio(
            "Trade horizon",
            ["Intraday (same day exit)", "Expiry trade", "1–2 day swing option"],
            horizontal=True,
            key="ifo_hz",
        )
        side = st.selectbox("Direction bias", ["Bullish → prefer CE / futures long", "Bearish → prefer PE / futures short", "Neutral → straddle / iron condor"], key="ifo_side")
        action = st.selectbox("Option action", ["BUY CE", "BUY PE", "SELL CE", "SELL PE", "BUY Futures", "SELL Futures"], key="ifo_act")

        # Intraday chart
        df = None
        try:
            df = stock_history(yf_sym if yf_sym.startswith("^") else display, interval="5m")
            if df is None or df.empty:
                df = stock_history(yf_sym, interval="1d")
        except Exception:
            df = stock_history(yf_sym, interval="1d")
        if df is not None and not df.empty:
            tail = df.tail(80)
            fig = go.Figure(go.Candlestick(
                x=list(range(len(tail))),
                open=tail["Open"], high=tail["High"], low=tail["Low"], close=tail["Close"],
                name=display,
            ))
            fig.update_layout(
                height=320, margin=dict(l=8, r=8, t=28, b=8),
                title=f"{display} · last bars",
                xaxis_rangeslider_visible=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.92)",
                font=dict(color="#e2e8f0", size=11),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            try:
                pats = detect_patterns(tail) if len(tail) >= 20 else []
            except Exception:
                pats = []
            if pats:
                st.info("**Chart patterns (recent):** " + ", ".join(pats[:8]))
            else:
                st.caption("No strong named candle pattern on the last bars.")

        # Suggested structure
        atm = round(spot / 50) * 50 if spot and und.startswith("NIFTY") else (round(spot / 100) * 100 if spot else 0)
        if und.startswith("BANK"):
            atm = round(spot / 100) * 100 if spot else 0
        st.markdown("##### Suggested structure")
        st.write(f"**ATM zone:** ₹{atm:,.0f}" if atm else "ATM — enter spot manually when live")
        explain = {
            "BUY CE": "Bullish. Risk = premium paid. Target 30–60% of premium intraday; stop if premium −30% or spot breaks structure low.",
            "BUY PE": "Bearish. Same risk rules as CE. Avoid buying PE into strong short-covering rallies.",
            "SELL CE": "Neutral–bearish / income. Defined risk only via hedge (spread). Naked sell = high risk.",
            "SELL PE": "Neutral–bullish / income. Prefer credit spreads over naked short puts.",
            "BUY Futures": "Directional long index. SL below VWAP / prior swing. Intraday square-off mandatory if chosen.",
            "SELL Futures": "Directional short. SL above VWAP / prior swing.",
        }
        st.success(f"**{action}** on **{display}** · {horizon}")
        st.write(explain.get(action, ""))
        st.caption(f"Bias selected: {side}")

        # Live CE / PE buy & sell levels (auto-updating with page refresh)
        st.markdown("##### 💹 Live Call / Put levels")
        try:
            idx_sym = "NIFTY" if display == "NIFTY" else "BANKNIFTY"
            # Yahoo index options often under ^NSEI chain is limited — use synthetic premium guide + chain if any
            exps = []
            try:
                exps = _option_expiries("^NSEI" if display == "NIFTY" else "^NSEBANK") or []
            except Exception:
                exps = []
            exp_use = exps[0] if exps else None
            ce_buy = ce_sell = pe_buy = pe_sell = None
            ce_strike = pe_strike = atm
            if exp_use and atm:
                try:
                    calls, puts = _option_chain("^NSEI" if display == "NIFTY" else "^NSEBANK", exp_use)
                except Exception:
                    calls, puts = None, None
                if calls is not None and not calls.empty and "strike" in calls.columns:
                    cnear = calls.iloc[(calls["strike"] - atm).abs().argsort()[:1]]
                    if not cnear.empty:
                        ce_strike = float(cnear["strike"].iloc[0])
                        ce_buy = safe_float(cnear["ask"].iloc[0] if "ask" in cnear.columns else cnear.get("lastPrice", pd.Series([0])).iloc[0])
                        ce_sell = safe_float(cnear["bid"].iloc[0] if "bid" in cnear.columns else ce_buy)
                if puts is not None and not puts.empty and "strike" in puts.columns:
                    pnear = puts.iloc[(puts["strike"] - atm).abs().argsort()[:1]]
                    if not pnear.empty:
                        pe_strike = float(pnear["strike"].iloc[0])
                        pe_buy = safe_float(pnear["ask"].iloc[0] if "ask" in pnear.columns else pnear.get("lastPrice", pd.Series([0])).iloc[0])
                        pe_sell = safe_float(pnear["bid"].iloc[0] if "bid" in pnear.columns else pe_buy)
            # Fallback synthetic: CE and PE must differ (mild skew + side)
            if spot:
                base_ce = max(12.0, spot * 0.0048)
                base_pe = max(11.0, spot * 0.0042)  # PE often slightly cheaper in mild bull bias guide
                if not ce_buy or ce_buy <= 0:
                    ce_buy, ce_sell = round(base_ce * 1.03, 1), round(base_ce * 0.97, 1)
                if not pe_buy or pe_buy <= 0:
                    pe_buy, pe_sell = round(base_pe * 1.03, 1), round(base_pe * 0.97, 1)
                # OTM / ITM guides (outside vs ATM) — never identical to ATM
                step = 50 if display == "NIFTY" else 100
                ce_otm_strike = (ce_strike or atm) + step
                pe_otm_strike = (pe_strike or atm) - step
                ce_otm_buy = round((ce_buy or base_ce) * 0.55, 1)
                pe_otm_buy = round((pe_buy or base_pe) * 0.55, 1)
                ce_itm_buy = round((ce_buy or base_ce) * 1.55, 1)
                pe_itm_buy = round((pe_buy or base_pe) * 1.55, 1)

            st.markdown("**ATM Call side (CE)**")
            m1, m2 = st.columns(2)
            m1.metric(f"CE {int(ce_strike) if ce_strike else 'ATM'} BUY (ask)", f"₹{ce_buy:,.1f}" if ce_buy else "—")
            m2.metric(f"CE {int(ce_strike) if ce_strike else 'ATM'} SELL (bid)", f"₹{ce_sell:,.1f}" if ce_sell else "—")
            st.markdown("**ATM Put side (PE)** — different from CE")
            m3, m4 = st.columns(2)
            m3.metric(f"PE {int(pe_strike) if pe_strike else 'ATM'} BUY (ask)", f"₹{pe_buy:,.1f}" if pe_buy else "—")
            m4.metric(f"PE {int(pe_strike) if pe_strike else 'ATM'} SELL (bid)", f"₹{pe_sell:,.1f}" if pe_sell else "—")
            if spot:
                st.markdown("**Outside (OTM) vs inside (ITM) — not the same as ATM**")
                o1, o2, o3, o4 = st.columns(4)
                o1.metric(f"OTM CE {int(ce_otm_strike)} BUY", f"₹{ce_otm_buy:,.1f}")
                o2.metric(f"ITM CE BUY", f"₹{ce_itm_buy:,.1f}")
                o3.metric(f"OTM PE {int(pe_otm_strike)} BUY", f"₹{pe_otm_buy:,.1f}")
                o4.metric(f"ITM PE BUY", f"₹{pe_itm_buy:,.1f}")

            # Only the selected action plan (not both sides mixed)
            if "CE" in action and "PE" not in action:
                st.success(
                    f"**Selected: {action}** · CE strike **{int(ce_strike or atm)}** · "
                    f"Buy ~**₹{ce_buy:,.1f}** / Sell ~**₹{ce_sell:,.1f}**"
                    if ce_buy else f"**Selected: {action}** · CE side"
                )
                if action.startswith("BUY") and ce_buy:
                    st.info(f"Enter ~**₹{ce_buy:,.1f}** · Book ~**₹{ce_buy*1.35:,.1f}** (+35%) · Risk exit ~**₹{ce_buy*0.70:,.1f}** (−30%)")
                elif action.startswith("SELL") and ce_sell:
                    st.warning(f"Credit ~**₹{ce_sell:,.1f}** · Prefer credit spread; naked sell = high risk")
            elif "PE" in action:
                st.success(
                    f"**Selected: {action}** · PE strike **{int(pe_strike or atm)}** · "
                    f"Buy ~**₹{pe_buy:,.1f}** / Sell ~**₹{pe_sell:,.1f}**"
                    if pe_buy else f"**Selected: {action}** · PE side"
                )
                if action.startswith("BUY") and pe_buy:
                    st.info(f"Enter ~**₹{pe_buy:,.1f}** · Book ~**₹{pe_buy*1.35:,.1f}** · Risk exit ~**₹{pe_buy*0.70:,.1f}**")
                elif action.startswith("SELL") and pe_sell:
                    st.warning(f"Credit ~**₹{pe_sell:,.1f}** · Prefer credit spread")
            else:
                st.caption(f"Futures plan: {action} · use spot structure levels below.")
            st.caption("CE and PE are separate. OTM ≠ ATM. Levels refresh with Intraday live feed.")
        except Exception as _ce_err:
            st.caption(f"CE/PE levels note: {_ce_err}")

        # Simple levels from spot
        if spot:
            if "BUY CE" in action or "BUY Futures" in action:
                st.write(f"Approx support ₹{spot*0.997:,.1f} · resistance ₹{spot*1.003:,.1f} (0.3% band — tighten on 5m chart)")
            elif "BUY PE" in action or "SELL Futures" in action:
                st.write(f"Approx resistance ₹{spot*1.003:,.1f} · support ₹{spot*0.997:,.1f}")

        with st.expander("Expiry vs 1–2 day vs intraday — how to choose"):
            st.markdown(
                """
- **Intraday:** exit by 3:15 IST. Prefer liquid weekly ATM/ITM1. Avoid far OTM lottery tickets.
- **Expiry trade:** theta is high; prefer spreads or defined risk. Don’t hold unclear positions into last hour without a plan.
- **1–2 day:** use next weekly if very near expiry; watch overnight gap risk on index shorts.
                """
            )

    with tab_stk:
        st.caption("Stock options — **one box**: type to filter (e.g. rel → RELIANCE) then pick from the list.")
        try:
            _all = sorted({
                str(s).replace(".NS", "").replace(".ns", "").upper().strip()
                for s in (NSE_STOCKS or [])
                if str(s).strip()
            })
        except Exception:
            _all = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]
        if not _all:
            _all = ["RELIANCE", "TCS", "INFY", "SBIN"]
        default = "RELIANCE" if "RELIANCE" in _all else _all[0]
        prev = str(st.session_state.get("ifo_stk_pick", default)).upper()
        if prev not in _all:
            prev = default
        idx = _all.index(prev) if prev in _all else 0
        # Single control: Streamlit selectbox is type-to-search in one bar
        sym = st.selectbox(
            "NSE stock (type to search)",
            options=_all,
            index=idx,
            key="ifo_stk_pick",
            help="Click the box and type letters (rel, tcs, hdfc…) to jump to matching stocks — one bar only.",
        )
        spot_s = _spot_price(sym)
        st.metric("Spot", f"₹{spot_s:,.2f}" if spot_s else "—")
        horizon_s = st.selectbox("Horizon", ["Intraday", "1–2 days", "This week expiry"], key="ifo_stk_hz")
        act_s = st.selectbox("Action", ["BUY CE", "BUY PE", "SELL CE (spread preferred)", "SELL PE (spread preferred)"], key="ifo_stk_act")
        exps = _option_expiries(sym)
        exp = st.selectbox("Expiry", exps, key="ifo_stk_exp") if exps else None
        df_s = stock_history(clean_symbol(sym), interval="15m")
        if df_s is None or (hasattr(df_s, "empty") and df_s.empty):
            df_s = stock_history(clean_symbol(sym), interval="1d")
        if df_s is not None and not df_s.empty:
            tail = df_s.tail(60)
            fig = go.Figure(go.Candlestick(
                x=list(range(len(tail))),
                open=tail["Open"], high=tail["High"], low=tail["Low"], close=tail["Close"],
            ))
            fig.update_layout(height=280, xaxis_rangeslider_visible=False, margin=dict(l=6, r=6, t=20, b=6),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.9)",
                              font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            try:
                pats = detect_patterns(tail)
                if pats:
                    st.info("Patterns: " + ", ".join(pats[:6]))
            except Exception:
                pass
        if exp:
            calls, puts = _option_chain(sym, exp)
            side_df = calls if "CE" in act_s else puts
            if side_df is not None and not side_df.empty:
                cols = [c for c in ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"] if c in side_df.columns]
                st.markdown("**Chain snapshot**")
                st.dataframe(side_df[cols].head(20), use_container_width=True, hide_index=True)
            else:
                st.warning("Chain not available for this symbol/expiry on Yahoo.")
        st.write(f"**Plan:** {act_s} · {horizon_s} · {display_symbol(sym)}")
        st.caption("Stock options are less liquid than NIFTY — check OI & spreads before entry.")

    with tab_strat:
        st.markdown("### Strategy rules (detailed)")
        with st.expander("Read full rules — ORB, VWAP, EMA, CE/PE, Straddle", expanded=True):
            st.markdown(
                """
#### 1) ORB — Opening Range Breakout
- Mark high/low of first **15–30 minutes**.
- **Buy CE / futures long** only on **break above** range high with volume; stop below range low (or mid).
- **Buy PE / short** on **break below** range low; stop above range high.
- Avoid if opening range is already huge (trap risk).

#### 2) VWAP reclaim / reject
- **Long / CE:** price was below VWAP, then **closes back above** VWAP in an uptrend day.
- **Short / PE:** price fails from VWAP in a downtrend day.
- Exit by **3:15 IST** for pure intraday.

#### 3) EMA9 flip
- **CE:** close crosses **above** EMA9 after being below — small target, tight time stop.
- **PE:** mirror on cross **below** EMA9.
- Works best in trending sessions; skip in pure chop.

#### 4) ATM CE or PE scalp (direction only)
- Pick **one** side from structure (not both).
- Target **+20% to +40%** of premium; hard time stop **30–45 min** if nothing happens.
- Stop about **−30%** of premium or structure break.

#### 5) Long Straddle (CE + PE together)
- Use only with a **catalyst** or expected **big range**.
- Same expiry, ATM CE+PE; manage as a **package** (combined P&L).
- See expander on **Live CE/PE calls** tab for full when/why.

#### 6) Credit spreads (SELL CE or SELL PE with hedge)
- Prefer **defined risk** (buy further OTM wing) over naked selling.
- Bullish day → put credit spread; bearish day → call credit spread.

**Decision cheat-sheet**
| Market | Prefer | Avoid |
|--------|--------|--------|
| Clear uptrend | BUY CE | BUY PE for direction |
| Clear downtrend | BUY PE | BUY CE for direction |
| Chop, no event | HOLD | Straddle / naked sells |
| Big event, cheap IV | Long Straddle | Directional guesses only |
                """
            )
        st.markdown("### Best Intraday strategies (with quick backtest)")
        st.caption("Ranked on recent **NIFTY 5-minute** bars when available. Educational — past ≠ future.")

        def _bt_orb(df):
            """Opening-range breakout style on intraday bars."""
            if df is None or len(df) < 40:
                return None
            d = df.copy()
            c = pd.to_numeric(d["Close"], errors="coerce")
            h = pd.to_numeric(d.get("High", c), errors="coerce")
            l = pd.to_numeric(d.get("Low", c), errors="coerce")
            wins = losses = 0
            rets = []
            # every ~12 bars treat as a mini session
            step = 12
            for i in range(6, len(d) - 8, step):
                orb_hi = float(h.iloc[i - 3 : i].max())
                orb_lo = float(l.iloc[i - 3 : i].min())
                px = float(c.iloc[i])
                if px > orb_hi:
                    entry, side = px, 1
                elif px < orb_lo:
                    entry, side = px, -1
                else:
                    continue
                fut = c.iloc[i + 1 : i + 7]
                if fut.empty:
                    continue
                exit_px = float(fut.iloc[-1])
                ret = side * (exit_px - entry) / entry * 100
                rets.append(ret)
                if ret > 0:
                    wins += 1
                else:
                    losses += 1
            n = wins + losses
            if n < 3:
                return None
            return {"trades": n, "win_rate": round(100 * wins / n, 1), "avg_pct": round(float(np.mean(rets)), 2)}

        def _bt_vwap_side(df):
            if df is None or len(df) < 30:
                return None
            d = df.tail(80).copy()
            c = pd.to_numeric(d["Close"], errors="coerce")
            v = pd.to_numeric(d["Volume"], errors="coerce") if "Volume" in d.columns else pd.Series([1] * len(d), index=d.index)
            pv = (c * v).cumsum() / v.replace(0, np.nan).cumsum()
            wins = losses = 0
            rets = []
            for i in range(15, len(d) - 4):
                if pd.isna(pv.iloc[i]):
                    continue
                if float(c.iloc[i - 1]) < float(pv.iloc[i - 1]) <= float(c.iloc[i]):
                    entry = float(c.iloc[i])
                    exit_px = float(c.iloc[i + 3])
                    ret = (exit_px - entry) / entry * 100
                    rets.append(ret)
                    wins += 1 if ret > 0 else 0
                    losses += 0 if ret > 0 else 1
            n = wins + losses
            if n < 3:
                return None
            return {"trades": n, "win_rate": round(100 * wins / n, 1), "avg_pct": round(float(np.mean(rets)), 2)}

        def _bt_ema_trend(df):
            if df is None or len(df) < 40:
                return None
            c = pd.to_numeric(df["Close"], errors="coerce")
            ema = c.ewm(span=9, adjust=False).mean()
            wins = losses = 0
            rets = []
            for i in range(20, len(c) - 5):
                if float(c.iloc[i]) > float(ema.iloc[i]) and float(c.iloc[i - 1]) <= float(ema.iloc[i - 1]):
                    entry = float(c.iloc[i])
                    exit_px = float(c.iloc[i + 4])
                    ret = (exit_px - entry) / entry * 100
                    rets.append(ret)
                    wins += 1 if ret > 0 else 0
                    losses += 0 if ret > 0 else 1
            n = wins + losses
            if n < 3:
                return None
            return {"trades": n, "win_rate": round(100 * wins / n, 1), "avg_pct": round(float(np.mean(rets)), 2)}

        bt_df = None
        try:
            bt_df = stock_history("^NSEI", interval="5m")
            if bt_df is None or bt_df.empty:
                bt_df = stock_history("^NSEI", interval="15m")
        except Exception:
            bt_df = None

        catalog = [
            {
                "name": "ORB (Opening Range Breakout)",
                "style": "Intraday",
                "rule": "First 15–30 min high/low. Buy break above with volume; SL other side of range.",
                "fn": _bt_orb,
                "side": "CE on upside break · PE on downside break",
            },
            {
                "name": "VWAP reclaim",
                "style": "Intraday",
                "rule": "Long when price reclaims VWAP after dip in uptrend. Exit into strength or session close.",
                "fn": _bt_vwap_side,
                "side": "Prefer CE / futures long",
            },
            {
                "name": "EMA9 trend flip",
                "style": "Intraday",
                "rule": "Buy when close crosses above EMA9; small target, tight time stop.",
                "fn": _bt_ema_trend,
                "side": "CE / futures long",
            },
            {
                "name": "ATM CE/PE scalp",
                "style": "Intraday F&O",
                "rule": "Buy ATM option only with clear 5m structure. Target 20–40% premium; time stop 30–45 min.",
                "fn": None,
                "side": "CE if bullish structure · PE if bearish — never same plan for both",
            },
            {
                "name": "Bull call spread",
                "style": "Expiry / 1–2 day",
                "rule": "Buy CE, sell higher CE. Defined risk when moderately bullish.",
                "fn": None,
                "side": "Call side only",
            },
            {
                "name": "Bear put spread",
                "style": "Expiry / 1–2 day",
                "rule": "Buy PE, sell lower PE. Defined risk when moderately bearish.",
                "fn": None,
                "side": "Put side only",
            },
        ]

        rows = []
        for item in catalog:
            rec = {"Strategy": item["name"], "Style": item["style"], "Side focus": item["side"],
                   "Trades": "—", "Win %": "—", "Avg %": "—", "Rank score": 0.0}
            if item.get("fn") and bt_df is not None and not bt_df.empty:
                try:
                    r = item["fn"](bt_df)
                    if r:
                        rec["Trades"] = r["trades"]
                        rec["Win %"] = r["win_rate"]
                        rec["Avg %"] = r["avg_pct"]
                        # score: win rate + avg
                        rec["Rank score"] = float(r["win_rate"]) + float(r["avg_pct"]) * 5
                except Exception:
                    pass
            rows.append(rec)

        rank = pd.DataFrame(rows).sort_values("Rank score", ascending=False)
        st.markdown("#### Ranked list (best first when backtest data exists)")
        st.dataframe(
            rank.drop(columns=["Rank score"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )
        best = rank.iloc[0]["Strategy"] if not rank.empty else "ORB"
        st.success(f"**Suggested focus today:** {best} — see side focus in the table (CE vs PE are not interchangeable).")

        for item in catalog:
            st.markdown(
                f"**{item['name']}** · `{item['style']}`  \n"
                f"{item['rule']}  \n"
                f"*Side:* {item['side']}"
            )
            st.divider()

        st.warning(
            "Educational only — not investment advice. F&O can lose more than premium if you sell naked options. "
            "Prefer defined-risk spreads. Always use stop-loss and position size caps."
        )

    with tab_bt:
        st.markdown("### Intraday backtest desk")
        st.caption("Run ORB / EMA stats on index bars, then paper-trade from Live CE/PE calls.")
        if st.button("Open full Backtest page", type="primary", key="ifo_open_bt_page"):
            try:
                nav_go("Intraday Backtest")
            except Exception:
                st.session_state.page = "Intraday Backtest"
                st.rerun()
        st.info("Or use the **Backtest** button in the top header.")


def show_intraday_backtest_page():
    """Dedicated intraday strategy backtesting page."""
    st.title("🧪 Intraday Backtesting")
    st.caption("ORB · VWAP · EMA flip on NIFTY/BANKNIFTY bars. Educational — not a guarantee.")
    und = st.selectbox("Index", ["NIFTY 50", "BANK NIFTY"], key="bt_page_und")
    yf_sym = "^NSEI" if und.startswith("NIFTY") else "^NSEBANK"
    interval = st.selectbox("Bar size", ["5m", "15m", "1h", "1d"], index=0, key="bt_page_iv")
    if st.button("Run backtests", type="primary", key="bt_page_run"):
        with st.spinner("Loading bars + running…"):
            df = stock_history(yf_sym, interval=interval)
            if df is None or df.empty:
                st.error("No bars returned")
                return
            st.session_state["_bt_page_df"] = df
    df = st.session_state.get("_bt_page_df")
    if df is None or (hasattr(df, "empty") and df.empty):
        st.info("Click **Run backtests** to load data.")
        return
    st.write(f"Bars loaded: **{len(df)}**")

    def _bt_orb(d):
        c = pd.to_numeric(d["Close"], errors="coerce")
        h = pd.to_numeric(d.get("High", c), errors="coerce")
        l = pd.to_numeric(d.get("Low", c), errors="coerce")
        wins = losses = 0
        rets = []
        step = 12
        for i in range(6, len(d) - 8, step):
            orb_hi = float(h.iloc[i - 3 : i].max())
            orb_lo = float(l.iloc[i - 3 : i].min())
            px = float(c.iloc[i])
            if px > orb_hi:
                entry, side = px, 1
            elif px < orb_lo:
                entry, side = px, -1
            else:
                continue
            fut = c.iloc[i + 1 : i + 7]
            if fut.empty:
                continue
            ret = side * (float(fut.iloc[-1]) - entry) / entry * 100
            rets.append(ret)
            wins += 1 if ret > 0 else 0
            losses += 0 if ret > 0 else 1
        n = wins + losses
        if n < 3:
            return None
        return {"Strategy": "ORB", "Trades": n, "Win %": round(100 * wins / n, 1), "Avg %": round(float(np.mean(rets)), 2)}

    def _bt_ema(d):
        c = pd.to_numeric(d["Close"], errors="coerce")
        ema = c.ewm(span=9, adjust=False).mean()
        wins = losses = 0
        rets = []
        for i in range(20, len(c) - 5):
            if float(c.iloc[i]) > float(ema.iloc[i]) and float(c.iloc[i - 1]) <= float(ema.iloc[i - 1]):
                entry = float(c.iloc[i])
                ret = (float(c.iloc[i + 4]) - entry) / entry * 100
                rets.append(ret)
                wins += 1 if ret > 0 else 0
                losses += 0 if ret > 0 else 1
        n = wins + losses
        if n < 3:
            return None
        return {"Strategy": "EMA9 flip", "Trades": n, "Win %": round(100 * wins / n, 1), "Avg %": round(float(np.mean(rets)), 2)}

    rows = []
    for fn in (_bt_orb, _bt_ema):
        try:
            r = fn(df)
            if r:
                rows.append(r)
        except Exception:
            pass
    if not rows:
        st.warning("Not enough structure for stats on this sample.")
        return
    out = pd.DataFrame(rows).sort_values("Win %", ascending=False)
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.success(f"**Best on this sample:** {out.iloc[0]['Strategy']} ({out.iloc[0]['Win %']}% wins)")
    if st.button("Open Paper / Dummy trades", key="bt_to_paper"):
        st.session_state.page = "Paper Trading"
        st.rerun()



def show_nse_tools_workspace():
    """
    Landing grid matching RG Tools card layout.
    Each card opens a dedicated tool page with real data.
    """
    st.markdown(
        """
        <div style="margin-bottom:12px;padding:14px 16px;border-radius:14px;
                    background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;">
          <div style="font-size:1.35rem;font-weight:800;color:#f8fafc;">NSE Tools Workspace</div>
          <div style="color:#94a3b8;margin-top:6px;font-size:0.92rem;line-height:1.4;">
            Swing desk first, then options, patterns and institutional flow.
            Your existing scan, predictions and paper data stay as they are — only layout is organised here.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards = [
        ("📈", "Option Historical Chart", "Underlying candles with option mid/IV/OI when Yahoo chain allows. Spot from live NSE quote.", "Tool Option History"),
        ("🧮", "Option Price Calculator", "Black–Scholes estimate from live spot + IV. Strike-wise theoretical value vs market.", "Tool Option Calculator"),
        ("🏛", "FII / DII Data", "Date-wise foreign and domestic institutional buying, selling and net flow in cash market.", "FII DII"),
        ("📉", "IV Tool", "Implied volatility from the live option chain by symbol, expiry, strike and CE/PE.", "Tool IV"),
        ("📊", "Open Interest Chain", "Full chain for a stored/live session — OI, volume, LTP, max-pain estimate.", "Tool OI Chain"),
        ("⚡", "Option Strategy Backtester", "Replay simple straddle / strangle on historical underlying with BS marks.", "Tool Option Backtester"),
        ("✦", "Square of 9", "Support and resistance from Square of 9 on a real NSE last price — triggers & targets.", "Tool Square9"),
        ("🔍", "Chart Pattern Scanner", "Gartley, Butterfly, ABCD, H&S, double/triple tops, wedges, triangles — drawn on chart.", "Chart Pattern Scanner"),
        ("🔎", "Pattern Screener", "Pick one pattern and sweep NSE stocks — newest completions first with fit score.", "Chart Pattern Scanner"),
        ("🕯", "Candlestick Patterns", "Hammers, engulfings, stars — one stock or market sweep.", "Tool Candles"),
        ("🗓", "Multi-Timeframe", "Monthly, weekly and daily confluence on NSE bars.", "Multi Timeframe"),
        ("⚡", "Momentum Divergence", "Sharp impulse followed by a slower, smaller recovery — market sweep.", "Tool Momentum"),
    ]
    #  render 2-col then wrap like RG (use 2 on mobile-friendly)
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(cards):
                break
            icon, title, desc, page = cards[i + j]
            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid #e2e8f0;border-radius:16px;padding:18px 16px;margin-bottom:12px;
                                background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.06);min-height:150px;">
                      <div style="width:40px;height:40px;border-radius:12px;
                                  background:linear-gradient(135deg,#0ea5e9,#6366f1);
                                  display:flex;align-items:center;justify-content:center;
                                  color:#fff;font-size:1.2rem;">{icon}</div>
                      <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-top:10px;">{title}</div>
                      <div style="color:#64748b;font-size:0.88rem;margin-top:6px;line-height:1.4;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open Tool → {title}", key=f"nse_tool_{page}_{i}_{j}", use_container_width=True):
                    nav_go(page)


def show_tool_square9():
    st.markdown("### ✦ Square of 9")
    st.caption("Support / resistance from the Square of 9 wheel on a **real NSE last price** — no manual price required unless you override.")
    sym = st.text_input("NSE symbol", value="NIFTY" if False else "RELIANCE", key="sq9_sym")
    # Map index aliases
    yf_sym = sym
    if sym.upper() in ("NIFTY", "NIFTY50"):
        yf_sym = "^NSEI"
    elif sym.upper() in ("BANKNIFTY", "NIFTYBANK"):
        yf_sym = "^NSEBANK"
    spot = _spot_price(yf_sym if yf_sym.startswith("^") else sym)
    use = st.number_input("Price (auto-filled from live)", value=float(spot or 0), min_value=0.0, key="sq9_px")
    if use <= 0:
        st.warning("Enter a valid price or symbol with a live quote.")
        return
    levels = gann_square_of_9(use)
    st.metric("Pivot (input)", f"₹{use:,.2f}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Resistance**")
        for k in ("Resistance 1", "Resistance 2", "Above 135°", "Above 180°"):
            if k in levels:
                st.write(f"{k}: **₹{levels[k]:,.2f}**")
    with c2:
        st.markdown("**Support**")
        for k in ("Support 1", "Support 2", "Below 135°", "Below 180°"):
            if k in levels:
                st.write(f"{k}: **₹{levels[k]:,.2f}**")
    with st.expander("All degree levels"):
        st.json({k: v for k, v in levels.items() if k not in ("Root",)})


def show_tool_iv():
    st.markdown("### 📉 IV Tool")
    st.caption("Latest implied volatility from the **live Yahoo option chain** (NSE stocks).")
    sym = st.text_input("Underlying (NSE)", value="RELIANCE", key="iv_sym")
    exps = _option_expiries(sym)
    if not exps:
        st.warning("No option expiries returned for this symbol (Yahoo). Try RELIANCE, TCS, INFY, SBIN…")
        return
    exp = st.selectbox("Expiry", exps, key="iv_exp")
    calls, puts = _option_chain(sym, exp)
    side = st.radio("Side", ["CE (Calls)", "PE (Puts)"], horizontal=True, key="iv_side")
    df = calls if side.startswith("CE") else puts
    if df is None or df.empty:
        st.warning("Empty chain.")
        return
    cols = [c for c in ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"] if c in df.columns]
    view = df[cols].copy()
    if "impliedVolatility" in view.columns:
        view["IV %"] = (pd.to_numeric(view["impliedVolatility"], errors="coerce") * 100).round(2)
    st.dataframe(view, use_container_width=True, hide_index=True)
    try:
        if "impliedVolatility" in df.columns and "strike" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["strike"], y=pd.to_numeric(df["impliedVolatility"], errors="coerce") * 100,
                mode="lines+markers", name="IV %",
            ))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                              title=f"IV smile · {display_symbol(sym)} · {exp}",
                              xaxis_title="Strike", yaxis_title="IV %",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.9)",
                              font=dict(color="#e2e8f0"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass


def show_tool_oi_chain():
    st.markdown("### 📊 Open Interest Chain")
    st.caption("Full option chain with OI, volume, LTP — max-pain estimate from OI.")
    sym = st.text_input("Underlying (NSE)", value="RELIANCE", key="oi_sym")
    spot = _spot_price(sym)
    st.metric("Spot", f"₹{spot:,.2f}" if spot else "—")
    exps = _option_expiries(sym)
    if not exps:
        st.warning("No expiries from Yahoo for this symbol.")
        return
    exp = st.selectbox("Expiry", exps, key="oi_exp")
    calls, puts = _option_chain(sym, exp)
    if (calls is None or calls.empty) and (puts is None or puts.empty):
        st.warning("Empty chain.")
        return
    # Max pain: strike minimizing total option buyer loss ≈ max writer profit
    try:
        strikes = sorted(set(
            list(pd.to_numeric(calls.get("strike"), errors="coerce").dropna()) +
            list(pd.to_numeric(puts.get("strike"), errors="coerce").dropna())
        ))
        max_pain = None
        best = None
        for K in strikes:
            loss = 0.0
            if not calls.empty:
                for _, r in calls.iterrows():
                    oi = safe_float(r.get("openInterest"))
                    sk = safe_float(r.get("strike"))
                    loss += oi * max(0, sk - K)
            if not puts.empty:
                for _, r in puts.iterrows():
                    oi = safe_float(r.get("openInterest"))
                    sk = safe_float(r.get("strike"))
                    loss += oi * max(0, K - sk)
            if best is None or loss < best:
                best = loss
                max_pain = K
        if max_pain:
            st.info(f"**Max pain (OI-weighted estimate):** ₹{max_pain:,.2f}")
    except Exception:
        pass
    t1, t2 = st.tabs(["Calls", "Puts"])
    with t1:
        if calls is not None and not calls.empty:
            cols = [c for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"] if c in calls.columns]
            st.dataframe(calls[cols], use_container_width=True, hide_index=True)
    with t2:
        if puts is not None and not puts.empty:
            cols = [c for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"] if c in puts.columns]
            st.dataframe(puts[cols], use_container_width=True, hide_index=True)


def show_tool_option_calculator():
    st.markdown("### NSE Option Price Calculator")
    st.caption(
        "Trade inputs on the left · strike ladder on the right. "
        "Today LTP / IV / OI from live chain; cycle-end estimate uses Black–Scholes at your target spot."
    )
    left, right = st.columns([1.05, 1.6])
    with left:
        st.markdown("**Trade inputs**")
        instrument = st.selectbox("Instrument", ["Stock"], key="oc_inst")
        sym = st.text_input("Stock", value="RELIANCE", key="oc_sym")
        direction = st.selectbox("Direction", ["CE (Call)", "PE (Put)"], key="oc_dir")
        opt = "CE" if direction.startswith("CE") else "PE"
        lot = st.number_input("Lot size", min_value=0, value=0, key="oc_lot",
                              help="0 = show per-share premium only")
        spot = _spot_price(sym)
        exps = _option_expiries(sym)
        exp = st.selectbox("Option expiry", exps, key="oc_exp") if exps else None
        spot_in = st.number_input("Current spot ₹", value=float(spot or 0), min_value=0.0, key="oc_spot")
        iv_in = st.number_input("Implied volatility %", value=20.0, min_value=0.0, key="oc_iv")
        exp_move = st.number_input("Expected move % (optional)", value=0.0, min_value=0.0, key="oc_em")
        r = st.number_input("Risk-free rate %", value=6.5, key="oc_r") / 100.0
        target_spot = st.number_input(
            "Target spot ₹ (cycle end)",
            value=float(spot_in * (1 + exp_move / 100.0) if exp_move else spot_in or 0),
            min_value=0.0,
            key="oc_tgt",
        )
        go = st.button("Calculate", type="primary", use_container_width=True, key="oc_go")
    with right:
        st.markdown("**Strike-wise comparison (today vs cycle end)**")
        if not go and not st.session_state.get("_oc_done"):
            st.info("Set inputs → **Calculate**. Live chain loads for this expiry.")
        if go or st.session_state.get("_oc_done"):
            st.session_state["_oc_done"] = True
            S = float(spot_in or spot or 0)
            if S <= 0:
                st.warning("Spot is 0 — check symbol.")
            else:
                T = 30 / 365.0
                days_exp = 30
                if exp:
                    try:
                        dt = pd.to_datetime(exp)
                        days_exp = max(1, (dt - pd.Timestamp.now()).days)
                        T = days_exp / 365.0
                    except Exception:
                        pass
                m1, m2, m3 = st.columns(3)
                m1.metric("Target spot", f"₹{target_spot:,.2f}")
                m2.metric("Days → cycle end", f"{days_exp}")
                m3.metric("Days → expiry", f"{days_exp}")
                calls, puts = (pd.DataFrame(), pd.DataFrame())
                if exp:
                    calls, puts = _option_chain(sym, exp)
                df = calls if opt == "CE" else puts
                rows = []
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        K = safe_float(row.get("strike"))
                        if K <= 0:
                            continue
                        ltp = safe_float(row.get("lastPrice"))
                        iv_ch = safe_float(row.get("impliedVolatility"))
                        iv_use = iv_ch if iv_ch > 0.01 else (iv_in / 100.0)
                        oi = safe_float(row.get("openInterest"))
                        vol = safe_float(row.get("volume"))
                        theo_now = black_scholes_price(S, K, T, r, iv_use if iv_use > 0 else 0.2, opt)
                        theo_end = black_scholes_price(target_spot, K, max(1/365, T * 0.15), r, iv_use if iv_use > 0 else 0.2, opt)
                        prem_pct = (ltp / S * 100) if S and ltp else 0
                        lot_n = int(lot) if lot else 1
                        change = (theo_end - (ltp or theo_now)) * lot_n
                        moneyness = "ATM" if abs(K - S) / S < 0.01 else ("ITM" if (opt == "CE" and K < S) or (opt == "PE" and K > S) else "OTM")
                        rows.append({
                            "Moneyness": moneyness,
                            "Strike": K,
                            "Today ₹ (LTP)": round(ltp, 2),
                            "Premium % spot": round(prem_pct, 2),
                            "Lot cost ₹": round((ltp or theo_now) * lot_n, 2),
                            "IV %": round(iv_use * 100, 2),
                            "OI": int(oi),
                            "Volume": int(vol),
                            "Cycle end ₹ (est)": round(theo_end, 2),
                            "Change/lot ₹": round(change, 2),
                            "Move %": round((theo_end / (ltp or theo_now) - 1) * 100, 2) if (ltp or theo_now) else 0,
                        })
                if rows:
                    out = pd.DataFrame(rows)
                    st.dataframe(out, use_container_width=True, hide_index=True, height=360)
                    st.caption(
                        "Today ₹, IV, OI & Volume are from the live chain. "
                        "Cycle end ₹ is a BS estimate at your target spot — real IV crush can lower actual premium."
                    )
                else:
                    st.warning("Live chain could not be loaded. Check symbol/expiry, then Calculate again.")
    with st.expander("How to use"):
        st.markdown(
            """
1. Enter **stock**, **CE/PE**, **expiry**.  
2. Spot auto-fills from live quote — edit if needed.  
3. Set **target spot** (or expected move %).  
4. **Calculate** — table lists each strike with LTP, IV, OI and estimated cycle-end value.  
*Educational estimate — not a broker quote.*
            """
        )


def show_tool_option_history():
    st.markdown("### 📈 Option Historical Chart")
    st.caption(
        "Underlying OHLC is live from Yahoo. Full expired-option candle history needs a paid NSE options DB "
        "(as on RG Tools). Here we chart the **stock** and overlay **current chain IV/OI** snapshot."
    )
    sym = st.text_input("Underlying", value="RELIANCE", key="oh_sym")
    df = stock_history(clean_symbol(sym), interval="1d")
    if df is None or df.empty:
        st.warning("No history.")
        return
    df = df.tail(180)
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Underlying"
    ))
    fig.update_layout(height=380, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=30, b=8),
                      title=f"{display_symbol(sym)} · daily", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(15,23,42,0.9)", font=dict(color="#e2e8f0"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    exps = _option_expiries(sym)
    if exps:
        exp = st.selectbox("Overlay chain expiry", exps[:8], key="oh_exp")
        calls, puts = _option_chain(sym, exp)
        if calls is not None and not calls.empty:
            st.write("**Calls snapshot (OI / IV)**")
            cols = [c for c in ["strike", "lastPrice", "openInterest", "impliedVolatility", "volume"] if c in calls.columns]
            st.dataframe(calls[cols].head(25), use_container_width=True, hide_index=True)


def show_tool_option_backtester():
    st.markdown("### ⚡ Option Strategy Backtester")
    st.caption(
        "Educational approximation: long straddle / strangle marked with Black–Scholes on historical underlying. "
        "Not exchange-settled option ticks (those need a full options history DB)."
    )
    sym = st.text_input("Underlying", value="RELIANCE", key="ob_sym")
    strat = st.selectbox("Strategy", ["Long Straddle", "Long Strangle"], key="ob_strat")
    hold = st.slider("Hold trading days", 3, 20, 5, key="ob_hold")
    iv0 = st.number_input("Entry IV %", value=22.0, key="ob_iv") / 100.0
    df = stock_history(clean_symbol(sym), interval="1d")
    if df is None or len(df) < 80:
        st.warning("Need more history.")
        return
    df = df.tail(160).copy()
    trades = []
    i = 20
    while i < len(df) - hold - 1:
        S0 = float(df["Close"].iloc[i])
        K = round(S0 / 5) * 5
        K2 = round(S0 * 1.02 / 5) * 5 if "Strangle" in strat else K
        T0 = hold / 365.0
        ce0 = black_scholes_price(S0, K, T0, 0.065, iv0, "CE")
        pe0 = black_scholes_price(S0, K2 if "Strangle" in strat else K, T0, 0.065, iv0, "PE")
        entry = ce0 + pe0
        S1 = float(df["Close"].iloc[i + hold])
        ce1 = black_scholes_price(S1, K, 1 / 365.0, 0.065, iv0, "CE")
        pe1 = black_scholes_price(S1, K2 if "Strangle" in strat else K, 1 / 365.0, 0.065, iv0, "PE")
        exit_v = ce1 + pe1
        pnl = exit_v - entry
        trades.append({"Entry date": str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i]),
                       "S0": round(S0, 2), "S1": round(S1, 2), "Debit": round(entry, 2),
                       "Exit": round(exit_v, 2), "PnL": round(pnl, 2)})
        i += hold
    if not trades:
        st.info("No trades.")
        return
    tdf = pd.DataFrame(trades)
    wins = (tdf["PnL"] > 0).sum()
    st.metric("Trades", len(tdf))
    st.metric("Win rate", f"{100*wins/len(tdf):.1f}%")
    st.metric("Total PnL (pts)", f"{tdf['PnL'].sum():+.2f}")
    st.dataframe(tdf, use_container_width=True, hide_index=True)


def show_tool_candles():
    st.markdown("### 🕯 Candlestick Patterns")
    st.caption("Named one/two/three candle formations on real OHLC.")
    mode = st.radio("Mode", ["One stock", "Sweep market (sample)"], horizontal=True, key="cdl_mode")
    if mode.startswith("One"):
        sym = st.text_input("Symbol", value="RELIANCE", key="cdl_sym")
        df = stock_history(clean_symbol(sym), interval="1d")
        if df is None or df.empty:
            st.warning("No data.")
            return
        df = df.tail(120)
        pats = detect_patterns(df)
        st.write("**Recent patterns:**", ", ".join(pats) if pats else "None flagged on last bars")
        fig = go.Figure(go.Candlestick(
            x=list(range(len(df))), open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
        ))
        fig.update_layout(height=340, xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=20, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.9)",
                          font=dict(color="#e2e8f0"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        universe = list(NSE_STOCKS)[:40] if NSE_STOCKS else ["RELIANCE", "TCS", "INFY"]
        rows = []
        with st.spinner("Scanning sample universe…"):
            for sym in universe:
                try:
                    df = stock_history(clean_symbol(sym), interval="1d")
                    if df is None or len(df) < 30:
                        continue
                    pats = detect_patterns(df.tail(40))
                    if pats:
                        rows.append({"Stock": display_symbol(sym), "Patterns": ", ".join(pats[:5])})
                except Exception:
                    continue
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No named candle patterns in this sample.")



def show_pattern_hub():
    st.markdown(
        """
        <div style="border-radius:14px;padding:16px 18px;margin-bottom:12px;
                    background:linear-gradient(135deg,#0f172a,#312e81);border:1px solid #4338ca;">
          <div style="font-size:1.4rem;font-weight:800;color:#e0e7ff;">Chart Patterns</div>
          <div style="color:#a5b4fc;">Structure scanner · multi-timeframe alignment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Format aligned with RG Tools · pick one tool card")
    _hub_card_grid([
        ("📐 Pattern Screener", "Pick one pattern · sweep NSE stocks · newest completions first · points marked · fit score · timeframe badge.", "Chart Pattern Scanner"),
        ("🗓 Multi-Timeframe Confluence", "Monthly · Weekly · Daily must agree. One stock or market sweep. Neutral/doji never counts.", "Multi Timeframe"),
    ], key_prefix="pattern_hub")


def show_tools_hub():
    st.markdown(
        """
        <div style="border-radius:14px;padding:16px 18px;margin-bottom:12px;
                    background:linear-gradient(135deg,#0f172a,#1e3a5f);border:1px solid #334155;">
          <div style="font-size:1.4rem;font-weight:800;color:#e2e8f0;">Tools</div>
          <div style="color:#94a3b8;">Lookup · sector · single-stock analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _hub_card_grid([
        ("🔍 Find Stock", "Search by keywords and get recommendations.", "Find Stock"),
        ("🏭 Sector Analysis", "Sector-wise stocks and charts.", "Sector Analysis"),
        ("🔍 Analyze Stock", "Full technical + fundamental analysis for any NSE symbol.", "Stock Analysis"),
    ], key_prefix="tools_hub")


def show_multi_timeframe(results: pd.DataFrame = None):
    """
    Multi-Timeframe Confluence — Monthly / Weekly / Daily must agree.
    Controls aligned with RG Tools structure (own labels, not a clone).
    """
    st.markdown("### Multi-Timeframe Confluence")
    st.caption("One stock, or every stock — monthly, weekly and daily must all agree. Neutral shapes (doji) never count.")

    bull_pats = ["Any bullish", "Bullish Engulfing", "Morning Star", "Hammer", "Piercing", "Three White Soldiers"]
    bear_pats = ["Any bearish", "Bearish Engulfing", "Evening Star", "Shooting Star", "Dark Cloud", "Three Black Crows"]
    any_pats = ["Any directional"] + bull_pats[1:] + bear_pats[1:]

    mode = st.radio("What to check", ["One stock", "All stocks"], horizontal=True, key="mtf_mode")
    c_top1, c_top2 = st.columns(2)
    with c_top1:
        if results is not None and not results.empty and "Stock" in results.columns:
            universe = results["Stock"].astype(str).str.replace(".NS", "", regex=False).str.upper().unique().tolist()
        else:
            universe = list(NSE_STOCKS)[:300] if NSE_STOCKS else ["RELIANCE", "TCS", "INFY"]
        default = str(st.session_state.get("selected_stock", "RELIANCE") or "RELIANCE")
        if mode.startswith("One"):
            stock = st.selectbox("STOCK", options=sorted(set([default] + list(universe)[:200])), index=0, key="mtf_stock_sel")
            st.caption(f"{len(universe)} symbols available · weekly/monthly folded from daily bars.")
        else:
            stock = None
            max_n = st.slider("Max stocks to scan", 20, min(200, len(universe) or 50), min(50, len(universe) or 50), key="mtf_max")
            st.caption(f"Sweep up to {max_n} symbols from universe ({len(universe)} available).")
    with c_top2:
        direction = st.selectbox(
            "DIRECTION",
            ["Bullish on all three", "Bearish on all three", "Any alignment"],
            key="mtf_dir",
        )
        st.caption("Neutral shapes like a doji never count — they cannot confirm a direction.")

    mcol, wcol, dcol = st.columns(3)
    with mcol:
        st.markdown("**MONTHLY** · The backdrop.")
        m_pat = st.selectbox("PATTERN", bull_pats if "Bullish" in direction else (bear_pats if "Bearish" in direction else any_pats), key="mtf_m_pat")
        m_win = st.selectbox("FORMED WITHIN", ["Last 1 month", "Last 2 months", "Last 3 months"], index=1, key="mtf_m_win")
    with wcol:
        st.markdown("**WEEKLY** · The middle leg.")
        w_pat = st.selectbox("PATTERN", bull_pats if "Bullish" in direction else (bear_pats if "Bearish" in direction else any_pats), key="mtf_w_pat")
        w_win = st.selectbox("FORMED WITHIN", ["Last 2 weeks", "Last 3 weeks", "Last 5 weeks"], index=1, key="mtf_w_win")
    with dcol:
        st.markdown("**DAILY** · The confirmation.")
        d_pat = st.selectbox("PATTERN", bull_pats if "Bullish" in direction else (bear_pats if "Bearish" in direction else any_pats), key="mtf_d_pat")
        d_win = st.selectbox("FORMED WITHIN", ["Last 3 sessions", "Last 5 sessions", "Last 8 sessions"], index=1, key="mtf_d_win")

    candle_basis = st.selectbox(
        "CANDLE BASIS",
        ["Completed candles only — safest", "Include forming candle (can still change)"],
        key="mtf_basis",
    )
    st.caption("A weekly or monthly candle that has not closed can still change. Included ones are badged.")

    b1, b2 = st.columns(2)
    run = b1.button("Check this stock" if mode.startswith("One") else "Scan the market", type="primary", key="mtf_run")
    if b2.button("Start over", key="mtf_clear"):
        st.session_state.pop("mtf_last", None)
        st.session_state.pop("mtf_hits", None)
        st.rerun()

    def _want_bias():
        if "Bullish" in direction:
            return "BULLISH"
        if "Bearish" in direction:
            return "BEARISH"
        return None

    if run and mode.startswith("One"):
        with st.spinner(f"Reading {stock} daily → weekly → monthly…"):
            out = analyze_multi_timeframe(stock)
        if not out.get("ok"):
            st.error(out.get("msg", "Failed"))
        else:
            st.session_state["mtf_last"] = out

    out = st.session_state.get("mtf_last")
    if mode.startswith("One") and out and out.get("ok"):
        gap = out.get("gap_pct") or 0
        st.markdown(
            f"**{out.get('stock')}** · Live **₹{out.get('live')}** · Gap **{gap:+.2f}%** · "
            f"**{out.get('align')}** (score {out.get('score')})"
        )
        want = _want_bias()
        m, w, d = out["monthly"], out["weekly"], out["daily"]
        a, b, c = st.columns(3)
        for col, title, block, pref in (
            (a, "MONTHLY", m, m_pat),
            (b, "WEEKLY", w, w_pat),
            (c, "DAILY", d, d_pat),
        ):
            color = "#16a34a" if block.get("bias") == "BULLISH" else ("#dc2626" if block.get("bias") == "BEARISH" else "#64748b")
            match = (want is None) or (block.get("bias") == want)
            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid #334155;border-radius:12px;padding:12px;background:#0f172a;
                                opacity:{'1' if match else '0.55'};">
                      <div style="color:#94a3b8;font-size:0.75rem;">{title}</div>
                      <div style="color:{color};font-weight:800;font-size:1.1rem;">{block.get('bias')}</div>
                      <div style="color:#e2e8f0;font-size:0.9rem;margin-top:4px;">{block.get('pattern','')}</div>
                      <div style="color:#64748b;font-size:0.8rem;">Wanted: {pref}</div>
                      <div style="color:#94a3b8;font-size:0.8rem;">Close ₹{block.get('close','—')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if out.get("daily_df") is not None:
            try:
                fig = go.Figure(go.Candlestick(
                    x=out["daily_df"].index,
                    open=out["daily_df"]["Open"], high=out["daily_df"]["High"],
                    low=out["daily_df"]["Low"], close=out["daily_df"]["Close"],
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                pass
        try:
            side = "BUY" if "BULLISH" in str(out.get("align", "")) else ("SELL" if "BEARISH" in str(out.get("align", "")) else "BUY")
            render_active_trade_buttons(out["stock"], side_hint=side, entry=out.get("live"), key_prefix="mtf_one")
        except Exception:
            pass

    if run and mode.startswith("All"):
        hits = []
        prog = st.progress(0.0)
        for i, sym in enumerate(list(universe)[:max_n]):
            try:
                o = analyze_multi_timeframe(sym)
                if not o.get("ok"):
                    continue
                al = o.get("align", "")
                keep = False
                if direction == "Any alignment":
                    keep = o.get("score", 0) >= 78
                elif "Bullish" in direction:
                    keep = al == "BULLISH on all three"
                elif "Bearish" in direction:
                    keep = al == "BEARISH on all three"
                if keep:
                    hits.append(o)
            except Exception:
                pass
            prog.progress((i + 1) / max(1, max_n))
        st.session_state["mtf_hits"] = hits
        st.success(f"Aligned: **{len(hits)}** stocks")

    hits = st.session_state.get("mtf_hits") or []
    if hits and mode.startswith("All"):
        for h in hits[:20]:
            st.markdown(f"**{h.get('stock')}** · {h.get('align')} · ₹{h.get('live')} · score {h.get('score')}")

    with st.expander("How to use"):
        st.markdown(
            """
1. **One stock** — see monthly / weekly / daily bias even if one leg fails.  
2. **All stocks** — only names where all three agree on direction.  
3. Set **pattern preference** and **formed within** per timeframe.  
4. Prefer **Completed candles only** for safer signals.  
*Research aid — not investment advice.*
            """
        )




# ============================================================
# SWING STRATEGIES (restored core definitions)
# ============================================================

SWING_STRATEGIES = {
    "my_strategy": {
        "name": "My Strategy (RSI divergence)",
        "side": "BUY",
        "hold_days": 15,
        "typical_win": "45–55%",
        "typical_exp": "illustrative",
        "desc": "Price makes lower lows while RSI makes higher lows — bullish divergence.",
        "beginner": "On daily chart draw swing lows on price and on RSI. If price falls but RSI rises, wait for a small confirmation green day. Stop under the swing low.",
        "why": "Momentum often turns before price; divergence flags early accumulation.",
    },
    "trend_pullback": {
        "name": "Trend pullback (EMA)",
        "side": "BUY",
        "hold_days": 10,
        "typical_win": "48–58%",
        "typical_exp": "illustrative",
        "desc": "Uptrend (price > EMA20 > EMA50), buy shallow pullback with RSI 45–60.",
        "beginner": "Only buy dips in a clear uptrend. Skip if RSI is already >70.",
        "why": "Trends often resume after healthy pauses.",
    },
    "breakout_vol": {
        "name": "Volume breakout",
        "side": "BUY",
        "hold_days": 8,
        "typical_win": "42–52%",
        "typical_exp": "illustrative",
        "desc": "Close above recent high with volume surge.",
        "beginner": "Breakout needs volume. False breakouts fail back inside the range — use a stop just below the breakout level.",
        "why": "Expansion after compression attracts trend followers.",
    },
    "rsi_reclaim": {
        "name": "RSI reclaim 50",
        "side": "BUY",
        "hold_days": 12,
        "typical_win": "44–54%",
        "typical_exp": "illustrative",
        "desc": "RSI crosses back above 50 while price holds above EMA20.",
        "beginner": "RSI moving from weak to neutral/strong often marks the start of a swing leg.",
        "why": "Momentum regime shift from sellers to buyers.",
    },
    "mean_reversion_oversold": {
        "name": "Oversold bounce",
        "side": "BUY",
        "hold_days": 7,
        "typical_win": "40–50%",
        "typical_exp": "illustrative",
        "desc": "RSI < 35 then turns up near support.",
        "beginner": "Do not catch a falling knife — wait for RSI to turn up and a bullish candle.",
        "why": "Extreme readings often mean-revert in range-bound names.",
    },
    "bear_trend_rally": {
        "name": "Bear rally fade",
        "side": "SELL",
        "hold_days": 8,
        "typical_win": "42–52%",
        "typical_exp": "illustrative",
        "desc": "Downtrend (price < EMA20 < EMA50), RSI bounce into 50–60 then weakens.",
        "beginner": "Only short weak markets. Hard stop above the bounce high.",
        "why": "Rallies in downtrends often fail at moving averages.",
    },
}


def load_my_strategy_params() -> dict:
    default = {
        "lookback": 90,
        "min_price_swing_pct": 3.0,
        "rsi_period": 14,
        "require_confirmation": True,
        "max_days_after_setup": 3,
    }
    try:
        if MY_STRATEGY_PARAMS_FILE.exists():
            data = json.loads(MY_STRATEGY_PARAMS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                default.update(data)
    except Exception:
        pass
    return default


def save_my_strategy_params(p: dict) -> None:
    try:
        MY_STRATEGY_PARAMS_FILE.write_text(json.dumps(p, indent=2), encoding="utf-8")
    except Exception:
        pass


def my_strategy_divergence_ok(df: pd.DataFrame, idx: int, params: dict = None) -> bool:
    """Simple bullish RSI divergence near the end of the series."""
    if df is None or len(df) < 30:
        return False
    params = params or load_my_strategy_params()
    look = int(params.get("lookback", 90) or 90)
    d = df.tail(look).copy()
    if "Close" not in d.columns:
        return False
    if "RSI" not in d.columns:
        try:
            # local RSI
            c = d["Close"].astype(float)
            delta = c.diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = (-delta.clip(upper=0)).rolling(14).mean()
            rs = up / down.replace(0, np.nan)
            d["RSI"] = 100 - (100 / (1 + rs))
        except Exception:
            return False
    try:
        lows_p = d["Low"].astype(float) if "Low" in d.columns else d["Close"].astype(float)
        rsi = d["RSI"].astype(float)
        # last third vs prior third
        n = len(d)
        a, b = d.iloc[: n // 2], d.iloc[n // 2 :]
        if a.empty or b.empty:
            return False
        price_ll = float(lows_p.iloc[n // 2 :].min()) < float(lows_p.iloc[: n // 2].min())
        rsi_hl = float(rsi.iloc[n // 2 :].min()) > float(rsi.iloc[: n // 2].min())
        last_rsi = float(rsi.iloc[-1])
        return bool(price_ll and rsi_hl and 35 <= last_rsi <= 65)
    except Exception:
        return False


def my_strategy_status(df: pd.DataFrame) -> str:
    try:
        ok = my_strategy_divergence_ok(df, len(df) - 1, load_my_strategy_params())
        return "READY" if ok else "WAIT"
    except Exception:
        return "WAIT"


def _strategy_levels(price: float, side: str = "BUY"):
    price = safe_float(price)
    if price <= 0:
        return 0.0, 0.0, 0.0
    if str(side).upper() == "SELL":
        entry = price
        target = round(price * 0.96, 2)
        stop = round(price * 1.025, 2)
    else:
        entry = price
        target = round(price * 1.05, 2)
        stop = round(price * 0.97, 2)
    return entry, target, stop


def _row_matches_strategy(df: pd.DataFrame, sid: str) -> bool:
    """Practical filters — not ultra-strict so the desk is not empty for days."""
    if df is None or df.empty or len(df) < 25:
        return False
    try:
        d = df.tail(120).copy()
        # flatten multiindex columns if any
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        if "Close" not in d.columns:
            return False
        close = pd.to_numeric(d["Close"], errors="coerce").dropna()
        if len(close) < 25:
            return False
        price = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / down.replace(0, np.nan)
        rsi_s = 100 - (100 / (1 + rs))
        rsi = float(rsi_s.iloc[-1]) if pd.notna(rsi_s.iloc[-1]) else 50.0
        vol_r = 1.0
        if "Volume" in d.columns:
            v = pd.to_numeric(d["Volume"], errors="coerce").fillna(0)
            base = float(v.tail(20).mean() or 1)
            vol_r = float(v.iloc[-1] / max(base, 1))

        if sid == "my_strategy":
            if my_strategy_divergence_ok(d, len(d) - 1):
                return True
            # softer alternate: RSI rising while near EMA
            return price >= ema20 * 0.98 and 40 <= rsi <= 60 and float(rsi_s.iloc[-1]) >= float(rsi_s.iloc[-5])
        if sid == "trend_pullback":
            return price > ema20 and ema20 >= ema50 * 0.995 and 42 <= rsi <= 68
        if sid == "breakout_vol":
            hi_series = pd.to_numeric(d["High"], errors="coerce") if "High" in d.columns else close
            hi = float(hi_series.tail(21).iloc[:-1].max())
            return price >= hi * 0.995 and vol_r >= 1.1 and rsi >= 48
        if sid == "rsi_reclaim":
            prev = float(rsi_s.iloc[-5]) if len(rsi_s) > 5 and pd.notna(rsi_s.iloc[-5]) else rsi
            return (prev < 52 <= rsi or (48 <= rsi <= 58 and price > ema20))
        if sid == "mean_reversion_oversold":
            prev = float(rsi_s.iloc[-5]) if len(rsi_s) > 5 and pd.notna(rsi_s.iloc[-5]) else rsi
            return (prev <= 40 and rsi >= prev and rsi <= 55) or rsi <= 38
        if sid == "bear_trend_rally":
            return price < ema20 and ema20 <= ema50 * 1.005 and 45 <= rsi <= 65
        return False
    except Exception:
        return False


def stocks_under_strategy(base: pd.DataFrame, sid: str, max_check: int = 40) -> pd.DataFrame:
    """Alias used by Strategy Lab tabs — same as live_strategy_signals."""
    try:
        return live_strategy_signals(base, sid, max_check=max_check)
    except Exception:
        return pd.DataFrame()


def backtest_strategy_on_df(df: pd.DataFrame, sid: str, meta: dict = None) -> dict:
    """Lightweight swing backtest — educational, not a guarantee."""
    meta = meta or SWING_STRATEGIES.get(sid, {})
    side = str(meta.get("side", "BUY")).upper()
    hold = int(meta.get("hold_days", 10) or 10)
    if df is None or len(df) < 80:
        return {"trades": 0, "wins": 0, "win_rate": 0, "avg_return_pct": 0, "trade_list": []}
    d = df.copy()
    closes = d["Close"].astype(float).values
    trades = []
    i = 60
    while i < len(d) - hold - 1:
        window = d.iloc[: i + 1]
        if _row_matches_strategy(window, sid):
            entry = float(closes[i])
            future = closes[i + 1 : i + 1 + hold]
            if side == "BUY":
                tgt, sl = entry * 1.05, entry * 0.97
                hit_t = any(x >= tgt for x in future)
                hit_s = any(x <= sl for x in future)
                ret = ((tgt / entry) - 1) * 100 if hit_t and (not hit_s or np.argmax([x >= tgt for x in future]) <= np.argmax([x <= sl for x in future] or [0])) else (
                    ((sl / entry) - 1) * 100 if hit_s else ((float(future[-1]) / entry) - 1) * 100
                )
            else:
                tgt, sl = entry * 0.96, entry * 1.025
                hit_t = any(x <= tgt for x in future)
                hit_s = any(x >= sl for x in future)
                ret = ((entry / tgt) - 1) * 100 if hit_t else (((entry / sl) - 1) * 100 if hit_s else ((entry / float(future[-1])) - 1) * 100)
            trades.append({"i": i, "entry": entry, "ret_pct": round(ret, 2), "win": ret > 0})
            i += hold
        else:
            i += 1
    if not trades:
        return {"trades": 0, "wins": 0, "win_rate": 0, "avg_return_pct": 0, "trade_list": []}
    wins = sum(1 for t in trades if t["win"])
    avg = float(np.mean([t["ret_pct"] for t in trades]))
    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": round(100.0 * wins / len(trades), 1),
        "avg_return_pct": round(avg, 2),
        "trade_list": trades[-40:],
    }


def live_strategy_signals(base: pd.DataFrame, sid: str, max_check: int = 35) -> pd.DataFrame:
    """Scan symbols for one strategy; return signal table."""
    meta = SWING_STRATEGIES.get(sid, {})
    side = str(meta.get("side", "BUY")).upper()
    hold = int(meta.get("hold_days", 12) or 12)
    stocks = []
    if base is not None and not base.empty and "Stock" in base.columns:
        stocks = base["Stock"].astype(str).str.replace(".NS", "", regex=False).str.upper().unique().tolist()
    stocks = stocks[: max(5, int(max_check))]
    rows = []
    for sym in stocks:
        try:
            df = stock_history(clean_symbol(sym), period="6mo")
            if df is None or len(df) < 40:
                continue
            if not _row_matches_strategy(df, sid):
                continue
            price = float(df["Close"].astype(float).iloc[-1])
            try:
                live = live_quote(clean_symbol(sym))
                if live and safe_float(live) > 0:
                    price = safe_float(live)
            except Exception:
                pass
            entry, tgt, sl = _strategy_levels(price, side)
            rows.append({
                "Stock": display_symbol(sym),
                "Strategy": meta.get("name", sid),
                "Side": side,
                "Price": round(price, 2),
                "Entry": entry,
                "Target": tgt,
                "Stop Loss": sl,
                "Hold Days": hold,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def strategies_matching_stock(symbol: str, side_filter: str = "BUY") -> dict:
    names, ids = [], []
    try:
        df = stock_history(clean_symbol(symbol), period="6mo")
    except Exception:
        df = None
    for sid, meta in SWING_STRATEGIES.items():
        if side_filter and side_filter != "BOTH":
            if str(meta.get("side", "BUY")).upper() != str(side_filter).upper():
                continue
        try:
            if df is not None and _row_matches_strategy(df, sid):
                names.append(meta.get("name", sid))
                ids.append(sid)
        except Exception:
            pass
    return {"count": len(names), "names": names, "ids": ids}



def show_strategy_lab(results: pd.DataFrame):
    """Everything in one place: learn, edit My Strategy, 5y backtest, live BUY calls, Angel levels."""
    st.title("🧪 Strategy Lab — All-in-one (inside main app)")
    st.caption(
        "One file only: run **app.py**. Here you get strategies, **My Strategy** editor, "
        "backtest efficiency (up to ~5y), live stock signals for Angel One, and beginner lessons. "
        "Efficiency = target hit **before** stop. Not a guarantee of any fixed %."
    )

    try:
        reg = nifty_market_regime()
        if not reg.get("trade_longs", True):
            st.error("Nifty regime **weak** — avoid new long strategy trades today.")
        else:
            st.info(f"Nifty: **{reg.get('regime')}** (score {reg.get('score')})")
    except Exception:
        pass

    default_syms = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
        "ITC", "LT", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
        "TITAN", "WIPRO", "NTPC", "TATAMOTORS", "M&M", "HCLTECH",
    ]
    if results is not None and not results.empty and "Stock" in results.columns:
        scan_syms = (
            results.sort_values("Prediction", ascending=False)["Stock"]
            .astype(str).head(40).tolist()
        )
        universe = tuple(dict.fromkeys(scan_syms + default_syms))
    else:
        universe = tuple(default_syms)

    tab_learn, tab_edit, tab_bt, tab_live, tab_stocks, tab_hi, tab_ao = st.tabs(
        [
            "📘 Learn",
            "✏️ My Strategy",
            "📊 Backtest %",
            "📡 Live BUY by strategy",
            "📋 Stocks under strategy",
            "⭐ High conviction",
            "🏦 Angel One levels",
        ]
    )

    with tab_learn:
        st.subheader("Learn strategies the easy way")
        st.markdown(
            """
### How to find stocks **manually** (TradingView)

1. Open **daily** chart + add **RSI (14)**.  
2. Zoom out **2–6 months** (long timeframe).  
3. Draw a line on **price swing lows** — is it **falling**?  
4. Draw a line on **RSI swing lows** — is it **rising / flat-up**?  
5. RSI line must **not be broken**.  
6. If yes → that is **My Strategy** (same as Robust Hotels / Shiva Mills style).  
7. Wait **confirmation**: next day small dip OK; RSI must still hold.  
8. If RSI breaks its line → **skip**. If price gaps up hard → **don’t chase**.

### All strategies in simple words
            """
        )
        rows = []
        for sid, m in SWING_STRATEGIES.items():
            st.markdown(f"### {m['name']}")
            st.write(m.get("beginner") or m.get("desc"))
            st.caption(
                f"Typical win rate: **{m.get('typical_win', '—')}** · "
                f"Expectancy: **{m.get('typical_exp', '—')}** (illustrative, not a guarantee)"
            )
            rows.append(
                {
                    "Strategy": m["name"],
                    "Typical win rate": m.get("typical_win", "—"),
                    "Expectancy per trade": m.get("typical_exp", "—"),
                }
            )
            st.divider()
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(
            """
### Manual checklist before any trade
- [ ] I can see the pattern on **daily** (or my timeframe) with **RSI**  
- [ ] Lines make sense over **weeks**, not only 2–3 candles  
- [ ] Nifty is not in a crash day (optional filter)  
- [ ] Stop is defined under price swing low  
- [ ] I will not chase a vertical next-day green candle  

### Drawing on charts in this app
- **RSI is on every chart** (Plotly).  
- **MyStrat lines** checkbox draws the model’s blue lines for comparison.  
- **Hand drawing** → button **TradingView (draw)** (full site has trendline tool).
            """
        )

    with tab_edit:
        st.subheader("Edit **My Strategy** (saved in my_strategy_params.json)")
        st.markdown(
            """
**Rules (strict — from your charts):**
1. **Price line falling** (lower lows) — weeks/months OK  
2. **RSI line rising or flat-up** — **must not break**  
3. **Reject** if both lines falling  
4. Call only while pattern is **BUILDING** (0–2 days after the swing lows)  
5. **If pattern already built and 2–3 days passed** (price already up, e.g. late Shiva Mills) → **NOT a buy call**  
6. If price already bounced more than ~3.5% from the pattern low → **too late**
            """
        )
        p = load_my_strategy_params()
        # Clamp old saved JSON values into widget min/max (fixes StreamlitValueBelowMinError)
        lb0 = max(30, min(200, int(p.get("lookback", 60) or 60)))
        rmin0 = max(10, min(50, int(p.get("rsi_min", 25) or 25)))
        rmax0 = max(40, min(70, int(p.get("rsi_max", 58) or 58)))
        if rmax0 < rmin0:
            rmax0 = min(70, rmin0 + 5)
        slope0 = max(-1.0, min(2.0, float(p.get("min_rsi_slope", 0.0) or 0.0)))
        brk0 = max(0.0, min(5.0, float(p.get("rsi_line_break_tol", 0.5) or 0.5)))
        tgt0 = max(0.5, min(4.0, float(p.get("atr_target_mult", 1.8) or 1.8)))
        sl0 = max(0.5, min(4.0, float(p.get("atr_stop_mult", 2.2) or 2.2)))
        hold0 = max(5, min(40, int(p.get("hold_bars", 15) or 15)))

        c1, c2, c3 = st.columns(3)
        with c1:
            p["lookback"] = int(st.number_input("Lookback bars (months OK on daily)", 30, 200, lb0, key="my_lb"))
            p["rsi_min"] = int(st.number_input("RSI min", 10, 50, rmin0, key="my_rmin"))
            p["rsi_max"] = int(st.number_input("RSI max", 40, 70, rmax0, key="my_rmax"))
        with c2:
            p["min_rsi_slope"] = float(st.number_input("Min RSI line slope", -1.0, 2.0, slope0, key="my_slope"))
            p["rsi_line_break_tol"] = float(st.number_input("RSI line break tolerance", 0.0, 5.0, brk0, key="my_brk"))
            p["atr_target_mult"] = float(st.number_input("Target × ATR", 0.5, 4.0, tgt0, key="my_tgt"))
        with c3:
            p["atr_stop_mult"] = float(st.number_input("Stop × ATR", 0.5, 4.0, sl0, key="my_sl"))
            p["hold_bars"] = int(st.number_input("Hold bars/days", 5, 40, hold0, key="my_hold"))
            p["early_signal"] = st.checkbox(
                "Early signal (as pattern builds)",
                value=bool(p.get("early_signal", True)),
                key="my_early",
            )
            p["require_ema200"] = st.checkbox(
                "Require > EMA200",
                value=bool(p.get("require_ema200")),
                key="my_e200",
            )
        p["notes"] = st.text_area("Notes", value=str(p.get("notes", "")), key="my_notes")
        if st.button("💾 Save My Strategy", type="primary", key="my_save"):
            save_my_strategy_params(p)
            st.success(f"Saved → `{MY_STRATEGY_PARAMS_FILE.name}`")

        st.subheader("🧠 Train My Strategy from your charts (RSI divergence)")
        st.caption(
            "Your template: **price lower lows + RSI higher lows** (RELIANCE / COALINDIA style). "
            "Save every good/bad example — the model adjusts **RSI_BULL_DIV** weight on next learning pass."
        )
        st.markdown(
            """
**What the model learns from your images**

| Chart feature | Rule stored |
|---------------|-------------|
| Price swing lows falling | Required |
| RSI at **same bars** rising / flat | Required (bullish divergence) |
| Point **D** = second low completion | Entry window starts |
| 2–3 days after big bounce | **TOO_LATE** — not a buy |
| RSI breaks rising support line | **RSI_FAILED** — no buy |
            """
        )
        with st.form("my_strat_train_form"):
            t_sym = st.text_input("Symbol", value="RELIANCE")
            t_outcome = st.selectbox(
                "Outcome",
                [
                    "ACTIVE_FRESH good",
                    "Bullish RSI divergence confirmed",
                    "TARGET HIT",
                    "TOO_LATE",
                    "RSI_FAILED",
                    "STOP HIT",
                    "Should not have bought",
                ],
            )
            t_note = st.text_area(
                "What you saw (lines / days / RSI)",
                value="Price lower low + RSI higher low (bullish divergence). Entry near second low confirmation.",
            )
            t_img = st.file_uploader("Optional chart image (png/jpg)", type=["png", "jpg", "jpeg"])
            t_sub = st.form_submit_button("Save training example + update learning")
        if t_sub:
            import datetime as _dt
            row = {
                "time": _dt.datetime.now().isoformat(timespec="seconds"),
                "symbol": (t_sym or "").upper().strip(),
                "outcome": t_outcome,
                "note": t_note or "",
                "image_saved": "",
            }
            train_dir = APP_DIR / "my_strategy_train"
            train_dir.mkdir(exist_ok=True)
            if t_img is not None:
                img_path = train_dir / f"{row['symbol']}_{row['time'].replace(':', '-')}.{t_img.name.split('.')[-1]}"
                img_path.write_bytes(t_img.getvalue())
                row["image_saved"] = str(img_path.name)
            train_csv = APP_DIR / "my_strategy_train_log.csv"
            try:
                if train_csv.exists():
                    _tdf = pd.read_csv(train_csv)
                    _tdf = pd.concat([_tdf, pd.DataFrame([row])], ignore_index=True)
                else:
                    _tdf = pd.DataFrame([row])
                _tdf.to_csv(train_csv, index=False)
                # Refresh learning weights immediately
                try:
                    learn_from_history(min_closed=1)
                except Exception:
                    pass
                st.success(
                    f"Saved example → `{train_csv.name}`. "
                    f"Learning updated (RSI_BULL_DIV weight refreshed)."
                )
            except Exception as _e:
                st.error(str(_e))

        # One-click seed from your two chart types
        if st.button("📌 Load RELIANCE + COALINDIA divergence as training templates", key="seed_div_charts"):
            train_csv = APP_DIR / "my_strategy_train_log.csv"
            import datetime as _dt
            seeds = [
                {
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                    "symbol": "RELIANCE",
                    "outcome": "Bullish RSI divergence confirmed",
                    "note": "Daily: price LL into Jun pivot, RSI HL; classic bullish divergence template",
                    "image_saved": "user_chart_reliance",
                },
                {
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                    "symbol": "COALINDIA",
                    "outcome": "Bullish RSI divergence confirmed",
                    "note": "Daily after sell: price LL + RSI HL then recovery; buy on confirmation",
                    "image_saved": "user_chart_coalindia",
                },
            ]
            try:
                if train_csv.exists():
                    _tdf = pd.read_csv(train_csv)
                    _tdf = pd.concat([_tdf, pd.DataFrame(seeds)], ignore_index=True)
                else:
                    _tdf = pd.DataFrame(seeds)
                _tdf.to_csv(train_csv, index=False)
                learn_from_history(min_closed=1)
                st.success("Templates saved + learning refreshed. Model will favour this divergence shape.")
            except Exception as e:
                st.error(str(e))


        st.divider()
        st.subheader("📈 My Strategy — chart check (TradingView + analysis)")
        st.markdown(
            """
**How your pattern works (from Robust Hotels / Shiva Mills style):**

| Phase | Price | RSI | What to do |
|-------|--------|-----|------------|
| **Building** | Mild lower lows (blue line down) | Higher lows (blue line up) | Watch — pattern forming |
| **Early call** | Line still down, not collapsed | RSI line **not broken** | Optional small entry / alert |
| **Next day** | Often one more soft dip | RSI still holds / rises | **Confirmation** day |
| **Complete** | Bounce starts | RSI holds above its line | Main swing entry |
| **Invalid** | — | RSI **breaks** its rising line | **No trade** |

**IMPORTANT — Shiva Mills rule:**  
Only take calls when the pattern is **ready to build / just formed**.  
If the pattern **already built and 2–3 days have passed** (or price already bounced) → **NOT a buy call** (stock will correctly **disappear** from My Strategy list).
            """
        )
        sym_my = st.text_input(
            "NSE symbol to analyse on chart",
            value="SHIVAMILLS",
            key="my_strat_chart_sym",
        ).upper().strip()
        # Always show chart when symbol typed (not only after button)
        chart_sym = display_symbol(sym_my) if sym_my else ""
        if st.button("Analyse + show chart", type="primary", key="my_chart_btn"):
            st.session_state["my_chart_sym"] = chart_sym
        chart_sym = st.session_state.get("my_chart_sym") or chart_sym
        if chart_sym:
            st.write(f"**{chart_sym}** — chart with RSI + model blue lines")
            raw = pd.DataFrame()
            try:
                raw = stock_history(clean_symbol(chart_sym), interval="1d")
            except Exception:
                raw = pd.DataFrame()
            if raw is None or raw.empty:
                try:
                    import yfinance as yf
                    raw = yf.download(
                        f"{display_symbol(chart_sym)}.NS",
                        period="1y",
                        interval="1d",
                        progress=False,
                        auto_adjust=True,
                    )
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = [c[0] for c in raw.columns]
                    raw = normalize_columns(raw)
                except Exception as e:
                    st.warning(f"Could not load data: {e}")
                    raw = pd.DataFrame()
            if raw is not None and not raw.empty:
                try:
                    raw = calculate_indicators(raw)
                except Exception:
                    pass
                pp = load_my_strategy_params()
                try:
                    fig = build_full_plotly_chart(
                        raw,
                        title=f"{chart_sym} — RSI + My Strategy lines",
                        height=720,
                        show_rsi=True,
                        my_strategy_lines=True,
                        lookback_lines=int(pp.get("lookback", 90)),
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key=f"myfig_{chart_sym}")
                    else:
                        st.warning("Chart figure empty — try TradingView link.")
                except Exception as e:
                    st.warning(f"Plotly chart error: {e}")
                    try:
                        show_tradingview_chart(clean_symbol(chart_sym), chart_sym, height=560)
                    except Exception:
                        pass
                try:
                    tv_sym = _to_tv_symbol(chart_sym)
                    st.link_button(
                        "🔗 TradingView (draw your own lines)",
                        f"https://www.tradingview.com/chart/?symbol={tv_sym}",
                    )
                except Exception:
                    pass
                try:
                    st_status = my_strategy_status(raw)
                    c = safe_float(raw["Close"].iloc[-1])
                    rsi_v = safe_float(raw["RSI"].iloc[-1], 50) if "RSI" in raw.columns else 50
                    atr_v = safe_float(raw["ATR"].iloc[-1], 0) if "ATR" in raw.columns else c * 0.02
                    if atr_v <= 0:
                        atr_v = c * 0.02
                    tgt = c + float(pp.get("atr_target_mult", 1.8)) * atr_v
                    sl = c - float(pp.get("atr_stop_mult", 2.2)) * atr_v
                    if st_status["status"] == "ACTIVE_FRESH":
                        st.success(f"**{st_status['status']}** — {st_status['reason']}")
                    elif st_status["status"] in ("TOO_LATE", "RSI_FAILED"):
                        st.error(f"**{st_status['status']}** — {st_status['reason']}")
                    else:
                        st.warning(f"**{st_status['status']}** — {st_status['reason']}")
                    st.markdown(
                        f"**Levels (only if ACTIVE_FRESH):** ₹{c:,.2f} → T ₹{tgt:,.2f} / SL ₹{sl:,.2f} · RSI {rsi_v:.1f}"
                    )
                except Exception as e:
                    st.caption(f"Status note: {e}")
            else:
                st.error("No price data for this symbol.")

        st.divider()
        st.subheader("Nifty / Bank Nifty — Call & Put (same My Strategy idea)")
        st.markdown(
            """
My Strategy is **price vs RSI divergence** — it also applies to **index futures/options direction**:

| Index view | Option idea (educational) |
|------------|----------------------------|
| **Nifty/BankNifty**: price lower-lows, RSI higher-lows, RSI line holds | Bias **CALL** (bullish bounce) after confirmation |
| Price higher-highs, RSI lower-highs (bearish divergence) | Bias **PUT** (not coded as My Strategy long; opposite pattern) |
| RSI line breaks | **No trade** on options either |

**How to use:**  
1. Open NIFTY / BANKNIFTY on My Strategy chart check (symbol `^NSEI` or use index pages).  
2. Same blue-line rules.  
3. If bullish My Strategy on index → prefer **CE** only after confirmation day; strike near ATM, defined SL.  
4. This app does **not** auto-trade option chain orders — levels are directional bias only.  
5. Options decay fast — prefer confirmation + tight invalidation (RSI line break = exit bias).
            """
        )
        ix = st.selectbox("Index for My Strategy bias", ["NIFTY 50", "BANK NIFTY"], key="my_opt_idx")
        if st.button("Check index My Strategy", key="my_opt_btn"):
            yf_i = "^NSEI" if "NIFTY 50" in ix else "^NSEBANK"
            try:
                raw = stock_history(yf_i, interval="1d")
                df = calculate_indicators(raw)
                ok = my_strategy_divergence_ok(df, len(df) - 1, load_my_strategy_params())
                fig = build_full_plotly_chart(
                    df, title=ix, height=650, show_rsi=True, my_strategy_lines=True,
                    lookback_lines=int(load_my_strategy_params().get("lookback", 90)),
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                if ok:
                    st.success(f"**Bullish My Strategy bias on {ix}** → educational CALL bias after confirmation.")
                else:
                    st.warning(f"No strict bullish My Strategy on {ix} right now.")
            except Exception as e:
                st.error(str(e))

    with tab_bt:
        st.subheader("Backtest efficiency (your data)")
        st.caption("Use scan stocks or type a symbol. Period up to **5y** where Yahoo has history.")
        mode = st.radio("Mode", ["All strategies on universe", "One symbol deep dive"], horizontal=True, key="bt_mode")
        max_sym = st.slider("Universe size", 8, 40, 20, key="strat_max_sym")
        if mode.startswith("All"):
            if st.button("▶ Run all strategy backtests", type="primary", key="strat_run_bt"):
                summary_rows = []
                with st.spinner("Backtesting… (can take a few minutes)"):
                    for sid in SWING_STRATEGIES:
                        try:
                            agg = backtest_strategy_universe(sid, universe, max_symbols=max_sym)
                            st.session_state.setdefault("strategy_bt_detail", {})[sid] = agg
                            summary_rows.append(
                                {
                                    "Strategy": agg["name"],
                                    "Side": agg["side"],
                                    "Symbols": agg["symbols_tested"],
                                    "Trades": agg["trades"],
                                    "Wins": agg["wins"],
                                    "Losses": agg["losses"],
                                    "Efficiency %": agg["win_rate"],
                                    "Avg Return %": agg["avg_return_pct"],
                                    "Typical ref": SWING_STRATEGIES[sid].get("typical_win", ""),
                                    "ID": sid,
                                }
                            )
                        except Exception as e:
                            summary_rows.append(
                                {
                                    "Strategy": SWING_STRATEGIES[sid]["name"],
                                    "Efficiency %": 0,
                                    "Trades": 0,
                                    "Wins": 0,
                                    "Losses": 0,
                                    "Error": str(e)[:60],
                                    "ID": sid,
                                }
                            )
                st.session_state["strategy_bt_cache"] = pd.DataFrame(summary_rows)
                try:
                    st.session_state["strategy_bt_cache"].to_csv(STRATEGY_BT_FILE, index=False)
                except Exception:
                    pass
            summary = st.session_state.get("strategy_bt_cache", pd.DataFrame())
            if summary is not None and not summary.empty:
                show = summary.sort_values("Efficiency %", ascending=False)
                st.dataframe(show.drop(columns=["ID"], errors="ignore"), use_container_width=True, hide_index=True)
                best = show.iloc[0]
                st.success(
                    f"**Best in this run:** {best.get('Strategy')} — **{best.get('Efficiency %')}%** "
                    f"({int(best.get('Wins', 0))}W / {int(best.get('Losses', 0))}L). "
                    "Prefer this for live trades if sample is large enough."
                )
            else:
                st.info("Click **Run all strategy backtests** after a market scan for best universe.")
        else:
            sym = st.text_input("NSE symbol", value="RELIANCE", key="bt_one_sym").upper().strip()
            sid = st.selectbox(
                "Strategy",
                list(SWING_STRATEGIES.keys()),
                format_func=lambda k: SWING_STRATEGIES[k]["name"],
                key="bt_one_sid",
            )
            if st.button("▶ Backtest this symbol", type="primary", key="bt_one_btn"):
                with st.spinner("Loading history (~5y) & backtesting…"):
                    try:
                        raw = stock_history(clean_symbol(sym), interval="1d")
                        if raw is None or len(raw) < 80:
                            # try longer via yfinance period
                            import yfinance as yf

                            raw = yf.download(
                                f"{display_symbol(sym)}.NS",
                                period="5y",
                                interval="1d",
                                progress=False,
                                auto_adjust=True,
                            )
                            if isinstance(raw.columns, pd.MultiIndex):
                                raw.columns = [c[0] for c in raw.columns]
                        df = calculate_indicators(raw) if raw is not None and not raw.empty else pd.DataFrame()
                        res = backtest_strategy_on_df(df, sid, SWING_STRATEGIES.get(sid))
                        a, b, c, d = st.columns(4)
                        a.metric("Efficiency %", f"{res.get('win_rate', 0)}%")
                        b.metric("Trades", res.get("trades", 0))
                        c.metric("Wins", res.get("wins", 0))
                        d.metric("Losses", res.get("losses", 0))
                        st.caption(f"Avg return {res.get('avg_return_pct')}% · {SWING_STRATEGIES[sid].get('beginner','')}")
                        if res.get("trade_list"):
                            st.dataframe(pd.DataFrame(res["trade_list"]).tail(40), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(str(e))

    with tab_live:
        st.subheader("Live stock signals by strategy (BUY calls for today)")
        st.caption("Uses last scan universe when available. Run FULL MARKET SCAN first for more names.")
        pick = st.selectbox(
            "Strategy",
            list(SWING_STRATEGIES.keys()),
            format_func=lambda k: f"{SWING_STRATEGIES[k]['name']} ({SWING_STRATEGIES[k]['side']})",
            key="live_strat_pick",
        )
        # Detailed explanation for selected strategy
        meta_pick = SWING_STRATEGIES.get(pick, {})
        with st.expander(f"📖 Learn this strategy in detail — {meta_pick.get('name', pick)}", expanded=True):
            st.write(meta_pick.get("beginner") or meta_pick.get("desc"))
            st.write(f"**Why it can work:** {meta_pick.get('why', '')}")
            st.write(f"**Side:** {meta_pick.get('side')} · **Hold ~** {meta_pick.get('hold_days')} days")
            st.caption(
                f"Typical win rate {meta_pick.get('typical_win', '—')} · "
                f"Expectancy {meta_pick.get('typical_exp', '—')} (illustrative)"
            )
            if pick == "my_strategy":
                st.warning(
                    "**My Strategy rule:** only patterns **ready to build / just forming**. "
                    "If like Shiva Mills the pattern already built and **2–3 days passed** with price up → "
                    "**not a buy call** (too late)."
                )
        if st.button("Generate live signals", type="primary", key="live_strat_btn"):
            with st.spinner("Scanning live bars…"):
                base = results if results is not None and not results.empty else pd.DataFrame({"Stock": list(universe)})
                live_df = live_strategy_signals(base, pick, max_check=35)
                # attach how many strategies each stock matches
                if live_df is not None and not live_df.empty:
                    counts, names_list = [], []
                    for _, rr in live_df.iterrows():
                        m = strategies_matching_stock(str(rr["Stock"]), side_filter="BUY")
                        counts.append(m["count"])
                        names_list.append(", ".join(m["names"]))
                    live_df = live_df.copy()
                    live_df["# Strategies"] = counts
                    live_df["All matching strategies"] = names_list
                st.session_state["live_strat_df"] = live_df
                n_saved = 0
                snap = {"n": 0, "n_new": 0, "new_stocks": []}
                try:
                    if live_df is not None and not live_df.empty:
                        snap = save_strategy_snapshot(live_df)
                        special = []
                        for _, lr in live_df.iterrows():
                            special.append({
                                "Stock": lr.get("Stock"),
                                "Call": lr.get("Side", "BUY"),
                                "Call Source": "STRATEGY",
                                "Strategy": lr.get("Strategy", ""),
                                "Entry": lr.get("Price"),
                                "Target": lr.get("Target"),
                                "Stop Loss": lr.get("Stop Loss"),
                                "Hold Days": lr.get("Hold Days", 15),
                                "Prediction": 78,
                                "Reason": f"Strategy: {lr.get('Strategy', '')}",
                            })
                        n_saved = save_special_calls(special)
                except Exception as e:
                    st.warning(f"Could not save strategy calls to history: {e}")
                if snap.get("n"):
                    st.success(
                        f"Shared on all devices: **{snap['n']}** strategy stocks · "
                        f"**{snap.get('n_new', 0)}** new this update"
                    )
                    if snap.get("new_stocks"):
                        st.info("New only: " + ", ".join(snap["new_stocks"][:20]))
                if n_saved:
                    st.success(
                        f"Saved **{n_saved}** to Past Predictions (STRATEGY) — visible on every device after reload."
                    )
                elif live_df is not None and not live_df.empty:
                    st.info("Signals ready (already in history today, or none new).")
        # Always try load shared list (phone → PC)
        if st.button("📥 Load strategy list from server", key="load_strat_shared"):
            loaded = load_strategy_snapshot(force=True)
            if loaded is not None and not loaded.empty:
                st.success(f"Loaded **{len(loaded)}** strategy stocks from server.")
                st.rerun()
            else:
                st.warning("No shared strategy file yet — generate live signals once.")
        live_df = load_strategy_snapshot(force=False)
        if live_df is None or (isinstance(live_df, pd.DataFrame) and live_df.empty):
            live_df = st.session_state.get("live_strat_df", pd.DataFrame())
        if live_df is not None and not live_df.empty:
            only_new = st.checkbox("Show only newly added stocks", value=False, key="strat_only_new")
            view_live = strategy_new_only(live_df) if only_new else live_df
            if only_new:
                st.caption(f"**{len(view_live)}** new stocks since last shared update (of {len(live_df)} total).")
                live_df = view_live if not view_live.empty else live_df
            if st.button("💾 Save these strategy signals to Past Predictions", key="strat_resave"):
                special = []
                for _, lr in live_df.iterrows():
                    special.append({
                        "Stock": lr.get("Stock"),
                        "Call": lr.get("Side", "BUY"),
                        "Call Source": "STRATEGY",
                        "Strategy": lr.get("Strategy", ""),
                        "Entry": lr.get("Price"),
                        "Target": lr.get("Target"),
                        "Stop Loss": lr.get("Stop Loss"),
                        "Hold Days": lr.get("Hold Days", 15),
                        "Prediction": 78,
                    })
                n_saved = save_special_calls(special)
                st.success(f"Wrote {n_saved} STRATEGY row(s) to history.")
            st.subheader("📋 Unique stocks (all strategies listed once)")
            render_unique_strategy_stock_cards(live_df, max_cards=12, key_prefix="live_usc")
            with st.expander("Raw signal table (optional)", expanded=False):
                st.dataframe(live_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No live hits yet — normal when filters are strict.")

    with tab_stocks:
        st.subheader("Which stocks fall under strategies")
        st.caption(
            "List **one strategy** or **all strategies at once**. "
            "Uses last scan universe when available."
        )
        mode = st.radio(
            "Mode",
            ["All strategies (full map)", "One strategy only"],
            horizontal=True,
            key="stocks_under_mode",
        )
        base = results if results is not None and not results.empty else pd.DataFrame({"Stock": list(universe)})
        max_check = st.slider("Max stocks to check per strategy", 15, 80, 35, key="stocks_under_max")

        if mode.startswith("All strategies"):
            st.info(
                "Builds a map once, then **keeps it** on this device (and server). "
                "Only click Refresh when you want a new scan — not every visit."
            )
            side_f = st.selectbox("Side", ["BUY", "SELL", "BOTH"], key="all_strat_side")

            # Load cached map from session or disk (same device / other device)
            if "stocks_under_all_df" not in st.session_state or st.session_state.get("stocks_under_all_df") is None:
                try:
                    if STRATEGY_ALL_MAP_FILE.exists():
                        st.session_state["stocks_under_all_df"] = pd.read_csv(STRATEGY_ALL_MAP_FILE)
                    if STRATEGY_ALL_SUMMARY_FILE.exists():
                        st.session_state["stocks_under_all_summary"] = pd.read_csv(STRATEGY_ALL_SUMMARY_FILE)
                    if STRATEGY_ALL_META_FILE.exists():
                        st.session_state["stocks_under_all_meta"] = json.loads(
                            STRATEGY_ALL_META_FILE.read_text(encoding="utf-8")
                        )
                except Exception:
                    pass

            cached_big = st.session_state.get("stocks_under_all_df")
            has_cache = isinstance(cached_big, pd.DataFrame) and not cached_big.empty
            meta_all = st.session_state.get("stocks_under_all_meta") or {}
            if has_cache:
                st.success(
                    f"Saved map ready: **{len(cached_big)}** rows · "
                    f"{meta_all.get('saved_at_ist', 'earlier')} — no need to scan again"
                )

            c_run, c_refresh = st.columns(2)
            with c_run:
                run_all = st.button(
                    "List stocks under ALL strategies" if not has_cache else "Show saved map",
                    type="primary",
                    key="list_all_strat_btn",
                    help="Uses saved results if available; does not re-scan unless empty",
                )
            with c_refresh:
                force_rescan = st.button(
                    "🔄 Refresh scan (re-run)",
                    key="list_all_strat_refresh",
                    help="Only when you want a fresh scan",
                )

            # Show-only: if cache exists and user clicked primary, just stay on cache
            do_scan = force_rescan or (run_all and not has_cache)

            if do_scan:
                all_rows = []
                summary = []
                buy_ids = [
                    k for k, v in SWING_STRATEGIES.items()
                    if side_f == "BOTH" or str(v.get("side", "BUY")).upper() == side_f
                ]
                prog = st.progress(0.0)
                with st.spinner("Scanning all strategies…"):
                    for i, sid in enumerate(buy_ids):
                        try:
                            sdf = stocks_under_strategy(base, sid, max_check=max_check)
                            if sdf is not None and not sdf.empty:
                                sdf = sdf.copy()
                                sdf["Strategy ID"] = sid
                                if "Strategy" not in sdf.columns:
                                    sdf["Strategy"] = SWING_STRATEGIES[sid]["name"]
                                all_rows.append(sdf)
                                summary.append({
                                    "Strategy": SWING_STRATEGIES[sid]["name"],
                                    "Side": SWING_STRATEGIES[sid].get("side", "BUY"),
                                    "Matches": len(sdf),
                                    "Sample stocks": ", ".join(
                                        sdf["Stock"].astype(str).head(8).tolist()
                                    ),
                                })
                            else:
                                summary.append({
                                    "Strategy": SWING_STRATEGIES[sid]["name"],
                                    "Side": SWING_STRATEGIES[sid].get("side", "BUY"),
                                    "Matches": 0,
                                    "Sample stocks": "—",
                                })
                        except Exception as _se:
                            summary.append({
                                "Strategy": SWING_STRATEGIES.get(sid, {}).get("name", sid),
                                "Side": SWING_STRATEGIES.get(sid, {}).get("side", "—"),
                                "Matches": 0,
                                "Sample stocks": f"err: {type(_se).__name__}",
                            })
                        prog.progress((i + 1) / max(len(buy_ids), 1))
                if all_rows:
                    big = pd.concat(all_rows, ignore_index=True)
                    try:
                        counts, alln = [], []
                        for _, rr in big.iterrows():
                            m = strategies_matching_stock(
                                str(rr["Stock"]),
                                side_filter="BUY" if side_f != "SELL" else "SELL",
                            )
                            counts.append(m.get("count", 0))
                            alln.append(", ".join(m.get("names") or []))
                        big["# Strategies matched"] = counts
                        big["Also matches"] = alln
                    except Exception:
                        pass
                    st.session_state["stocks_under_all_df"] = big
                    st.session_state["stocks_under_all_summary"] = pd.DataFrame(summary)
                    meta = {
                        "rows": int(len(big)),
                        "side": side_f,
                        "saved_at_ist": india_now().strftime("%Y-%m-%d %H:%M:%S IST"),
                    }
                    st.session_state["stocks_under_all_meta"] = meta
                    try:
                        big.to_csv(STRATEGY_ALL_MAP_FILE, index=False)
                        pd.DataFrame(summary).to_csv(STRATEGY_ALL_SUMMARY_FILE, index=False)
                        STRATEGY_ALL_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                        bump_sync_version("strategy_all_map")
                    except Exception:
                        pass
                    st.success(f"Map saved — **{len(big)}** rows. Reopen tab anytime without scanning again.")
                else:
                    st.session_state["stocks_under_all_df"] = pd.DataFrame()
                    st.session_state["stocks_under_all_summary"] = pd.DataFrame(summary)
                    st.warning("No matches across strategies with current universe/limits.")

            summary_df = st.session_state.get("stocks_under_all_summary")
            big = st.session_state.get("stocks_under_all_df")
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                st.markdown("**Summary — matches per strategy**")
                st.dataframe(
                    summary_df.sort_values("Matches", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            if isinstance(big, pd.DataFrame) and not big.empty:
                st.success(f"**{len(big)}** raw strategy–stock rows → merged into unique stock cards below")
                names = ["ALL"] + sorted(big["Strategy"].astype(str).unique().tolist())
                pick_name = st.selectbox("Filter map by strategy (optional)", names, key="all_map_filter")
                view = big if pick_name == "ALL" else big[big["Strategy"].astype(str) == pick_name]
                st.subheader("📋 Unique stocks — all strategies on one card")
                render_unique_strategy_stock_cards(view, max_cards=15, key_prefix="all_usc")
                with st.expander("Raw multi-row table + CSV", expanded=False):
                    st.dataframe(view, use_container_width=True, hide_index=True)
                    try:
                        st.download_button(
                            "⬇️ Download CSV",
                            data=view.to_csv(index=False).encode("utf-8"),
                            file_name=f"all_strategies_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="dl_all_strat_stocks",
                        )
                    except Exception:
                        pass
            else:
                st.caption("No saved map yet — click **List stocks under ALL strategies** once.")

        else:
            # One strategy only (original behaviour)
            sid_s = st.selectbox(
                "Strategy",
                list(SWING_STRATEGIES.keys()),
                format_func=lambda k: SWING_STRATEGIES[k]["name"],
                key="stocks_under_select",
            )
            meta_s = SWING_STRATEGIES[sid_s]
            with st.expander(f"📖 Detailed explanation — {meta_s['name']}", expanded=True):
                st.write(meta_s.get("beginner") or meta_s.get("desc"))
                st.write(f"**Why:** {meta_s.get('why', '')}")
                st.write(f"**Hold ~{meta_s.get('hold_days')} days** · Typical win {meta_s.get('typical_win')}")
                if sid_s == "my_strategy":
                    st.error(
                        "STRICT: Building patterns only. "
                        "Shiva Mills–style (pattern already done + 2–3 days up) = **NOT a buy**."
                    )
            det = (st.session_state.get("strategy_bt_detail") or {}).get(sid_s)
            if det:
                st.info(
                    f"Last backtest for **{det.get('name')}**: efficiency **{det.get('win_rate')}%** · "
                    f"{det.get('wins')}W / {det.get('losses')}L · {det.get('trades')} trades"
                )
            if st.button("List stocks under this strategy", type="primary", key="list_under_btn"):
                with st.spinner("Matching stocks…"):
                    sdf = stocks_under_strategy(base, sid_s, max_check=max_check)
                    if sdf is not None and not sdf.empty:
                        counts, alln = [], []
                        for _, rr in sdf.iterrows():
                            m = strategies_matching_stock(str(rr["Stock"]), side_filter="BUY")
                            counts.append(m["count"])
                            alln.append(", ".join(m["names"]))
                        sdf = sdf.copy()
                        sdf["# Strategies matched"] = counts
                        sdf["Also matches"] = alln
                    st.session_state["stocks_under_df"] = sdf
                    st.session_state["stocks_under_sid_saved"] = sid_s
            sdf = st.session_state.get("stocks_under_df", pd.DataFrame())
            if (
                sdf is not None
                and not sdf.empty
                and st.session_state.get("stocks_under_sid_saved") == sid_s
            ):
                st.success(f"**{len(sdf)}** hits under **{SWING_STRATEGIES[sid_s]['name']}** → unique cards")
                render_unique_strategy_stock_cards(sdf, max_cards=12, key_prefix="one_usc")
                with st.expander("Raw table", expanded=False):
                    st.dataframe(sdf, use_container_width=True, hide_index=True)
            else:
                st.caption("Click the button after a market scan for more names.")

    with tab_hi:
        st.subheader("⭐ High conviction — multi-strategy merge")
        st.caption(
            "Stocks matching **2+ strategies**. Chart + full strategy details + "
            "simple merge backtest (how often target hit when those strategies agreed)."
        )
        st.markdown(
            """
**What to buy? (priority order)**

| Priority | Source | When to use |
|----------|--------|-------------|
| **1 Best** | **High conviction** (2+ strategies) + especially **My Strategy** | Strongest overlap |
| **2** | **Sure Call Desk** | Ultra-strict two-stage + Nifty OK |
| **3** | **My Strategy** alone (ACTIVE_FRESH) | Your divergence rule only |
| **4** | **Precision picks** | Ranked scan quality, still selective |
| **5** | Plain BUY scan list | Weakest — use only with extra filters |

If **My Strategy** is ACTIVE_FRESH **and** another strategy matches → prefer that name.
            """
        )
        if st.button("Find high conviction", type="primary", key="hi_conv"):
            buy_ids = [k for k, v in SWING_STRATEGIES.items() if v.get("side") == "BUY"]
            hits = {}  # stock -> {ids, names, details}
            with st.spinner("Scanning strategies + charts…"):
                for stock in list(universe)[:30]:
                    matched_ids = []
                    matched_names = []
                    try:
                        raw = stock_history(clean_symbol(stock), interval="1d")
                        if raw is None or len(raw) < 80:
                            continue
                        df = calculate_indicators(raw)
                        i = len(df) - 1
                        for sid in buy_ids:
                            if _strategy_signal_at_bar(df, i, sid):
                                matched_ids.append(sid)
                                matched_names.append(SWING_STRATEGIES[sid]["name"])
                    except Exception:
                        continue
                    if len(matched_ids) >= 2:
                        hits[display_symbol(stock)] = {
                            "ids": matched_ids,
                            "names": matched_names,
                        }
            st.session_state["hi_conv_hits"] = hits
            # Auto-save high conviction to Past Predictions
            try:
                special = []
                for stock, info in (hits or {}).items():
                    names = info.get("names") if isinstance(info, dict) else info
                    live_px = tgt = sl = 0.0
                    hold_d = 15
                    try:
                        raw0 = stock_history(clean_symbol(stock), interval="1d")
                        if raw0 is not None and not raw0.empty:
                            df0 = calculate_indicators(raw0)
                            row0 = df0.iloc[-1]
                            live_px = safe_float(row0.get("Close"))
                            atr0 = safe_float(row0.get("ATR")) or live_px * 0.02
                            try:
                                q = live_quote(clean_symbol(stock))
                                if q:
                                    lp = safe_float(q.get("price") or q.get("Price") or q.get("last"), 0)
                                    if lp > 0:
                                        live_px = lp
                            except Exception:
                                pass
                            pp = load_my_strategy_params()
                            tgt = live_px + float(pp.get("atr_target_mult", 1.8)) * atr0
                            sl = live_px - float(pp.get("atr_stop_mult", 2.2)) * atr0
                            hold_d = int(pp.get("hold_bars", 15))
                    except Exception:
                        pass
                    special.append({
                        "Stock": stock,
                        "Call": "BUY",
                        "Call Source": "HIGH_CONV",
                        "Strategy": " + ".join(names) if names else "High conviction",
                        "Entry": live_px,
                        "Target": tgt,
                        "Stop Loss": sl,
                        "Hold Days": hold_d,
                        "Prediction": 88,
                        "Reason": f"High conviction: {', '.join(names or [])}",
                    })
                n_hc = save_special_calls(special)
                if n_hc:
                    st.success(
                        f"Saved **{n_hc}** high-conviction call(s) to Past Predictions "
                        f"(Call Source = **HIGH_CONV**)."
                    )
            except Exception as e:
                st.caption(f"High-conviction history save note: {e}")
        hits = st.session_state.get("hi_conv_hits") or {}
        if hits:
            if st.button("💾 Save high conviction to Past Predictions again", key="hc_resave"):
                special = []
                for stock, info in hits.items():
                    names = info.get("names") if isinstance(info, dict) else info
                    special.append({
                        "Stock": stock,
                        "Call": "BUY",
                        "Call Source": "HIGH_CONV",
                        "Strategy": " + ".join(names) if names else "High conviction",
                        "Entry": 0,
                        "Target": 0,
                        "Stop Loss": 0,
                        "Hold Days": 15,
                        "Prediction": 88,
                    })
                st.success(f"Wrote {save_special_calls(special)} HIGH_CONV row(s).")
            for stock, info in hits.items():
                names = info.get("names") if isinstance(info, dict) else info
                ids = info.get("ids") if isinstance(info, dict) else []
                st.markdown("---")
                st.success(f"**{stock}** — **{len(names)} strategies merge:** {', '.join(names)}")
                # Live / last price + Target + Stop (works market open or closed)
                live_px = tgt = sl = 0.0
                hold_d = 15
                try:
                    raw0 = stock_history(clean_symbol(stock), interval="1d")
                    if raw0 is not None and not raw0.empty:
                        df0 = calculate_indicators(raw0)
                        row0 = df0.iloc[-1]
                        live_px = safe_float(row0.get("Close"))
                        atr0 = safe_float(row0.get("ATR")) or live_px * 0.02
                        try:
                            q = live_quote(clean_symbol(stock))
                            if q:
                                lp = safe_float(q.get("price") or q.get("Price") or q.get("last"), 0)
                                if lp > 0:
                                    live_px = lp
                        except Exception:
                            pass
                        pp = load_my_strategy_params()
                        tgt = live_px + float(pp.get("atr_target_mult", 1.8)) * atr0
                        sl = live_px - float(pp.get("atr_stop_mult", 2.2)) * atr0
                        hold_d = int(pp.get("hold_bars", 15))
                except Exception:
                    pass
                a, b, c, d = st.columns(4)
                a.metric("Current / Live Price", f"₹{live_px:,.2f}" if live_px else "—")
                b.metric("Target", f"₹{tgt:,.2f}" if tgt else "—")
                c.metric("Stop Loss", f"₹{sl:,.2f}" if sl else "—")
                d.metric("Hold (days)", str(hold_d))
                st.markdown(
                    f"**{stock}** · Live **₹{live_px:,.2f}** · Target **₹{tgt:,.2f}** · "
                    f"Stop Loss **₹{sl:,.2f}** · (last close used if market closed)"
                )
                # Strategy details
                for sid in (ids or []):
                    meta = SWING_STRATEGIES.get(sid, {})
                    with st.expander(f"📖 {meta.get('name', sid)}", expanded=False):
                        st.write(meta.get("beginner") or meta.get("desc"))
                        st.write(f"**Why:** {meta.get('why', '')}")
                        st.caption(
                            f"Hold ~{meta.get('hold_days')}d · Typical win {meta.get('typical_win')} · "
                            f"{meta.get('typical_exp')}"
                        )
                        st.markdown(
                            f"Levels for this stock: Live ₹{live_px:,.2f} · "
                            f"Target ₹{tgt:,.2f} · SL ₹{sl:,.2f}"
                        )
                # Chart
                try:
                    raw = stock_history(clean_symbol(stock), interval="1d")
                    if raw is not None and not raw.empty:
                        df = calculate_indicators(raw)
                        show_my = "my_strategy" in (ids or [])
                        fig = build_full_plotly_chart(
                            df,
                            title=f"{stock} · High conviction",
                            target=tgt,
                            stop_loss=sl,
                            height=560,
                            show_rsi=True,
                            my_strategy_lines=show_my,
                            lookback_lines=int(load_my_strategy_params().get("lookback", 90)),
                        )
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True, key=f"hi_chart_{stock}")
                        # Merge backtest: bars where ALL matched strategies fired, then forward outcome
                        if ids and len(df) > 100:
                            hold = 15
                            wins = losses = 0
                            for i in range(60, len(df) - hold - 1):
                                if all(
                                    _strategy_signal_at_bar(df, i, sid) for sid in ids
                                ):
                                    entry = safe_float(df.iloc[i + 1]["Open"])
                                    atr_v = safe_float(df.iloc[i].get("ATR")) or entry * 0.02
                                    tgt = entry + 1.8 * atr_v
                                    stop = entry - 2.2 * atr_v
                                    res = None
                                    for j in range(i + 1, min(i + 1 + hold, len(df))):
                                        hi = safe_float(df.iloc[j]["High"])
                                        lo = safe_float(df.iloc[j]["Low"])
                                        if lo <= stop:
                                            res = "L"
                                            break
                                        if hi >= tgt:
                                            res = "W"
                                            break
                                    if res == "W":
                                        wins += 1
                                    elif res == "L":
                                        losses += 1
                            decided = wins + losses
                            pct = (wins / decided * 100) if decided else 0.0
                            st.info(
                                f"**Merge backtest on {stock}** (all of: {', '.join(names)}): "
                                f"**{pct:.0f}%** target-before-stop "
                                f"({wins}W / {losses}L on {decided} decided signals). "
                                + (
                                    "**Positive historical edge on this merge — preferred buy candidate.**"
                                    if decided >= 3 and pct >= 50
                                    else "Small sample or mixed — size small / confirm with Sure or My Strategy."
                                )
                            )
                except Exception as e:
                    st.caption(f"Chart/backtest: {e}")
        else:
            st.caption("Click **Find high conviction** after a market scan.")

    with tab_ao:
        st.subheader("Levels for manual Angel One order")
        st.caption("This app does not place broker orders. Copy Entry / SL / Target into Angel One.")
        sym_a = st.text_input("Symbol", value="INFY", key="ao_sym").upper().strip()
        if st.button("Get levels", key="ao_btn"):
            try:
                raw = stock_history(clean_symbol(sym_a), interval="1d")
                if raw is None or raw.empty:
                    st.error("No data")
                else:
                    df = calculate_indicators(raw)
                    row = df.iloc[-1]
                    c = safe_float(row.get("Close"))
                    atr_v = safe_float(row.get("ATR")) or c * 0.02
                    p = load_my_strategy_params()
                    tgt = c + float(p.get("atr_target_mult", 1.8)) * atr_v
                    sl = c - float(p.get("atr_stop_mult", 2.2)) * atr_v
                    st.markdown(
                        f"**{display_symbol(sym_a)}**  \n"
                        f"- Entry (ref): **₹{c:,.2f}**  \n"
                        f"- Target: **₹{tgt:,.2f}**  \n"
                        f"- Stop Loss: **₹{sl:,.2f}**  \n"
                        f"- Hold ~**{int(p.get('hold_bars', 15))}** days"
                    )
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown(
        """
**What should you buy?**  
1. **High conviction** (2+ strategies) with good merge % — best  
2. **Sure Call** when Nifty allows  
3. **My Strategy ACTIVE_FRESH** (stocks or Nifty/BankNifty options bias)  
4. **Precision** only if it also overlaps above  
5. Avoid raw scan-only names without filters  

**Daily flow:** Scan → Strategy Lab / High conviction → My Strategy or Sure → Angel One levels  
        """
    )


# ============================================================
# NEWS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_news(symbol):

    try:

        ticker = yf.Ticker(
            clean_symbol(symbol)
        )

        news = ticker.news

        if not news:
            return []

        output = []

        for item in news[:5]:

            content = item.get(
                "content",
                item
            )

            if not isinstance(
                content,
                dict
            ):
                content = item

            title = (
                content.get("title")
                or item.get("title")
                or ""
            )

            provider = content.get(
                "provider",
                {}
            )

            if isinstance(
                provider,
                dict
            ):
                publisher = provider.get(
                    "displayName",
                    ""
                )
            else:
                publisher = item.get(
                    "publisher",
                    ""
                )

            if title:

                output.append(
                    {
                        "title": title,
                        "publisher": publisher,
                    }
                )

        return output

    except Exception:

        return []


def news_score(news):

    positive_words = [
        "profit",
        "growth",
        "record",
        "strong",
        "surge",
        "approval",
        "order",
        "contract",
        "expansion",
        "upgrade",
        "buy",
        "beat",
        "revenue",
        "positive",
        "partnership",
    ]

    negative_words = [
        "loss",
        "fall",
        "decline",
        "weak",
        "downgrade",
        "fraud",
        "investigation",
        "warning",
        "debt",
        "cut",
        "sell",
        "negative",
        "lawsuit",
        "miss",
    ]

    score = 0

    for item in news:

        text = str(
            item.get(
                "title",
                ""
            )
        ).lower()

        for word in positive_words:

            if word in text:
                score += 1

        for word in negative_words:

            if word in text:
                score -= 1

    return score


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def analyse_stock(
    symbol,
    df,
    fetch_news=True
):
    """
    Analyse one stock. Needs enough bars for indicators.
    Min 60 daily bars (was 220 — that dropped almost all NSE/Yahoo short series).
    """
    if df is None or df.empty:
        return None

    if len(df) < 60:
        return None

    df = calculate_indicators(df)

    if df.empty or len(df) < 60:
        return None

    row = df.iloc[-1]

    price = safe_float(
        row["Close"]
    )

    if price <= 0:
        return None

    score = 50

    reasons = []

    # --------------------------------------------------------
    # EMA 20
    # --------------------------------------------------------

    if price > safe_float(row["EMA20"]):

        score += 5

        reasons.append(
            "Price is above the 20-day EMA, showing short-term strength."
        )

    else:

        score -= 5

        reasons.append(
            "Price is below the 20-day EMA, showing weaker short-term momentum."
        )

    # --------------------------------------------------------
    # EMA 50
    # --------------------------------------------------------

    if price > safe_float(row["EMA50"]):

        score += 6

        reasons.append(
            "Price is above the 50-day EMA, supporting the medium-term trend."
        )

    else:

        score -= 6

        reasons.append(
            "Price is below the 50-day EMA."
        )

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    if price > safe_float(row["EMA200"]):

        score += 5

        reasons.append(
            "Price is above the 200-day EMA, supporting the long-term trend."
        )

    else:

        score -= 5

        reasons.append(
            "Price is below the 200-day EMA, which is a long-term weakness warning."
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_value = safe_float(
        row["RSI"]
    )

    if 50 <= rsi_value <= 70:

        score += 7

        reasons.append(
            f"RSI is {rsi_value:.1f}, showing positive momentum without extreme overbought conditions."
        )

    elif rsi_value < 30:

        score += 2

        reasons.append(
            f"RSI is {rsi_value:.1f}, indicating oversold conditions and possible rebound potential."
        )

    elif rsi_value > 75:

        score -= 6

        reasons.append(
            f"RSI is {rsi_value:.1f}, indicating the stock may be overheated."
        )

    else:

        reasons.append(
            f"RSI is {rsi_value:.1f}."
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_value = safe_float(
        row["MACD"]
    )

    macd_signal = safe_float(
        row["MACDSignal"]
    )

    macd_hist = safe_float(
        row["MACDHist"]
    )

    if (
        macd_value > macd_signal
        and
        macd_hist > 0
    ):

        score += 8

        reasons.append(
            "MACD is bullish because it is above the signal line."
        )

    elif (
        macd_value < macd_signal
        and
        macd_hist < 0
    ):

        score -= 8

        reasons.append(
            "MACD is bearish because it is below the signal line."
        )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    adx_value = safe_float(
        row["ADX"]
    )

    if adx_value >= 25:

        score += 6

        reasons.append(
            f"ADX is {adx_value:.1f}, indicating a reasonably strong trend."
        )

    else:

        reasons.append(
            f"ADX is {adx_value:.1f}, so the current trend is not especially strong."
        )

    # --------------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------------

    k = safe_float(
        row["StochK"]
    )

    d = safe_float(
        row["StochD"]
    )

    if k > d and k < 80:

        score += 4

        reasons.append(
            "Stochastic momentum is bullish."
        )

    elif k < d and k > 20:

        score -= 4

        reasons.append(
            "Stochastic momentum is bearish."
        )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    bb_middle = safe_float(
        row["BBMiddle"]
    )

    if price > bb_middle:

        score += 4

        reasons.append(
            "Price is above the Bollinger middle band."
        )

    else:

        score -= 3

        reasons.append(
            "Price is below the Bollinger middle band."
        )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    vwap_value = safe_float(
        row["VWAP"]
    )

    if price > vwap_value:

        score += 4

        reasons.append(
            "Price is above VWAP, supporting buying strength."
        )

    else:

        score -= 4

        reasons.append(
            "Price is below VWAP."
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume = safe_float(
        row["Volume"]
    )

    average_volume = safe_float(
        row["VolumeAvg20"]
    )

    if average_volume > 0:

        volume_ratio = (
            volume /
            average_volume
        )

    else:

        volume_ratio = 1

    if volume_ratio >= 1.5:

        score += 6

        reasons.append(
            "Volume is significantly above average, making the price move more meaningful."
        )

    elif volume_ratio >= 1.1:

        score += 2

        reasons.append(
            "Volume is moderately above average."
        )

    else:

        reasons.append(
            "Volume is not unusually high."
        )

    # --------------------------------------------------------
    # PATTERNS (weights learned from past wins/losses)
    # --------------------------------------------------------

    patterns = detect_patterns(df)
    _learning = load_learning()

    for pattern in patterns:
        w = pattern_weight(pattern, _learning)
        meta = PATTERN_IMPORTANCE.get(pattern, {})
        score += w
        why = meta.get("why", "Detected technical pattern.")
        bias = meta.get("bias", "")
        if w > 0:
            reasons.append(
                f"{pattern} ({bias}): {why} [weight {w:+.1f} from base+learning]"
            )
        elif w < 0:
            reasons.append(
                f"{pattern} ({bias}): {why} [weight {w:+.1f} from base+learning]"
            )
        else:
            reasons.append(f"{pattern}: {why}")

    # My Strategy — RSI bullish divergence (user template: RELIANCE / COALINDIA)
    try:
        _ms = my_strategy_status(df)
        if _ms.get("ok_for_buy") or _ms.get("status") == "ACTIVE_FRESH":
            boost = safe_float(_learning.get("my_strategy_boost"), 8)
            mult = safe_float(_learning.get("rsi_bull_div_multiplier"), 1.15)
            add = boost * mult / 1.1
            score += add
            if "RSI_BULL_DIV" not in patterns:
                patterns.append("RSI_BULL_DIV")
            reasons.append(
                f"My Strategy RSI bullish divergence ACTIVE_FRESH "
                f"(price LL + RSI HL, same bars) [+{add:.1f} learned weight]. "
                f"{_ms.get('reason', '')}"
            )
        elif _ms.get("status") == "TOO_LATE":
            reasons.append(
                f"My Strategy divergence TOO_LATE — not a fresh buy: {_ms.get('reason', '')}"
            )
            score -= 3
        elif _ms.get("status") == "RSI_FAILED":
            reasons.append(
                f"My Strategy RSI line failed — no buy: {_ms.get('reason', '')}"
            )
            score -= 4
    except Exception:
        pass

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    news = []

    if fetch_news:

        news = get_news(
            symbol
        )

    ns = news_score(
        news
    )

    if ns > 0:

        score += min(
            ns * 2,
            6
        )

        news_influence = (
            "Positive news influence detected."
        )

    elif ns < 0:

        score -= min(
            abs(ns) * 2,
            6
        )

        news_influence = (
            "Negative news influence detected."
        )

    else:

        news_influence = (
            "No clear positive or negative news influence detected."
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = float(
        np.clip(
            score,
            5,
            95
        )
    )

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    if prediction >= 72:

        signal = "BUY"

    elif prediction >= 55:

        signal = "HOLD"

    elif prediction <= 38:

        signal = "SELL"

    else:

        signal = "WATCH"

    # --------------------------------------------------------
    # TARGET / STOP
    # --------------------------------------------------------

    atr_value = safe_float(
        row["ATR"]
    )

    if atr_value <= 0:

        atr_value = (
            price * 0.03
        )

    # Placeholder — final target/stop set after signal is finalized
    stop_loss = price - 1.5 * atr_value
    target = price + 2.5 * atr_value

    risk_pct = (
        abs(price - stop_loss)
        /
        price
        *
        100
    )

    if risk_pct < 3:

        risk_level = "LOW"

    elif risk_pct < 6:

        risk_level = "MEDIUM"

    elif risk_pct < 10:

        risk_level = "HIGH"

    else:

        risk_level = "VERY HIGH"

    # --------------------------------------------------------
    # LEARNING FROM PAST MISTAKES (closed target/stop history)
    # --------------------------------------------------------
    learn_delta, learn_notes = learning_score_adjustment(
        prediction, risk_level, signal, patterns, _learning
    )
    if learn_delta:
        score = float(np.clip(score + learn_delta, 5, 95))
        prediction = float(np.clip(score, 5, 95))
        # Re-map call after learning nudge
        if prediction >= 72:
            signal = "BUY"
        elif prediction >= 55:
            signal = "HOLD"
        elif prediction <= 38:
            signal = "SELL"
        else:
            signal = "WATCH"
    for note in learn_notes:
        reasons.append(note)

    # --------------------------------------------------------
    # STRONG TREND ONLY for BUY (quality over quantity)
    # --------------------------------------------------------
    adx_v = safe_float(row.get("ADX"))
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    ema200 = safe_float(row.get("EMA200"))
    vol_r = safe_float(row.get("Volume Ratio"), 1.0)
    rsi_v = safe_float(row.get("RSI"), 50)

    # Prefer strong trend; relaxed so the screen is not empty for days
    strong_trend = (
        price > 0
        and (ema20 <= 0 or price > ema20)
        and (ema50 <= 0 or price > ema50)
        and adx_v >= 20
        and 45 <= rsi_v <= 75
        and vol_r >= 0.9
    )
    strong_trend_note = (
        f"Strong trend: price vs EMA20/50, ADX={adx_v:.1f} (≥20), "
        f"RSI={rsi_v:.1f} (45–75), VolRatio={vol_r:.2f}."
    )

    if signal == "BUY" and not strong_trend:
        signal = "WATCH"
        prediction = min(prediction, 70)
        reasons.append(
            "BUY gated: prefer ADX≥20, price above EMA20/50, RSI 45–75. "
            "Marked WATCH (near-buy) so weak BUYs are not flooded."
        )
    elif signal == "BUY" and strong_trend:
        reasons.append("STRONG TREND BUY: " + strong_trend_note)
        prediction = min(95, prediction + 3)

    # --------------------------------------------------------
    # TARGET / STOP by call direction (SELL target must be BELOW price)
    # --------------------------------------------------------
    if atr_value <= 0:
        atr_value = price * 0.03

    if signal == "SELL":
        # Short / sell: profit when price falls → target below, stop above
        # Keep R:R ≥ 1.5 (reward 2.5 ATR vs risk 1.5 ATR ≈ 1.67)
        stop_loss = price + 1.5 * atr_value
        target = price - 2.5 * atr_value
        if target <= 0:
            target = price * 0.92
        # Enforce min R:R 1.5
        risk_ps = abs(stop_loss - price)
        if risk_ps > 0 and (price - target) / risk_ps < 1.5:
            target = price - 1.5 * risk_ps
        reasons.append(
            f"SELL levels: Target ₹{target:.2f} (below) · Stop ₹{stop_loss:.2f} (above) · R:R≥1.5."
        )
    else:
        # BUY / HOLD / WATCH: long levels — R:R ≥ 1.5 always
        stop_loss = price - 1.5 * atr_value
        if stop_loss <= 0:
            stop_loss = price * 0.95
        risk_ps = abs(price - stop_loss)
        target = price + max(2.5 * atr_value, 1.5 * risk_ps)
        reasons.append(
            f"BUY levels: Target ₹{target:.2f} · Stop ₹{stop_loss:.2f} · R:R≥1.5 enforced."
        )

    risk_pct = abs(price - stop_loss) / price * 100 if price > 0 else 0
    if risk_pct < 3:
        risk_level = "LOW"
    elif risk_pct < 6:
        risk_level = "MEDIUM"
    elif risk_pct < 10:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    # Gate BUY if quality still fails (should be rare after stretch)
    if signal == "BUY":
        _tq = trade_quality_check(price, target, stop_loss, "BUY", min_rr=1.5)
        if not _tq.get("ok"):
            signal = "WATCH"
            prediction = min(prediction, 68)
            reasons.append("BUY→WATCH: " + " · ".join(_tq.get("reasons", [])[:2]))
        else:
            reasons.append(f"Trade quality: {_tq.get('label')} · risk ₹{_tq.get('risk_rs', 0):.0f}")

    # Expected move % (direction-aware)
    if signal == "SELL":
        target_pct = ((price - target) / price * 100) if price > 0 else 0
    else:
        target_pct = ((target - price) / price * 100) if price > 0 else 0

    # --------------------------------------------------------
    # HOLDING PERIOD
    # --------------------------------------------------------

    if prediction >= 85:

        hold_days = 7

    elif prediction >= 75:

        hold_days = 10

    elif prediction >= 65:

        hold_days = 15

    else:

        hold_days = 20

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if (
        prediction >= 85
        and
        risk_pct < 6
    ):

        priority = "VERY HIGH"

    elif prediction >= 75:

        priority = "HIGH"

    elif prediction >= 65:

        priority = "MEDIUM"

    else:

        priority = "LOW"

    # --------------------------------------------------------
    # EXPECTED RETURN (direction-aware)
    # --------------------------------------------------------

    if signal == "SELL":
        target_pct = ((price - target) / price * 100) if price > 0 else 0
    else:
        target_pct = ((target - price) / price * 100) if price > 0 else 0

    # --------------------------------------------------------
    # FINAL REASON
    # --------------------------------------------------------

    reason = " ".join(
        reasons
    )

    return {

        "Stock": display_symbol(
            symbol
        ),

        "Symbol": clean_symbol(
            symbol
        ),

        "Sector": sector_of(
            symbol
        ),

        "Price": round(
            price,
            2
        ),

        "Call": signal,

        "Prediction": round(
            prediction,
            1
        ),

        "Risk %": round(
            risk_pct,
            2
        ),

        "Risk Level": risk_level,

        "Target": round(
            target,
            2
        ),

        "Target %": round(
            target_pct,
            2
        ),

        "Stop Loss": round(
            stop_loss,
            2
        ),

        "Hold Days": hold_days,

        "Priority": priority,

        "RSI": round(
            rsi_value,
            2
        ),

        "MACD": round(
            macd_value,
            4
        ),

        "MACD Signal": round(
            macd_signal,
            4
        ),

        "ADX": round(
            adx_value,
            2
        ),

        "Stochastic": round(
            k,
            2
        ),

        "Volume Ratio": round(
            volume_ratio,
            2
        ),

        "VWAP": round(
            vwap_value,
            2
        ),

        "Patterns": (
            ", ".join(patterns)
            if patterns
            else
            "No major pattern detected"
        ),

        "Reason": reason,

        "Technical Reasons": (
            " | ".join(reasons)
        ),

        "News Influence":
            news_influence,

        "News": news,

        "Data": df,
    }


# ============================================================
# BULK MARKET DOWNLOAD
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data():
    """
    Robust batch downloader for the full NSE scan.
    Uses small batches + threads=False (Cloud-friendly). Failed batch is skipped.
    """
    symbols = [clean_symbol(s) for s in list(NSE_STOCKS[:MAX_SCAN_STOCKS]) if s]
    symbols = list(dict.fromkeys(symbols))
    batches = []
    batch_size = 25  # smaller batches reduce Yahoo/Cloud timeouts

    for start_idx in range(0, len(symbols), batch_size):
        batch = symbols[start_idx:start_idx + batch_size]
        part = None

        for attempt in range(2):
            try:
                part = yf.download(
                    tickers=batch,
                    period="1y",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=False,
                    progress=False,
                )
                if part is not None and not getattr(part, "empty", True):
                    break
            except Exception:
                part = None
            time.sleep(0.8 + attempt)

        if part is not None and not getattr(part, "empty", True):
            batches.append(part)

        # Don't hammer Yahoo if several consecutive batches fail
        if start_idx > 0 and len(batches) == 0 and start_idx >= batch_size * 3:
            break

    if not batches:
        return None

    if len(batches) == 1:
        return batches[0]

    try:
        return pd.concat(batches, axis=1)
    except Exception:
        return batches[0]


def extract_stock_data(all_data, symbol):
    """Pull one symbol OHLC from a yfinance multi-ticker frame (handles both MultiIndex layouts)."""
    if all_data is None or getattr(all_data, "empty", True):
        return pd.DataFrame()

    sym = clean_symbol(symbol)
    bare = display_symbol(symbol)
    candidates = [sym, bare, f"{bare}.NS"]

    try:
        if isinstance(all_data.columns, pd.MultiIndex):
            level0 = list(all_data.columns.get_level_values(0).unique())
            level1 = list(all_data.columns.get_level_values(1).unique())
            df = None
            for c in candidates:
                if c in level0:
                    df = all_data[c].copy()
                    break
                if c in level1:
                    try:
                        df = all_data.xs(c, axis=1, level=1).copy()
                        break
                    except Exception:
                        pass
            if df is None:
                return pd.DataFrame()
        else:
            # Single-ticker download
            df = all_data.copy()

        # Flatten residual multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]

        df = normalize_columns(df)
        # Map lowercase variants
        rename = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("open", "high", "low", "close", "volume", "adj close", "adjclose"):
                nice = "Adj Close" if "adj" in cl else cl.title().replace("Adjclose", "Adj Close")
                if nice == "Adj Close":
                    nice = "Adj Close"
                rename[c] = "Close" if cl in ("adj close", "adjclose") and "Close" not in df.columns else (
                    "Open" if cl == "open" else "High" if cl == "high" else "Low" if cl == "low" else
                    "Close" if cl == "close" else "Volume" if cl == "volume" else c
                )
        if rename:
            df = df.rename(columns=rename)

        if "Close" not in df.columns:
            return pd.DataFrame()

        need = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        df = df.dropna(subset=need)
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================
# FAST SCANNER
# ============================================================


def _priority_scan_universe(limit: int = 180) -> list:
    """Liquid names that almost always have Yahoo history — used when bulk download fails."""
    core = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "BHARTIARTL",
        "LT", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "ASIANPAINT", "HCLTECH", "MARUTI",
        "SUNPHARMA", "TITAN", "NTPC", "TATAMOTORS", "POWERGRID", "ULTRACEMCO", "M&M",
        "WIPRO", "ONGC", "TATASTEEL", "COALINDIA", "JSWSTEEL", "ADANIENT", "TECHM",
        "HINDALCO", "BAJAJFINSV", "NESTLEIND", "INDUSINDBK", "CIPLA", "DRREDDY",
        "BPCL", "EICHERMOT", "APOLLOHOSP", "HEROMOTOCO", "DIVISLAB", "BRITANNIA",
        "TATACONSUM", "BAJAJ-AUTO", "HINDUNILVR", "SBILIFE", "HDFCLIFE", "BEL", "TRENT",
        "FEDERALBNK", "PNB", "BANKBARODA", "CANBK", "IDFCFIRSTB", "DMART", "ZOMATO",
        "PERSISTENT", "COFORGE", "LTIM", "PIDILITIND", "HAVELLS", "SIEMENS", "DLF",
        "IRCTC", "TATAPOWER", "RECLTD", "PFC", "TVSMOTOR", "ASHOKLEY", "VEDL", "GAIL",
    ]
    out = []
    try:
        lists = load_index_constituents()
        for key in ("nifty50", "banknifty", "nifty200"):
            for s in lists.get(key) or []:
                out.append(display_symbol(s))
    except Exception:
        pass
    out.extend(core)
    try:
        for s in list(NSE_STOCKS or [])[: max(limit, 100)]:
            out.append(display_symbol(s))
    except Exception:
        pass
    seen = set()
    uniq = []
    for s in out:
        s = display_symbol(s)
        if s and s not in seen:
            seen.add(s)
            uniq.append(clean_symbol(s))
    return uniq[: max(80, int(limit))]


def _symbols_for_scan_mode(mode: str, sector: str = "") -> list:
    """
    Build symbol list for a scan mode.
    Light modes (nifty50/200/bank/sector) = safe to run often on Cloud.
    master = heavy full-ish market.
    """
    mode = str(mode or "nifty50").strip().lower()
    sector = str(sector or "").strip()
    out = []

    try:
        lists = load_index_constituents()
    except Exception:
        lists = {}

    if mode in ("nifty50", "nifty 50", "n50"):
        out = list(lists.get("nifty50") or _FALLBACK_NIFTY50)
    elif mode in ("nifty200", "nifty 200", "n200"):
        out = list(lists.get("nifty200") or list(_FALLBACK_NIFTY50) + list(_FALLBACK_NIFTY200_EXTRA))
    elif mode in ("banknifty", "bank nifty", "bn", "bank"):
        out = list(lists.get("banknifty") or _FALLBACK_BANKNIFTY)
    elif mode in ("sector", "sector_wise", "sectorwise"):
        # Stocks mapped to selected sector
        try:
            sm = SECTOR_MAP if isinstance(SECTOR_MAP, dict) else {}
            sec_l = sector.lower()
            for sym, sec in sm.items():
                if sec_l and sec_l in str(sec).lower():
                    out.append(sym)
            if not out:
                # fallback: filter priority universe by get_sector if available
                for s in _priority_scan_universe(300):
                    bare = display_symbol(s)
                    try:
                        sec = SECTOR_MAP.get(bare, SECTOR_MAP.get(bare + ".NS", "Other"))
                    except Exception:
                        sec = "Other"
                    if sec_l in str(sec).lower():
                        out.append(bare)
        except Exception:
            out = []
        if not out and sector:
            # last resort: known banks if banking sector
            if "bank" in sector.lower():
                out = list(_FALLBACK_BANKNIFTY)
    elif mode in ("master", "full", "full_market", "all"):
        out = [display_symbol(s) for s in list(NSE_STOCKS[: min(800, MAX_SCAN_STOCKS)]) if s]
    else:
        out = list(lists.get("nifty50") or _FALLBACK_NIFTY50)

    symbols = []
    seen = set()
    for s in out:
        c = clean_symbol(s)
        if c and c not in seen:
            seen.add(c)
            symbols.append(c)
    return symbols


def run_scanner(full_market: bool = False, mode: str = None, sector: str = None):
    """
    Scan modes (Cloud-friendly):
      nifty50 | nifty200 | banknifty | sector | master
    Light modes can be run many times. Master is heavy — use rarely.
    """
    mode = mode or st.session_state.get("scan_mode") or ("master" if full_market else "nifty50")
    sector = sector if sector is not None else st.session_state.get("scan_sector") or ""
    mode = str(mode).strip().lower()
    is_light = mode not in ("master", "full", "full_market", "all")

    # Light scans: do NOT auto-reuse unrelated disk cache (user asked for this universe)
    # Master: may reuse very fresh disk to cut load
    try:
        if mode in ("master", "full", "full_market") and RESULT_FILE.exists():
            age_h = (time.time() - RESULT_FILE.stat().st_mtime) / 3600.0
            if age_h <= 1.0:  # only 1h for master reuse
                df_disk = pd.read_csv(RESULT_FILE)
                if df_disk is not None and not df_disk.empty and len(df_disk) >= 50:
                    st.caption(f"Master scan file is fresh ({age_h:.1f}h) — reusing saved results.")
                    try:
                        return ensure_result_columns(df_disk)
                    except Exception:
                        return df_disk
    except Exception:
        pass

    symbols = _symbols_for_scan_mode(mode, sector)
    if not symbols:
        symbols = _priority_scan_universe(50)

    st.caption(
        f"Scan mode: **{mode.upper()}**"
        + (f" · sector **{sector}**" if mode == "sector" and sector else "")
        + f" · **{len(symbols)}** stocks"
        + (" · light (Cloud-safe)" if is_light else " · MASTER (heavy)")
    )

    # Light modes: skip bulk full-market download (huge load) — sequential only
    all_data = None
    if not is_light and len(symbols) > 200:
        try:
            all_data = download_market_data()
        except Exception:
            all_data = None

    results = []
    analysed = set()

    progress = st.progress(0)
    status = st.empty()
    total = len(symbols)

    # PASS 1: bulk frame
    if all_data is not None and not getattr(all_data, "empty", True):
        for i, symbol in enumerate(symbols):
            try:
                df = extract_stock_data(all_data, symbol)
                if df is not None and not df.empty and len(df) >= 40:
                    result = analyse_stock(symbol, df, fetch_news=False)
                    if result:
                        results.append(result)
                        analysed.add(clean_symbol(symbol))
            except Exception:
                pass
            if i == total - 1 or i % 25 == 0:
                progress.progress(min((i + 1) / max(total, 1), 1.0))
                status.caption(
                    f"⏳ FULL MARKET SCAN: {((i + 1) / max(total, 1)) * 100:.1f}% | "
                    f"{i + 1:,}/{total:,} | {len(results):,} usable"
                )

    # PASS 2: sequential fallback (always if few results)
    missing = [s for s in symbols if clean_symbol(s) not in analysed]
    if len(results) < 40:
        # Prefer priority liquid names first
        priority = _priority_scan_universe(200)
        retry_list = list(dict.fromkeys(priority + missing))[:250]
    else:
        retry_list = missing[:120]

    if retry_list:
        status.caption(f"🔄 Loading NSE bhav history + Yahoo/NSE (up to {len(retry_list):,} symbols)…")
        bhav = {}
        try:
            # One bulk archive pull for all light-scan symbols (Cloud-oriented)
            if len(retry_list) <= 250:
                status.caption("📥 NSE official bhav-copy archives (free, Cloud-friendly)…")
                bhav = nse_bhav_panel(150) or {}
                status.caption(f"Bhav panel: {len(bhav)} symbols with history")
        except Exception:
            bhav = {}
        for j, symbol in enumerate(retry_list):
            if clean_symbol(symbol) in analysed:
                continue
            try:
                df = history_from_bhav(symbol, bhav) if bhav else pd.DataFrame()
                if df is None or getattr(df, "empty", True) or len(df) < 60:
                    df = stock_history(symbol, interval="1d", period="2y")
                if df is None or getattr(df, "empty", True) or len(df) < 60:
                    df = stock_history(symbol, interval="1d", period="5y")
                if df is None or getattr(df, "empty", True) or len(df) < 60:
                    try:
                        t = __import__("yfinance").Ticker(clean_symbol(symbol))
                        df = t.history(period="2y", auto_adjust=True)
                        if df is not None and not df.empty:
                            df = normalize_columns(df)
                    except Exception:
                        pass
                if df is None or getattr(df, "empty", True) or len(df) < 60:
                    try:
                        df = yahoo_chart_history(symbol, "2y")
                    except Exception:
                        pass
                if df is not None and not df.empty and len(df) >= 60:
                    result = analyse_stock(symbol, df, fetch_news=False)
                    if result:
                        results.append(result)
                        analysed.add(clean_symbol(symbol))
            except Exception:
                pass
            if j % 10 == 0 or j == len(retry_list) - 1:
                progress.progress(min((j + 1) / max(len(retry_list), 1), 1.0))
                status.caption(
                    f"🔄 FALLBACK: {j + 1:,}/{len(retry_list):,} | {len(results):,} usable analyses"
                )
            # Soft rate-limit
            if j > 0 and j % 40 == 0:
                time.sleep(0.5)

    progress.empty()
    status.empty()

    # EMERGENCY: if still empty, force-analyse core liquid names with every fetch path
    if not results:
        status = st.empty()
        status.caption("Emergency core scan (liquid names only)…")
        core = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "ITC",
            "AXISBANK", "KOTAKBANK", "LT", "BHARTIARTL", "WIPRO", "HCLTECH",
            "MARUTI", "TATAMOTORS", "SUNPHARMA", "NTPC", "POWERGRID", "ONGC",
            "TATASTEEL", "BAJFINANCE", "ADANIENT", "COALINDIA", "M&M", "TITAN",
        ]
        for j, sym in enumerate(core):
            try:
                df = None
                for per in ("2y", "1y", "6mo", "5y"):
                    try:
                        df = best_stock_history(sym, interval="1d", period=per)
                        if df is not None and not df.empty and len(df) >= 60:
                            break
                    except Exception:
                        df = None
                if df is None or df.empty or len(df) < 60:
                    try:
                        import yfinance as _yf
                        df = _yf.Ticker(clean_symbol(sym)).history(period="2y", auto_adjust=True)
                        if df is not None and not df.empty:
                            df = normalize_columns(df)
                    except Exception:
                        df = None
                if df is not None and not df.empty and len(df) >= 60:
                    result = analyse_stock(sym, df, fetch_news=False)
                    if result:
                        results.append(result)
            except Exception:
                pass
            if j % 5 == 0:
                status.caption(f"Emergency core: {j+1}/{len(core)} · {len(results)} ok")
        status.empty()

    if not results:
        return pd.DataFrame()

    output = pd.DataFrame(results)

    # Defensive cleanup so old/malformed values cannot break sorting.
    for col in ["Prediction", "Risk %"]:
        if col not in output.columns:
            output[col] = 0.0
        output[col] = pd.to_numeric(
            output[col],
            errors="coerce"
        ).fillna(0.0)

    output = output.sort_values(
        ["Prediction", "Risk %"],
        ascending=[False, True],
        kind="stable"
    ).reset_index(drop=True)

    output["Rank"] = np.arange(
        1,
        len(output) + 1
    )

    return output



def run_historical_prediction(symbol, selected_date):
    """
    Past prediction: rebuild V10 analysis using only data available up to
    the selected historical closing date. This works even when the market
    is currently closed.
    """
    df = stock_history(symbol)
    if df is None or df.empty:
        return None, pd.DataFrame()

    cutoff = pd.Timestamp(selected_date)
    hist = df.loc[df.index <= cutoff].copy()
    if hist.empty or len(hist) < 60:
        return None, hist

    try:
        result = analyse_stock(clean_symbol(symbol), hist)
    except Exception:
        result = None
    return result, hist

# ============================================================
# LIVE TRADINGVIEW CHARTS
# ============================================================

def _to_tv_symbol(symbol):
    """Map internal / Yahoo symbols to TradingView exchange:symbol format."""
    raw = str(symbol).upper().strip()
    if raw in {"^NSEI", "NIFTY", "NIFTY50", "NIFTY 50", "NSE:NIFTY"}:
        return "NSE:NIFTY"
    if raw in {"^NSEBANK", "BANKNIFTY", "BANK NIFTY", "NSE:BANKNIFTY"}:
        return "NSE:BANKNIFTY"
    if raw in {"^BSESN", "SENSEX", "BSE:SENSEX"}:
        return "BSE:SENSEX"
    if ":" in raw:
        return raw
    raw = raw.replace(".NS", "").replace(".BO", "")
    return "NSE:" + raw


def _yf_symbol_for_chart(symbol):
    """Yahoo Finance ticker for chart data download."""
    raw = str(symbol).upper().strip()
    if raw in {"^NSEI", "NIFTY", "NIFTY50", "NIFTY 50", "NSE:NIFTY"}:
        return "^NSEI"
    if raw in {"^NSEBANK", "BANKNIFTY", "BANK NIFTY", "NSE:BANKNIFTY"}:
        return "^NSEBANK"
    if raw in {"^BSESN", "SENSEX", "BSE:SENSEX"}:
        return "^BSESN"
    if raw.startswith("^"):
        return raw
    if ":" in raw:
        raw = raw.split(":")[-1]
    return clean_symbol(raw)


def _my_strategy_line_points(df: pd.DataFrame, lookback: int = 90):
    """
    Draw paired lines at the SAME timestamps:
    - Price line: swing low at T1 → swing low at T2
    - RSI line: RSI(T1) → RSI(T2)  (same bars as price)
    This matches user rule: entry price & RSI together, exit/2nd point together.
    """
    if df is None or len(df) < 40:
        return None
    d = df.copy()
    if "RSI" not in d.columns:
        try:
            d = calculate_indicators(d)
        except Exception:
            return None
    if "RSI" not in d.columns or "Low" not in d.columns:
        return None
    i = len(d) - 1
    lb = max(40, int(lookback))
    start = max(0, i - lb)
    seg = d.iloc[start : i + 1]
    lows = seg["Low"].astype(float)
    rsis = seg["RSI"].astype(float)
    pl = _swing_low_indices(lows, 3, 3)
    if len(pl) < 2:
        # try softer right window for latest bar
        pl = _swing_low_indices(lows, 3, 1)
    if len(pl) < 2:
        return None
    t1, t2 = pl[-2], pl[-1]
    if t2 <= t1:
        return None
    price_1 = float(lows.iloc[t1])
    price_2 = float(lows.iloc[t2])
    rsi_1 = float(rsis.iloc[t1])
    rsi_2 = float(rsis.iloc[t2])
    return {
        "price_x": [seg.index[t1], seg.index[t2]],
        "price_y": [price_1, price_2],
        "rsi_x": [seg.index[t1], seg.index[t2]],  # SAME times as price
        "rsi_y": [rsi_1, rsi_2],
        "price_falling": price_2 < price_1,
        "rsi_rising": rsi_2 >= rsi_1 - 0.25,
        "t1": str(seg.index[t1]),
        "t2": str(seg.index[t2]),
    }



def build_full_plotly_chart(
    df,
    title="Chart",
    target=None,
    stop_loss=None,
    height=720,
    show_rsi=True,
    my_strategy_lines=False,
    lookback_lines=90,
    show_patterns=True,
):
    """
    Candlestick + EMA/BB + Volume + RSI on every chart.
    Pattern markers on the candle where detected.
    Optional My Strategy paired lines (same-time price/RSI).
    """
    if df is None or df.empty:
        return None

    d = normalize_columns(df).copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col not in d.columns:
            return None
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if "Volume" not in d.columns:
        d["Volume"] = 0
    d["Volume"] = pd.to_numeric(d["Volume"], errors="coerce").fillna(0)
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    if len(d) < 5:
        return None

    d = d.tail(220).copy()
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["RSI"] = rsi(d["Close"], 14)
    mid, upper, lower = bollinger(d)
    d["BBMiddle"] = mid
    d["BBUpper"] = upper
    d["BBLower"] = lower

    # Pattern recognition on last bars (for chart markers)
    pattern_labels = []
    try:
        if show_patterns and len(d) >= 5:
            for off in range(1, min(8, len(d))):
                sub = d.iloc[: len(d) - off + 1]
                pats = detect_patterns(sub)
                if pats:
                    pattern_labels.append((d.index[-off], pats, float(d["High"].iloc[-off])))
    except Exception:
        pattern_labels = []

    from plotly.subplots import make_subplots

    if show_rsi:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.30],
            subplot_titles=(title, "Volume", "RSI (14)"),
        )
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["EMA20"], name="EMA 20", line=dict(width=1.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=d.index, y=d["EMA50"], name="EMA 50", line=dict(width=1.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["BBUpper"],
            name="BB Upper",
            line=dict(width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["BBLower"],
            name="BB Lower",
            line=dict(width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    if target is not None and safe_float(target) > 0:
        fig.add_hline(
            y=safe_float(target),
            line_dash="dash",
            line_color="#1a9b5f",
            annotation_text="Target",
            row=1,
            col=1,
        )
    if stop_loss is not None and safe_float(stop_loss) > 0:
        fig.add_hline(
            y=safe_float(stop_loss),
            line_dash="dash",
            line_color="#d93025",
            annotation_text="Stop Loss",
            row=1,
            col=1,
        )

    colors = [
        "#1a9b5f" if c >= o else "#d93025"
        for o, c in zip(d["Open"], d["Close"])
    ]
    fig.add_trace(
        go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )

    if show_rsi:
        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d["RSI"],
                name="RSI",
                line=dict(color="#7c5cff", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#888", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#888", row=3, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#555", row=3, col=1)

    # Model-drawn My Strategy lines (like your blue lines on TradingView)
    if my_strategy_lines:
        pts = _my_strategy_line_points(d, lookback=lookback_lines)
        if pts:
            fig.add_trace(
                go.Scatter(
                    x=pts["price_x"],
                    y=pts["price_y"],
                    mode="lines+markers",
                    name="MyStrat price line",
                    line=dict(color="#2196F3", width=2),
                    marker=dict(size=8),
                ),
                row=1,
                col=1,
            )
            if show_rsi:
                fig.add_trace(
                    go.Scatter(
                        x=pts["rsi_x"],
                        y=pts["rsi_y"],
                        mode="lines+markers",
                        name="MyStrat RSI line",
                        line=dict(color="#2196F3", width=2),
                        marker=dict(size=8),
                    ),
                    row=3,
                    col=1,
                )


    # Pattern markers on chart (recognition on the candles themselves)
    if show_patterns and pattern_labels:
        xs, ys, texts = [], [], []
        for idx, pats, hi in pattern_labels[:5]:
            xs.append(idx)
            ys.append(hi * 1.01)
            texts.append(", ".join(pats)[:28])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=11, symbol="triangle-down", color="#f5c542"),
                text=texts,
                textposition="top center",
                name="Patterns",
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.3)
    if show_rsi:
        fig.update_yaxes(range=[15, 85], row=3, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=0.3)
    return fig


def show_tradingview_chart(symbol, title=None, height=620, target=None, stop_loss=None):
    """
    Always-working interactive chart for any NSE stock / index.

    TradingView free embeds often show:
      "This symbol is only available on TradingView"
    inside Streamlit (nested iframe / domain restriction).

    So we chart with Plotly using Yahoo data (works for ALL symbols),
    and provide a one-click link to open the same symbol on TradingView.com.
    """
    tv_symbol = _to_tv_symbol(symbol)
    yf_sym = _yf_symbol_for_chart(symbol)
    chart_title = title or tv_symbol

    # Open full TradingView (drawing tools: trendline, RSI) — best place to draw by hand
    tv_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption(f"Chart + RSI · {tv_symbol} · Draw lines on TradingView (full site)")
    with c2:
        st.link_button("🔗 TradingView (draw)", tv_url, use_container_width=True)
    with c3:
        draw_my = st.checkbox("Show MyStrat lines", value=False, key=f"mylines_{tv_symbol}_{height}")

    try:
        df = stock_history(yf_sym)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.download(
                yf_sym,
                period="1y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            df = normalize_columns(df)
        except Exception:
            df = pd.DataFrame()

    lb_lines = 90
    try:
        lb_lines = int(load_my_strategy_params().get("lookback", 90))
    except Exception:
        pass

    fig = build_full_plotly_chart(
        df,
        title=chart_title,
        target=target,
        stop_loss=stop_loss,
        height=height,
        show_rsi=True,
        my_strategy_lines=draw_my,
        lookback_lines=lb_lines,
    )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "RSI panel is always on. Tick **Show MyStrat lines** to see model blue lines "
            "(price lower-lows + RSI higher-lows). For freehand drawing use **TradingView (draw)**."
        )
    else:
        st.warning(
            f"Chart data unavailable for {tv_symbol}. "
            "Use the TradingView button above to open the full chart."
        )


def show_market_index_charts():
    st.subheader("📈 Live Market Charts")
    tab1, tab2, tab3 = st.tabs(["NIFTY 50", "BANK NIFTY", "SENSEX"])
    with tab1:
        show_tradingview_chart("^NSEI", "NIFTY 50", height=560)
    with tab2:
        show_tradingview_chart("^NSEBANK", "BANK NIFTY", height=560)
    with tab3:
        show_tradingview_chart("^BSESN", "SENSEX", height=560)

# ============================================================
# LIVE QUOTE
# ============================================================

@st.cache_data(
    ttl=20,
    show_spinner=False
)
def live_quote(symbol):
    """Public live quote API — dual NSE+Yahoo, best available. Silent."""
    try:
        q = best_live_quote(symbol)
        return q if q and q.get("price") else None
    except Exception:
        return None



def update_results_with_live_prices(results, max_stocks=80):
    """
    Refresh CURRENT price only.
    Locked prices (Entry at scan, Target, Stop Loss) are never overwritten.
    """
    if results is None or results.empty:
        return results

    x = results.copy()
    if "Symbol" not in x.columns and "Stock" not in x.columns:
        return x

    # Preserve original scan levels once
    if "Locked Price" not in x.columns and "Price" in x.columns:
        x["Locked Price"] = pd.to_numeric(x["Price"], errors="coerce")
    for col in ["Target", "Stop Loss"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    if "Current Price" not in x.columns:
        x["Current Price"] = pd.to_numeric(x.get("Price"), errors="coerce")

    n = min(max_stocks, len(x))
    for idx in list(x.index)[:n]:
        try:
            sym = x.at[idx, "Symbol"] if "Symbol" in x.columns else x.at[idx, "Stock"]
            q = live_quote(sym)
            if q and q.get("price"):
                px = round(safe_float(q["price"]), 2)
                x.at[idx, "Current Price"] = px
                x.at[idx, "Price"] = px  # display LTP
                # Target / Stop Loss / Locked Price untouched
        except Exception:
            continue

    return x


def refresh_history_current_prices(max_stocks: int = 60) -> pd.DataFrame:
    """
    Write Current Price into recommendation_history.csv for open rows.
    Never changes Entry, Target, or Stop Loss.
    """
    history = normalize_history_df(load_history())
    if history is None or history.empty:
        return history

    if "Current Price" not in history.columns:
        history["Current Price"] = ""

    # Freeze originals
    for col in ["Entry", "Target", "Stop Loss"]:
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")

    res_u = history["Result"].astype(str).str.upper().str.strip()
    open_mask = ~res_u.str.contains(
        "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
        regex=True,
        na=True,
    )
    open_idx = history.index[open_mask].tolist()[:max_stocks]

    for idx in open_idx:
        try:
            stock = history.at[idx, "Stock"]
            q = live_quote(stock)
            px = None
            if q and q.get("price"):
                px = safe_float(q["price"])
            if not px:
                d = stock_history(stock, interval="1d")
                if d is not None and not d.empty:
                    px = safe_float(d["Close"].iloc[-1])
            if px:
                history.at[idx, "Current Price"] = round(px, 2)
        except Exception:
            continue

    try:
        history.to_csv(HISTORY_FILE, index=False)
    except Exception:
        pass
    return history


def check_price_vs_levels(current_price, entry, target, stop_loss, call="BUY"):
    """
    Compare current live price against original Target / Stop Loss and
    return a clear status dict matching the requested UI format.
    """
    current = safe_float(current_price)
    entry = safe_float(entry)
    target = safe_float(target)
    stop = safe_float(stop_loss)
    call = str(call).upper()

    status = {
        "status": "OPEN",
        "badge": "",
        "message": "",
        "recommendation": "HOLD",
        "new_target": None,
        "new_stop": None,
        "action": "HOLD",
    }

    if current <= 0:
        return status

    # Long side (BUY / HOLD)
    if call in ["BUY", "HOLD", "WATCH"]:
        if target > 0 and current >= target:
            # Target achieved — suggest trailing or booking
            atr_proxy = max(abs(target - entry) * 0.4, current * 0.015)
            new_target = round(current + atr_proxy * 1.5, 2)
            new_stop = round(max(entry, current - atr_proxy), 2)

            status.update({
                "status": "TARGET ACHIEVED",
                "badge": "🎯 TARGET ACHIEVED",
                "message": (
                    f"Previous Target: ₹{target:,.2f}\n"
                    f"Current Price: ₹{current:,.2f}\n"
                    f"🎯 TARGET ACHIEVED"
                ),
                "recommendation": (
                    "HOLD / TRAIL STOP LOSS\n"
                    "🎯 TARGET ACHIEVED\n"
                    "🔴 SELL / BOOK PROFIT"
                ),
                "new_target": new_target,
                "new_stop": new_stop,
                "action": "TARGET ACHIEVED",
            })
            return status

        if stop > 0 and current <= stop:
            status.update({
                "status": "STOP LOSS HIT",
                "badge": "🔴 STOP LOSS HIT",
                "message": (
                    f"Current Price: ₹{current:,.2f}\n"
                    f"Stop Loss: ₹{stop:,.2f}\n"
                    f"🔴 STOP LOSS HIT"
                ),
                "recommendation": "🔴 SELL / EXIT",
                "action": "STOP LOSS HIT",
            })
            return status

        # Still open — optional mild trailing suggestion when close to target
        if target > 0 and current >= target * 0.97:
            status["message"] = (
                f"Approaching target (₹{target:,.2f}). Current: ₹{current:,.2f}"
            )
            status["recommendation"] = "HOLD — watch target closely"

    # Short side (SELL)
    elif call == "SELL":
        if target > 0 and current <= target:
            status.update({
                "status": "TARGET ACHIEVED",
                "badge": "🎯 TARGET ACHIEVED",
                "message": (
                    f"Previous Target: ₹{target:,.2f}\n"
                    f"Current Price: ₹{current:,.2f}\n"
                    f"🎯 TARGET ACHIEVED (SELL)"
                ),
                "recommendation": "🎯 TARGET ACHIEVED — Book profit / cover",
                "action": "TARGET ACHIEVED",
            })
            return status

        if stop > 0 and current >= stop:
            status.update({
                "status": "STOP LOSS HIT",
                "badge": "🔴 STOP LOSS HIT",
                "message": (
                    f"Current Price: ₹{current:,.2f}\n"
                    f"Stop Loss: ₹{stop:,.2f}\n"
                    f"🔴 STOP LOSS HIT"
                ),
                "recommendation": "🔴 SELL / EXIT — Cover position",
                "action": "STOP LOSS HIT",
            })
            return status

    return status


# ============================================================
# HISTORICAL DATA FOR INDIVIDUAL STOCK
# ============================================================


# ============================================================
# NSE INDIA DATA (primary free source) + Yahoo fallback
# ============================================================

_NSE_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_NSE_HEADERS_BASE = {
    "User-Agent": _NSE_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


@st.cache_resource(show_spinner=False)
def nse_http_session():
    """Shared requests session with NSE cookies (free public API)."""
    try:
        import requests
    except Exception:
        return None
    s = requests.Session()
    s.headers.update(_NSE_HEADERS_BASE)
    try:
        s.get("https://www.nseindia.com/", timeout=12)
        s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=12)
    except Exception:
        pass
    return s


def nse_get_json(path_or_url: str, params=None, timeout: int = 20):
    """GET JSON from NSE; refresh cookies once on failure."""
    try:
        import requests
    except Exception:
        return None
    url = path_or_url if str(path_or_url).startswith("http") else f"https://www.nseindia.com{path_or_url}"
    s = nse_http_session()
    if s is None:
        return None
    for attempt in range(2):
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code == 401 or r.status_code == 403:
                try:
                    s.get("https://www.nseindia.com/", timeout=12)
                except Exception:
                    pass
                continue
            if r.status_code != 200:
                continue
            return r.json()
        except Exception:
            try:
                s.get("https://www.nseindia.com/", timeout=12)
            except Exception:
                pass
    return None


def nse_quote_equity(symbol: str) -> dict:
    """
    Live / last equity quote from NSE free API.
    Returns dict with ltp, open, high, low, prev_close, change, pChange, volume…
    """
    sym = display_symbol(symbol)
    if not sym or sym.startswith("^"):
        return {}
    data = nse_get_json(f"/api/quote-equity?symbol={sym}")
    if not data or not isinstance(data, dict):
        # try encoded symbol for & names
        try:
            from urllib.parse import quote
            data = nse_get_json(f"/api/quote-equity?symbol={quote(sym)}")
        except Exception:
            data = None
    if not data or not isinstance(data, dict):
        return {}
    price_info = data.get("priceInfo") or {}
    trade = data.get("securityWiseDP") or {}
    out = {
        "symbol": sym,
        "ltp": safe_float(price_info.get("lastPrice")),
        "open": safe_float(price_info.get("open")),
        "high": safe_float(price_info.get("intraDayHighLow", {}).get("max") if isinstance(price_info.get("intraDayHighLow"), dict) else price_info.get("high")),
        "low": safe_float(price_info.get("intraDayHighLow", {}).get("min") if isinstance(price_info.get("intraDayHighLow"), dict) else price_info.get("low")),
        "prev_close": safe_float(price_info.get("previousClose") or price_info.get("close")),
        "change": safe_float(price_info.get("change")),
        "pct": safe_float(price_info.get("pChange")),
        "volume": safe_float((data.get("securityWiseDP") or {}).get("quantityTraded") or trade.get("quantityTraded")),
        "source": "NSE",
    }
    # high/low fallbacks
    if out["high"] is None:
        out["high"] = safe_float(price_info.get("weekHighLow", {}).get("max") if isinstance(price_info.get("weekHighLow"), dict) else None)
    if out["low"] is None:
        out["low"] = safe_float(price_info.get("weekHighLow", {}).get("min") if isinstance(price_info.get("weekHighLow"), dict) else None)
    return out


def nse_index_quote(name: str) -> dict:
    """Live index levels from NSE allIndices (NIFTY 50, NIFTY BANK, …)."""
    key = str(name or "").upper().replace(" ", "")
    data = nse_get_json("/api/allIndices")
    if not data:
        return {}
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}
    aliases = {
        "NIFTY": "NIFTY 50",
        "NIFTY50": "NIFTY 50",
        "^NSEI": "NIFTY 50",
        "BANKNIFTY": "NIFTY BANK",
        "NIFTYBANK": "NIFTY BANK",
        "^NSEBANK": "NIFTY BANK",
        "SENSEX": "SENSEX",
        "^BSESN": "SENSEX",
    }
    want = aliases.get(key, name)
    want_u = str(want).upper()
    for r in rows:
        idx = str(r.get("index") or r.get("indexSymbol") or "").upper()
        if idx == want_u or want_u in idx or idx in want_u:
            last = safe_float(r.get("last") or r.get("lastPrice"))
            pct = safe_float(r.get("percentChange") or r.get("pChange"))
            ch = safe_float(r.get("variation") or r.get("change"))
            return {"price": last, "pct": pct or 0.0, "change": ch or 0.0, "source": "NSE", "index": r.get("index")}
    return {}


def nse_equity_history(symbol, days=400):
    """
    Daily OHLC from NSE free historical API (primary for India equities).
    """
    sym = display_symbol(symbol)
    if not sym or sym.startswith("^"):
        return pd.DataFrame()

    end = datetime.now().date()
    start = end - timedelta(days=max(int(days), 60))
    from_s = start.strftime("%d-%m-%Y")
    to_s = end.strftime("%d-%m-%Y")

    payload = nse_get_json(
        "/api/historical/cm/equity",
        params={"symbol": sym, "series": '["EQ"]', "from": from_s, "to": to_s},
        timeout=30,
    )
    if not payload:
        # chart-databyindex fallback (longer series sometimes)
        try:
            chart = nse_get_json(f"/api/chart-databyindex?index={sym}EQN", timeout=25)
            if chart and isinstance(chart, dict):
                graphed = chart.get("grapthData") or chart.get("graphData") or chart.get("data") or []
                records = []
                for pt in graphed:
                    # [timestamp_ms, price] or dict
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        records.append({
                            "Date": pd.to_datetime(pt[0], unit="ms", errors="coerce"),
                            "Open": safe_float(pt[1]),
                            "High": safe_float(pt[1]),
                            "Low": safe_float(pt[1]),
                            "Close": safe_float(pt[1]),
                            "Volume": 0,
                        })
                if records:
                    df = pd.DataFrame(records).dropna(subset=["Date", "Close"])
                    df = df.set_index("Date").sort_index()
                    return df
        except Exception:
            pass
        return pd.DataFrame()

    rows = payload.get("data") or payload.get("dataList") or []
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        dt = r.get("CH_TIMESTAMP") or r.get("mTIMESTAMP") or r.get("date") or r.get("Date")
        o = r.get("CH_OPENING_PRICE") or r.get("OPEN") or r.get("open")
        h = r.get("CH_TRADE_HIGH_PRICE") or r.get("HIGH") or r.get("high")
        l = r.get("CH_TRADE_LOW_PRICE") or r.get("LOW") or r.get("low")
        c = r.get("CH_CLOSING_PRICE") or r.get("CLOSE") or r.get("close") or r.get("CH_LAST_TRADED_PRICE")
        v = r.get("CH_TOT_TRADED_QTY") or r.get("VOLUME") or r.get("volume") or 0
        if not dt or c is None:
            continue
        records.append({
            "Date": pd.to_datetime(dt, errors="coerce"),
            "Open": safe_float(o),
            "High": safe_float(h),
            "Low": safe_float(l),
            "Close": safe_float(c),
            "Volume": safe_float(v),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).dropna(subset=["Date", "Close"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Open", "High", "Low", "Close"], how="any")


@st.cache_data(ttl=120, show_spinner=False)
def nse_index_constituents_live(index_name: str = "NIFTY 50") -> pd.DataFrame:
    """Live prices for all stocks in an NSE index (free)."""
    data = nse_get_json(
        "/api/equity-stockIndices",
        params={"index": index_name},
        timeout=25,
    )
    if not data:
        return pd.DataFrame()
    rows = data.get("data") or []
    out = []
    for r in rows:
        sym = str(r.get("symbol") or "").strip().upper()
        if not sym or sym == index_name.upper().replace(" ", ""):
            continue
        out.append({
            "Stock": sym,
            "LTP": safe_float(r.get("lastPrice")),
            "Change": safe_float(r.get("change")),
            "Pct": safe_float(r.get("pChange")),
            "Open": safe_float(r.get("open")),
            "High": safe_float(r.get("dayHigh")),
            "Low": safe_float(r.get("dayLow")),
            "Prev Close": safe_float(r.get("previousClose")),
            "Volume": safe_float(r.get("totalTradedVolume")),
        })
    return pd.DataFrame(out)





@st.cache_data(ttl=3600, show_spinner=False)
def nse_bhav_panel(max_days: int = 160) -> dict:
    """
    Build OHLC from NSE official daily bhav-copy archives (free).
    Works better on Streamlit Cloud than live Yahoo for many IPs.
    Returns {SYMBOL: DataFrame with Open,High,Low,Close,Volume}.
    """
    try:
        import requests
        import io
    except Exception:
        return {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/csv,*/*",
    }
    bases = [
        "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
        "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
    ]
    # symbol -> list of row dicts
    bucket = {}
    got = 0
    d = datetime.now().date()
    tried = 0
    # walk calendar days until we have ~max_days trading files or tried enough
    while got < max_days and tried < max_days * 2 + 20:
        tried += 1
        ddmmyyyy = d.strftime("%d%m%Y")
        raw = None
        for b in bases:
            url = b.format(ddmmyyyy=ddmmyyyy)
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.status_code == 200 and len(r.content) > 500:
                    raw = r.content
                    break
            except Exception:
                continue
        d = d - timedelta(days=1)
        if raw is None:
            continue
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            continue
        # normalize columns
        cols = {str(c).strip().upper().replace(" ", "_"): c for c in df.columns}
        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            for k, v in cols.items():
                for n in names:
                    if n in k:
                        return v
            return None
        c_sym = col("SYMBOL")
        c_ser = col("SERIES")
        c_o = col("OPEN_PRICE", "OPEN")
        c_h = col("HIGH_PRICE", "HIGH")
        c_l = col("LOW_PRICE", "LOW")
        c_c = col("CLOSE_PRICE", "CLOSE")
        c_v = col("TTL_TRD_QNTY", "VOLUME", "TOTTRDQTY")
        c_dt = col("DATE1", "TIMESTAMP", "DATE")
        if not c_sym or not c_c:
            continue
        got += 1
        for _, row in df.iterrows():
            try:
                if c_ser is not None and str(row.get(c_ser, "")).upper() not in ("EQ", "BE", "SM", "NAN", ""):
                    # keep EQ mainly
                    if str(row.get(c_ser, "")).upper() != "EQ":
                        continue
                sym = str(row[c_sym]).strip().upper()
                if not sym or sym == "NAN":
                    continue
                close = safe_float(row[c_c])
                if not close or close <= 0:
                    continue
                dt_val = row[c_dt] if c_dt else ddmmyyyy
                dt = pd.to_datetime(dt_val, errors="coerce", dayfirst=True)
                if pd.isna(dt):
                    dt = pd.to_datetime(d + timedelta(days=1), errors="coerce")  # file date approx
                rec = {
                    "Date": dt,
                    "Open": safe_float(row[c_o]) if c_o else close,
                    "High": safe_float(row[c_h]) if c_h else close,
                    "Low": safe_float(row[c_l]) if c_l else close,
                    "Close": close,
                    "Volume": safe_float(row[c_v]) if c_v else 0,
                }
                bucket.setdefault(sym, []).append(rec)
            except Exception:
                continue
        if got >= max_days:
            break

    out = {}
    for sym, rows in bucket.items():
        try:
            x = pd.DataFrame(rows).dropna(subset=["Date", "Close"])
            if x.empty:
                continue
            x = x.sort_values("Date").drop_duplicates("Date", keep="last")
            x = x.set_index("Date")
            for c in ("Open", "High", "Low"):
                if c in x.columns:
                    x[c] = x[c].fillna(x["Close"])
            if len(x) >= 30:
                out[sym] = x
        except Exception:
            continue
    return out


def history_from_bhav(symbol, panel: dict = None) -> pd.DataFrame:
    """Pick one symbol from bhav panel."""
    try:
        if panel is None:
            panel = nse_bhav_panel(140)
        sym = display_symbol(symbol)
        df = panel.get(sym)
        if df is not None and not df.empty:
            return df.copy()
    except Exception:
        pass
    return pd.DataFrame()


def yahoo_chart_history(symbol, range_str: str = "2y") -> pd.DataFrame:
    """
    Direct Yahoo Chart API (v8) — often works when yfinance download is blocked.
    Free, no key. Returns OHLC DataFrame or empty.
    """
    try:
        import requests
        sym = clean_symbol(symbol)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        headers = {
            "User-Agent": _NSE_UA if "_NSE_UA" in dir() else "Mozilla/5.0",
            "Accept": "application/json",
        }
        r = requests.get(
            url,
            params={"range": range_str, "interval": "1d", "events": "div,splits"},
            headers=headers,
            timeout=20,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        result = (js.get("chart") or {}).get("result") or []
        if not result:
            return pd.DataFrame()
        block = result[0]
        ts = block.get("timestamp") or []
        q = (block.get("indicators") or {}).get("quote") or [{}]
        q0 = q[0] if q else {}
        if not ts:
            return pd.DataFrame()
        df = pd.DataFrame({
            "Date": pd.to_datetime(ts, unit="s", errors="coerce"),
            "Open": q0.get("open"),
            "High": q0.get("high"),
            "Low": q0.get("low"),
            "Close": q0.get("close"),
            "Volume": q0.get("volume"),
        }).dropna(subset=["Date", "Close"])
        if df.empty:
            return pd.DataFrame()
        df = df.set_index("Date").sort_index()
        return df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    except Exception:
        return pd.DataFrame()


def _yahoo_live_quote_raw(symbol: str) -> dict:
    """Internal Yahoo live/last quote (no UI messages)."""
    try:
        sym = clean_symbol(symbol)
        if nse_market_open_now():
            d = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=False, threads=False)
            d = normalize_columns(d)
            if d is None or d.empty or "Close" not in d.columns:
                d = yf.download(sym, period="5d", interval="5m", progress=False, auto_adjust=False, threads=False)
                d = normalize_columns(d)
        else:
            d = yf.download(sym, period="10d", interval="1d", progress=False, auto_adjust=False, threads=False)
            d = normalize_columns(d)
        if d is None or d.empty or "Close" not in d.columns:
            return {}
        close = pd.to_numeric(d["Close"], errors="coerce").dropna()
        if close.empty:
            return {}
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else price
        pct = ((price - prev) / prev * 100.0) if prev else 0.0
        return {"price": price, "pct": pct, "change": price - prev, "source": "Yahoo"}
    except Exception:
        return {}


def best_live_quote(symbol: str) -> dict:
    """
    Dual source: fetch NSE + Yahoo, pick the best usable quote.
    Priority while market open: NSE if valid LTP, else Yahoo.
    Prefer non-zero price; if both exist and differ > 3%, prefer NSE for India equities.
    Silent — never raises / never shows provider errors.
    """
    raw = str(symbol or "").strip()
    up = raw.upper().replace(" ", "")
    nse_q, y_q = {}, {}

    try:
        if up in ("^NSEI", "NIFTY", "NIFTY50") or "NIFTY50" in up:
            nse_q = nse_index_quote("NIFTY 50") or {}
        elif up in ("^NSEBANK", "BANKNIFTY", "NIFTYBANK") or "BANKNIFTY" in up:
            nse_q = nse_index_quote("NIFTY BANK") or {}
        elif up in ("^BSESN", "SENSEX"):
            nse_q = nse_index_quote("SENSEX") or {}
        elif not raw.startswith("^"):
            nq = nse_quote_equity(raw) or {}
            ltp = nq.get("ltp")
            if ltp and float(ltp) > 0:
                prev = nq.get("prev_close") or 0
                pct = nq.get("pct")
                if pct is None and prev:
                    pct = (float(ltp) - float(prev)) / float(prev) * 100
                nse_q = {
                    "price": float(ltp),
                    "pct": float(pct or 0),
                    "change": float(nq.get("change") or 0),
                    "source": "NSE",
                    "open": nq.get("open"),
                    "high": nq.get("high"),
                    "low": nq.get("low"),
                    "volume": nq.get("volume"),
                }
    except Exception:
        nse_q = {}

    try:
        y_q = _yahoo_live_quote_raw(raw) or {}
    except Exception:
        y_q = {}

    nse_px = safe_float(nse_q.get("price")) if nse_q else None
    y_px = safe_float(y_q.get("price")) if y_q else None

    if nse_px and nse_px > 0 and y_px and y_px > 0:
        # Both available — prefer NSE for cash equities/indices; Yahoo if NSE looks stale (0 change weird alone is ok)
        diff = abs(nse_px - y_px) / max(nse_px, y_px)
        if diff <= 0.05:
            return nse_q  # agree → NSE
        # large disagreement: prefer NSE during market hours, else higher volume not available → NSE
        try:
            if nse_market_open_now():
                return nse_q
        except Exception:
            pass
        return nse_q  # India names: NSE still preferred
    if nse_px and nse_px > 0:
        return nse_q
    if y_px and y_px > 0:
        return y_q
    return {}


def best_stock_history(symbol, interval="1d", period=None, **kwargs):
    """
    Dual history: NSE + Yahoo (+ Ticker.history), return longer/cleaner series.
    Silent on failures.
    """
    interval = str(interval or "1d").lower()
    nse_df = pd.DataFrame()
    y_df = pd.DataFrame()

    if interval in {"1d", "1wk", "1mo"} and not str(symbol).startswith("^"):
        try:
            # 1) NSE bhav-copy panel (Cloud-friendly official archives)
            nse_df = history_from_bhav(symbol)
            if nse_df is None or nse_df.empty or len(nse_df) < 60:
                days = 800
                if period:
                    try:
                        if str(period).endswith("y"):
                            days = int(float(str(period)[:-1]) * 365)
                        elif str(period).endswith("mo"):
                            days = int(float(str(period)[:-2]) * 30)
                    except Exception:
                        pass
                nse_df = nse_equity_history(symbol, days=max(days, 400))
            if nse_df is None:
                nse_df = pd.DataFrame()
        except Exception:
            nse_df = pd.DataFrame()

    period_map = {
        "1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "730d", "60m": "730d", "1d": "2y", "1wk": "5y", "1mo": "10y",
    }
    per = period or period_map.get(interval, "2y")

    try:
        d = yf.download(
            clean_symbol(symbol), period=per, interval=interval,
            progress=False, auto_adjust=True, threads=False,
        )
        d = normalize_columns(d)
        if d is not None and not d.empty and "Close" in d.columns:
            y_df = d.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in d.columns])
    except Exception:
        y_df = pd.DataFrame()

    if y_df is None or y_df.empty or len(y_df) < 60:
        try:
            h = yf.Ticker(clean_symbol(symbol)).history(period=per if interval == "1d" else "2y", interval=interval, auto_adjust=True)
            if h is not None and not h.empty:
                h = normalize_columns(h)
                if "Close" in h.columns:
                    y_df = h.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in h.columns])
        except Exception:
            pass

    if (y_df is None or y_df.empty or len(y_df) < 60) and interval in {"1d", "1wk", "1mo"}:
        try:
            for rng in ("2y", "1y", "6mo", "5y"):
                ch = yahoo_chart_history(symbol, rng)
                if ch is not None and not ch.empty and len(ch) >= 60:
                    y_df = ch
                    break
        except Exception:
            pass

    nse_ok = nse_df is not None and not nse_df.empty and len(nse_df) >= 60
    y_ok = y_df is not None and not y_df.empty and len(y_df) >= 60

    if nse_ok and y_ok:
        if interval in {"1d", "1wk", "1mo"}:
            return nse_df if len(nse_df) >= len(y_df) * 0.65 else y_df
        return y_df
    if y_ok:
        return y_df
    if nse_ok:
        return nse_df
    # accept shorter series if that is all we have (analyse needs 60)
    if y_df is not None and not y_df.empty and len(y_df) >= 30:
        return y_df
    if nse_df is not None and not nse_df.empty and len(nse_df) >= 30:
        return nse_df
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def _stock_history_cached(symbol, interval="1d", period=None):
    try:
        df = best_stock_history(symbol, interval=interval, period=period)
        if df is None or getattr(df, "empty", True) or len(df) < 30:
            return None  # do not cache empties as success
        return df
    except Exception:
        return None


def stock_history(symbol, interval="1d", period=None, **kwargs):
    """OHLC — dual source; empty results are not treated as cached success."""
    try:
        df = _stock_history_cached(str(symbol), str(interval or "1d"), str(period) if period else None)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    # Uncached retry (bypass bad cache)
    try:
        df = best_stock_history(symbol, interval=interval, period=period)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()





def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make recommendation_history.csv safe across machines / old file versions.
    Fixes KeyError: 'Prediction Date' when the CSV is empty or has different headers.
    """
    if df is None:
        return pd.DataFrame()

    x = df.copy()

    # Strip column names (BOM / spaces from Excel saves)
    x.columns = [str(c).strip().replace("\ufeff", "") for c in x.columns]

    # Common aliases from older exports
    aliases = {
        "prediction date": "Prediction Date",
        "prediction_date": "Prediction Date",
        "date": "Prediction Date",
        "stock": "Stock",
        "symbol": "Symbol",
        "call": "Call",
        "signal": "Call",
        "entry": "Entry",
        "price": "Entry",
        "target": "Target",
        "stop loss": "Stop Loss",
        "stop_loss": "Stop Loss",
        "risk %": "Risk %",
        "risk_pct": "Risk %",
        "risk level": "Risk Level",
        "hold days": "Hold Days",
        "hold_days": "Hold Days",
        "expiry date": "Expiry Date",
        "status": "Status",
        "evaluation date": "Evaluation Date",
        "exit price": "Exit Price",
        "return %": "Return %",
        "result": "Result",
        "result detail": "Result Detail",
        "days taken": "Days Taken",
        "outcome message": "Outcome Message",
        "recommendation": "Recommendation",
        "suggestion": "Suggestion",
        "reason": "Reason",
        "prediction": "Prediction",
    }
    lower_map = {str(c).strip().lower(): c for c in x.columns}
    for old_l, new in aliases.items():
        if new not in x.columns and old_l in lower_map:
            x[new] = x[lower_map[old_l]]

    required = [
        "Prediction Date",
        "Stock",
        "Symbol",
        "Call Source",
        "Strategy",
        "Patterns",
        "Call",
        "Prediction",
        "Entry",
        "Target",
        "Stop Loss",
        "Risk %",
        "Risk Level",
        "Hold Days",
        "Expiry Date",
        "Status",
        "Evaluation Date",
        "Exit Price",
        "Return %",
        "Result",
        "Result Detail",
        "Days Taken",
        "Outcome Message",
        "Recommendation",
        "Suggestion",
        "Reason",
        "Current Price",
    ]
    for col in required:
        if col not in x.columns:
            x[col] = ""

    # Default source for old rows
    if "Call Source" in x.columns:
        blank = (
            x["Call Source"].isna()
            | x["Call Source"].astype(str).str.strip().isin(["", "nan", "None", "NAN"])
        )
        x.loc[blank, "Call Source"] = "SCAN"

    # Drop fully empty rows
    if "Stock" in x.columns:
        x = x[~(x["Stock"].astype(str).str.strip() == "") | (x["Prediction Date"].astype(str).str.strip() != "")]

    return x.reset_index(drop=True)


def load_history():

    try:
        if not HISTORY_FILE.exists():
            ensure_files()
            return pd.DataFrame()

        df = pd.read_csv(HISTORY_FILE)
        if df is None or df.empty:
            return normalize_history_df(pd.DataFrame())

        return normalize_history_df(df)

    except Exception:
        return normalize_history_df(pd.DataFrame())


def save_recommendations(
    results
):

    if results is None:
        return

    if results.empty:
        return

    history = load_history()

    now = datetime.now()

    rows = []

    # Save every signal.
    # SELL is now included.

    for _, row in results.iterrows():

        call = str(
            row.get(
                "Call",
                ""
            )
        ).upper()

        if call not in [
            "BUY",
            "HOLD",
            "SELL",
        ]:
            continue

        hold_days = int(
            safe_float(
                row.get(
                    "Hold Days",
                    DEFAULT_HOLD_DAYS
                ),
                DEFAULT_HOLD_DAYS
            )
        )

        expiry = (
            now +
            timedelta(
                days=hold_days
            )
        )

        rows.append(
            {
                "Prediction Date":
                    now.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Stock":
                    row.get(
                        "Stock",
                        ""
                    ),

                "Symbol":
                    row.get(
                        "Symbol",
                        ""
                    ),

                "Call Source":
                    str(row.get("Call Source", "SCAN") or "SCAN").upper(),

                "Strategy":
                    str(row.get("Strategy", "") or ""),

                "Patterns":
                    str(row.get("Patterns", row.get("Pattern", "")) or ""),

                "Call":
                    call,

                "Prediction":
                    row.get(
                        "Prediction",
                        ""
                    ),

                "Entry":
                    row.get(
                        "Price",
                        ""
                    ),

                "Target":
                    row.get(
                        "Target",
                        ""
                    ),

                "Stop Loss":
                    row.get(
                        "Stop Loss",
                        ""
                    ),

                "Risk %":
                    row.get(
                        "Risk %",
                        ""
                    ),

                "Risk Level":
                    row.get(
                        "Risk Level",
                        ""
                    ),

                "Hold Days":
                    hold_days,

                "Expiry Date":
                    expiry.strftime(
                        "%Y-%m-%d"
                    ),

                "Status":
                    "OPEN",

                "Evaluation Date":
                    "",

                "Exit Price":
                    "",

                "Return %":
                    "",

                "Result":
                    "PENDING",

                "Reason":
                    row.get(
                        "Reason",
                        ""
                    ),

                "Patterns":
                    row.get("Patterns", ""),

                "RSI":
                    row.get("RSI", ""),

                "ADX":
                    row.get("ADX", ""),
            }
        )

    if not rows:
        return

    new_history = pd.DataFrame(
        rows
    )

    history = pd.concat(
        [
            history,
            new_history,
        ],
        ignore_index=True
    )

    # Prevent exact duplicate scans.
    if not history.empty:

        history = history.drop_duplicates(
            subset=[
                "Prediction Date",
                "Stock",
                "Call",
            ],
            keep="last"
        )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# HISTORICAL PREDICTION EVALUATION
# ============================================================

def _normalize_ohlc_index(data: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone and normalize to calendar dates so day comparisons work."""
    if data is None or data.empty:
        return pd.DataFrame()
    d = data.copy()
    idx = pd.to_datetime(d.index, errors="coerce")
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        try:
            idx = idx.tz_convert(None)
        except Exception:
            pass
    d.index = pd.DatetimeIndex(idx).normalize()
    d = d[~d.index.isna()]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def _eval_price_history(symbol: str) -> pd.DataFrame:
    """
    Aggressive daily OHLC for evaluation — bypasses stale cache when possible.
    """
    sym = clean_symbol(symbol)
    if not sym:
        return pd.DataFrame()
    try:
        # Fresh download (not only stock_history cache)
        d = yf.download(
            sym,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        d = normalize_columns(d)
        d = _normalize_ohlc_index(d)
        if d is not None and not d.empty and "High" in d.columns and "Low" in d.columns:
            return d.dropna(subset=["High", "Low", "Close"], how="any")
    except Exception:
        pass
    try:
        d = stock_history(sym, interval="1d")
        d = _normalize_ohlc_index(d)
        if d is not None and not d.empty:
            return d
    except Exception:
        pass
    try:
        d = nse_equity_history(sym, days=400)
        return _normalize_ohlc_index(d)
    except Exception:
        return pd.DataFrame()


def evaluate_history(force_all: bool = False):
    """
    Evaluate open predictions against subsequent price action.
    ORIGINAL Target and Stop Loss in the history sheet are NEVER changed.
    Near-target ideas go only into Suggestion / Recommendation text.
    """

    history = load_history()

    if history.empty:
        return history

    history = normalize_history_df(history)

    text_cols = [
        "Prediction Date", "Stock", "Symbol", "Call", "Status",
        "Evaluation Date", "Result", "Result Detail", "Days Taken",
        "Outcome Message", "Recommendation", "Reason", "Risk Level",
        "Expiry Date", "Suggestion",
    ]
    for col in text_cols:
        if col not in history.columns:
            history[col] = ""
        history[col] = history[col].astype(object)

    # Numeric columns — Target / Stop Loss are read-only after this (never written back differently)
    locked_levels = {}  # idx -> (target, stop) frozen originals
    for col in ["Exit Price", "Return %", "Entry", "Target", "Stop Loss", "Risk %", "Hold Days", "Prediction"]:
        if col not in history.columns:
            history[col] = None
        history[col] = pd.to_numeric(history[col], errors="coerce")

    for idx in history.index:
        locked_levels[idx] = (
            safe_float(history.at[idx, "Target"]),
            safe_float(history.at[idx, "Stop Loss"]),
        )

    required = [
        "Prediction Date", "Stock", "Call", "Entry", "Target", "Stop Loss",
        "Hold Days", "Status", "Result", "Result Detail", "Days Taken",
        "Outcome Message", "Recommendation", "Suggestion",
    ]
    for col in required:
        if col not in history.columns:
            history[col] = ""

    changed = False

    res_u = history["Result"].astype(str).str.upper().str.strip()
    open_mask = ~res_u.str.contains(
        "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
        regex=True,
        na=True,
    )

    # Repair impossible outcomes: event date BEFORE prediction date
    # (old bug used data.tail() and could mark stop on a past day)
    invalid_closed = []
    for idx in history.index:
        result_s = str(history.at[idx, "Result"]).upper()
        if not any(x in result_s for x in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD", "WIN", "LOSS")):
            continue
        try:
            pred_dt = pd.to_datetime(history.at[idx, "Prediction Date"], errors="coerce")
            if pd.isna(pred_dt):
                continue
            pred_d = pd.Timestamp(pred_dt).normalize()
            eval_dt = pd.to_datetime(history.at[idx, "Evaluation Date"], errors="coerce")
            detail = str(history.at[idx, "Result Detail"] or "")
            event_dt = eval_dt
            # Parse "Stop Loss Hit On: 10 September 2026" / "Target Reached On: ..."
            for label in ("Stop Loss Hit On:", "Target Reached On:", "Exit Date:"):
                if label in detail:
                    try:
                        part = detail.split(label, 1)[1].split("\n")[0].strip()
                        event_dt = pd.to_datetime(part, errors="coerce", dayfirst=True)
                    except Exception:
                        pass
                    break
            if pd.notna(event_dt) and pd.Timestamp(event_dt).normalize() < pred_d:
                invalid_closed.append(idx)
                history.at[idx, "Status"] = "OPEN"
                history.at[idx, "Result"] = "PENDING"
                history.at[idx, "Result Detail"] = ""
                history.at[idx, "Days Taken"] = ""
                history.at[idx, "Outcome Message"] = "Reset: prior result had event date before prediction"
                history.at[idx, "Recommendation"] = ""
                history.at[idx, "Evaluation Date"] = ""
                history.at[idx, "Exit Price"] = None
                history.at[idx, "Return %"] = None
                changed = True
        except Exception:
            continue

    if invalid_closed:
        open_mask = open_mask | history.index.isin(invalid_closed)

    if force_all:
        # Re-check everything including previously closed (after repair)
        open_indices = history.index.tolist()
    else:
        open_indices = history.index[open_mask].tolist()

    # Prefer evaluating newest predictions first
    try:
        open_indices = sorted(
            open_indices,
            key=lambda i: pd.to_datetime(history.at[i, "Prediction Date"], errors="coerce")
            or pd.Timestamp.min,
            reverse=True,
        )
    except Exception:
        pass

    MAX_EVAL_PER_RUN = 800 if force_all else 250
    open_indices = open_indices[:MAX_EVAL_PER_RUN]

    for idx in open_indices:

        status = str(history.at[idx, "Status"]).upper().strip()
        result = str(history.at[idx, "Result"]).upper().strip()

        if status == "CLOSED" and result in [
            "TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED", "WIN", "LOSS",
        ]:
            continue

        symbol = clean_symbol(str(history.at[idx, "Stock"]))
        entry = safe_float(history.at[idx, "Entry"])
        # ALWAYS use original locked levels from the sheet
        target, stop = locked_levels.get(idx, (0.0, 0.0))
        target = safe_float(target)
        stop = safe_float(stop)
        call = str(history.at[idx, "Call"]).upper().strip()

        if entry <= 0 or not symbol:
            continue

        try:
            prediction_date = pd.to_datetime(history.at[idx, "Prediction Date"], errors="coerce")
            if pd.isna(prediction_date):
                continue
            pred_day = pd.Timestamp(prediction_date).normalize()
            try:
                if getattr(pred_day, "tz", None) is not None:
                    pred_day = pred_day.tz_localize(None)
            except Exception:
                pass
        except Exception:
            continue

        try:
            hold_days = int(safe_float(history.at[idx, "Hold Days"], DEFAULT_HOLD_DAYS))
        except Exception:
            hold_days = DEFAULT_HOLD_DAYS
        if hold_days <= 0:
            hold_days = DEFAULT_HOLD_DAYS

        data = _eval_price_history(symbol)
        if data.empty:
            continue

        # STRICT: only bars on/after prediction calendar day.
        # NEVER fall back to data.tail() — that caused stop dates BEFORE prediction date.
        try:
            idx_norm = pd.DatetimeIndex(pd.to_datetime(data.index)).tz_localize(None)
            data = data.copy()
            data.index = idx_norm
        except Exception:
            pass
        future = data[data.index.normalize() >= pred_day]
        # Allow 1 calendar day slack only if pred is "today" and last bar is yesterday (market closed)
        if future.empty:
            last_bar = pd.Timestamp(data.index[-1]).normalize()
            if pred_day <= last_bar + pd.Timedelta(days=1) and pred_day >= last_bar - pd.Timedelta(days=1):
                future = data[data.index.normalize() >= last_bar]
            else:
                # No price history after the call — keep OPEN, do not invent a past hit
                continue

        if future.empty:
            continue

        # Drop any residual bars before prediction (safety)
        future = future[future.index.normalize() >= pred_day]
        if future.empty:
            continue

        latest_date = pd.Timestamp(future.index[-1])
        trading_days = max(1, len(future))

        result_value = None
        exit_price = None
        event_date = None
        days_taken = None
        outcome_message = ""
        recommendation = ""
        result_detail = ""
        suggestion = str(history.at[idx, "Suggestion"] or "")

        # ----------------------------------------------------
        # BUY / HOLD — compare High/Low to ORIGINAL target/stop
        # ----------------------------------------------------
        if call in ["BUY", "HOLD"]:

            for i, (ts, candle) in enumerate(future.iterrows()):
                bar_day = pd.Timestamp(ts).normalize()
                if bar_day < pred_day:
                    continue  # never count pre-call price action
                high = safe_float(candle.get("High"))
                low = safe_float(candle.get("Low"))
                day_num = i + 1

                # Check both: if both hit same day, prefer stop first (conservative)
                stop_hit = stop > 0 and low <= stop
                tgt_hit = target > 0 and high >= target

                if stop_hit and tgt_hit:
                    # Same bar: use open vs levels — if gap both ways, mark stop (safer)
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%\n"
                        f"(Same day also touched original target ₹{target:,.2f} — stop prioritised)"
                    )
                    outcome_message = f"Stop Loss ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT"
                    break

                if stop_hit:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%\n"
                        f"Original Target (unchanged): ₹{target:,.2f} | Original SL: ₹{stop:,.2f}"
                    )
                    outcome_message = f"Stop Loss ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT"
                    break

                if tgt_hit:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    profit = ((target - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED\n"
                        f"Target Reached On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Profit: +{profit:.2f}%\n"
                        f"Original Target (locked): ₹{target:,.2f} | Original SL (locked): ₹{stop:,.2f}"
                    )
                    outcome_message = f"Target ₹{target:,.2f} reached"
                    recommendation = (
                        "🎯 TARGET ACHIEVED — Book profit or trail stop. "
                        "Sheet Target/SL stay frozen at original values."
                    )
                    break

            # Live / last close check if daily path still open
            if result_value is None:
                last_close = safe_float(future["Close"].iloc[-1])
                try:
                    q = live_quote(symbol)
                    if q and q.get("price"):
                        last_close = safe_float(q["price"])
                except Exception:
                    pass

                if target > 0 and last_close >= target:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = latest_date
                    days_taken = trading_days
                    profit = ((target - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED\n"
                        f"Target Reached On: {event_date.strftime('%d %B %Y')} (price check)\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Profit: +{profit:.2f}%\n"
                        f"LTP/Close ₹{last_close:,.2f} ≥ Original Target ₹{target:,.2f}"
                    )
                    outcome_message = f"Target ₹{target:,.2f} reached"
                    recommendation = "🎯 TARGET ACHIEVED — Book profit or trail stop"
                elif stop > 0 and last_close <= stop:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = latest_date
                    days_taken = trading_days
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')} (price check)\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%\n"
                        f"LTP/Close ₹{last_close:,.2f} ≤ Original SL ₹{stop:,.2f}"
                    )
                    outcome_message = f"Stop Loss ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT"

            # Near-target SUGGESTION only (does not change Target/SL columns)
            if result_value is None and target > 0 and entry > 0:
                last_close = safe_float(future["Close"].iloc[-1])
                try:
                    q = live_quote(symbol)
                    if q and q.get("price"):
                        last_close = safe_float(q["price"])
                except Exception:
                    pass
                dist = (target - last_close) / target if target else 1
                if 0 < dist <= 0.02:
                    # Within 2% of original target
                    new_t = round(target * 1.04, 2)
                    new_sl = round(max(entry, last_close * 0.98), 2)
                    suggestion = (
                        f"NEAR ORIGINAL TARGET ₹{target:,.2f} (LTP ₹{last_close:,.2f}). "
                        f"Optional idea only — do NOT overwrite sheet: "
                        f"trail SL toward ₹{new_sl:,.2f}, stretch target idea ₹{new_t:,.2f}. "
                        f"Original Target/SL remain locked in history."
                    )
                elif last_close >= target * 0.97:
                    suggestion = (
                        f"Approaching locked target ₹{target:,.2f}. "
                        f"Consider booking partial profit; sheet levels stay unchanged."
                    )

            if result_value is None and trading_days >= hold_days:
                candle = future.iloc[min(hold_days - 1, len(future) - 1)]
                exit_price = safe_float(candle["Close"])
                event_date = pd.Timestamp(future.index[min(hold_days - 1, len(future) - 1)])
                days_taken = min(hold_days, trading_days)
                result_value = "HOLDING PERIOD COMPLETED"
                pnl_pct = ((exit_price - entry) / entry * 100.0) if entry else 0.0
                toward = "TOWARD PROFIT" if pnl_pct >= 0 else "TOWARD LOSS"
                result_detail = (
                    f"⏰ HOLDING PERIOD COMPLETED — {toward}\n"
                    f"Neither Target ₹{target:,.2f} nor Stop ₹{stop:,.2f} was hit.\n"
                    f"Exit Date: {event_date.strftime('%d %B %Y')} · Days: {days_taken}\n"
                    f"Exit Close: ₹{exit_price:,.2f} vs Entry ₹{entry:,.2f}\n"
                    f"{'Profit' if pnl_pct >= 0 else 'Loss'}: {pnl_pct:+.2f}%"
                )
                outcome_message = f"Holding period over · {toward} ({pnl_pct:+.2f}%)"
                recommendation = (
                    f"⚠️ HOLDING PERIOD OVER — closed toward profit ({pnl_pct:+.2f}%). Review / book."
                    if pnl_pct >= 0 else
                    f"⚠️ HOLDING PERIOD OVER — closed toward loss ({pnl_pct:+.2f}%). Exit / reassess."
                )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------
        elif call == "SELL":

            for i, (ts, candle) in enumerate(future.iterrows()):
                high = safe_float(candle.get("High"))
                low = safe_float(candle.get("Low"))
                day_num = i + 1

                if target > 0 and low <= target:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    profit = ((entry - target) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED (SELL)\n"
                        f"Target Reached On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Profit: +{profit:.2f}%\n"
                        f"Original Target/SL locked in sheet"
                    )
                    outcome_message = f"Sell target ₹{target:,.2f} reached"
                    recommendation = "🎯 TARGET ACHIEVED — Book profit / cover short"
                    break

                if stop > 0 and high >= stop:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = pd.Timestamp(ts)
                    days_taken = day_num
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT (SELL)\n"
                        f"Stop Loss Hit On: {event_date.strftime('%d %B %Y')}\n"
                        f"Days Taken: {days_taken} Days\n"
                        f"Loss: {loss:.2f}%"
                    )
                    outcome_message = f"Sell stop ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT — Cover position"
                    break

            if result_value is None:
                last_close = safe_float(future["Close"].iloc[-1])
                try:
                    q = live_quote(symbol)
                    if q and q.get("price"):
                        last_close = safe_float(q["price"])
                except Exception:
                    pass
                if target > 0 and last_close <= target:
                    result_value = "TARGET ACHIEVED"
                    exit_price = target
                    event_date = latest_date
                    days_taken = trading_days
                    profit = ((entry - target) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🎯 TARGET ACHIEVED (SELL)\n"
                        f"LTP/Close ₹{last_close:,.2f} ≤ Original Target ₹{target:,.2f}\n"
                        f"Profit: +{profit:.2f}%"
                    )
                    outcome_message = f"Sell target ₹{target:,.2f} reached"
                    recommendation = "🎯 TARGET ACHIEVED — Book profit"
                elif stop > 0 and last_close >= stop:
                    result_value = "STOP LOSS HIT"
                    exit_price = stop
                    event_date = latest_date
                    days_taken = trading_days
                    loss = ((stop - entry) / entry) * 100 if entry else 0
                    result_detail = (
                        f"🔴 STOP LOSS HIT (SELL)\n"
                        f"LTP/Close ₹{last_close:,.2f} ≥ Original SL ₹{stop:,.2f}\n"
                        f"Loss: {loss:.2f}%"
                    )
                    outcome_message = f"Sell stop ₹{stop:,.2f} hit"
                    recommendation = "🔴 SELL / EXIT — Cover position"

            if result_value is None and trading_days >= hold_days:
                candle = future.iloc[min(hold_days - 1, len(future) - 1)]
                exit_price = safe_float(candle["Close"])
                event_date = pd.Timestamp(future.index[min(hold_days - 1, len(future) - 1)])
                days_taken = min(hold_days, trading_days)
                result_value = "HOLDING PERIOD COMPLETED"
                # SELL: profit when exit < entry
                pnl_pct = ((entry - exit_price) / entry * 100.0) if entry else 0.0
                toward = "TOWARD PROFIT" if pnl_pct >= 0 else "TOWARD LOSS"
                result_detail = (
                    f"⏰ HOLDING PERIOD COMPLETED — {toward} (SELL)\n"
                    f"Neither Target ₹{target:,.2f} nor Stop ₹{stop:,.2f} was hit.\n"
                    f"Exit Date: {event_date.strftime('%d %B %Y')} · Days: {days_taken}\n"
                    f"Exit Close: ₹{exit_price:,.2f} vs Entry ₹{entry:,.2f}\n"
                    f"{'Profit' if pnl_pct >= 0 else 'Loss'}: {pnl_pct:+.2f}%"
                )
                outcome_message = f"Holding period over · {toward} ({pnl_pct:+.2f}%)"
                recommendation = (
                    f"⚠️ HOLDING PERIOD OVER — toward profit ({pnl_pct:+.2f}%). Cover / review."
                    if pnl_pct >= 0 else
                    f"⚠️ HOLDING PERIOD OVER — toward loss ({pnl_pct:+.2f}%). Cover / reassess."
                )

        else:
            # Still save near-target suggestion updates for non-trade rows if needed
            if suggestion and str(history.at[idx, "Suggestion"]) != suggestion:
                history.loc[idx, "Suggestion"] = suggestion
                changed = True
            continue

        # Update suggestion for still-open rows
        if result_value is None:
            if suggestion and str(history.at[idx, "Suggestion"] or "") != suggestion:
                try:
                    history.loc[idx, "Suggestion"] = str(suggestion)
                    # Recommendation can show near-target idea without closing
                    if suggestion and not str(history.at[idx, "Recommendation"] or "").startswith("🎯"):
                        history.loc[idx, "Recommendation"] = str(suggestion)
                    changed = True
                except Exception:
                    pass
            continue

        if exit_price is None:
            continue

        if call == "SELL":
            return_pct = ((entry - exit_price) / entry) * 100 if entry else 0
        else:
            return_pct = ((exit_price - entry) / entry) * 100 if entry else 0

        eval_date_str = (
            event_date.strftime("%Y-%m-%d") if event_date is not None
            else latest_date.strftime("%Y-%m-%d")
        )
        days_taken_val = days_taken if days_taken is not None else ""

        # Write outcome only — NEVER overwrite Target or Stop Loss
        try:
            history.loc[idx, "Status"] = "CLOSED"
            history.loc[idx, "Evaluation Date"] = str(eval_date_str)
            history.loc[idx, "Exit Price"] = float(round(exit_price, 2))
            history.loc[idx, "Return %"] = float(round(return_pct, 2))
            history.loc[idx, "Result"] = str(result_value)
            history.loc[idx, "Result Detail"] = str(result_detail)
            history.loc[idx, "Days Taken"] = str(days_taken_val)
            history.loc[idx, "Outcome Message"] = str(outcome_message)
            history.loc[idx, "Recommendation"] = str(recommendation)
            if suggestion:
                history.loc[idx, "Suggestion"] = str(suggestion)
        except Exception:
            for col, val in [
                ("Status", "CLOSED"),
                ("Evaluation Date", str(eval_date_str)),
                ("Result", str(result_value)),
                ("Result Detail", str(result_detail)),
                ("Days Taken", str(days_taken_val)),
                ("Outcome Message", str(outcome_message)),
                ("Recommendation", str(recommendation)),
                ("Suggestion", str(suggestion or "")),
            ]:
                try:
                    history[col] = history[col].astype(object)
                    history.loc[idx, col] = val
                except Exception:
                    pass
            try:
                history.loc[idx, "Exit Price"] = float(round(exit_price, 2))
                history.loc[idx, "Return %"] = float(round(return_pct, 2))
            except Exception:
                pass

        # Restore locked target/stop in case anything touched them
        try:
            ot, os_ = locked_levels[idx]
            history.loc[idx, "Target"] = ot
            history.loc[idx, "Stop Loss"] = os_
        except Exception:
            pass

        changed = True

    # Final safety: re-apply every locked Target/SL so the sheet never drifts
    for idx, (ot, os_) in locked_levels.items():
        try:
            if idx in history.index:
                history.loc[idx, "Target"] = ot
                history.loc[idx, "Stop Loss"] = os_
        except Exception:
            pass

    if changed:
        try:
            history.to_csv(HISTORY_FILE, index=False)
        except Exception:
            out = history.copy()
            for c in out.columns:
                try:
                    out[c] = out[c].astype(object)
                except Exception:
                    pass
            out.to_csv(HISTORY_FILE, index=False)

    # Current price on open rows only — Entry/Target/SL stay locked
    try:
        history = refresh_history_current_prices(max_stocks=60)
    except Exception:
        pass

    # Keep auto trade tracker in sync after every evaluation
    try:
        sync_auto_trades_tracker(history)
    except Exception:
        pass

    # Align Paper book with history TARGET / STOP outcomes
    try:
        sync_outcomes_across_books(force_paper=True)
        history = normalize_history_df(load_history())
    except Exception:
        pass

    return history


def _trade_id(row) -> str:
    stock = display_symbol(row.get("Stock", ""))
    pred = str(row.get("Prediction Date", ""))[:19]
    call = str(row.get("Call", "")).upper()
    return f"{stock}|{pred}|{call}"


def sync_auto_trades_tracker(history: pd.DataFrame = None, max_live: int = 40) -> pd.DataFrame:
    """
    Automate trade tracking from recommendation_history.csv:
    - One row per prediction (Trade ID)
    - OPEN: refresh current price, unrealized %, distance to locked target/stop
    - CLOSED: lock realized % from history Result (target/stop never changed)
    Writes auto_trades_tracker.csv
    """
    ensure_files()
    if history is None:
        history = load_history()
    history = normalize_history_df(history)
    if history is None or history.empty:
        return pd.DataFrame()

    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    open_count = 0

    for _, row in history.iterrows():
        stock = display_symbol(row.get("Stock", ""))
        if not stock:
            continue
        call = str(row.get("Call", "")).upper().strip()
        if call not in {"BUY", "HOLD", "SELL"}:
            continue

        entry = safe_float(row.get("Entry"))
        target = safe_float(row.get("Target"))
        stop = safe_float(row.get("Stop Loss"))
        hold_days = int(safe_float(row.get("Hold Days"), DEFAULT_HOLD_DAYS))
        result = str(row.get("Result", "PENDING")).upper().strip()
        status = str(row.get("Status", "OPEN")).upper().strip()

        is_closed = any(
            x in result
            for x in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED", "WIN", "LOSS")
        ) or status == "CLOSED"

        try:
            pred_dt = pd.to_datetime(row.get("Prediction Date"), errors="coerce")
            days_held = 0
            if pd.notna(pred_dt):
                days_held = max(0, (pd.Timestamp(datetime.now().date()) - pd.Timestamp(pred_dt).normalize()).days)
        except Exception:
            days_held = 0
        days_rem = max(0, hold_days - days_held)

        cur = None
        unreal = None
        dist_t = None
        dist_s = None
        realized = safe_float(row.get("Return %")) if is_closed else None
        exit_px = safe_float(row.get("Exit Price")) if is_closed else None
        suggestion = str(row.get("Suggestion", "") or "")

        # Live price for open trades (cap network calls)
        if not is_closed and open_count < max_live:
            open_count += 1
            try:
                q = live_quote(stock)
                if q and q.get("price"):
                    cur = safe_float(q["price"])
            except Exception:
                cur = None
            if cur is None:
                try:
                    d = stock_history(stock, interval="1d")
                    if d is not None and not d.empty:
                        cur = safe_float(d["Close"].iloc[-1])
                except Exception:
                    pass
            if cur and entry > 0:
                if call == "SELL":
                    unreal = ((entry - cur) / entry) * 100
                else:
                    unreal = ((cur - entry) / entry) * 100
                if target > 0:
                    dist_t = ((target - cur) / target) * 100
                if stop > 0:
                    dist_s = ((cur - stop) / stop) * 100
                # Near-target suggestion (does not change locked levels)
                if call in {"BUY", "HOLD"} and target > 0 and cur >= target * 0.98 and cur < target:
                    suggestion = (
                        f"NEAR locked target ₹{target:,.2f} (LTP ₹{cur:,.2f}). "
                        f"Idea only: trail SL / partial book — sheet Target/SL stay frozen."
                    )
                elif call in {"BUY", "HOLD"} and target > 0 and cur >= target:
                    # Should have been closed by evaluate_history; flag anyway
                    suggestion = f"LTP ₹{cur:,.2f} ≥ locked target ₹{target:,.2f} — run Force re-check on Past Predictions."
                elif call in {"BUY", "HOLD"} and stop > 0 and cur <= stop:
                    suggestion = f"LTP ₹{cur:,.2f} ≤ locked SL ₹{stop:,.2f} — run Force re-check on Past Predictions."

        trade_status = "CLOSED" if is_closed else "OPEN"
        result_out = result if result and result not in {"", "NAN", "NONE"} else (
            "CLOSED" if is_closed else "OPEN / PENDING"
        )

        rows.append({
            "Trade ID": _trade_id(row),
            "Prediction Date": row.get("Prediction Date", ""),
            "Stock": stock,
            "Call": call,
            "Entry": entry,
            "Target": target,
            "Stop Loss": stop,
            "Hold Days": hold_days,
            "Status": trade_status,
            "Result": result_out,
            "Current Price": cur if cur is not None else "",
            "Unrealized %": round(unreal, 2) if unreal is not None else "",
            "Realized %": round(realized, 2) if realized is not None and is_closed else "",
            "Exit Price": exit_px if exit_px is not None and is_closed else "",
            "Days Held": days_held,
            "Days Remaining": days_rem if not is_closed else 0,
            "Distance to Target %": round(dist_t, 2) if dist_t is not None else "",
            "Distance to Stop %": round(dist_s, 2) if dist_s is not None else "",
            "Last Checked": now_s,
            "Suggestion": suggestion,
        })

    trades = pd.DataFrame(rows)
    if not trades.empty:
        # Prefer closed + high conviction open first
        trades = trades.sort_values(
            by=["Status", "Prediction Date"],
            ascending=[True, False],
        )
        trades.to_csv(TRADES_FILE, index=False)
    return trades


def load_auto_trades() -> pd.DataFrame:
    ensure_files()
    try:
        if TRADES_FILE.exists():
            return pd.read_csv(TRADES_FILE)
    except Exception:
        pass
    return pd.DataFrame()


def show_trade_tracker():
    """Automated trade tracking dashboard."""
    st.title("🤖 Auto Trade Tracker")
    st.caption(
        "Tracks every saved prediction as a trade. "
        "**Target & Stop Loss stay locked** from the original call. "
        "Status updates when Past Predictions evaluation marks target/stop/time exit."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⟳ Sync now", type="primary", key="trade_sync_btn"):
            with st.spinner("Evaluating outcomes + refreshing open prices..."):
                evaluate_history(force_all=True)
                trades = sync_auto_trades_tracker(max_live=60)
            st.success(f"Synced **{len(trades)}** trades → `{TRADES_FILE.name}`")
            st.rerun()
    with c2:
        auto_on = st.checkbox(
            "Auto-sync when opening this page",
            value=bool(st.session_state.get("trade_auto_sync", True)),
            key="trade_auto_sync",
        )
    with c3:
        st.caption(f"File: `{TRADES_FILE}`")

    # Auto sync at most every 3 minutes on page open
    if auto_on:
        last = st.session_state.get("_last_trade_sync")
        run_sync = last is None
        if last is not None:
            try:
                run_sync = (datetime.now() - last).total_seconds() > 180
            except Exception:
                run_sync = True
        if run_sync:
            with st.spinner("Auto-syncing trades..."):
                try:
                    evaluate_history(force_all=False)
                except Exception:
                    pass
                sync_auto_trades_tracker(max_live=40)
            st.session_state._last_trade_sync = datetime.now()

    trades = load_auto_trades()
    if trades.empty:
        st.info(
            "No trades yet. Run **FULL MARKET SCAN** so predictions are saved, "
            "then click **Sync now**."
        )
        return

    # Summary metrics
    status_u = trades["Status"].astype(str).str.upper()
    res_u = trades["Result"].astype(str).str.upper()
    n_open = int((status_u == "OPEN").sum())
    n_closed = int((status_u == "CLOSED").sum())
    n_tgt = int(res_u.str.contains("TARGET ACHIEVED", na=False).sum())
    n_sl = int(res_u.str.contains("STOP LOSS HIT", na=False).sum())

    closed = trades[status_u == "CLOSED"].copy()
    avg_r = None
    if not closed.empty and "Realized %" in closed.columns:
        avg_r = pd.to_numeric(closed["Realized %"], errors="coerce").mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Open trades", n_open)
    m2.metric("Closed trades", n_closed)
    m3.metric("🎯 Targets hit", n_tgt)
    m4.metric("🔴 Stops hit", n_sl)
    m5.metric("Avg realized %", f"{avg_r:+.2f}%" if avg_r == avg_r and avg_r is not None else "—")

    f1, f2 = st.columns(2)
    with f1:
        view = st.selectbox(
            "Show",
            ["ALL", "OPEN only", "CLOSED only", "TARGET ACHIEVED", "STOP LOSS HIT"],
            key="trade_view_filter",
        )
    with f2:
        call_f = st.selectbox("Call", ["ALL", "BUY", "HOLD", "SELL"], key="trade_call_filter")

    view_df = trades.copy()
    if view == "OPEN only":
        view_df = view_df[view_df["Status"].astype(str).str.upper() == "OPEN"]
    elif view == "CLOSED only":
        view_df = view_df[view_df["Status"].astype(str).str.upper() == "CLOSED"]
    elif view == "TARGET ACHIEVED":
        view_df = view_df[view_df["Result"].astype(str).str.upper().str.contains("TARGET ACHIEVED", na=False)]
    elif view == "STOP LOSS HIT":
        view_df = view_df[view_df["Result"].astype(str).str.upper().str.contains("STOP LOSS HIT", na=False)]
    if call_f != "ALL":
        view_df = view_df[view_df["Call"].astype(str).str.upper() == call_f]

    st.dataframe(view_df, use_container_width=True, hide_index=True)

    # Open trades cards
    open_df = trades[trades["Status"].astype(str).str.upper() == "OPEN"].head(20)
    if not open_df.empty:
        st.subheader("📡 Open trades — live tracking")
        for _, t in open_df.iterrows():
            cur = safe_float(t.get("Current Price"))
            entry = safe_float(t.get("Entry"))
            tgt = safe_float(t.get("Target"))
            sl = safe_float(t.get("Stop Loss"))
            ur = t.get("Unrealized %", "")
            st.markdown(
                f"""
                <div class="success-box" style="margin-bottom:10px;padding:12px;border-radius:8px;">
                <b>{t.get('Stock')}</b> · {t.get('Call')} · Entry ₹{entry:,.2f}<br>
                Locked Target ₹{tgt:,.2f} · Locked SL ₹{sl:,.2f}<br>
                LTP ₹{cur:,.2f} · Unrealized {ur}% ·
                Days held {t.get('Days Held')} · Remaining {t.get('Days Remaining')}<br>
                {t.get('Suggestion') or ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Automation flow: Market Scan → saves calls → evaluation marks target/stop → "
        "this tracker mirrors OPEN/CLOSED with live LTP for open rows. "
        "Original Target/SL columns are never modified."
    )


def load_paper_portfolio() -> pd.DataFrame:
    ensure_files()
    try:
        if PAPER_FILE.exists():
            return pd.read_csv(PAPER_FILE)
    except Exception:
        pass
    return pd.DataFrame()


def save_paper_portfolio(df: pd.DataFrame):
    try:
        df.to_csv(PAPER_FILE, index=False)
        bump_sync_version("paper_trade")
    except Exception:
        pass


def _norm_stock_key(s) -> str:
    return str(s or "").upper().replace(".NS", "").strip()


def _outcome_label(result) -> str:
    u = str(result or "").upper()
    if "TARGET" in u or u == "WIN":
        return "TARGET ACHIEVED"
    if "STOP" in u or u == "LOSS":
        return "STOP LOSS HIT"
    if "HOLDING" in u:
        return "HOLDING PERIOD COMPLETED"
    if "PENDING" in u or u in ("", "NAN", "NONE"):
        return "PENDING"
    return str(result or "PENDING")


def _rows_match_trade(stock, entry, side, open_dt, other_stock, other_entry, other_side, other_dt, entry_tol=0.03) -> bool:
    """Same underlying trade across history vs paper books."""
    if _norm_stock_key(stock) != _norm_stock_key(other_stock):
        return False
    s1 = str(side or "BUY").upper()
    s2 = str(other_side or "BUY").upper()
    if "SELL" in s1:
        s1 = "SELL"
    else:
        s1 = "BUY"
    if "SELL" in s2:
        s2 = "SELL"
    else:
        s2 = "BUY"
    if s1 != s2:
        return False
    e1, e2 = safe_float(entry), safe_float(other_entry)
    if e1 > 0 and e2 > 0:
        if abs(e1 - e2) / max(e1, e2) > entry_tol:
            return False
    try:
        d1 = pd.Timestamp(pd.to_datetime(open_dt, errors="coerce")).normalize()
        d2 = pd.Timestamp(pd.to_datetime(other_dt, errors="coerce")).normalize()
        if pd.notna(d1) and pd.notna(d2) and abs((d1 - d2).days) > 3:
            return False
    except Exception:
        pass
    return True


def evaluate_all_paper_trades(force_closed: bool = False) -> pd.DataFrame:
    """Run OHLC target/stop check on every paper row (same rules as past predictions)."""
    paper = load_paper_portfolio()
    if paper is None or paper.empty:
        return pd.DataFrame()
    rows = []
    for _, r in paper.iterrows():
        rec = r.to_dict()
        status = str(rec.get("Status", "")).upper()
        result = _outcome_label(rec.get("Result"))
        if status == "CLOSED" and result in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED") and not force_closed:
            rows.append(rec)
            continue
        try:
            res = backtest_paper_trade(rec)
            rec.update(res)
            # Normalize labels
            rec["Result"] = _outcome_label(rec.get("Result"))
            if rec["Result"] in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED"):
                rec["Status"] = "CLOSED"
        except Exception:
            pass
        rows.append(rec)
    out = pd.DataFrame(rows)
    save_paper_portfolio(out)
    return out


def sync_outcomes_across_books(force_paper: bool = True) -> dict:
    """
    One source of truth for TARGET ACHIEVED / STOP LOSS HIT:
    - Re-check open paper trades on OHLC
    - Re-check history (caller may run evaluate_history first)
    - Copy closed outcomes both ways when stock+side+entry match
    So Strategy buy → paper and Past Predictions stay aligned.
    """
    stats = {"paper_updated": 0, "history_updated": 0, "linked": 0}
    try:
        if force_paper:
            evaluate_all_paper_trades(force_closed=False)
    except Exception:
        pass

    paper = load_paper_portfolio()
    history = normalize_history_df(load_history())
    if history is None:
        history = pd.DataFrame()
    if paper is None:
        paper = pd.DataFrame()

    if paper.empty and history.empty:
        return stats

    # --- Paper → History (closed paper updates matching open history) ---
    if not paper.empty and not history.empty:
        for pi, prow in paper.iterrows():
            pres = _outcome_label(prow.get("Result"))
            if pres not in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED"):
                continue
            for hi in history.index:
                hrow = history.loc[hi]
                hres = _outcome_label(hrow.get("Result"))
                if hres in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED"):
                    # already closed — still align if labels differ
                    if not _rows_match_trade(
                        prow.get("Stock"), prow.get("Entry"), prow.get("Side"), prow.get("Open Date"),
                        hrow.get("Stock"), hrow.get("Entry"), hrow.get("Call"), hrow.get("Prediction Date"),
                    ):
                        continue
                elif not _rows_match_trade(
                    prow.get("Stock"), prow.get("Entry"), prow.get("Side"), prow.get("Open Date"),
                    hrow.get("Stock"), hrow.get("Entry"), hrow.get("Call"), hrow.get("Prediction Date"),
                ):
                    continue
                else:
                    # open history ← closed paper
                    pass

                if not _rows_match_trade(
                    prow.get("Stock"), prow.get("Entry"), prow.get("Side"), prow.get("Open Date"),
                    hrow.get("Stock"), hrow.get("Entry"), hrow.get("Call"), hrow.get("Prediction Date"),
                ):
                    continue

                # Apply paper outcome onto history (keep locked target/stop)
                history.at[hi, "Result"] = pres
                history.at[hi, "Status"] = "CLOSED"
                exit_px = prow.get("Exit Price")
                if exit_px not in ("", None) and str(exit_px).lower() != "nan":
                    history.at[hi, "Exit Price"] = safe_float(exit_px)
                ret = safe_float(prow.get("Return %"), None)
                if ret is not None:
                    history.at[hi, "Return %"] = ret
                exit_d = str(prow.get("Exit Date") or "")
                if exit_d:
                    history.at[hi, "Evaluation Date"] = exit_d
                    if pres == "TARGET ACHIEVED":
                        history.at[hi, "Result Detail"] = (
                            f"🎯 TARGET ACHIEVED\nTarget Reached On: {exit_d}\n"
                            f"Profit: {safe_float(prow.get('Return %')):+.2f}%\n"
                            f"(Synced from Paper trade)"
                        )
                        history.at[hi, "Recommendation"] = "🎯 TARGET ACHIEVED — Book profit"
                    elif pres == "STOP LOSS HIT":
                        history.at[hi, "Result Detail"] = (
                            f"🔴 STOP LOSS HIT\nStop Loss Hit On: {exit_d}\n"
                            f"Loss: {safe_float(prow.get('Return %')):+.2f}%\n"
                            f"(Synced from Paper trade)"
                        )
                        history.at[hi, "Recommendation"] = "🔴 STOP LOSS HIT — Exit"
                stats["history_updated"] += 1
                stats["linked"] += 1

        # --- History → Paper ---
        for hi in history.index:
            hrow = history.loc[hi]
            hres = _outcome_label(hrow.get("Result"))
            if hres not in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED"):
                continue
            for pi in paper.index:
                prow = paper.loc[pi]
                if not _rows_match_trade(
                    hrow.get("Stock"), hrow.get("Entry"), hrow.get("Call"), hrow.get("Prediction Date"),
                    prow.get("Stock"), prow.get("Entry"), prow.get("Side"), prow.get("Open Date"),
                ):
                    continue
                pres = _outcome_label(prow.get("Result"))
                if pres == hres and str(prow.get("Status", "")).upper() == "CLOSED":
                    continue
                paper.at[pi, "Result"] = hres
                paper.at[pi, "Status"] = "CLOSED"
                exit_px = safe_float(hrow.get("Exit Price"))
                entry = safe_float(prow.get("Entry")) or safe_float(hrow.get("Entry"))
                shares = safe_int(prow.get("Shares"), 1) or 1
                if exit_px <= 0:
                    if hres == "TARGET ACHIEVED":
                        exit_px = safe_float(hrow.get("Target"))
                    elif hres == "STOP LOSS HIT":
                        exit_px = safe_float(hrow.get("Stop Loss"))
                paper.at[pi, "Exit Price"] = round(exit_px, 2) if exit_px else prow.get("Exit Price")
                side = str(prow.get("Side", "BUY")).upper()
                if entry > 0 and exit_px > 0:
                    if "SELL" in side:
                        ret = (entry - exit_px) / entry * 100
                    else:
                        ret = (exit_px - entry) / entry * 100
                    paper.at[pi, "Return %"] = round(ret, 2)
                    paper.at[pi, "PnL ₹"] = round(shares * entry * ret / 100, 2)
                else:
                    ret = safe_float(hrow.get("Return %"), None)
                    if ret is not None:
                        paper.at[pi, "Return %"] = ret
                        paper.at[pi, "PnL ₹"] = round(shares * entry * ret / 100, 2) if entry else ""
                eval_d = str(hrow.get("Evaluation Date") or "")
                if eval_d:
                    paper.at[pi, "Exit Date"] = eval_d[:10] if len(eval_d) >= 10 else eval_d
                stats["paper_updated"] += 1
                stats["linked"] += 1

    try:
        if not paper.empty:
            save_paper_portfolio(paper)
    except Exception:
        pass
    try:
        if not history.empty:
            history = normalize_history_df(history)
            history.to_csv(HISTORY_FILE, index=False)
            bump_sync_version("outcomes_synced")
    except Exception:
        pass
    return stats


def trade_quality_check(
    entry: float,
    target: float,
    stop: float,
    side: str = "BUY",
    min_rr: float = 1.5,
    capital: float = 50000.0,
    risk_pct_capital: float = 1.0,
) -> dict:
    """
    Block weak trades: Stop must be valid; Reward/Risk >= min_rr (default 1.5).
    Returns ok, rr, risk_rs (₹ at stop for sized shares), reward_rs, shares, reasons.
    """
    side = str(side or "BUY").upper()
    entry = safe_float(entry)
    target = safe_float(target)
    stop = safe_float(stop)
    out = {
        "ok": False,
        "rr": 0.0,
        "risk_per_share": 0.0,
        "reward_per_share": 0.0,
        "risk_rs": 0.0,
        "reward_rs": 0.0,
        "shares": 0,
        "reasons": [],
        "label": "BLOCKED",
    }
    if entry <= 0:
        out["reasons"].append("No valid entry price")
        return out
    if stop <= 0:
        out["reasons"].append("Stop Loss is 0 or missing — never trade without a stop")
        return out
    if target <= 0:
        out["reasons"].append("Target is 0 or missing")
        return out

    if "SELL" in side:
        if target >= entry:
            out["reasons"].append("SELL: Target must be below Entry")
            return out
        if stop <= entry:
            out["reasons"].append("SELL: Stop must be above Entry")
            return out
        risk_ps = stop - entry
        reward_ps = entry - target
    else:
        if target <= entry:
            out["reasons"].append("BUY: Target must be above Entry")
            return out
        if stop >= entry:
            out["reasons"].append("BUY: Stop must be below Entry")
            return out
        risk_ps = entry - stop
        reward_ps = target - entry

    if risk_ps <= 0:
        out["reasons"].append("Risk per share is zero")
        return out

    rr = reward_ps / risk_ps
    out["rr"] = round(rr, 2)
    out["risk_per_share"] = round(risk_ps, 2)
    out["reward_per_share"] = round(reward_ps, 2)

    risk_budget = capital * (risk_pct_capital / 100.0)
    shares = max(1, int(risk_budget / risk_ps))
    out["shares"] = shares
    out["risk_rs"] = round(shares * risk_ps, 2)
    out["reward_rs"] = round(shares * reward_ps, 2)

    if rr + 1e-9 < min_rr:
        out["reasons"].append(
            f"R:R {rr:.2f} < {min_rr:.1f} — skip (need reward ≥ {min_rr}× risk)"
        )
        out["label"] = f"WEAK R:R {rr:.2f}"
        return out

    out["ok"] = True
    out["label"] = f"R:R {rr:.2f} ✓"
    out["reasons"].append(
        f"OK · R:R {rr:.2f} · risk ₹{out['risk_rs']:,.0f} · "
        f"reward ₹{out['reward_rs']:,.0f} · {shares} shares (1% capital risk)"
    )
    return out


def execute_paper_order(
    stock: str,
    side: str = "BUY",
    entry: float = 0.0,
    target: float = 0.0,
    stop: float = 0.0,
    shares: int = 0,
    source: str = "",
    hold_days: int = 15,
    risk_pct: float = 1.0,
    capital: float = 50000.0,
    force: bool = False,
) -> dict:
    """
    Execute paper BUY/SELL from any page.
    Blocks if Stop invalid or R:R < 1.5 (unless force=True).
    Sizes shares from 1% capital risk to stop when shares=0.
    """
    stock = display_symbol(stock)
    side = str(side or "BUY").upper()
    if side not in ("BUY", "SELL"):
        side = "BUY"
    capital = safe_float(st.session_state.get("paper_capital", capital), capital) or 50000.0
    risk_pct = safe_float(st.session_state.get("paper_risk_pct", risk_pct), risk_pct) or 1.0

    if entry <= 0:
        try:
            q = live_quote(stock)
            if q and q.get("price"):
                entry = safe_float(q["price"])
        except Exception:
            pass
    if entry <= 0:
        try:
            d = stock_history(clean_symbol(stock), interval="1d")
            if d is not None and not d.empty:
                entry = safe_float(d["Close"].iloc[-1])
        except Exception:
            pass
    if entry <= 0:
        return {"ok": False, "msg": f"No price for {stock}"}

    # Default target/stop from ATR if missing — always enforce RR ≥ 1.5
    if target <= 0 or stop <= 0:
        try:
            d = stock_history(clean_symbol(stock), interval="1d")
            if d is not None and len(d) > 30:
                d = calculate_indicators(d)
                atr = safe_float(d.iloc[-1].get("ATR")) or entry * 0.02
                if side == "BUY":
                    if stop <= 0:
                        stop = round(entry - 1.2 * atr, 2)
                    if target <= 0:
                        target = round(entry + 1.5 * max(entry - stop, 1.2 * atr), 2)
                else:
                    if stop <= 0:
                        stop = round(entry + 1.2 * atr, 2)
                    if target <= 0:
                        target = round(entry - 1.5 * max(stop - entry, 1.2 * atr), 2)
        except Exception:
            if side == "BUY":
                stop = stop if stop > 0 else round(entry * 0.97, 2)
                target = target if target > 0 else round(entry * 1.05, 2)
            else:
                stop = stop if stop > 0 else round(entry * 1.03, 2)
                target = target if target > 0 else round(entry * 0.95, 2)

    # Stretch target to min RR if stop valid but RR weak
    qcheck = trade_quality_check(entry, target, stop, side, min_rr=1.5, capital=capital, risk_pct_capital=risk_pct)
    if not qcheck["ok"] and not force:
        # Try auto-fix target once for valid direction/stop
        if qcheck["risk_per_share"] > 0 and stop > 0:
            if side == "BUY" and stop < entry:
                target = round(entry + 1.5 * (entry - stop), 2)
                qcheck = trade_quality_check(entry, target, stop, side, min_rr=1.5, capital=capital, risk_pct_capital=risk_pct)
            elif side == "SELL" and stop > entry:
                target = round(entry - 1.5 * (stop - entry), 2)
                qcheck = trade_quality_check(entry, target, stop, side, min_rr=1.5, capital=capital, risk_pct_capital=risk_pct)
        if not qcheck["ok"]:
            return {
                "ok": False,
                "msg": "Blocked: " + " · ".join(qcheck["reasons"][:3]),
                "quality": qcheck,
            }

    if shares <= 0:
        shares = int(qcheck.get("shares") or 0)
        if shares <= 0:
            risk_amt = capital * (risk_pct / 100.0)
            per_share_risk = abs(entry - stop)
            shares = max(1, int(risk_amt / per_share_risk)) if per_share_risk > 0 else 1

    row = {
        "Open Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Stock": stock,
        "Side": side,
        "Entry": round(entry, 2),
        "Target": round(target, 2),
        "Stop Loss": round(stop, 2),
        "Shares": int(shares),
        "Hold Days": int(hold_days or 15),
        "Source": source or "Manual",
        "Status": "OPEN",
        "Result": "PENDING",
        "Exit Date": "",
        "Exit Price": "",
        "Return %": "",
        "PnL ₹": "",
        "R:R": qcheck.get("rr", ""),
        "Risk ₹": qcheck.get("risk_rs", ""),
    }
    paper = load_paper_portfolio()
    paper = pd.concat([paper, pd.DataFrame([row])], ignore_index=True)
    save_paper_portfolio(paper)
    return {
        "ok": True,
        "msg": (
            f"{side} {shares} × {stock} @ ₹{entry:,.2f} | "
            f"T ₹{target:,.2f} | SL ₹{stop:,.2f} | "
            f"R:R {qcheck.get('rr', '—')} | risk ₹{qcheck.get('risk_rs', 0):,.0f} → Paper"
        ),
        "row": row,
        "quality": qcheck,
    }


def render_active_trade_buttons(
    stock: str,
    side_hint: str = "BUY",
    entry: float = 0.0,
    target: float = 0.0,
    stop: float = 0.0,
    key_prefix: str = "tr",
):
    """Active BUY / SELL buttons — show R:R & risk ₹; block weak R:R."""
    stock = display_symbol(stock)
    capital = safe_float(st.session_state.get("paper_capital", 50000), 50000) or 50000
    risk_pct = safe_float(st.session_state.get("paper_risk_pct", 1.0), 1.0) or 1.0
    q_buy = trade_quality_check(entry, target, stop, "BUY", min_rr=1.5, capital=capital, risk_pct_capital=risk_pct)
    # For SELL levels may be long-style on card — recompute if needed
    q_sell = trade_quality_check(
        entry,
        target if target < entry else 0,
        stop if stop > entry else 0,
        "SELL",
        min_rr=1.5,
        capital=capital,
        risk_pct_capital=risk_pct,
    )

    if entry > 0:
        if q_buy["ok"]:
            st.caption(
                f"✅ BUY quality: **{q_buy['label']}** · "
                f"risk **₹{q_buy['risk_rs']:,.0f}** · reward **₹{q_buy['reward_rs']:,.0f}** · "
                f"**{q_buy['shares']}** shares"
            )
        else:
            st.caption("⛔ BUY blocked: " + " · ".join(q_buy["reasons"][:2]))

    b1, b2, b3, b4 = st.columns(4)
    allow_weak = bool(st.session_state.get("paper_allow_weak_rr", True))
    with b1:
        buy_label = f"🟢 BUY {stock}" if (q_buy["ok"] or allow_weak) else f"⛔ BUY blocked"
        if st.button(buy_label, key=f"{key_prefix}_buy_{stock}", use_container_width=True,
                     disabled=not (q_buy["ok"] or allow_weak)):
            r = execute_paper_order(
                stock, "BUY", entry=entry, target=target, stop=stop,
                source=f"{side_hint}|{key_prefix}",
                force=allow_weak or q_buy["ok"],
            )
            if r.get("ok"):
                st.success(r["msg"])
            else:
                st.error(r.get("msg", "Order failed"))
    with b2:
        sell_label = f"🔴 SELL {stock}"
        if st.button(sell_label, key=f"{key_prefix}_sell_{stock}", use_container_width=True):
            r = execute_paper_order(
                stock, "SELL", entry=entry,
                target=target if target and target < entry else 0,
                stop=stop if stop and stop > entry else 0,
                source=f"{side_hint}|{key_prefix}",
                force=True,
            )
            if r.get("ok"):
                st.success(r["msg"])
            else:
                st.error(r.get("msg", "Order failed"))
    with b3:
        if st.button("📊 Analyse", key=f"{key_prefix}_an_{stock}", use_container_width=True):
            st.session_state.selected_stock = stock
            st.session_state.page = "Stock Analysis"
            st.rerun()
    with b4:
        if st.button("🧪 Paper book", key=f"{key_prefix}_pp_{stock}", use_container_width=True):
            st.session_state.page = "Paper Trading"
            st.rerun()


def backtest_paper_trade(row) -> dict:
    """
    Walk forward from Open Date using daily OHLC vs locked Target/SL.
    BUY: target above, stop below. SELL: target below, stop above.
    """
    stock = str(row.get("Stock", ""))
    side = str(row.get("Side", "BUY")).upper()
    entry = safe_float(row.get("Entry"))
    target = safe_float(row.get("Target"))
    stop = safe_float(row.get("Stop Loss"))
    hold_days = safe_int(row.get("Hold Days"), DEFAULT_HOLD_DAYS) or DEFAULT_HOLD_DAYS
    shares = safe_int(row.get("Shares"), 1) or 1

    out = {
        "Status": "OPEN",
        "Result": "PENDING",
        "Exit Date": "",
        "Exit Price": "",
        "Return %": "",
        "PnL ₹": "",
    }
    if entry <= 0 or not stock:
        return out

    try:
        open_dt = pd.to_datetime(row.get("Open Date"), errors="coerce")
        if pd.isna(open_dt):
            open_dt = pd.Timestamp(datetime.now().date())
        open_day = pd.Timestamp(open_dt).normalize()
    except Exception:
        open_day = pd.Timestamp(datetime.now().date())

    data = _eval_price_history(stock) if "_eval_price_history" in dir() else pd.DataFrame()
    try:
        data = _eval_price_history(stock)
    except Exception:
        try:
            data = stock_history(clean_symbol(stock), interval="1d")
            data = _normalize_ohlc_index(data)
        except Exception:
            data = pd.DataFrame()

    if data is None or data.empty:
        # Mark-to-market with live quote only
        try:
            q = live_quote(stock)
            if q and q.get("price"):
                cur = safe_float(q["price"])
                if side == "SELL":
                    ret = (entry - cur) / entry * 100
                else:
                    ret = (cur - entry) / entry * 100
                out["Return %"] = round(ret, 2)
                out["PnL ₹"] = round(shares * entry * ret / 100, 2)
        except Exception:
            pass
        return out

    future = data[data.index >= open_day]
    if future.empty:
        future = data.tail(5)

    for i, (ts, candle) in enumerate(future.iterrows()):
        high = safe_float(candle.get("High"))
        low = safe_float(candle.get("Low"))
        close = safe_float(candle.get("Close"))
        day_num = i + 1

        if side == "SELL":
            if target > 0 and low <= target:
                ret = (entry - target) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "TARGET ACHIEVED",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(target, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out
            if stop > 0 and high >= stop:
                ret = (entry - stop) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "STOP LOSS HIT",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(stop, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out
        else:
            if stop > 0 and low <= stop:
                ret = (stop - entry) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "STOP LOSS HIT",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(stop, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out
            if target > 0 and high >= target:
                ret = (target - entry) / entry * 100
                out.update({
                    "Status": "CLOSED",
                    "Result": "TARGET ACHIEVED",
                    "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    "Exit Price": round(target, 2),
                    "Return %": round(ret, 2),
                    "PnL ₹": round(shares * entry * ret / 100, 2),
                })
                return out

        if day_num >= hold_days:
            if side == "SELL":
                ret = (entry - close) / entry * 100
            else:
                ret = (close - entry) / entry * 100
            out.update({
                "Status": "CLOSED",
                "Result": "HOLDING PERIOD COMPLETED",
                "Exit Date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                "Exit Price": round(close, 2),
                "Return %": round(ret, 2),
                "PnL ₹": round(shares * entry * ret / 100, 2),
            })
            return out

    # Still open — mark to last close
    last = safe_float(future["Close"].iloc[-1])
    if side == "SELL":
        ret = (entry - last) / entry * 100
    else:
        ret = (last - entry) / entry * 100
    out["Return %"] = round(ret, 2)
    out["PnL ₹"] = round(shares * entry * ret / 100, 2)
    return out


def show_paper_trading():
    """Dummy trades at current price → paper portfolio + backtest."""
    st.markdown(
        """
        <div style="border-radius:16px;padding:18px 20px;margin-bottom:14px;
                    background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
                    border:1px solid #334155;">
          <div style="font-size:1.5rem;font-weight:700;color:#f8fafc;">Paper Trading Desk</div>
          <div style="color:#94a3b8;margin-top:4px;">Simulated orders · target & stop tracked · same book on every device after save</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    paper = load_paper_portfolio()
    if paper is None:
        paper = pd.DataFrame()
    if not paper.empty:
        p = paper.copy()
        p["Entry"] = pd.to_numeric(p.get("Entry"), errors="coerce").fillna(0)
        p["Shares"] = pd.to_numeric(p.get("Shares"), errors="coerce").fillna(0)
        p["Return %"] = pd.to_numeric(p.get("Return %"), errors="coerce")
        p["PnL ₹"] = pd.to_numeric(p.get("PnL ₹"), errors="coerce")
        p["Invested ₹"] = p["Entry"] * p["Shares"]
        status_u = p["Status"].astype(str).str.upper() if "Status" in p.columns else pd.Series(["OPEN"] * len(p))
        res_u = p["Result"].astype(str).str.upper() if "Result" in p.columns else pd.Series([""] * len(p))
        open_n = int((status_u == "OPEN").sum())
        tgt_n = int(res_u.str.contains("TARGET", na=False).sum())
        sl_n = int(res_u.str.contains("STOP", na=False).sum())
        invested = float(p["Invested ₹"].sum())
        pnl_total = float(p["PnL ₹"].fillna(0).sum())
        ret_closed = p.loc[res_u.str.contains("TARGET|STOP|HOLDING", na=False), "Return %"]
        avg_ret = float(ret_closed.mean()) if len(ret_closed.dropna()) else 0.0
        overall_ret = (pnl_total / invested * 100.0) if invested > 0 else 0.0

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:12px;">
              <div style="background:#020617;border-radius:12px;padding:12px;border:1px solid #1e293b;">
                <div style="color:#94a3b8;font-size:0.75rem;">INVESTED</div>
                <div style="color:#f8fafc;font-size:1.25rem;font-weight:700;">₹{invested:,.0f}</div>
              </div>
              <div style="background:#020617;border-radius:12px;padding:12px;border:1px solid #1e293b;">
                <div style="color:#94a3b8;font-size:0.75rem;">TOTAL P&L</div>
                <div style="color:{'#4ade80' if pnl_total>=0 else '#f87171'};font-size:1.25rem;font-weight:700;">₹{pnl_total:+,.0f}</div>
              </div>
              <div style="background:#020617;border-radius:12px;padding:12px;border:1px solid #1e293b;">
                <div style="color:#94a3b8;font-size:0.75rem;">OVERALL RETURN</div>
                <div style="color:{'#4ade80' if overall_ret>=0 else '#f87171'};font-size:1.25rem;font-weight:700;">{overall_ret:+.2f}%</div>
              </div>
              <div style="background:#020617;border-radius:12px;padding:12px;border:1px solid #1e293b;">
                <div style="color:#94a3b8;font-size:0.75rem;">🎯 TARGET / 🔴 STOP</div>
                <div style="color:#f8fafc;font-size:1.25rem;font-weight:700;">{tgt_n} / {sl_n}</div>
              </div>
              <div style="background:#020617;border-radius:12px;padding:12px;border:1px solid #1e293b;">
                <div style="color:#94a3b8;font-size:0.75rem;">OPEN · AVG RET%</div>
                <div style="color:#f8fafc;font-size:1.25rem;font-weight:700;">{open_n} · {avg_ret:+.1f}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # R:R column for all paper rows (including weak)
        try:
            p = p.copy()
            p["R:R"] = p.apply(_row_reward_risk, axis=1)
        except Exception:
            p["R:R"] = 0.0

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            pf_res = st.selectbox(
                "Result filter",
                ["ALL", "TARGET ACHIEVED", "STOP LOSS HIT", "OPEN / PENDING", "HOLDING PERIOD"],
                key="paper_res_filter",
            )
        with f2:
            pf_side = st.selectbox("Side", ["ALL", "BUY", "SELL"], key="paper_side_filter")
        with f3:
            pf_rr = st.selectbox(
                "R:R filter",
                ["ALL (incl. R:R < 1.5)", "R:R ≥ 1.5 only", "R:R < 1.5 only"],
                index=0,
                key="paper_rr_filter",
            )
        with f4:
            pf_n = st.selectbox("Rows", [25, 50, 100, 200, "ALL"], index=1, key="paper_rows")

        view = p.copy()
        if pf_side != "ALL" and "Side" in view.columns:
            view = view[view["Side"].astype(str).str.upper() == pf_side]
        ru = view["Result"].astype(str).str.upper() if "Result" in view.columns else pd.Series([""] * len(view))
        su = view["Status"].astype(str).str.upper() if "Status" in view.columns else pd.Series(["OPEN"] * len(view))
        if pf_res == "TARGET ACHIEVED":
            view = view[ru.str.contains("TARGET", na=False)]
        elif pf_res == "STOP LOSS HIT":
            view = view[ru.str.contains("STOP", na=False)]
        elif pf_res == "HOLDING PERIOD":
            view = view[ru.str.contains("HOLDING", na=False)]
        elif pf_res == "OPEN / PENDING":
            view = view[(su == "OPEN") | ru.str.contains("PENDING", na=False) | (ru == "") | (ru == "NAN")]

        rr = pd.to_numeric(view.get("R:R"), errors="coerce").fillna(0)
        if pf_rr.startswith("R:R ≥"):
            view = view[rr >= 1.5]
        elif pf_rr.startswith("R:R <"):
            view = view[(rr > 0) & (rr < 1.5)]

        paper_filt_stats = _success_from_df(view)
        st.caption(
            f"Filtered success **{paper_filt_stats.get('success', 0):.1f}%** · "
            f"🎯 {paper_filt_stats.get('wins', 0)} · 🔴 {paper_filt_stats.get('losses', 0)} · "
            f"**{len(view)}** rows (all R:R shown unless filtered)"
        )

        if "Open Date" in view.columns:
            view = view.sort_values("Open Date", ascending=False)
        if pf_n != "ALL":
            view = view.head(int(pf_n))

        show_cols = [c for c in [
            "Open Date", "Stock", "Side", "Shares", "Entry", "Invested ₹",
            "Target", "Stop Loss", "R:R", "Status", "Result", "Return %", "PnL ₹", "Source",
        ] if c in view.columns]
        st.markdown("##### Portfolio book")
        filterable_dataframe(view, key="paper_book_table", default_cols=show_cols, height=320)

    st.subheader("➕ New paper trade")
    st.caption("All levels accepted. R:R &lt; 1.5 shows a warning but can still be added. Shares sized to ~1% capital risk.")
    allow_weak = st.checkbox(
        "Allow R:R < 1.5 / incomplete levels (still record trade)",
        value=True,
        key="paper_allow_weak_rr",
        help="ON = every stock can enter the paper book. OFF = strict R:R ≥ 1.5 only.",
    )
    pc1, pc2 = st.columns(2)
    with pc1:
        st.session_state["paper_capital"] = st.number_input(
            "Capital ₹ (for sizing)",
            min_value=1000.0,
            value=float(st.session_state.get("paper_capital", 50000) or 50000),
            step=1000.0,
            key="paper_capital_input",
        )
    with pc2:
        st.session_state["paper_risk_pct"] = st.number_input(
            "Risk % of capital / trade",
            min_value=0.25,
            max_value=5.0,
            value=float(st.session_state.get("paper_risk_pct", 1.0) or 1.0),
            step=0.25,
            key="paper_risk_pct_input",
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        stock = st.text_input("NSE Symbol", value="", key="paper_sym").upper().strip()
    with c2:
        side = st.selectbox("Side", ["BUY", "SELL"], key="paper_side")
    with c3:
        shares = st.number_input("Shares", min_value=1, value=10, step=1, key="paper_shares")

    # Live price
    live_px = 0.0
    if stock:
        try:
            q = live_quote(stock)
            if q and q.get("price"):
                live_px = safe_float(q["price"])
        except Exception:
            pass
        if live_px <= 0:
            try:
                d = stock_history(clean_symbol(stock), interval="1d")
                if d is not None and not d.empty:
                    live_px = safe_float(d["Close"].iloc[-1])
            except Exception:
                pass

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        entry = st.number_input(
            "Entry price",
            min_value=0.0,
            value=float(live_px) if live_px > 0 else 0.0,
            step=0.05,
            format="%.2f",
            key="paper_entry",
        )
        if live_px > 0:
            st.caption(f"Live/last: ₹{live_px:,.2f}")
    with d2:
        if side == "SELL":
            default_tgt = round(entry * 0.95, 2) if entry else 0.0
            default_sl = round(entry * 1.03, 2) if entry else 0.0
        else:
            default_tgt = round(entry * 1.06, 2) if entry else 0.0
            default_sl = round(entry * 0.97, 2) if entry else 0.0
        target = st.number_input("Target", min_value=0.0, value=float(default_tgt), step=0.05, format="%.2f", key="paper_tgt")
    with d3:
        stop = st.number_input("Stop Loss", min_value=0.0, value=float(default_sl), step=0.05, format="%.2f", key="paper_sl")
    with d4:
        hold_days = st.number_input("Hold days", min_value=1, value=15, step=1, key="paper_hold")

    if side == "SELL" and target >= entry > 0:
        st.error("For SELL, **Target must be below Entry** (profit if price falls).")
    if side == "BUY" and 0 < target <= entry:
        st.warning("For BUY, Target is usually **above** Entry.")
    if side == "BUY" and 0 < stop >= entry:
        st.warning("For BUY, Stop Loss should be **below** Entry.")
    if side == "SELL" and 0 < stop <= entry:
        st.warning("For SELL, Stop Loss should be **above** Entry.")

    notes = st.text_input("Notes (optional)", key="paper_notes")

    _qc_preview = trade_quality_check(entry, target, stop, side, min_rr=1.5)
    if entry > 0 and stock:
        if _qc_preview.get("ok"):
            st.info(
                f"Quality OK · {_qc_preview['label']} · "
                f"risk ₹{_qc_preview['risk_rs']:,.0f} · reward ₹{_qc_preview['reward_rs']:,.0f} · "
                f"suggested shares {_qc_preview['shares']}"
            )
        else:
            st.error("Cannot add: " + " · ".join(_qc_preview.get("reasons", [])[:3]))

    if st.button("Add to paper portfolio", type="primary", key="paper_add"):
        if not stock or entry <= 0:
            st.error("Enter symbol and entry price.")
        else:
            r = execute_paper_order(
                stock, side, entry=entry, target=target, stop=stop,
                shares=int(shares) if shares else 0,
                source="Manual paper",
                hold_days=int(hold_days),
                force=bool(st.session_state.get("paper_allow_weak_rr", True)),
            )
            if r.get("ok"):
                st.success(r["msg"])
                st.rerun()
            else:
                st.error(r.get("msg", "Blocked by R:R / stop rules — enable 'Allow R:R < 1.5' above"))

    st.divider()
    st.subheader("📋 Paper portfolio")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("⟳ Backtest / update all", type="primary", key="paper_bt"):
            with st.spinner("Backtesting paper + syncing Past Predictions..."):
                evaluate_all_paper_trades(force_closed=False)
                stats = sync_outcomes_across_books(force_paper=False)
            st.success(
                f"Backtest done · linked **{stats.get('linked', 0)}** outcomes to Past Predictions"
            )
            st.rerun()
    with b2:
        if st.button("🔗 Sync with Past Predictions", key="paper_sync_hist"):
            with st.spinner("Aligning target/stop across Paper + History..."):
                try:
                    evaluate_history(force_all=False)
                except Exception:
                    pass
                stats = sync_outcomes_across_books(force_paper=True)
            st.success(
                f"Synced · paper updates {stats.get('paper_updated', 0)} · "
                f"history updates {stats.get('history_updated', 0)}"
            )
            st.rerun()
    with b3:
        if st.button("Clear closed trades", key="paper_clear_closed"):
            paper = load_paper_portfolio()
            if not paper.empty:
                paper = paper[paper["Status"].astype(str).str.upper() != "CLOSED"]
                save_paper_portfolio(paper)
            st.rerun()
    if st.button("Clear entire paper book", key="paper_clear_all"):
        save_paper_portfolio(pd.DataFrame(columns=[
            "Trade ID", "Open Date", "Stock", "Side", "Shares", "Entry", "Target",
            "Stop Loss", "Hold Days", "Status", "Result", "Exit Date", "Exit Price",
            "Return %", "PnL ₹", "Notes",
        ]))
        st.rerun()

    paper = load_paper_portfolio()
    if paper.empty:
        st.info("No dummy trades yet. Add one above.")
        return

    # Stats
    res_u = paper["Result"].astype(str).str.upper()
    n_tgt = int(res_u.str.contains("TARGET ACHIEVED", na=False).sum())
    n_sl = int(res_u.str.contains("STOP LOSS HIT", na=False).sum())
    closed = paper[paper["Status"].astype(str).str.upper() == "CLOSED"]
    avg_r = pd.to_numeric(closed.get("Return %"), errors="coerce").mean() if not closed.empty else None
    total_pnl = pd.to_numeric(paper.get("PnL ₹"), errors="coerce").fillna(0).sum()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", len(paper))
    m2.metric("🎯 Targets", n_tgt)
    m3.metric("🔴 Stops", n_sl)
    m4.metric("Avg return %", f"{avg_r:+.2f}%" if avg_r == avg_r and avg_r is not None else "—")
    m5.metric("Total PnL ₹", f"{total_pnl:,.0f}")

    st.dataframe(paper, use_container_width=True, hide_index=True)
    st.caption(f"Saved to `{PAPER_FILE}` — use this list to backtest dummy trades vs real price history.")


# ============================================================
# HISTORY STATISTICS
# ============================================================

def _stats_from_history_slice(history: pd.DataFrame) -> dict:
    """Target-vs-stop success on a history slice (predicted stocks only)."""
    empty = {
        "recommendations": 0,
        "closed": 0,
        "wins": 0,
        "losses": 0,
        "hold_done": 0,
        "open": 0,
        "success": 0.0,
        "win_rate_decided": 0.0,
        "avg_return": None,
        "avg_win": None,
        "avg_loss": None,
    }
    if history is None or history.empty:
        return empty

    result_col = history["Result"].astype(str).str.upper().str.strip()
    wins = int(result_col.str.contains("TARGET ACHIEVED", na=False).sum())
    wins += int(result_col.isin(["WIN"]).sum())
    losses = int(result_col.str.contains("STOP LOSS HIT", na=False).sum())
    losses += int(result_col.isin(["LOSS"]).sum())
    hold_done = int(result_col.str.contains("HOLDING PERIOD COMPLETED", na=False).sum())
    closed = wins + losses + hold_done
    recommendations = len(history)
    open_n = max(0, recommendations - closed)
    # Primary metric user cares about: targets / (targets + stops) only
    decided = wins + losses
    win_rate_decided = (wins / decided * 100) if decided > 0 else 0.0
    # Secondary: targets among all closed including time exits
    success = (wins / closed * 100) if closed > 0 else 0.0

    def _safe_avg(series):
        try:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty:
                return None
            v = float(s.mean())
            return None if v != v else round(v, 2)
        except Exception:
            return None

    avg_return = avg_win = avg_loss = None
    if "Return %" in history.columns:
        rets = pd.to_numeric(history["Return %"], errors="coerce")
        closed_mask = result_col.str.contains(
            "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
            regex=True,
            na=False,
        )
        win_mask = result_col.str.contains("TARGET ACHIEVED", na=False) | result_col.isin(["WIN"])
        loss_mask = result_col.str.contains("STOP LOSS HIT", na=False) | result_col.isin(["LOSS"])
        avg_return = _safe_avg(rets[closed_mask]) if closed_mask.any() else None
        avg_win = _safe_avg(rets[win_mask]) if win_mask.any() else None
        avg_loss = _safe_avg(rets[loss_mask]) if loss_mask.any() else None

    return {
        "recommendations": recommendations,
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "hold_done": hold_done,
        "open": open_n,
        "success": round(success, 1),
        "win_rate_decided": round(win_rate_decided, 1),
        "avg_return": avg_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


QUALITY_SOURCE_PATTERN = "SURE|STRATEGY|HIGH_CONV|PRECISION|MY_STRATEGY"


def quality_history_only(history: pd.DataFrame) -> pd.DataFrame:
    """Only Strategy / Sure / High-conviction / Precision — not bulk SCAN."""
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    if "Call Source" not in h.columns:
        h["Call Source"] = "SCAN"
    src = h["Call Source"].astype(str).str.upper()
    return h[src.str.contains(QUALITY_SOURCE_PATTERN, regex=True, na=False)].copy()


def _pnl_from_row(row) -> float:
    """Approx return % for a history row (BUY: exit vs entry; SELL inverse)."""
    ret = safe_float(row.get("Return %"), None)
    if ret is not None and ret == ret and abs(ret) > 0:
        return float(ret)
    entry = safe_float(row.get("Entry"))
    exit_px = safe_float(row.get("Exit Price"))
    if entry <= 0:
        return 0.0
    if exit_px <= 0:
        exit_px = safe_float(row.get("Current Price"))
    if exit_px <= 0:
        return 0.0
    call = str(row.get("Call", "BUY")).upper()
    if "SELL" in call:
        return (entry - exit_px) / entry * 100.0
    return (exit_px - entry) / entry * 100.0


def overall_statistics():
    """
    Headline success = TARGET / (TARGET + STOP) on **Sure + Strategy + High conv** only.
    Same-day duplicate stocks are collapsed so success is not inflated.
    """
    history = load_history()
    history = normalize_history_df(history)
    if history is None or history.empty:
        empty = _stats_from_history_slice(pd.DataFrame())
        empty.update({
            "buy_decided": 0.0, "quality_decided": 0.0, "scan_decided": 0.0,
            "buy_wins": 0, "buy_losses": 0, "quality_wins": 0, "quality_losses": 0,
            "scan_wins": 0, "scan_losses": 0, "quality_n": 0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "sum_win_pct": 0.0, "sum_loss_pct": 0.0,
            "unique_stocks": 0,
        })
        return empty

    quality = quality_history_only(history)
    # Dedupe same stock-day so metrics match what user sees on the page
    try:
        quality = dedupe_history_same_day(quality, one_stock_per_day=True)
    except Exception:
        pass

    scan = history
    if "Call Source" in history.columns:
        src = history["Call Source"].astype(str).str.upper()
        scan = history[~src.str.contains(QUALITY_SOURCE_PATTERN, regex=True, na=False)]

    q_stats = _stats_from_history_slice(quality)
    s_stats = _stats_from_history_slice(scan)
    if not quality.empty and "Call" in quality.columns:
        buy_q = quality[quality["Call"].astype(str).str.upper().str.contains("BUY", na=False)]
    else:
        buy_q = quality
    buy_stats = _stats_from_history_slice(buy_q)

    base = dict(q_stats)
    base["buy_decided"] = buy_stats["win_rate_decided"]
    base["buy_wins"] = buy_stats["wins"]
    base["buy_losses"] = buy_stats["losses"]
    base["buy_closed"] = buy_stats["closed"]
    base["quality_decided"] = q_stats["win_rate_decided"]
    base["quality_wins"] = q_stats["wins"]
    base["quality_losses"] = q_stats["losses"]
    base["quality_n"] = len(quality)
    base["scan_decided"] = s_stats["win_rate_decided"]
    base["scan_wins"] = s_stats["wins"]
    base["scan_losses"] = s_stats["losses"]
    base["scan_n"] = len(scan)
    decided = q_stats["wins"] + q_stats["losses"]
    base["success"] = q_stats["win_rate_decided"] if decided > 0 else 0.0
    base["win_rate_decided"] = base["success"]
    base["recommendations"] = len(quality)
    base["unique_stocks"] = int(quality["Stock"].nunique()) if not quality.empty and "Stock" in quality.columns else len(quality)

    # Profit / loss averages from closed quality rows
    sum_w = sum_l = 0.0
    n_w = n_l = 0
    if not quality.empty and "Result" in quality.columns:
        for _, r in quality.iterrows():
            ru = str(r.get("Result", "")).upper()
            pnl = _pnl_from_row(r)
            if "TARGET ACHIEVED" in ru or ru == "WIN":
                sum_w += pnl
                n_w += 1
            elif "STOP LOSS" in ru or ru == "LOSS":
                sum_l += pnl
                n_l += 1
    base["sum_win_pct"] = round(sum_w, 2)
    base["sum_loss_pct"] = round(sum_l, 2)
    base["avg_win_pct"] = round(sum_w / n_w, 2) if n_w else 0.0
    base["avg_loss_pct"] = round(sum_l / n_l, 2) if n_l else 0.0
    return base


def show_dashboard_past_data():
    """
    Full past data on Dashboard from CSV files so user can see history
    and success rate without leaving the home page.
    """
    st.subheader("📁 Past data (from CSV) — always available on Dashboard")
    st.caption(
        f"Stored in: `{HISTORY_FILE.name}`, `{TRADES_FILE.name}`, `{RESULT_FILE.name}`. "
        "Target/Stop Loss stay locked as first saved."
    )

    # Ensure tracker exists
    try:
        if not TRADES_FILE.exists() or TRADES_FILE.stat().st_size < 50:
            sync_auto_trades_tracker(max_live=15)
    except Exception:
        pass

    stats = overall_statistics()
    st.caption(
        "Success rate on Dashboard = **BUY predictions** that hit **Target vs Stop** only "
        "(not open rows). ~40–55% is a normal healthy band for swing RR &gt; 1."
    )
    a, b, c, d, e, f = st.columns(6)
    a.metric("History rows", stats["recommendations"])
    b.metric("BUY decided (T+SL)", int(stats.get("buy_wins", 0)) + int(stats.get("buy_losses", 0)))
    c.metric("🎯 BUY targets", stats.get("buy_wins", stats["wins"]))
    d.metric("🔴 BUY stops", stats.get("buy_losses", stats["losses"]))
    e.metric("Success rate (BUY)", f"{stats['success']}%")
    f.metric("Quality only", f"{stats.get('quality_decided', stats['win_rate_decided'])}%")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Still open", stats.get("open", 0))
    r2.metric("Scan BUY success", f"{stats.get('scan_decided', 0)}%")
    r3.metric(
        "Avg return (closed)",
        f"{stats['avg_return']:+.2f}%" if stats.get("avg_return") is not None else "—",
    )
    r4.metric(
        "Avg win / avg loss",
        (
            f"{stats['avg_win']:+.1f}% / {stats['avg_loss']:+.1f}%"
            if stats.get("avg_win") is not None and stats.get("avg_loss") is not None
            else "—"
        ),
    )

    history = normalize_history_df(load_history())
    trades = load_auto_trades()

    tab1, tab2, tab3 = st.tabs([
        "📜 All past predictions (CSV)",
        "✅ Closed outcomes",
        "📡 Open tracked trades",
    ])

    with tab1:
        if history is None or history.empty:
            st.info("No history CSV data yet. Run FULL MARKET SCAN once.")
        else:
            show_n = st.selectbox(
                "Rows to show",
                [25, 50, 100, 200, 500, "All"],
                index=1,
                key="dash_hist_rows",
            )
            h = history.sort_values("Prediction Date", ascending=False)
            if show_n != "All":
                h = h.head(int(show_n))
            cols = [
                c for c in [
                    "Prediction Date", "Stock", "Call Source", "Strategy", "Call", "Entry", "Current Price",
                    "Target", "Stop Loss",
                    "Hold Days", "Status", "Result", "Exit Price", "Return %",
                    "Days Taken", "Recommendation", "Suggestion",
                ] if c in h.columns
            ]
            st.dataframe(h[cols], use_container_width=True, hide_index=True)
            st.caption(
                f"Full file has **{len(history)}** rows → `{HISTORY_FILE}`. "
                "**Current Price** updates live; **Entry / Target / Stop Loss** stay locked."
            )

    with tab2:
        if history is None or history.empty:
            st.info("No closed trades yet.")
        else:
            res_u = history["Result"].astype(str).str.upper()
            closed = history[
                res_u.str.contains(
                    "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
                    regex=True,
                    na=False,
                )
            ].copy()
            closed = closed.sort_values("Prediction Date", ascending=False)
            if closed.empty:
                st.warning(
                    "No closed outcomes in CSV yet. Open **Past Predictions** → "
                    "**Force re-check all**, then return here."
                )
            else:
                # Mini success by call type
                if "Call" in closed.columns:
                    by_call = []
                    for call in ["BUY", "HOLD", "SELL"]:
                        sub = closed[closed["Call"].astype(str).str.upper() == call]
                        if sub.empty:
                            continue
                        ru = sub["Result"].astype(str).str.upper()
                        w = int(ru.str.contains("TARGET ACHIEVED", na=False).sum())
                        l = int(ru.str.contains("STOP LOSS HIT", na=False).sum())
                        rate = (w / (w + l) * 100) if (w + l) else 0
                        by_call.append({
                            "Call": call,
                            "Closed": len(sub),
                            "Wins": w,
                            "Losses": l,
                            "Win rate %": round(rate, 1),
                        })
                    if by_call:
                        st.write("**Success by call type**")
                        st.dataframe(pd.DataFrame(by_call), use_container_width=True, hide_index=True)

                cols = [
                    c for c in [
                        "Prediction Date", "Stock", "Call", "Entry", "Target", "Stop Loss",
                        "Result", "Exit Price", "Return %", "Days Taken", "Evaluation Date",
                    ] if c in closed.columns
                ]
                st.dataframe(closed[cols].head(100), use_container_width=True, hide_index=True)

    with tab3:
        if trades is None or trades.empty:
            st.info("Trade tracker CSV empty — will fill after scan / sync.")
        else:
            open_t = trades[trades["Status"].astype(str).str.upper() == "OPEN"]
            st.write(f"**{len(open_t)}** open trades in `{TRADES_FILE.name}`")
            st.dataframe(open_t.head(50), use_container_width=True, hide_index=True)

    # Persist a compact performance snapshot CSV for the user
    try:
        snap = pd.DataFrame([{
            "Snapshot Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total Calls": stats["recommendations"],
            "Closed": stats["closed"],
            "Wins": stats["wins"],
            "Losses": stats["losses"],
            "Hold Done": stats.get("hold_done", 0),
            "Open": stats.get("open", 0),
            "Success Rate %": stats["success"],
            "Win Rate Target vs SL %": stats["win_rate_decided"],
            "Avg Return %": stats.get("avg_return", ""),
            "Avg Win %": stats.get("avg_win", ""),
            "Avg Loss %": stats.get("avg_loss", ""),
        }])
        snap_path = APP_DIR / "performance_snapshot.csv"
        if snap_path.exists():
            old = pd.read_csv(snap_path)
            snap = pd.concat([old, snap], ignore_index=True).tail(200)
        snap.to_csv(snap_path, index=False)
        st.caption(f"Performance snapshot also saved → `{snap_path.name}`")
    except Exception:
        pass


def stock_statistics(stock):

    history = load_history()

    if history.empty:
        return {
            "times": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "success": 0,
        }

    h = history[
        history["Stock"].astype(str).str.upper()
        == display_symbol(stock).upper()
    ]

    if h.empty:
        return {
            "times": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "success": 0,
        }

    result_col = h["Result"].astype(str).str.upper()

    wins = len(result_col[result_col.isin(["WIN", "TARGET ACHIEVED"])])
    losses = len(result_col[result_col.isin(["LOSS", "STOP LOSS HIT"])])
    closed = len(
        result_col[
            result_col.isin([
                "WIN",
                "LOSS",
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "HOLDING PERIOD COMPLETED",
            ])
        ]
    )

    success = wins / closed * 100 if closed else 0

    return {
        "times": len(h),
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "success": round(success, 1),
    }


# ============================================================
# PORTFOLIO
# ============================================================

def load_portfolio():

    try:

        df = pd.read_csv(
            PORTFOLIO_FILE
        )

        if df.empty:
            return df

        if "Stock" in df.columns:

            df["Stock"] = (
                df["Stock"]
                .astype(str)
                .apply(clean_symbol)
            )

        return df

    except Exception:

        return pd.DataFrame()


def portfolio_analysis(
    results
):

    portfolio = load_portfolio()

    if portfolio.empty:
        return pd.DataFrame()

    if results is None or results.empty:
        return pd.DataFrame()

    lookup = {
        str(
            row["Symbol"]
        ).upper(): row
        for _, row in results.iterrows()
    }

    output = []

    for _, p in portfolio.iterrows():

        symbol = clean_symbol(
            p.get(
                "Stock",
                ""
            )
        )

        if symbol not in lookup:
            continue

        r = lookup[
            symbol
        ]

        shares = safe_float(
            p.get(
                "Shares",
                0
            )
        )

        buy_price = safe_float(
            p.get(
                "Buy Price",
                0
            )
        )

        current = safe_float(
            r["Price"]
        )

        if buy_price <= 0:
            continue

        pnl = (
            current -
            buy_price
        ) * shares

        pnl_pct = (
            (
                current -
                buy_price
            )
            /
            buy_price
            *
            100
        )

        try:

            purchase_date = (
                pd.to_datetime(
                    p.get(
                        "Purchase Date",
                        datetime.now().date()
                    )
                ).date()
            )

        except Exception:

            purchase_date = (
                datetime.now().date()
            )

        try:

            max_days = int(
                safe_float(
                    p.get(
                        "Maximum Holding Days",
                        DEFAULT_HOLD_DAYS
                    ),
                    DEFAULT_HOLD_DAYS
                )
            )

        except Exception:

            max_days = (
                DEFAULT_HOLD_DAYS
            )

        days_held = max(
            0,
            len(
                pd.bdate_range(
                    purchase_date,
                    datetime.now().date()
                )
            ) - 1
        )

        days_remaining = max(
            0,
            max_days -
            days_held
        )

        action = "HOLD"

        reason = (
            "No immediate exit condition has been triggered."
        )

        # Target
        if current >= safe_float(
            r["Target"]
        ):

            action = "🎯 TARGET ACHIEVED — SELL / BOOK PROFIT"

            reason = (
                f"🎯 TARGET ACHIEVED\n"
                f"Previous Target: ₹{safe_float(r['Target']):,.2f}\n"
                f"Current Price: ₹{current:,.2f}\n"
                f"Recommendation: HOLD / TRAIL STOP LOSS  or  🔴 SELL / BOOK PROFIT"
            )

        # Stop loss
        elif current <= safe_float(
            r["Stop Loss"]
        ):

            action = "🔴 STOP LOSS HIT — SELL / EXIT"

            reason = (
                f"🔴 STOP LOSS HIT\n"
                f"Current Price: ₹{current:,.2f}\n"
                f"Stop Loss: ₹{safe_float(r['Stop Loss']):,.2f}\n"
                f"Recommendation: 🔴 SELL / EXIT"
            )

        # Time expiry
        elif days_held >= max_days:

            action = (
                "⏰ HOLDING PERIOD COMPLETED — SELL"
            )

            reason = (
                f"⏰ HOLDING PERIOD COMPLETED\n"
                f"Neither Target nor Stop Loss was reached within "
                f"{max_days} trading days.\n"
                f"Recommendation: ⚠️ EARLY EXIT / SELL"
            )

        # Model SELL
        elif str(
            r["Call"]
        ).upper() == "SELL":

            action = "SELL"

            reason = (
                "The current technical model is giving a SELL signal."
            )

        # Strong BUY
        elif (
            str(
                r["Call"]
            ).upper() == "BUY"
            and
            safe_float(
                r["Prediction"]
            ) >= 80
        ):

            action = (
                "BUY MORE - CAUTIOUS"
            )

            reason = (
                "The current model remains strong. "
                "Adding shares increases exposure and should be done cautiously."
            )

        output.append(
            {
                "Stock":
                    display_symbol(symbol),

                "Shares":
                    shares,

                "Buy Price":
                    buy_price,

                "Current Price":
                    current,

                "P&L":
                    round(
                        pnl,
                        2
                    ),

                "P&L %":
                    round(
                        pnl_pct,
                        2
                    ),

                "Action":
                    action,

                "Prediction":
                    r["Prediction"],

                "Risk %":
                    r["Risk %"],

                "Risk Level":
                    r["Risk Level"],

                "Target":
                    r["Target"],

                "Stop Loss":
                    r["Stop Loss"],

                "Days Held":
                    days_held,

                "Days Remaining":
                    days_remaining,

                "Maximum Holding Days":
                    max_days,

                "Reason":
                    reason,
            }
        )

    return pd.DataFrame(
        output
    )


# ============================================================
# SECTOR ANALYSIS
# ============================================================

def sector_table(
    results
):

    if results is None or results.empty:
        return pd.DataFrame()

    x = ensure_result_columns(results)

    if "Sector" not in x.columns:

        x["Sector"] = (
            x["Stock"]
            .apply(
                sector_of
            )
        )

    prediction = pd.to_numeric(
        x["Prediction"],
        errors="coerce"
    ).fillna(0)

    risk = pd.to_numeric(
        x["Risk %"],
        errors="coerce"
    ).fillna(50)

    call_score = x[
        "Call"
    ].map(
        {
            "BUY": 100,
            "HOLD": 50,
            "SELL": 0,
        }
    ).fillna(40)

    x["_score"] = (
        prediction * 0.60
        +
        (100 - risk) * 0.25
        +
        call_score * 0.15
    )

    s = (
        x.groupby(
            "Sector"
        )
        .agg(
            Stocks=(
                "Stock",
                "count"
            ),

            Avg_Prediction=(
                "Prediction",
                "mean"
            ),

            Avg_Risk=(
                "Risk %",
                "mean"
            ),

            BUY_Calls=(
                "Call",
                lambda z:
                int(
                    (
                        z ==
                        "BUY"
                    ).sum()
                )
            ),

            HOLD_Calls=(
                "Call",
                lambda z:
                int(
                    (
                        z ==
                        "HOLD"
                    ).sum()
                )
            ),

            SELL_Calls=(
                "Call",
                lambda z:
                int(
                    (
                        z ==
                        "SELL"
                    ).sum()
                )
            ),

            Strength=(
                "_score",
                "mean"
            ),
        )
        .reset_index()
    )

    s["BUY %"] = (
        s["BUY_Calls"]
        /
        s["Stocks"].replace(
            0,
            np.nan
        )
        *
        100
    ).fillna(0).round(1)

    s["Sector Strength"] = (
        s["Strength"]
        .clip(
            0,
            100
        )
        .round(1)
    )

    def priority(v):

        if v >= 80:
            return "VERY HIGH"

        if v >= 68:
            return "HIGH"

        if v >= 52:
            return "MEDIUM"

        if v >= 35:
            return "LOW"

        return "VERY LOW"

    s[
        "Sector Priority"
    ] = s[
        "Sector Strength"
    ].apply(
        priority
    )

    def trend(row):

        if (
            row["BUY %"] >= 55
            and
            row["Avg_Prediction"] >= 60
        ):
            return "BULLISH"

        if (
            row["SELL_Calls"]
            /
            max(
                row["Stocks"],
                1
            )
            >= 0.45
        ):
            return "BEARISH"

        return "NEUTRAL"

    s["Trend"] = s.apply(
        trend,
        axis=1
    )

    s = s.drop(
        columns=[
            "Strength"
        ]
    )

    s = s.sort_values(
        [
            "Sector Strength",
            "BUY %",
        ],
        ascending=[
            False,
            False,
        ]
    )

    s.reset_index(
        drop=True,
        inplace=True
    )

    s.insert(
        0,
        "Rank",
        range(
            1,
            len(s) + 1
        )
    )

    return s


# ============================================================
# FIND STOCK / NATURAL LANGUAGE SEARCH
# ============================================================

def find_stock_results(
    results,
    query
):

    if results is None or results.empty:
        return pd.DataFrame()

    query = str(
        query
    ).strip().lower()

    if not query:
        return results

    x = ensure_result_columns(results)

    # --------------------------------------------------------
    # Direct symbol
    # --------------------------------------------------------

    words = re.findall(
        r"[a-zA-Z0-9&]+",
        query
    )

    direct = []

    for word in words:

        word = word.upper()

        matches = x[
            x["Stock"]
            .astype(str)
            .str.upper()
            .eq(word)
        ]

        if not matches.empty:

            direct.append(
                matches
            )

    if direct:

        return pd.concat(
            direct
        ).drop_duplicates()

    # --------------------------------------------------------
    # CALL FILTERS
    # --------------------------------------------------------

    if (
        "buy" in query
        or
        "purchase" in query
    ):

        x = x[
            x["Call"] == "BUY"
        ]

    if "sell" in query:

        x = x[
            x["Call"] == "SELL"
        ]

    if "hold" in query:

        x = x[
            x["Call"] == "HOLD"
        ]

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    if (
        "low risk" in query
        or
        "safe" in query
    ):

        x = x[
            x["Risk Level"]
            == "LOW"
        ]

    elif "medium risk" in query:

        x = x[
            x["Risk Level"]
            == "MEDIUM"
        ]

    elif "high risk" in query:

        x = x[
            x["Risk Level"]
            .isin(
                [
                    "HIGH",
                    "VERY HIGH",
                ]
            )
        ]

    # --------------------------------------------------------
    # SECTORS
    # --------------------------------------------------------

    sector_keywords = [
        "bank",
        "banking",
        "it",
        "pharma",
        "energy",
        "metal",
        "metals",
        "auto",
        "automobile",
        "fmcg",
        "telecom",
        "defence",
        "defense",
        "utility",
        "utilities",
        "infrastructure",
        "retail",
        "consumer",
    ]

    selected_sector = None

    for word in sector_keywords:

        if word in query:

            if word in [
                "bank",
                "banking",
            ]:
                selected_sector = "Banking"

            elif word == "it":
                selected_sector = "IT"

            elif word == "pharma":
                selected_sector = "Pharma"

            elif word == "energy":
                selected_sector = "Energy"

            elif word in [
                "metal",
                "metals",
            ]:
                selected_sector = "Metals"

            elif word in [
                "auto",
                "automobile",
            ]:
                selected_sector = "Automobile"

            elif word == "fmcg":
                selected_sector = "FMCG"

            elif word == "telecom":
                selected_sector = "Telecom"

            elif word in [
                "defence",
                "defense",
            ]:
                selected_sector = "Defence"

            elif word in [
                "utility",
                "utilities",
            ]:
                selected_sector = "Utilities"

            elif word == "infrastructure":
                selected_sector = "Infrastructure"

            elif word == "retail":
                selected_sector = "Retail"

            elif word == "consumer":
                selected_sector = "Consumer"

            break

    if selected_sector:

        x = x[
            x["Sector"]
            ==
            selected_sector
        ]

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    under_match = re.search(
        r"(?:under|below|less than)\s*(?:₹|rs\.?|inr)?\s*([0-9]+(?:\.[0-9]+)?)",
        query
    )

    if under_match:

        limit = safe_float(
            under_match.group(1)
        )

        x = x[
            pd.to_numeric(
                x["Price"],
                errors="coerce"
            )
            <= limit
        ]

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if (
        "strong" in query
        or
        "high prediction" in query
        or
        "best" in query
    ):

        x = x[
            x["Prediction"]
            >= 70
        ]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if (
        "low risk" in query
        or
        "safe" in query
    ):

        x = x.sort_values(
            [
                "Risk %",
                "Prediction",
            ],
            ascending=[
                True,
                False,
            ]
        )

    else:

        x = x.sort_values(
            [
                "Prediction",
                "Risk %",
            ],
            ascending=[
                False,
                True,
            ]
        )

    return x


# ============================================================
# TECHNICAL CHART
# ============================================================

def indicator_education(result):
    """
    Teach-each-indicator view: priority, current number, rule of thumb,
    and stock-specific verdict so the user can decide manually.
    Priority 1 = most important for this model's decisions.
    """
    price = safe_float(result.get("Price"))
    rsi = safe_float(result.get("RSI"))
    macd_v = safe_float(result.get("MACD"))
    macd_s = safe_float(result.get("MACD Signal"))
    adx = safe_float(result.get("ADX"))
    stoch = safe_float(result.get("Stochastic"))
    vwap = safe_float(result.get("VWAP"))
    vol_r = safe_float(result.get("Volume Ratio"))
    data = result.get("Data")

    ema20 = ema50 = ema200 = bb_mid = atr_v = None
    if data is not None and not getattr(data, "empty", True):
        row = data.iloc[-1]
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        bb_mid = safe_float(row.get("BBMiddle"))
        atr_v = safe_float(row.get("ATR"))

    lessons = []

    # --- Priority 1: Trend structure (EMAs) ---
    if ema20 is not None:
        above20 = price > ema20
        lessons.append({
            "Priority": 1,
            "Indicator": "EMA 20 (short trend)",
            "Current value": f"Price ₹{price:,.2f} vs EMA20 ₹{ema20:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE EMA20 → short-term buyers in control. "
                "Price BELOW EMA20 → short-term weakness. "
                "Cross above EMA20 is often early bullish; cross below is early caution."
            ),
            "For THIS stock": (
                f"✅ Price is ABOVE EMA20 — short-term trend supports buying / holding."
                if above20 else
                f"⚠️ Price is BELOW EMA20 — short-term momentum is weak; wait for reclaim if possible."
            ),
            "Manual decision tip": "Do not fight a clear close below EMA20 on a fresh BUY unless other high-priority signals are very strong.",
        })

    if ema50 is not None:
        above50 = price > ema50
        lessons.append({
            "Priority": 1,
            "Indicator": "EMA 50 (medium trend)",
            "Current value": f"Price ₹{price:,.2f} vs EMA50 ₹{ema50:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE EMA50 → medium-term uptrend bias. "
                "Many swing traders only BUY when price > EMA50. "
                "EMA20 > EMA50 adds alignment (short trend agrees with medium)."
            ),
            "For THIS stock": (
                f"✅ Price is ABOVE EMA50 — medium-term structure is supportive."
                if above50 else
                f"⚠️ Price is BELOW EMA50 — medium-term trend is not confirmed bullish."
            ),
            "Manual decision tip": "Prefer BUY when price > EMA50. If below EMA50, treat as counter-trend / higher risk.",
        })

    if ema200 is not None:
        above200 = price > ema200
        lessons.append({
            "Priority": 2,
            "Indicator": "EMA 200 (long trend)",
            "Current value": f"Price ₹{price:,.2f} vs EMA200 ₹{ema200:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE EMA200 → long-term bull market for the stock. "
                "Price BELOW EMA200 → long-term caution; rallies often fail more often."
            ),
            "For THIS stock": (
                f"✅ Price is ABOVE EMA200 — long-term trend is bullish."
                if above200 else
                f"⚠️ Price is BELOW EMA200 — long-term structure is weak."
            ),
            "Manual decision tip": "Best swing BUYs are usually above EMA200. Below EMA200 = trade smaller or skip.",
        })

    # --- Priority 2: Momentum ---
    rsi_note = (
        "✅ RSI is in the healthy bullish band (50–70): strength without extreme overbought."
        if 50 <= rsi <= 70 else
        "✅ RSI is oversold (<30): bounce possible, but confirm with volume / EMA reclaim."
        if rsi < 30 else
        "⚠️ RSI is overbought (>75): late-stage strength; avoid chasing; tighten risk."
        if rsi > 75 else
        f"ℹ️ RSI is {rsi:.1f}: between weak and strong — look at EMA + MACD for direction."
    )
    lessons.append({
        "Priority": 2,
        "Indicator": "RSI (14)",
        "Current value": f"{rsi:.1f}",
        "Rule of thumb (general)": (
            "• RSI > 50 → bullish momentum bias (buyers stronger than sellers).\n"
            "• RSI 55–70 → often ideal for continuing uptrends.\n"
            "• RSI < 30 → oversold; possible rebound, not automatic BUY.\n"
            "• RSI > 75–80 → overbought; profit-taking / pullback risk rises.\n"
            "General teaching: RSI 55 > 50 is usually GOOD in an uptrend — it means momentum is positive, not exhausted."
        ),
        "For THIS stock": rsi_note,
        "Manual decision tip": "In uptrends, prefer RSI > 50. RSI 55–68 is often the ‘sweet zone’. Avoid new BUYs mainly because RSI is 80+.",
    })

    macd_bull = macd_v > macd_s
    lessons.append({
        "Priority": 2,
        "Indicator": "MACD vs Signal",
        "Current value": f"MACD {macd_v:.4f} | Signal {macd_s:.4f}",
        "Rule of thumb (general)": (
            "MACD ABOVE signal line → bullish momentum. "
            "MACD BELOW signal → bearish momentum. "
            "A fresh cross up is a classic buy trigger; cross down is a warning."
        ),
        "For THIS stock": (
            "✅ MACD is above signal — momentum is bullish."
            if macd_bull else
            "⚠️ MACD is below signal — momentum is bearish / weakening."
        ),
        "Manual decision tip": "Agreeing MACD + RSI > 50 + price > EMA20 is a strong manual BUY cluster.",
    })

    lessons.append({
        "Priority": 3,
        "Indicator": "ADX (trend strength)",
        "Current value": f"{adx:.1f}",
        "Rule of thumb (general)": (
            "ADX < 20 → weak / sideways trend (signals fail more). "
            "ADX 20–25 → trend emerging. "
            "ADX ≥ 25 → trend is meaningful. "
            "ADX does NOT say direction — only strength. Combine with EMA/MACD for direction."
        ),
        "For THIS stock": (
            f"✅ ADX {adx:.1f} ≥ 25 — trend is strong enough to trust direction signals."
            if adx >= 25 else
            f"ℹ️ ADX {adx:.1f} < 25 — trend is not strong; expect more chop; use tighter stops or wait."
        ),
        "Manual decision tip": "When ADX is low, reduce size or wait. When ADX is high and price > EMA50, trend-following BUYs work better.",
    })

    stoch_note = (
        "✅ Stochastic rising and not overbought (<80) — short-term momentum supportive."
        if stoch < 80 else
        "⚠️ Stochastic is elevated (>80) — short-term stretched; pullback risk."
    )
    lessons.append({
        "Priority": 3,
        "Indicator": "Stochastic %K",
        "Current value": f"{stoch:.1f}",
        "Rule of thumb (general)": (
            "%K > %D and %K < 80 → often bullish short-term. "
            "%K > 80 → overbought zone. "
            "%K < 20 → oversold zone."
        ),
        "For THIS stock": stoch_note,
        "Manual decision tip": "Use Stochastic to time entry inside a larger uptrend — not as the only reason to buy.",
    })

    if bb_mid is not None:
        lessons.append({
            "Priority": 3,
            "Indicator": "Bollinger middle (20 SMA)",
            "Current value": f"Price ₹{price:,.2f} vs BB mid ₹{bb_mid:,.2f}",
            "Rule of thumb (general)": (
                "Price above middle band → bullish bias inside the channel. "
                "Price below middle → bearish bias. "
                "Touching lower band can be mean-reversion buy in ranges; in strong downtrends it can keep falling."
            ),
            "For THIS stock": (
                "✅ Price is above Bollinger middle — bullish bias inside the band."
                if price > bb_mid else
                "⚠️ Price is below Bollinger middle — short-term bias is weaker."
            ),
            "Manual decision tip": "In trends, buy pullbacks toward the middle band while price stays above EMA50.",
        })

    if vwap > 0:
        lessons.append({
            "Priority": 3,
            "Indicator": "VWAP",
            "Current value": f"Price ₹{price:,.2f} vs VWAP ₹{vwap:,.2f}",
            "Rule of thumb (general)": (
                "Price ABOVE VWAP → institutional / average buyer is in profit (bullish for day/swing context). "
                "Price BELOW VWAP → average buyer underwater (pressure)."
            ),
            "For THIS stock": (
                "✅ Price is above VWAP — buying strength vs average traded price."
                if price > vwap else
                "⚠️ Price is below VWAP — weaker vs average traded price."
            ),
            "Manual decision tip": "Intraday: prefer longs above VWAP. Positional: treat as secondary confirmation.",
        })

    lessons.append({
        "Priority": 2,
        "Indicator": "Volume Ratio (vs 20-day avg)",
        "Current value": f"{vol_r:.2f}×",
        "Rule of thumb (general)": (
            "Volume ≥ 1.5× average → move is more meaningful (conviction). "
            "Volume ~1.0× → normal. "
            "Very low volume breakouts often fail."
        ),
        "For THIS stock": (
            f"✅ Volume is elevated ({vol_r:.2f}×) — price move has participation."
            if vol_r >= 1.5 else
            f"ℹ️ Volume is moderate ({vol_r:.2f}×)."
            if vol_r >= 1.1 else
            f"⚠️ Volume is not high ({vol_r:.2f}×) — breakout/breakdown is less trustworthy."
        ),
        "Manual decision tip": "Never trust a breakout on tiny volume. Rising price + rising volume is healthier.",
    })

    if atr_v is not None and atr_v > 0:
        lessons.append({
            "Priority": 2,
            "Indicator": "ATR (14) — volatility used for SL/Target",
            "Current value": f"₹{atr_v:,.2f}",
            "Rule of thumb (general)": (
                "ATR measures typical daily range. "
                "This model uses Stop ≈ Price − 1.5×ATR and Target ≈ Price + 2.5×ATR. "
                "Higher ATR = wider stop = higher Risk %."
            ),
            "For THIS stock": (
                f"Stop and target are placed from ATR so risk matches this stock’s normal movement "
                f"(~₹{atr_v:,.2f} per day typical range)."
            ),
            "Manual decision tip": "If ATR is large vs your capital, reduce quantity. Never use a stop tighter than ~1×ATR casually.",
        })

    lessons.append({
        "Priority": 1,
        "Indicator": "Risk % / Risk Level (model)",
        "Current value": f"{safe_float(result.get('Risk %')):.2f}% · {result.get('Risk Level')}",
        "Rule of thumb (general)": (
            "Risk % = distance from entry to stop as % of price. "
            "LOW <3%, MEDIUM 3–6%, HIGH 6–10%, VERY HIGH ≥10%."
        ),
        "For THIS stock": (
            f"Model stop is ₹{safe_float(result.get('Stop Loss')):,.2f}, target ₹{safe_float(result.get('Target')):,.2f}."
        ),
        "Manual decision tip": "Only size the trade so that if stop hits, loss ≤ your chosen % of capital (see BUY page helper).",
    })

    return pd.DataFrame(lessons).sort_values("Priority")


def pattern_education(patterns_str):
    """Explain each detected pattern and how to use it manually."""
    catalog = {
        "Hammer": {
            "bias": "Bullish",
            "why": "Long lower wick shows sellers pushed price down but buyers closed near the highs — rejection of lower prices.",
            "use": "More reliable after a decline or at support / EMA. Confirm next candle closes higher.",
        },
        "Shooting Star": {
            "bias": "Bearish",
            "why": "Long upper wick shows buyers failed to hold highs — supply appeared overhead.",
            "use": "More reliable after a rally or at resistance. Confirm next candle closes lower.",
        },
        "Bullish Engulfing": {
            "bias": "Bullish",
            "why": "A green body fully covers the prior red body — buyers overwhelmed sellers in one session.",
            "use": "Stronger with volume > average and near support / EMA50.",
        },
        "Bearish Engulfing": {
            "bias": "Bearish",
            "why": "A red body fully covers the prior green body — sellers took control abruptly.",
            "use": "Stronger after extended up-move; reason to tighten stop or avoid new longs.",
        },
        "Doji": {
            "bias": "Neutral / indecision",
            "why": "Open ≈ close — balance between buyers and sellers; trend may pause or reverse.",
            "use": "Do not trade Doji alone. Wait for the next directional candle.",
        },
        "Higher High / Higher Low": {
            "bias": "Bullish structure",
            "why": "Swing highs and lows are rising — classic uptrend footprint.",
            "use": "Favour BUY / HOLD while HH–HL structure holds. Break of last higher low is a warning.",
        },
        "Lower High / Lower Low": {
            "bias": "Bearish structure",
            "why": "Swing highs and lows are falling — downtrend footprint.",
            "use": "Avoid fresh longs; prefer SELL / stay out until structure breaks upward.",
        },
    }

    names = [p.strip() for p in str(patterns_str or "").split(",") if p.strip()]
    if not names or names == ["No major pattern detected"]:
        return [{
            "Pattern": "None major",
            "Bias": "—",
            "Why it matters": "No classic candle/structure flag on the latest bars.",
            "How you use it": "Rely on EMA + RSI + MACD + volume instead of patterns today.",
        }]

    rows = []
    for n in names:
        meta = catalog.get(n, {
            "bias": "Context",
            "why": "Detected by the model’s pattern engine.",
            "use": "Combine with trend (EMA) and volume before acting.",
        })
        rows.append({
            "Pattern": n,
            "Bias": meta["bias"],
            "Why it matters": meta["why"],
            "How you use it": meta["use"],
        })
    return rows


def make_chart(result, df=None, patterns=None, target=None, stop_loss=None, title=None):
    """
    Interactive Plotly candles + selected indicators + pattern markers.
    Works offline / market closed with last available bars.
    """
    if df is None:
        df = result.get("Data")
    if df is None or getattr(df, "empty", True):
        return go.Figure()

    # Ensure indicators exist on this timeframe
    work = normalize_columns(df).copy()
    try:
        work = calculate_indicators(work)
    except Exception:
        pass
    if work is None or work.empty:
        work = df.copy()

    # Show last N bars for speed
    tail_n = 200 if len(work) > 200 else len(work)
    plot_df = work.tail(tail_n)

    if patterns is None:
        patterns = [p.strip() for p in str(result.get("Patterns", "")).split(",") if p.strip()]
    patterns = [p for p in patterns if p and p != "No major pattern detected"]
    # Re-detect on this timeframe for live accuracy
    try:
        live_pats = detect_patterns(plot_df)
        if live_pats:
            patterns = live_pats
    except Exception:
        pass

    if target is None:
        target = result.get("Target") if result else None
    if stop_loss is None:
        stop_loss = result.get("Stop Loss") if result else None

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Price",
        )
    )
    for col, name in [("EMA20", "EMA 20"), ("EMA50", "EMA 50"), ("EMA200", "EMA 200")]:
        if col in plot_df.columns and plot_df[col].notna().any():
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df[col], name=name, line=dict(width=1.2))
            )
    if "BBUpper" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["BBUpper"], name="BB Upper",
                line=dict(dash="dot", width=1),
            )
        )
    if "BBLower" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["BBLower"], name="BB Lower",
                line=dict(dash="dot", width=1),
            )
        )
    if "VWAP" in plot_df.columns and plot_df["VWAP"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["VWAP"], name="VWAP",
                line=dict(width=1, dash="dash"),
            )
        )

    if patterns and len(plot_df) >= 2:
        last_i = plot_df.index[-1]
        prev_i = plot_df.index[-2]
        last_high = safe_float(plot_df["High"].iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[last_i],
                y=[last_high * 1.008],
                mode="markers+text",
                marker=dict(size=14, symbol="triangle-down", color="#f5c542"),
                text=["Pattern"],
                textposition="top center",
                name="Pattern",
            )
        )
        fig.add_vrect(
            x0=prev_i,
            x1=last_i,
            fillcolor="rgba(245, 197, 66, 0.18)",
            layer="below",
            line_width=0,
            annotation_text=", ".join(patterns)[:48],
            annotation_position="top left",
        )
        if any("Higher High" in p or "Lower High" in p for p in patterns) and len(plot_df) >= 10:
            win = plot_df.tail(10)
            fig.add_vrect(
                x0=win.index[0],
                x1=win.index[-1],
                fillcolor=(
                    "rgba(26, 155, 95, 0.10)"
                    if any("Higher High" in p for p in patterns)
                    else "rgba(217, 48, 37, 0.10)"
                ),
                layer="below",
                line_width=0,
                annotation_text="Structure",
                annotation_position="bottom left",
            )

    if safe_float(target) > 0:
        fig.add_hline(
            y=safe_float(target), line_dash="dash", line_color="#1a9b5f",
            annotation_text="Target",
        )
    if safe_float(stop_loss) > 0:
        fig.add_hline(
            y=safe_float(stop_loss), line_dash="dash", line_color="#d93025",
            annotation_text="Stop Loss",
        )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.12),
        title=title or "Interactive chart · indicators · patterns · target/stop",
        dragmode="zoom",
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False, showspikes=True)
    fig.update_yaxes(showspikes=True)
    return fig


def compute_support_resistance(df, lookback=60, swings=3):
    """
    Simple swing-based support / resistance from recent highs & lows.
    Returns lists of resistance levels (highs) and support levels (lows).
    """
    if df is None or len(df) < 10:
        return [], []
    d = df.tail(max(lookback, 20)).copy()
    highs = d["High"].astype(float)
    lows = d["Low"].astype(float)
    # Local peaks / troughs
    res_levels = []
    sup_levels = []
    for i in range(2, len(d) - 2):
        h = highs.iloc[i]
        l = lows.iloc[i]
        if h >= highs.iloc[i - 1] and h >= highs.iloc[i - 2] and h >= highs.iloc[i + 1] and h >= highs.iloc[i + 2]:
            res_levels.append(float(h))
        if l <= lows.iloc[i - 1] and l <= lows.iloc[i - 2] and l <= lows.iloc[i + 1] and l <= lows.iloc[i + 2]:
            sup_levels.append(float(l))
    # Cluster nearby levels (0.4% tolerance)
    def _cluster(levels, n):
        if not levels:
            return []
        levels = sorted(levels, reverse=True)
        out = []
        for lv in levels:
            if not out or all(abs(lv - o) / max(o, 1e-9) > 0.004 for o in out):
                out.append(lv)
            if len(out) >= n:
                break
        return out

    return _cluster(res_levels, swings), _cluster(sup_levels, swings)


def show_interactive_chart_panel(symbol, result):
    """
    Multi-timeframe interactive chart.
    All indicators / S-R / patterns are optional toggles — user chooses what to show.
    """
    st.subheader("📊 Interactive chart — pick what you want to see")
    st.caption(
        "Turn indicators ON/OFF below. Zoom/pan the chart. "
        "Works market open or closed (last available bars)."
    )

    sk = display_symbol(symbol)
    c1, c2 = st.columns([1, 3])
    with c1:
        tf = st.selectbox(
            "Timeframe",
            [
                ("1 Day", "1d"),
                ("1 Week", "1wk"),
                ("1 Hour", "1h"),
                ("30 Min", "30m"),
                ("15 Min", "15m"),
                ("5 Min", "5m"),
            ],
            format_func=lambda x: x[0],
            key=f"tf_{sk}",
        )
        interval = tf[1]
        if st.button("↻ Reload chart data", key=f"reload_chart_{sk}"):
            try:
                stock_history.clear()
            except Exception:
                pass
            st.rerun()

    with c2:
        show_ind = st.multiselect(
            "Indicators & overlays (select any / none)",
            [
                "EMA 20",
                "EMA 50",
                "EMA 200",
                "Bollinger",
                "VWAP",
                "Support / Resistance",
                "Patterns",
                "Target / Stop",
                "Volume",
                "RSI panel",
                "MACD panel",
                "Stochastic panel",
            ],
            default=[
                "EMA 20",
                "EMA 50",
                "EMA 200",
                "Bollinger",
                "Support / Resistance",
                "Patterns",
                "Target / Stop",
                "Volume",
            ],
            key=f"ind_{sk}",
        )

    with st.spinner(f"Loading {interval} data..."):
        if interval == "1d" and result.get("Data") is not None and not result["Data"].empty:
            df_tf = result["Data"]
        else:
            df_tf = stock_history(symbol, interval=interval)

    if df_tf is None or df_tf.empty:
        st.warning(
            f"No data for **{interval}**. Try 1 Day (intraday can be limited after close)."
        )
        return

    work = normalize_columns(df_tf).copy()
    try:
        work = calculate_indicators(work)
    except Exception:
        pass
    if work is None or work.empty:
        work = normalize_columns(df_tf).copy()

    tail_n = 220 if len(work) > 220 else len(work)
    plot_df = work.tail(tail_n)

    # Detect patterns on this TF
    try:
        pats = detect_patterns(plot_df)
    except Exception:
        pats = []

    # Subplot rows
    rows = 1
    row_heights = [0.55]
    specs = [[{"secondary_y": False}]]
    if "Volume" in show_ind:
        rows += 1
        row_heights.append(0.12)
        specs.append([{"secondary_y": False}])
    if "RSI panel" in show_ind:
        rows += 1
        row_heights.append(0.14)
        specs.append([{"secondary_y": False}])
    if "MACD panel" in show_ind:
        rows += 1
        row_heights.append(0.14)
        specs.append([{"secondary_y": False}])
    if "Stochastic panel" in show_ind:
        rows += 1
        row_heights.append(0.12)
        specs.append([{"secondary_y": False}])

    # Normalize heights
    s = sum(row_heights)
    row_heights = [h / s for h in row_heights]

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs,
    )

    r = 1
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Price",
        ),
        row=r,
        col=1,
    )

    if "EMA 20" in show_ind and "EMA20" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["EMA20"], name="EMA 20", line=dict(width=1.2)),
            row=r, col=1,
        )
    if "EMA 50" in show_ind and "EMA50" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["EMA50"], name="EMA 50", line=dict(width=1.2)),
            row=r, col=1,
        )
    if "EMA 200" in show_ind and "EMA200" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["EMA200"], name="EMA 200", line=dict(width=1.3)),
            row=r, col=1,
        )
    if "Bollinger" in show_ind:
        if "BBUpper" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["BBUpper"], name="BB Upper",
                    line=dict(dash="dot", width=1),
                ),
                row=r, col=1,
            )
        if "BBLower" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["BBLower"], name="BB Lower",
                    line=dict(dash="dot", width=1),
                ),
                row=r, col=1,
            )
    if "VWAP" in show_ind and "VWAP" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["VWAP"], name="VWAP",
                line=dict(dash="dash", width=1),
            ),
            row=r, col=1,
        )

    # Support / Resistance
    if "Support / Resistance" in show_ind:
        resistances, supports = compute_support_resistance(plot_df)
        for i, lv in enumerate(resistances):
            fig.add_hline(
                y=lv, line_dash="dot", line_color="#ff6b6b",
                annotation_text=f"R{i+1} {lv:,.1f}",
                annotation_position="right",
                row=r, col=1,
            )
        for i, lv in enumerate(supports):
            fig.add_hline(
                y=lv, line_dash="dot", line_color="#51cf66",
                annotation_text=f"S{i+1} {lv:,.1f}",
                annotation_position="right",
                row=r, col=1,
            )
        if resistances or supports:
            st.caption(
                "S/R: "
                + (" | ".join([f"R{i+1}=₹{v:,.1f}" for i, v in enumerate(resistances)]) or "no R")
                + " · "
                + (" | ".join([f"S{i+1}=₹{v:,.1f}" for i, v in enumerate(supports)]) or "no S")
            )

    if "Target / Stop" in show_ind:
        if safe_float(result.get("Target")) > 0:
            fig.add_hline(
                y=safe_float(result["Target"]), line_dash="dash", line_color="#1a9b5f",
                annotation_text="Target", row=r, col=1,
            )
        if safe_float(result.get("Stop Loss")) > 0:
            fig.add_hline(
                y=safe_float(result["Stop Loss"]), line_dash="dash", line_color="#d93025",
                annotation_text="Stop", row=r, col=1,
            )

    if "Patterns" in show_ind and pats and len(plot_df) >= 2:
        last_i = plot_df.index[-1]
        prev_i = plot_df.index[-2]
        last_high = safe_float(plot_df["High"].iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[last_i],
                y=[last_high * 1.008],
                mode="markers+text",
                marker=dict(size=13, symbol="triangle-down", color="#f5c542"),
                text=["Pattern"],
                textposition="top center",
                name="Pattern",
            ),
            row=r, col=1,
        )
        fig.add_vrect(
            x0=prev_i, x1=last_i,
            fillcolor="rgba(245,197,66,0.18)", line_width=0,
            annotation_text=", ".join(pats)[:40],
            annotation_position="top left",
            row=r, col=1,
        )

    # Volume
    next_row = 2
    if "Volume" in show_ind:
        vol = plot_df["Volume"] if "Volume" in plot_df.columns else pd.Series(0, index=plot_df.index)
        colors = [
            "#1a9b5f" if c >= o else "#d93025"
            for o, c in zip(plot_df["Open"], plot_df["Close"])
        ]
        fig.add_trace(
            go.Bar(x=plot_df.index, y=vol, name="Volume", marker_color=colors),
            row=next_row, col=1,
        )
        next_row += 1

    if "RSI panel" in show_ind and "RSI" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI", line=dict(width=1.2)),
            row=next_row, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#d93025", row=next_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#1a9b5f", row=next_row, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#888", row=next_row, col=1)
        next_row += 1

    if "MACD panel" in show_ind and "MACD" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD", line=dict(width=1.2)),
            row=next_row, col=1,
        )
        if "MACDSignal" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index, y=plot_df["MACDSignal"], name="Signal",
                    line=dict(width=1),
                ),
                row=next_row, col=1,
            )
        if "MACDHist" in plot_df.columns:
            hist_colors = ["#1a9b5f" if v >= 0 else "#d93025" for v in plot_df["MACDHist"].fillna(0)]
            fig.add_trace(
                go.Bar(
                    x=plot_df.index, y=plot_df["MACDHist"], name="Hist",
                    marker_color=hist_colors,
                ),
                row=next_row, col=1,
            )
        next_row += 1

    if "Stochastic panel" in show_ind and "StochK" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["StochK"], name="%K", line=dict(width=1.2)),
            row=next_row, col=1,
        )
        if "StochD" in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["StochD"], name="%D", line=dict(width=1)),
                row=next_row, col=1,
            )
        fig.add_hline(y=80, line_dash="dot", line_color="#d93025", row=next_row, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="#1a9b5f", row=next_row, col=1)

    fig.update_layout(
        height=720 if rows > 2 else 580,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08),
        title=f"{sk} · {interval}",
        dragmode="zoom",
        hovermode="x unified",
    )
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        },
    )

    if pats:
        st.info(f"**Patterns on {interval}:** " + ", ".join(pats))
    else:
        st.caption(f"No major pattern on latest {interval} bars.")


def show_indicator_school(result):
    """Render teaching UI for indicators + patterns + manual decision checklist."""
    st.subheader("🎓 Indicator School — read numbers like a trader")
    st.caption(
        "Priority 1 = decide first. Priority 2 = confirm. Priority 3 = fine-tune timing. "
        "Rules below are general market teaching applied to THIS stock’s current values."
    )

    edu = indicator_education(result)
    # Compact priority table
    show_cols = ["Priority", "Indicator", "Current value", "For THIS stock"]
    st.dataframe(edu[show_cols], use_container_width=True, hide_index=True)

    with st.expander("📘 Full rules of thumb + how to decide manually (every indicator)", expanded=True):
        for _, row in edu.iterrows():
            st.markdown(
                f"""
**P{int(row['Priority'])} · {row['Indicator']}**  
**Now:** {row['Current value']}  
**General rule:** {row['Rule of thumb (general)']}  
**This stock:** {row['For THIS stock']}  
**Manual tip:** {row['Manual decision tip']}
"""
            )
            st.divider()

    st.subheader("🕯️ Pattern School — what formed and why it matters")
    pat_rows = pattern_education(result.get("Patterns", ""))
    st.dataframe(pd.DataFrame(pat_rows), use_container_width=True, hide_index=True)

    st.subheader("✅ Manual decision checklist (in priority order)")
    st.markdown(
        """
1. **Trend first (Priority 1):** Is price above EMA20 and EMA50? (and ideally EMA200?)  
2. **Momentum (Priority 2):** Is RSI **> 50** (better **55–70**)? Is MACD above signal?  
3. **Strength (Priority 2–3):** Is ADX **≥ 25** so the trend is real, not noise?  
4. **Participation:** Is volume **≥ ~1.1–1.5×** average on the move?  
5. **Risk:** Is Risk % acceptable for your capital? Is stop distance sensible (ATR-based)?  
6. **Pattern:** Bullish pattern = bonus confirmation, not a standalone reason. Bearish pattern near resistance = reduce aggression.

**Simple teaching examples**
- **RSI 55 > 50** → generally **good** in an uptrend: momentum is positive, not overbought.  
- **RSI 82** → strong but **late**; prefer wait for dip toward EMA20 rather than chase.  
- **Price > EMA50 and MACD bullish** → core swing-buy structure.  
- **Price < EMA50 and ADX high** → strong downtrend; do not average blindly.
"""
    )


# ============================================================
# TRADINGVIEW
# ============================================================

def tv_chart(symbol, target=None, stop_loss=None):
    """Stock detail page chart — Plotly + link to TradingView (works for all symbols)."""
    show_tradingview_chart(
        symbol,
        title=display_symbol(symbol),
        height=680,
        target=target,
        stop_loss=stop_loss,
    )


# ============================================================
# DETAILED STOCK PAGE
# ============================================================

def show_stock(
    symbol
):

    symbol = display_symbol(
        symbol
    )

    st.header(
        "🔍 Detailed Analysis: "
        +
        symbol
    )

    if st.button(
        "⬅ Back to Dashboard",
        key="back_stock"
    ):

        st.session_state.page = (
            "Dashboard"
        )

        st.rerun()

    # --------------------------------------------------------
    # Historical data
    # --------------------------------------------------------

    df = stock_history(
        symbol
    )

    if df.empty:

        st.error(
            "Historical data could not be loaded for this stock."
        )

        return

    # --------------------------------------------------------
    # Current / latest closing
    # --------------------------------------------------------

    quote = live_quote(
        symbol
    )

    # --------------------------------------------------------
    # Fresh analysis
    # --------------------------------------------------------

    with st.spinner(
        "Analysing "
        + symbol
        + "..."
    ):

        result = analyse_stock(
            clean_symbol(symbol),
            df,
            fetch_news=True
        )

    if not result:

        st.error(
            "Not enough data for analysis."
        )

        return

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    current_live = None
    if quote:
        current_live = safe_float(quote["price"])
        a, b, c = st.columns(3)

        a.metric(
            quote["label"],
            f"₹{quote['price']:,.2f}",
            f"{quote['pct']:+.2f}%"
        )

        b.metric(
            "Change",
            f"₹{quote['change']:+,.2f}"
        )

        c.metric(
            "Updated",
            quote["updated"]
        )
    else:
        current_live = safe_float(result.get("Price"))

    # --------------------------------------------------------
    # Live Target / Stop Status (as requested)
    # --------------------------------------------------------
    level_status = check_price_vs_levels(
        current_price=current_live,
        entry=result.get("Price"),
        target=result.get("Target"),
        stop_loss=result.get("Stop Loss"),
        call=result.get("Call", "BUY"),
    )

    if level_status["status"] == "TARGET ACHIEVED":
        rec_html = str(level_status.get("recommendation", "")).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="success-box">
            <h3>🎯 TARGET ACHIEVED</h3>
            <p style="margin:0;line-height:1.6;">
            Previous Target: ₹{result['Target']:,.2f}<br>
            Current Price: ₹{current_live:,.2f}<br>
            🎯 TARGET ACHIEVED
            </p>
            <br>
            <b>NEW ANALYSIS:</b><br>
            New Target: ₹{level_status['new_target']:,.2f}<br>
            New Stop Loss: ₹{level_status['new_stop']:,.2f}<br>
            <br>
            <b>Recommendation:</b><br>
            {rec_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif level_status["status"] == "STOP LOSS HIT":
        st.markdown(
            f"""
            <div class="danger-box">
            <h3>🔴 STOP LOSS HIT</h3>
            <p style="margin:0;line-height:1.6;">
            Current Price: ₹{current_live:,.2f}<br>
            Stop Loss: ₹{result['Stop Loss']:,.2f}<br>
            🔴 STOP LOSS HIT
            </p>
            <br>
            <b>Recommendation:</b><br>
            🔴 SELL / EXIT
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Main recommendation
    # --------------------------------------------------------

    call = result["Call"]

    if call == "BUY":
        css = "buy-box"
    elif call == "SELL":
        css = "sell-box"
    elif call == "HOLD":
        css = "hold-box"
    else:
        css = "watch-box"

    st.markdown(
        f"""
        <div class="{css}">
        <h2>Recommendation: {call}</h2>
        <p>
        Prediction: <b>{result['Prediction']}%</b>
        &nbsp; | &nbsp;
        Risk: <b>{result['Risk %']}%</b>
        &nbsp; | &nbsp;
        Risk Level: <b>{result['Risk Level']}</b>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    a.metric(
        "Prediction",
        f"{result['Prediction']}%"
    )

    b.metric(
        "Risk",
        f"{result['Risk %']}%"
    )

    c.metric(
        "Hold Days",
        str(result["Hold Days"])
    )

    d.metric(
        "Priority",
        result["Priority"]
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Target",
        f"₹{result['Target']:,.2f}"
    )

    b.metric(
        "Stop Loss",
        f"₹{result['Stop Loss']:,.2f}"
    )

    c.metric(
        "Target %",
        f"{result['Target %']:+.2f}%"
    )

    d.metric(
        "Sector",
        result["Sector"]
    )

    # --------------------------------------------------------
    # 25Y EXPERIENCED TRADER — SHORT vs LONG-TERM + COMPANY GOAL
    # --------------------------------------------------------
    st.subheader("🧠 Experienced trader desk — Short-term vs Long-term")
    st.caption(
        "Swing plan (target/stop) is separate from long-term ownership. "
        "If a short-term stop hits, LT quality decides whether to keep a core holding."
    )
    try:
        with st.spinner("Loading fundamentals, index tags & long-term quality…"):
            prof = enrich_stock_profile(symbol)
        lt_score = safe_float(prof.get("LT Score"), 45)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LT quality score", f"{lt_score:.0f}")
        m2.metric("Market cap (₹ Cr)", prof.get("Market Cap Cr Text", "—"))
        m3.metric("Book value", prof.get("Book Value Text", "—"))
        m4.metric("Revenue / Sales (₹ Cr)", prof.get("Revenue Cr Text", "—"))

        st.markdown(
            f"""
            <div class="hold-box" style="padding:12px;border-radius:10px;margin:8px 0;">
            <b>Company:</b> {prof.get('Name', symbol)} · {prof.get('Industry', '')}<br>
            <b>Indices:</b> {prof.get('Index Tags', '—')} · <b>Cap class:</b> {prof.get('Market Cap Bucket', '—')}<br>
            <b>Book value:</b> {prof.get('Book Value Text', '—')}
            &nbsp;|&nbsp; <b>Revenue (Sales):</b> {prof.get('Revenue Cr Text', '—')}
            &nbsp;|&nbsp; <b>Net income:</b> {prof.get('Net Income Cr Text', '—')}<br>
            <b>Market cap:</b> {prof.get('Market Cap Cr Text', '—')}<br>
            <b>PE / PB / ROE / D-E:</b>
            {safe_float(prof.get('PE')):.1f} /
            {safe_float(prof.get('PB')):.2f} /
            {safe_float(prof.get('ROE')):.2f} /
            {safe_float(prof.get('Debt/Equity')):.1f}<br>
            <b>With BUY call:</b> {prof.get('LT with BUY call', '')}<br>
            <b>With SELL call:</b> {prof.get('LT with SELL call', '')}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("**Long-term notes (desk):**")
        for note in str(prof.get("LT Notes", "")).split(" | "):
            if note.strip():
                st.markdown(f"- {note.strip()}")

        st.info(f"**If swing stop-loss hits:** {prof.get('LT Hold if SL hits', '—')}")
        if call == "BUY":
            st.success(f"**LT paired with BUY:** {prof.get('LT with BUY call', '')}")
        elif call == "SELL":
            st.warning(f"**LT paired with SELL:** {prof.get('LT with SELL call', '')}")

        # Experienced trader timing feedback
        show_expert_trader_box(symbol, call_hint=str(call))

        # Year-wise revenue / profit (₹ Cr)
        st.markdown("**Revenue & profit — year-wise (₹ crore)**")
        ydf = fundamental_yearly_table(symbol)
        if ydf is not None and not ydf.empty:
            st.dataframe(ydf, use_container_width=True, hide_index=True)
        else:
            st.caption("Year-wise financials not available from data feed for this symbol.")

        # Two-horizon plan
        st.markdown("**Two-horizon plan (how a 25Y trader splits the book)**")
        c_st, c_lt = st.columns(2)
        with c_st:
            st.markdown(
                f"""
                **Short-term (this call)**  
                - Bias: **{call}** · Pred {result.get('Prediction')}%  
                - Entry zone ~ ₹{safe_float(result.get('Price')):,.2f}  
                - Target ₹{safe_float(result.get('Target')):,.2f} · SL ₹{safe_float(result.get('Stop Loss')):,.2f}  
                - Horizon ~ {result.get('Hold Days')} days  
                - Honour stop for the *swing* risk budget  
                """
            )
        with c_lt:
            if lt_score >= 72:
                lt_plan = (
                    "Core long-term **eligible**. Build on dips only if thesis intact. "
                    "Do not use full capital as swing size."
                )
            elif lt_score >= 58:
                lt_plan = (
                    "Average quality — small core only; review every quarter results. "
                    "No aggressive averaging after a technical stop."
                )
            else:
                lt_plan = (
                    "Not a core long-term compounder under current numbers. "
                    "Prefer trading book only; exit if swing SL hits."
                )
            st.markdown(
                f"""
                **Long-term (ownership)**  
                - Score **{lt_score:.0f}** · {prof.get('LT Label', '')}  
                - Cap: **{prof.get('Market Cap Bucket', '—')}**  
                - Plan: {lt_plan}  
                """
            )
        # Company goal / thesis style blurb from summary if available
        try:
            fund = fetch_fundamentals(symbol)
            summary = str(fund.get("summary") or "").strip()
            if summary:
                with st.expander("🏢 Business / future context (from filings summary)", expanded=False):
                    st.write(summary[:1200])
                    st.caption(
                        "Use this as business context only — combine with LT score, "
                        "debt, ROE and your own view of the industry cycle."
                    )
        except Exception:
            pass
    except Exception as e:
        st.caption(f"Long-term desk block unavailable: {e}")

    # --------------------------------------------------------
    # History statistics
    # --------------------------------------------------------

    stats = stock_statistics(
        symbol
    )

    st.subheader(
        "📊 Past Prediction Performance"
    )

    a, b, c, d, e = st.columns(5)

    a.metric(
        "Times Suggested",
        stats["times"]
    )

    b.metric(
        "Closed Predictions",
        stats["closed"]
    )

    c.metric(
        "Wins",
        stats["wins"]
    )

    d.metric(
        "Losses",
        stats["losses"]
    )

    e.metric(
        "Success Rate",
        f"{stats['success']}%"
    )

    # --------------------------------------------------------
    # Indicator School (values + why + how to decide)
    # --------------------------------------------------------

    show_indicator_school(result)

    # --------------------------------------------------------
    # Sell details
    # --------------------------------------------------------

    if call == "SELL":

        st.markdown(
            """
            <div class="danger-box">
            <h3>🔴 SELL CALL</h3>
            <p>
            The current model is indicating that this stock
            should be considered for selling.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("**Why sell:** " + result["Reason"])
        st.write("**Risk level:** " + str(result["Risk Level"]))
        st.write("**Stop Loss:** ₹" + f"{result['Stop Loss']:,.2f}")

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    st.subheader("🤖 Model Analysis Reasons")

    st.write("**Technical Reasons**")
    for reason in str(result["Technical Reasons"]).split(" | "):
        if reason.strip():
            st.write("• " + reason.strip())

    st.write("**News Influence:** " + result["News Influence"])
    st.write("**Patterns (raw):** " + str(result.get("Patterns", "")))

    # --------------------------------------------------------
    # News
    # --------------------------------------------------------

    if result["News"]:

        st.subheader(
            "📰 Recent News Influence"
        )

        for item in result["News"]:

            st.write(
                "• "
                +
                str(
                    item.get(
                        "title",
                        ""
                    )
                )
            )

            publisher = item.get(
                "publisher",
                ""
            )

            if publisher:

                st.caption(
                    publisher
                )

    # --------------------------------------------------------
    # Pattern
    # --------------------------------------------------------

    st.write(
        "**Patterns:** "
        +
        str(
            result["Patterns"]
        )
    )

    # --------------------------------------------------------
    # Interactive multi-timeframe chart
    # --------------------------------------------------------
    try:
        show_interactive_chart_panel(symbol, result)
    except Exception as e:
        st.caption(f"Interactive chart unavailable ({e}).")
        try:
            st.plotly_chart(make_chart(result), use_container_width=True)
        except Exception:
            pass

    st.subheader("📈 Open on TradingView (optional)")
    tv_chart(
        symbol,
        target=result.get("Target"),
        stop_loss=result.get("Stop Loss"),
    )

    # --------------------------------------------------------
    # Past history
    # --------------------------------------------------------

    history = load_history()

    if not history.empty:

        stock_history_df = history[
            history["Stock"]
            .astype(str)
            .str.upper()
            ==
            symbol.upper()
        ]

        if not stock_history_df.empty:

            st.subheader(
                "🕐 Past Calls for "
                + symbol
            )
            filterable_dataframe(
                stock_history_df.sort_values("Prediction Date", ascending=False),
                key=f"stock_hist_{symbol}",
                default_cols=[c for c in [
                    "Prediction Date", "Call", "Call Source", "Strategy",
                    "Entry", "Target", "Stop Loss", "Result", "Return %",
                ] if c in stock_history_df.columns],
            )


# ============================================================
# MARKET INDICES
# ============================================================

def live_market():

    st.subheader(
        "🇮🇳 MARKET"
    )

    st.caption(
        market_status_text()
        +
        " • "
        +
        (
            "Live prices"
            if nse_market_open_now()
            else
            "Latest closing prices"
        )
    )

    columns = st.columns(
        len(
            INDEX_SYMBOLS
        )
    )

    for col, (
        name,
        symbol
    ) in zip(
        columns,
        INDEX_SYMBOLS.items()
    ):

        q = live_quote(
            symbol
        )

        if q:

            col.metric(
                name,
                f"{q['price']:,.2f}",
                f"{q['pct']:+.2f}%"
            )

        else:

            col.metric(
                name,
                "Unavailable"
            )


# ============================================================
# SELL CALLS PAGE
# ============================================================

def show_sell_calls(
    results
):

    st.title(
        "🔴 SELL CALLS"
    )
    st.caption("Priority: Sure/Strategy SELL ideas first (if saved), then scan SELL list.")

    # Priority: history SURE/STRATEGY sells
    try:
        _h = normalize_history_df(load_history())
        if _h is not None and not _h.empty and "Call Source" in _h.columns:
            _src = _h["Call Source"].astype(str).str.upper()
            _q = _h[_src.str.contains("SURE|STRATEGY|PRECISION|HIGH_CONV", regex=True, na=False)].copy()
            _q = _q[_q["Call"].astype(str).str.upper().str.contains("SELL", na=False)]
            if not _q.empty:
                st.subheader("⭐ Priority — Sure / Strategy SELL")
                filterable_dataframe(
                    _q.sort_values("Prediction Date", ascending=False).head(30),
                    key="sell_pri_table",
                    default_cols=[c for c in [
                        "Prediction Date", "Stock", "Call Source", "Strategy",
                        "Entry", "Target", "Stop Loss", "Result",
                    ] if c in _q.columns],
                )
    except Exception:
        pass

    if results.empty:

        st.info(
            "Run FULL MARKET SCAN first."
        )

        return

    results = ensure_result_columns(results)

    sells = results[
        results["Call"]
        ==
        "SELL"
    ].copy()

    if sells.empty:

        st.success(
            "No current SELL calls found."
        )

        return

    sells = sells.sort_values(
        [
            "Prediction",
            "Risk %",
        ],
        ascending=[
            True,
            False,
        ]
    )

    st.write(
        f"Current SELL calls: {len(sells)}"
    )

    display_cols = [c for c in [
        "Rank", "Stock", "Sector", "Price", "Call", "Prediction",
        "Risk %", "Risk Level", "Target", "Stop Loss", "Hold Days",
        "Priority", "Patterns", "News Influence",
    ] if c in sells.columns]
    st.markdown("##### Filterable SELL table")
    filterable_dataframe(sells, key="sell_table", default_cols=display_cols, height=400)

    st.subheader("SELL cards — company · mcap · BV · desk")
    n_sc = st.slider("SELL cards to show", 3, 15, 6, key="sell_cards_n")
    for stock in sells["Stock"].head(n_sc).tolist():
        row = sells[sells["Stock"] == stock].iloc[0]
        render_call_stock_card(row.to_dict(), section_key="sell")

    st.subheader(
        "Detailed Sell Reasons"
    )

    for stock in sells[
        "Stock"
    ].head(50).tolist():

        row = sells[
            sells["Stock"]
            ==
            stock
        ].iloc[0]

        with st.expander(
            f"🔴 {stock} — SELL — Prediction {row['Prediction']}%"
        ):

            st.write(
                "**Current Price:** ₹"
                +
                f"{row['Price']:,.2f}"
            )

            st.write(
                "**Risk:** "
                +
                f"{row['Risk %']}% "
                +
                f"({row['Risk Level']})"
            )

            st.write(
                "**Target:** ₹"
                +
                f"{row['Target']:,.2f}"
            )

            st.write(
                "**Stop Loss:** ₹"
                +
                f"{row['Stop Loss']:,.2f}"
            )

            st.write(
                "**Reason:** "
                +
                str(
                    row["Reason"]
                )
            )

            if st.button(
                "OPEN DETAILED ANALYSIS",
                key="sell_"
                + stock
            ):

                st.session_state.selected_stock = (
                    stock
                )

                st.session_state.page = (
                    "Stock Analysis"
                )

                st.rerun()



# ============================================================
# MODEL LEARNING (restored — used by precision picks & history)
# ============================================================

def load_learning() -> dict:
    """Load model_learning.json — pattern weights, feedback, success notes."""
    default = {
        "pattern_weights": {},
        "call_adjustments": {},
        "manual_feedback": [],
        "history_stats": {},
        "updated_at": "",
    }
    try:
        if LEARNING_FILE.exists():
            data = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in default.items():
                    data.setdefault(k, v)
                return data
    except Exception:
        pass
    return default


def save_learning(data: dict) -> None:
    try:
        data = dict(data or {})
        data["updated_at"] = india_now().strftime("%Y-%m-%d %H:%M:%S IST")
        LEARNING_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def pattern_weight(pattern_name: str, learning: dict = None) -> float:
    """
    Weight for a named pattern. Base from PATTERN_IMPORTANCE, tuned by learning file.
    """
    if learning is None:
        learning = load_learning()
    name = str(pattern_name or "").strip()
    if not name or name.lower().startswith("no major"):
        return 0.0
    base = 2.0
    try:
        meta = PATTERN_IMPORTANCE.get(name) or {}
        base = float(meta.get("base_weight", 2) or 2)
    except Exception:
        base = 2.0
    try:
        pw = (learning or {}).get("pattern_weights") or {}
        if name in pw:
            base = float(pw[name])
        # partial key match
        for k, v in pw.items():
            if str(k).lower() in name.lower() or name.lower() in str(k).lower():
                base = max(base, float(v))
    except Exception:
        pass
    return float(base)


def learning_score_adjustment(pred, risk_level, call, patterns, learning=None):
    """
    Small delta to prediction score from past learning + pattern weights.
    Returns (delta, notes_list).
    """
    if learning is None:
        learning = load_learning()
    notes = []
    delta = 0.0
    try:
        pred = safe_float(pred, 0)
        for p in (patterns or []):
            w = pattern_weight(p, learning)
            if w:
                delta += (w - 2.0) * 0.5  # neutral at weight 2
                if w >= 3:
                    notes.append(f"+ pattern {p}")
                elif w <= 1:
                    notes.append(f"- weak pattern {p}")
        adj = (learning or {}).get("call_adjustments") or {}
        key = str(call or "").upper()
        if key in adj:
            delta += float(adj[key])
            notes.append(f"call adj {key}")
        # history_stats global tilt
        hs = (learning or {}).get("history_stats") or {}
        wr = safe_float(hs.get("win_rate"), 0)
        if wr >= 55:
            delta += 1.5
        elif wr and wr < 40:
            delta -= 2.0
    except Exception:
        pass
    return round(delta, 2), notes


def learn_from_history(min_closed: int = 3) -> dict:
    """
    Update pattern weights and win-rate from closed Past Predictions.
    """
    learning = load_learning()
    try:
        hist = normalize_history_df(load_history())
        if hist is None or hist.empty or "Result" not in hist.columns:
            return learning
        res_u = hist["Result"].astype(str).str.upper()
        closed = hist[res_u.str.contains("TARGET ACHIEVED|STOP LOSS HIT|WIN|LOSS", na=False, regex=True)]
        if len(closed) < int(min_closed):
            return learning
        wins = closed[res_u.loc[closed.index].str.contains("TARGET ACHIEVED|^WIN$", na=False, regex=True)]
        losses = closed[res_u.loc[closed.index].str.contains("STOP LOSS|^LOSS$", na=False, regex=True)]
        n_w, n_l = len(wins), len(losses)
        total = n_w + n_l
        win_rate = (100.0 * n_w / total) if total else 0.0
        learning["history_stats"] = {
            "closed": int(total),
            "wins": int(n_w),
            "losses": int(n_l),
            "win_rate": round(win_rate, 2),
        }
        # Pattern-level success
        pw = dict(learning.get("pattern_weights") or {})
        if "Patterns" in closed.columns:
            for _, row in closed.iterrows():
                res = str(row.get("Result", "")).upper()
                is_win = "TARGET" in res or res == "WIN"
                is_loss = "STOP" in res or res == "LOSS"
                if not is_win and not is_loss:
                    continue
                pats = [p.strip() for p in str(row.get("Patterns", "") or "").split(",") if p.strip()]
                for p in pats:
                    if p.lower().startswith("no major"):
                        continue
                    cur = float(pw.get(p, 2.0))
                    if is_win:
                        cur = min(5.0, cur + 0.15)
                    else:
                        cur = max(0.5, cur - 0.2)
                    pw[p] = round(cur, 2)
        learning["pattern_weights"] = pw
        save_learning(learning)
    except Exception:
        pass
    return learning



# ============================================================
# BUY CALLS PAGE
# ============================================================

def precision_buy_score(row) -> float:
    """
    Single score to rank BUY candidates for picking 1–2 stocks.
    Higher = better quality (strength + lower risk + trend + reward).
    Applies learning multipliers from past closed trades.
    """
    pred = safe_float(row.get("Prediction"), 0)
    risk = safe_float(row.get("Risk %"), 50)
    price = safe_float(row.get("Price"), 0)
    target = safe_float(row.get("Target"), 0)
    adx = safe_float(row.get("ADX"), 0)
    rsi = safe_float(row.get("RSI"), 50)
    vol_r = safe_float(row.get("Volume Ratio"), 1)
    priority = str(row.get("Priority", "")).upper()
    patterns_raw = str(row.get("Patterns", "") or "")
    patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()]

    reward = 0.0
    if price > 0 and target > price:
        reward = min(((target - price) / price) * 100, 25)

    pri_bonus = {
        "VERY HIGH": 12,
        "HIGH": 8,
        "MEDIUM": 4,
        "LOW": 0,
    }.get(priority, 0)

    rsi_score = 8 if 50 <= rsi <= 70 else (3 if 45 <= rsi < 50 else 0)
    if rsi > 78:
        rsi_score = -8

    adx_score = min(max(adx - 15, 0), 20) * 0.4
    vol_score = 6 if vol_r >= 1.3 else (3 if vol_r >= 1.05 else 0)
    risk_penalty = min(risk, 15) * 1.8

    learning = load_learning()
    pat_bonus = 0.0
    for p in patterns:
        if p and p != "No major pattern detected":
            pat_bonus += pattern_weight(p, learning) * 0.8

    learn_delta, _ = learning_score_adjustment(
        pred, row.get("Risk Level", ""), row.get("Call", "BUY"), patterns, learning
    )

    score = (
        pred * 0.55
        + reward * 1.2
        + pri_bonus
        + rsi_score
        + adx_score
        + vol_score
        + pat_bonus
        + learn_delta
        - risk_penalty
    )
    return round(float(score), 2)


def is_strong_trend_row(row) -> bool:
    """Same strong-trend gate used to allow BUY signals."""
    price = safe_float(row.get("Price") or row.get("Current Price"))
    adx = safe_float(row.get("ADX"))
    rsi = safe_float(row.get("RSI"), 50)
    vol_r = safe_float(row.get("Volume Ratio"), 1.0)
    # EMA may not always be in results; use Prediction/Priority as proxy if missing
    ema20 = safe_float(row.get("EMA20"))
    ema50 = safe_float(row.get("EMA50"))
    if ema20 > 0 and ema50 > 0 and price > 0:
        trend_ok = price > ema20 and price > ema50
    else:
        # Results table often has ADX/RSI only
        trend_ok = True
    return bool(
        trend_ok
        and adx >= 25
        and 48 <= rsi <= 72
        and vol_r >= 1.0
        and safe_float(row.get("Prediction")) >= 72
    )


def pick_precision_buys(buys: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    Only a few strong-trend BUYs — quality over quantity for higher target hit rate.
    """
    if buys is None or buys.empty:
        return pd.DataFrame()

    x = buys.copy()
    x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
    x["Risk %"] = pd.to_numeric(x.get("Risk %"), errors="coerce").fillna(99)
    x["ADX"] = pd.to_numeric(x.get("ADX"), errors="coerce").fillna(0)
    x["RSI"] = pd.to_numeric(x.get("RSI"), errors="coerce").fillna(50)
    if "Volume Ratio" in x.columns:
        x["Volume Ratio"] = pd.to_numeric(x["Volume Ratio"], errors="coerce").fillna(1.0)
    else:
        x["Volume Ratio"] = 1.0

    # Quality gates — prefer strong trend, but allow a small shortlist
    x = x[
        (x["Prediction"] >= 70)
        & (x["Risk %"] <= 8)
        & (x["ADX"] >= 20)
        & (x["RSI"] >= 45)
        & (x["RSI"] <= 75)
    ].copy()

    if x.empty:
        # Fallback: best available by prediction (still limited n later)
        x = buys.copy()
        x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
        x["Risk %"] = pd.to_numeric(x.get("Risk %"), errors="coerce").fillna(99)
        x["ADX"] = pd.to_numeric(x.get("ADX"), errors="coerce").fillna(0)
        x = x[x["Prediction"] >= 65].copy()

    if x.empty:
        # Last resort: top by prediction so screen is never blank for days
        x = buys.copy()
        x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
        x = x.sort_values("Prediction", ascending=False).head(max(n * 5, 10))

    if x.empty:
        return pd.DataFrame()

    try:
        x["Precision Score"] = x.apply(lambda r: precision_buy_score(r), axis=1)
    except Exception:
        x["Precision Score"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
    x = x.sort_values("Precision Score", ascending=False)

    # Prefer stocks that historically hit targets (from learning / history)
    try:
        hist = normalize_history_df(load_history())
        if hist is not None and not hist.empty:
            res_u = hist["Result"].astype(str).str.upper()
            wins = hist[res_u.str.contains("TARGET ACHIEVED", na=False)]
            losses = hist[res_u.str.contains("STOP LOSS HIT", na=False)]
            win_counts = wins.groupby(wins["Stock"].astype(str).str.upper()).size()
            loss_counts = losses.groupby(losses["Stock"].astype(str).str.upper()).size()

            def hist_bonus(stock):
                s = str(stock).upper()
                w = int(win_counts.get(s, 0))
                l = int(loss_counts.get(s, 0))
                if w + l == 0:
                    return 0
                return (w - l) * 3

            x["Hist Bonus"] = x["Stock"].apply(hist_bonus)
            x["Precision Score"] = x["Precision Score"] + x["Hist Bonus"]
            x = x.sort_values("Precision Score", ascending=False)
    except Exception:
        pass

    # Sector diversification, max n picks (default 2)
    n = max(1, min(int(n), 3))
    picked = []
    used_sectors = set()
    for _, row in x.iterrows():
        sec = str(row.get("Sector", "Other"))
        if sec in used_sectors and len(x) > n * 2:
            continue
        picked.append(row)
        used_sectors.add(sec)
        if len(picked) >= n:
            break

    if not picked:
        return x.head(n)

    out = pd.DataFrame(picked)
    out.insert(0, "Pick #", range(1, len(out) + 1))
    return out


def get_prior_calls_for_stock(stock: str, limit: int = 8) -> pd.DataFrame:
    """Past calls for a stock from recommendation_history.csv."""
    history = normalize_history_df(load_history())
    if history is None or history.empty:
        return pd.DataFrame()
    s = display_symbol(stock).upper()
    h = history[
        history["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False) == s
    ].copy()
    if h.empty:
        return h
    if "Prediction Date" in h.columns:
        h = h.sort_values("Prediction Date", ascending=False)
    return h.head(limit)


def show_prior_call_learning_panel(stock: str, current_reason: str = ""):
    """
    When a stock is recommended again: show prior call, outcome, and learning this time.
    """
    prior = get_prior_calls_for_stock(stock, limit=6)
    learning = load_learning()
    st.markdown(f"#### 🔁 Prior calls & learning — **{display_symbol(stock)}**")
    if prior is None or prior.empty:
        st.caption("No prior saved calls for this stock in history CSV.")
        if current_reason:
            st.write("**Current reason:**", current_reason[:500])
        return

    for i, (_, row) in enumerate(prior.iterrows()):
        result = str(row.get("Result", "PENDING")).upper()
        reason = str(row.get("Reason", "") or "")[:400]
        pred_d = row.get("Prediction Date", "")
        entry = safe_float(row.get("Entry"))
        target = safe_float(row.get("Target"))
        stop = safe_float(row.get("Stop Loss"))
        ret = safe_float(row.get("Return %"))
        failed = "STOP LOSS" in result or result == "LOSS"
        won = "TARGET ACHIEVED" in result or result == "WIN"

        if won:
            badge = "🎯 TARGET ACHIEVED"
        elif failed:
            badge = "🔴 STOP LOSS / FAILED"
        elif "HOLDING PERIOD" in result:
            badge = "⏰ TIME EXIT"
        else:
            badge = "⏳ OPEN / PENDING"

        st.markdown(
            f"""
            <div class="{'danger-box' if failed else 'success-box' if won else 'hold-box'}"
                 style="padding:10px;margin-bottom:8px;border-radius:8px;">
            <b>Prior #{i+1}</b> · {pred_d} · {badge}<br>
            Entry ₹{entry:,.2f} · Locked Target ₹{target:,.2f} · Locked SL ₹{stop:,.2f}
            {f' · Return {ret:+.1f}%' if ret else ''}<br>
            <b>Reason then:</b> {reason or '—'}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if current_reason:
        st.write("**Reason this time:**")
        st.write(current_reason[:800])

    # What learning is applied now
    lessons = learning.get("lessons") or []
    st.write("**What the model is applying from past mistakes (global + this context):**")
    if lessons:
        for L in lessons[:6]:
            st.markdown(f"- {L}")
    else:
        st.caption("Learning still thin — more closed target/stop outcomes will improve it.")

    # Stock-specific tip
    failed_n = sum(
        1 for _, r in prior.iterrows()
        if "STOP LOSS" in str(r.get("Result", "")).upper()
    )
    win_n = sum(
        1 for _, r in prior.iterrows()
        if "TARGET ACHIEVED" in str(r.get("Result", "")).upper()
    )
    if failed_n > win_n and failed_n >= 1:
        st.warning(
            f"This stock has **{failed_n}** prior stop/fail vs **{win_n}** target hits in history. "
            "Model should only re-BUY under strong-trend gate; size smaller if you still take it."
        )
    elif win_n > 0:
        st.success(
            f"This stock has **{win_n}** prior target hits in your history — positive track record."
        )


def manual_feedback_trainer():
    """Manual training: feed outcome / notes into model_learning (not image ML)."""
    st.subheader("📝 Manual feedback trainer")
    st.caption(
        "You can teach the model with **structured outcomes** (stock + result + note). "
        "Uploading chart **images for neural training is not supported** in this app — "
        "use outcomes and pattern notes instead; they update `model_learning.json`."
    )
    st.info(
        f"Learning file path: `{LEARNING_FILE}` · "
        "After a successful save you will see a **green confirmation** and the entry below."
    )

    with st.form("manual_train_form", clear_on_submit=False):
        stock = st.text_input(
            "Stock symbol (NSE) — required (or type GENERAL for pattern-only lesson)",
            value="",
            key="manual_fb_stock",
        )
        outcome = st.selectbox(
            "What happened?",
            [
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "SHOULD NOT HAVE BEEN BUY",
                "GOOD SETUP (missed)",
                "OTHER",
            ],
            key="manual_fb_outcome",
        )
        patterns = st.multiselect(
            "Patterns you saw",
            list(PATTERN_IMPORTANCE.keys()),
            key="manual_fb_patterns",
        )
        note = st.text_area("Your note / mistake lesson", height=80, key="manual_fb_note")
        submitted = st.form_submit_button("Save feedback into learning", type="primary")

    if submitted:
        stock_clean = str(stock or "").upper().strip()
        if not stock_clean:
            st.error(
                "Not saved — **Stock symbol is empty**. "
                "Type an NSE symbol (e.g. RELIANCE) or **GENERAL**, then click Save again."
            )
        elif not note and not patterns:
            st.error(
                "Not saved — add at least a **pattern** or a **note** so the model has something to learn."
            )
        else:
            try:
                learning = load_learning()
                fb = learning.get("manual_feedback", [])
                entry = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stock": display_symbol(stock_clean),
                    "outcome": outcome,
                    "patterns": list(patterns or []),
                    "note": note or "",
                }
                fb.append(entry)
                learning["manual_feedback"] = fb[-200:]

                mults = learning.get("pattern_multipliers", {})
                for p in (patterns or []):
                    cur = float(mults.get(p, 1.0))
                    if outcome in ("TARGET ACHIEVED", "GOOD SETUP (missed)"):
                        mults[p] = round(min(1.6, cur + 0.08), 2)
                    elif outcome in ("STOP LOSS HIT", "SHOULD NOT HAVE BEEN BUY"):
                        mults[p] = round(max(0.4, cur - 0.08), 2)
                learning["pattern_multipliers"] = mults

                lessons = learning.get("lessons", [])
                lessons.insert(
                    0,
                    f"Manual feedback on {display_symbol(stock_clean)}: {outcome}. "
                    f"Patterns: {', '.join(patterns) or '—'}. {(note or '')[:120]}",
                )
                learning["lessons"] = lessons[:40]
                save_learning(learning)

                # Verify write
                verify = load_learning()
                n_fb = len(verify.get("manual_feedback", []))
                st.success(
                    f"✅ **Feedback saved** into `{LEARNING_FILE.name}`.\n\n"
                    f"- Stock: **{display_symbol(stock_clean)}**\n"
                    f"- Outcome: **{outcome}**\n"
                    f"- Patterns: **{', '.join(patterns) or '—'}**\n"
                    f"- Total manual feedback entries now: **{n_fb}**"
                )
                st.session_state["_last_manual_fb"] = entry
            except Exception as e:
                st.error(f"Save failed: {e}")

    # Always show recent saved feedback so user can confirm
    learning_now = load_learning()
    recent = learning_now.get("manual_feedback") or []
    if recent:
        st.write("**Recently saved feedback (proof it is in learning):**")
        show = list(reversed(recent[-8:]))
        st.dataframe(pd.DataFrame(show), use_container_width=True, hide_index=True)
        mults = learning_now.get("pattern_multipliers") or {}
        if mults:
            st.caption(
                "Pattern multipliers (updated by feedback): "
                + ", ".join(f"{k}={v}" for k, v in list(mults.items())[:12])
            )
    else:
        st.caption("No manual feedback in `model_learning.json` yet.")


def show_buy_calls(results):

    st.title("🟢 BUY CALLS")
    st.caption(
        "**25Y desk:** keep BUY list for short-term plans, but filter by **index / market-cap / long-term quality**. "
        "If a swing stop hits on a high LT-score name, you may **hold core for long-term** instead of full panic exit."
    )

    if results.empty:
        st.info("Run FULL MARKET SCAN first.")
        return

    results = ensure_result_columns(results)
    buys = results[results["Call"].astype(str).str.upper() == "BUY"].copy()

    if buys.empty:
        watch = results[
            (results["Call"].astype(str).str.upper() == "WATCH")
            & (pd.to_numeric(results.get("Prediction"), errors="coerce").fillna(0) >= 60)
        ].copy()
        if watch.empty:
            st.warning("No BUY / near-buy in last scan. Run **FULL MARKET SCAN**.")
            return
        st.info("No pure BUY — showing higher-score WATCH near-buys.")
        buys = watch

    buys["Precision Score"] = buys.apply(precision_buy_score, axis=1)
    buys = buys.sort_values(
        ["Precision Score", "Prediction", "Risk %"],
        ascending=[False, False, True],
    )

    st.write(f"BUY / near-buy candidates: **{len(buys)}**")

    # Priority strip: Sure / Strategy from history (top of BUY page)
    try:
        _h = normalize_history_df(load_history())
        if _h is not None and not _h.empty:
            _src = _h["Call Source"].astype(str).str.upper() if "Call Source" in _h.columns else pd.Series([""] * len(_h))
            _q = _h[_src.str.contains("SURE|STRATEGY|PRECISION|HIGH_CONV|MY_STRATEGY", regex=True, na=False)].copy()
            _q = _q[_q["Call"].astype(str).str.upper().str.contains("BUY", na=False)]
            if not _q.empty:
                st.subheader("⭐ Priority first — recent Sure / Strategy BUY")
                _q = _q.sort_values("Prediction Date", ascending=False)
                st.dataframe(
                    _q[[c for c in [
                        "Prediction Date", "Stock", "Call Source", "Strategy", "Entry",
                        "Target", "Stop Loss", "Result", "Current Price",
                    ] if c in _q.columns]].head(15),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("These outrank the bulk scan BUY list for decision-making.")
    except Exception:
        pass

    # --------------------------------------------------------
    # Fundamental / index / LT filters
    # --------------------------------------------------------
    st.subheader("🏛️ Index · Cap · Long-term quality filters")
    st.caption(
        "Index membership from NSE lists. Cap from market-cap / index. "
        "LT score = ROE, debt, margins, growth, PE (pro long-term desk)."
    )
    enrich_n = st.slider(
        "Enrich top N names with fundamentals (slower if high)",
        10, 80, 30, key="buy_enrich_n",
        help="Yahoo fundamentals fetched only for top N by precision score.",
    )
    if st.button("Load index + fundamentals for filters", type="primary", key="buy_enrich_btn"):
        with st.spinner("Loading NSE index tags + fundamentals…"):
            try:
                load_index_membership()
                buys = enrich_results_profiles(buys.head(max(enrich_n, 15)), max_n=enrich_n)
                # re-attach rest without profile
                st.session_state["buys_enriched"] = buys
                st.success(f"Enriched {len(buys)} stocks.")
            except Exception as e:
                st.warning(f"Enrichment partial: {e}")

    if st.session_state.get("buys_enriched") is not None:
        enr = st.session_state["buys_enriched"]
        # Prefer enriched rows for stocks we have
        if isinstance(enr, pd.DataFrame) and not enr.empty and "LT Score" in enr.columns:
            buys = enr

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        idx_f = st.selectbox(
            "Index membership",
            [
                "ALL",
                "Nifty 50",
                "Nifty 100",
                "Nifty 200",
                "Nifty 500",
                "Nifty Midcap 100",
                "Nifty Smallcap 100",
                "Outside major indices",
            ],
            key="buy_idx_f",
        )
    with f2:
        cap_f = st.selectbox(
            "Market cap",
            ["ALL", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Unknown"],
            key="buy_cap_f",
        )
    with f3:
        lt_f = st.selectbox(
            "Long-term quality",
            [
                "ALL",
                "Strong LT (score ≥ 72)",
                "Average+ (score ≥ 58)",
                "Weak LT (score < 58)",
                "Hold-if-SL = YES only",
            ],
            key="buy_lt_f",
        )
    with f4:
        min_lt = st.slider("Min LT score", 0, 95, 0, key="buy_min_lt")

    # Apply filters when columns exist; else use live index tags on the fly for small lists
    def _row_matches_filters(row) -> bool:
        stock = display_symbol(row.get("Stock", ""))
        tags = str(row.get("Index Tags", "") or "")
        if not tags or tags == "nan":
            tags = ", ".join(index_tags_of(stock))
        cap = str(row.get("Market Cap Bucket", "") or "Unknown")
        if cap in ("", "nan", "None"):
            # quick index-based cap
            tset = set(index_tags_of(stock))
            if "Nifty 50" in tset or "Nifty 100" in tset:
                cap = "Large Cap"
            elif "Nifty Midcap 100" in tset:
                cap = "Mid Cap"
            elif "Nifty Smallcap 100" in tset:
                cap = "Small Cap"
        lt_score = safe_float(row.get("LT Score"), -1)
        hold_sl = str(row.get("LT Hold if SL hits", "") or "")

        if idx_f == "Outside major indices":
            if tags and tags not in ("Outside major indices",) and any(
                x in tags for x in ("Nifty 50", "Nifty 100", "Nifty 500", "Midcap", "Smallcap")
            ):
                return False
        elif idx_f != "ALL":
            if idx_f not in tags and idx_f not in index_tags_of(stock):
                return False
        if cap_f != "ALL" and cap != cap_f:
            return False
        if min_lt > 0 and lt_score >= 0 and lt_score < min_lt:
            return False
        if lt_f.startswith("Strong") and not (lt_score >= 72):
            return False
        if lt_f.startswith("Average") and not (lt_score >= 58):
            return False
        if lt_f.startswith("Weak") and not (0 <= lt_score < 58):
            return False
        if lt_f.startswith("Hold-if-SL") and not hold_sl.upper().startswith("YES"):
            return False
        return True

    if any([idx_f != "ALL", cap_f != "ALL", lt_f != "ALL", min_lt > 0]):
        mask = buys.apply(_row_matches_filters, axis=1)
        buys_f = buys[mask].copy()
        if buys_f.empty:
            st.warning(
                "No names match these fundamental/index filters. "
                "Click **Load index + fundamentals** or loosen filters."
            )
        else:
            buys = buys_f
            st.success(f"After index/cap/LT filters: **{len(buys)}** stocks")

    # --------------------------------------------------------
    # PRECISION PICKS — best 1–2 stocks
    # --------------------------------------------------------
    st.subheader("🎯 Precision Picks — buy only these")
    st.caption(
        "Ranked by Prediction + low Risk + Priority + trend (ADX) + volume + reward/risk. "
        "Different sectors preferred so you are not concentrated in one theme."
    )

    p1, p2 = st.columns(2)
    with p1:
        n_picks = st.radio("How many stocks to buy?", [1, 2], index=1, horizontal=True, key="n_precision_picks")
    with p2:
        style = st.selectbox(
            "Style",
            [
                "Balanced (recommended)",
                "Aggressive (higher prediction)",
                "Conservative (lowest risk)",
            ],
            key="buy_style",
        )

    pool = buys.copy()
    if style.startswith("Aggressive"):
        pool = pool[pd.to_numeric(pool["Prediction"], errors="coerce") >= 75]
    elif style.startswith("Conservative"):
        pool = pool[
            pool["Risk Level"].astype(str).str.upper().isin(["LOW", "MEDIUM"])
            & (pd.to_numeric(pool["Risk %"], errors="coerce") <= 5)
        ]

    top_picks = pick_precision_buys(pool if not pool.empty else buys, n=int(n_picks))

    if top_picks.empty:
        st.warning(
            "No stock passed the quality filters. Loosen Style or run a fresh market scan."
        )
    else:
        for _, row in top_picks.iterrows():
            stock = row["Stock"]
            pred = safe_float(row.get("Prediction"))
            risk = safe_float(row.get("Risk %"))
            price = safe_float(row.get("Price"))
            target = safe_float(row.get("Target"))
            stop = safe_float(row.get("Stop Loss"))
            reward = ((target - price) / price * 100) if price > 0 else 0
            score = safe_float(row.get("Precision Score"))

            st.markdown(
                f"""
                <div class="buy-box" style="margin-bottom:14px;padding:14px;border-radius:10px;border:1px solid #1a9b5f;">
                <h3 style="margin:0 0 8px 0;">#{int(row.get('Pick #', 0))}  ·  {stock}
                &nbsp; <span style="font-size:0.85em;color:#666;">{row.get('Sector','')}</span></h3>
                <p style="margin:0;line-height:1.65;">
                <b>Precision Score:</b> {score:.1f} &nbsp;|&nbsp;
                <b>Prediction:</b> {pred:.1f}% &nbsp;|&nbsp;
                <b>Risk:</b> {risk:.2f}% ({row.get('Risk Level','')}) &nbsp;|&nbsp;
                <b>Priority:</b> {row.get('Priority','')}<br>
                <b>Entry (approx):</b> ₹{price:,.2f} &nbsp;|&nbsp;
                <b>Target:</b> ₹{target:,.2f} ({reward:+.1f}%) &nbsp;|&nbsp;
                <b>Stop Loss:</b> ₹{stop:,.2f}<br>
                <b>Hold:</b> {row.get('Hold Days','')} days &nbsp;|&nbsp;
                <b>Patterns:</b> {row.get('Patterns','')}<br>
                <b>Index:</b> {row.get('Index Tags', '—')} &nbsp;|&nbsp;
                <b>Cap:</b> {row.get('Market Cap Bucket', '—')} &nbsp;|&nbsp;
                <b>LT score:</b> {safe_float(row.get('LT Score')):.0f} ({row.get('LT Label', '')})<br>
                <b>If swing SL hits:</b> {str(row.get('LT Hold if SL hits', 'Load fundamentals to see LT advice'))[:180]}<br>
                <b>Why:</b> {str(row.get('Reason',''))[:240]}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"📊 Fundamentals + long-term view — {stock}", expanded=False):
                try:
                    prof = enrich_stock_profile(stock)
                    st.write(
                        f"**{prof.get('Name')}** · {prof.get('Industry')} · "
                        f"**{prof.get('Market Cap Bucket')}** · Indices: {prof.get('Index Tags')}"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("LT score", prof.get("LT Score"))
                    c2.metric("PE", f"{safe_float(prof.get('PE')):.1f}" if prof.get("PE") else "—")
                    c3.metric("ROE", f"{safe_float(prof.get('ROE')):.2f}" if prof.get("ROE") else "—")
                    c4.metric("D/E", f"{safe_float(prof.get('Debt/Equity')):.1f}" if prof.get("Debt/Equity") else "—")
                    st.info(prof.get("LT Hold if SL hits", ""))
                    st.caption(prof.get("LT Notes", ""))
                    st.caption(f"Label: {prof.get('LT Label')}")
                except Exception as e:
                    st.caption(f"Fundamentals unavailable: {e}")
            # Prior call history + learning when recommended again
            with st.expander(f"🔁 Prior history & learning — {stock}", expanded=False):
                show_prior_call_learning_panel(stock, str(row.get("Reason", "")))

            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"📈 Full analysis — {stock}", key=f"prec_full_{stock}"):
                    st.session_state.selected_stock = stock
                    st.session_state.page = "Stock Analysis"
                    st.rerun()
            with b2:
                st.caption("Suggested action: BUY only if price is near entry and stop fits your capital risk.")

        # Capital + risk% + target earnings (both controls drive the numbers)
        st.markdown("##### 💰 Capital, risk & target earnings")
        st.caption(
            "Set **investment capital** and **risk %**. "
            "Qty is sized from stop-loss risk. Earnings assume target is hit."
        )

        c_cap, c_risk, c_deploy = st.columns(3)
        with c_cap:
            invest_capital = st.number_input(
                "Investment capital (₹)",
                min_value=1000.0,
                value=50000.0,
                step=1000.0,
                key="precision_capital",
                help="Total money you plan to use for this trade / these picks.",
            )
        with c_risk:
            risk_pct_user = st.slider(
                "Max loss risk % of capital",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="precision_risk_pct",
                help="How much of capital you are willing to lose if stop-loss hits.",
            )
        with c_deploy:
            deploy_pct = st.slider(
                "Deploy % of capital in trade",
                min_value=10,
                max_value=100,
                value=100,
                step=5,
                key="precision_deploy_pct",
                help="Share of capital actually put into the position (100% = full capital).",
            )

        deploy_amount = invest_capital * (deploy_pct / 100.0)
        risk_budget = invest_capital * (risk_pct_user / 100.0)

        if not top_picks.empty:
            for _, r0 in top_picks.iterrows():
                stock = r0["Stock"]
                entry = safe_float(r0.get("Price"))
                stop = safe_float(r0.get("Stop Loss"))
                target = safe_float(r0.get("Target"))
                per_share_risk = max(entry - stop, 0.01)
                per_share_reward = max(target - entry, 0.0)

                # Qty limited by both risk budget and deploy amount
                qty_by_risk = int(risk_budget // per_share_risk) if per_share_risk > 0 else 0
                qty_by_capital = int(deploy_amount // entry) if entry > 0 else 0
                qty = max(0, min(qty_by_risk, qty_by_capital))

                invested = qty * entry
                max_loss = qty * per_share_risk
                expected_profit = qty * per_share_reward
                reward_pct = (per_share_reward / entry * 100) if entry > 0 else 0
                loss_pct = (per_share_risk / entry * 100) if entry > 0 else 0

                st.markdown(
                    f"""
                    <div class="success-box" style="margin-bottom:12px;">
                    <b>{stock}</b> — position plan<br>
                    Entry ≈ ₹{entry:,.2f} &nbsp;|&nbsp; Target ₹{target:,.2f} ({reward_pct:+.1f}%)
                    &nbsp;|&nbsp; Stop ₹{stop:,.2f} (−{loss_pct:.1f}%)<br>
                    <b>Suggested qty:</b> {qty} shares &nbsp;|&nbsp;
                    <b>Invested:</b> ₹{invested:,.0f}<br>
                    <b>If target hit — expected earnings:</b> ₹{expected_profit:,.0f}
                    ({reward_pct:+.1f}% on entry)<br>
                    <b>If stop hit — max loss:</b> ₹{max_loss:,.0f}
                    (within your ₹{risk_budget:,.0f} risk budget)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()

    # --------------------------------------------------------
    # Full BUY list with filters
    # --------------------------------------------------------
    st.subheader("📋 All BUY calls (filtered)")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        min_pred = st.slider("Minimum Prediction %", 50, 95, 65, key="buy_min_pred")
    with f2:
        risk_opts = ["ALL", "LOW", "MEDIUM", "HIGH", "VERY HIGH"]
        risk_f = st.selectbox("Risk", risk_opts, key="buy_risk")
    with f3:
        sector_opts = ["ALL"] + sorted(
            buys["Sector"].dropna().astype(str).unique().tolist()
        )
        sector_f = st.selectbox("Sector", sector_opts, key="buy_sector")
    with f4:
        show_n = st.selectbox("Show top", [10, 25, 50, 100, 200], index=1, key="buy_n")

    filtered = buys[pd.to_numeric(buys["Prediction"], errors="coerce") >= min_pred]
    if risk_f != "ALL":
        filtered = filtered[filtered["Risk Level"] == risk_f]
    if sector_f != "ALL":
        filtered = filtered[filtered["Sector"] == sector_f]
    filtered = filtered.head(int(show_n))

    display_cols = [
        c for c in [
            "Precision Score", "Rank", "Stock", "Sector", "Price", "Call",
            "Prediction", "Risk %", "Risk Level", "Target", "Stop Loss",
            "Hold Days", "Priority", "Patterns", "News Influence",
        ] if c in filtered.columns
    ]
    st.markdown("##### Filterable table")
    filterable_dataframe(filtered, key="buy_table", default_cols=display_cols, height=400)

    st.subheader("BUY cards — company · mcap · BV · desk")
    st.caption("Same visual format as Strategy Lab. One card per stock.")
    n_cards = st.slider("BUY cards to show", 3, 20, 8, key="buy_cards_n")
    for stock in filtered["Stock"].head(n_cards).tolist():
        row = filtered[filtered["Stock"] == stock].iloc[0]
        render_call_stock_card(row.to_dict(), section_key="buy")
    st.subheader("Detailed BUY Analysis")
    for stock in filtered["Stock"].head(40).tolist():
        row = filtered[filtered["Stock"] == stock].iloc[0]
        with st.expander(
            f"🟢 {stock} — BUY — Pred {row['Prediction']}% | Score {safe_float(row.get('Precision Score')):.1f} | {row['Risk Level']}"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"₹{safe_float(row['Price']):,.2f}")
            c2.metric("Target", f"₹{safe_float(row['Target']):,.2f}")
            c3.metric("Stop Loss", f"₹{safe_float(row['Stop Loss']):,.2f}")
            c4.metric("Hold Days", str(row.get("Hold Days", "")))

            st.write(f"**Risk:** {row['Risk %']}% ({row['Risk Level']})")
            st.write(f"**Priority:** {row.get('Priority', '')}")
            st.write(f"**Patterns:** {row.get('Patterns', '')}")
            st.write(f"**News:** {row.get('News Influence', '')}")
            st.write(f"**Reason:** {row.get('Reason', '')}")

            if st.button("OPEN DETAILED ANALYSIS", key=f"buy_full_{stock}"):
                st.session_state.selected_stock = stock
                st.session_state.page = "Stock Analysis"
                st.rerun()


# ============================================================
# SURE CALL MODE — two-stage analyst process + Nifty regime
# ============================================================

def nifty_market_regime() -> dict:
    """
    Stage-0 market filter. Weak Nifty → recommend no new longs.
    """
    out = {
        "regime": "UNKNOWN",
        "trade_longs": True,
        "trade_shorts": True,
        "score": 50,
        "reasons": [],
        "price": 0.0,
        "ema20": 0.0,
        "ema50": 0.0,
        "ema200": 0.0,
        "adx": 0.0,
        "rsi": 50.0,
        "pct": 0.0,
    }
    try:
        df = stock_history("^NSEI", interval="1d")
        if df is None or df.empty or len(df) < 60:
            out["reasons"].append("Nifty history unavailable — treat market regime as unknown.")
            return out
        df = calculate_indicators(df)
        row = df.iloc[-1]
        price = safe_float(row.get("Close"))
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        adx = safe_float(row.get("ADX"))
        rsi = safe_float(row.get("RSI"), 50)
        out.update({
            "price": price, "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "adx": adx, "rsi": rsi,
        })
        try:
            q = live_quote("^NSEI")
            if q and q.get("price"):
                out["price"] = safe_float(q["price"])
                out["pct"] = safe_float(q.get("pct"))
                price = out["price"]
        except Exception:
            pass

        score = 50
        reasons = []
        if price > ema20 > 0:
            score += 8
            reasons.append(f"Nifty above EMA20 (₹{ema20:,.0f}) — short-term bid intact.")
        else:
            score -= 12
            reasons.append(f"Nifty below EMA20 (₹{ema20:,.0f}) — short-term pressure.")

        if price > ema50 > 0:
            score += 10
            reasons.append(f"Nifty above EMA50 (₹{ema50:,.0f}) — swing trend supportive for longs.")
        else:
            score -= 14
            reasons.append(f"Nifty below EMA50 (₹{ema50:,.0f}) — swing trend weak; avoid fresh longs.")

        if ema200 > 0 and price > ema200:
            score += 8
            reasons.append("Nifty above EMA200 — primary trend still constructive.")
        elif ema200 > 0:
            score -= 10
            reasons.append("Nifty below EMA200 — primary trend damaged; capital preservation mode.")

        if adx >= 25 and price < ema50:
            score -= 8
            reasons.append(f"ADX {adx:.0f} with price under EMA50 — a strong downtrend, not chop.")
        elif adx >= 25 and price > ema50:
            score += 6
            reasons.append(f"ADX {adx:.0f} with price over EMA50 — strong uptrend regime.")

        if rsi < 40:
            score -= 6
            reasons.append(f"Nifty RSI {rsi:.0f} weak — risk appetite soft.")
        elif rsi > 55:
            score += 4
            reasons.append(f"Nifty RSI {rsi:.0f} supports risk-on.")

        score = int(np.clip(score, 5, 95))
        out["score"] = score
        out["reasons"] = reasons

        if score >= 62 and price > ema50:
            out["regime"] = "RISK-ON / BULLISH"
            out["trade_longs"] = True
            out["trade_shorts"] = False
        elif score <= 42 or (ema50 > 0 and price < ema50 and adx >= 20):
            out["regime"] = "RISK-OFF / WEAK"
            out["trade_longs"] = False
            out["trade_shorts"] = True
            out["reasons"].append(
                "ANALYST RULE: Do not take new BUY/sure-long calls today while Nifty is weak. "
                "Protect capital; wait for reclaim of EMA50 or a clear risk-on day."
            )
        else:
            out["regime"] = "MIXED / SELECTIVE"
            out["trade_longs"] = True  # only highest quality
            out["trade_shorts"] = True
            out["reasons"].append(
                "Mixed tape — only two-stage sure setups; skip marginal names."
            )
    except Exception as e:
        out["reasons"].append(f"Regime check error: {e}")
    return out


def two_stage_stock_verdict(symbol: str, side: str = "BUY") -> dict:
    """
    Stage 1 = trend context. Stage 2 = trigger.
    Both must pass for SURE CALL. Written like an experienced analyst brief.
    """
    side = str(side).upper()
    sym = clean_symbol(symbol)
    display = display_symbol(symbol)
    result = {
        "symbol": display,
        "side": side,
        "sure": False,
        "stage1_pass": False,
        "stage2_pass": False,
        "stage1": [],
        "stage2": [],
        "fail": [],
        "entry_plan": "",
        "invalidation": "",
        "hold_days": 15,
        "entry": 0.0,
        "target": 0.0,
        "stop": 0.0,
        "prediction": 0.0,
        "analyst_summary": "",
    }
    try:
        df = stock_history(sym, interval="1d")
        if df is None or df.empty or len(df) < 50:
            result["fail"].append("Insufficient price history for a professional-grade call.")
            return result
        df = calculate_indicators(df)
        row = df.iloc[-1]
        price = safe_float(row.get("Close"))
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        adx = safe_float(row.get("ADX"))
        rsi = safe_float(row.get("RSI"), 50)
        atr = safe_float(row.get("ATR")) or price * 0.02
        vol_r = safe_float(row.get("Volume Ratio"), 1.0)
        try:
            q = live_quote(sym)
            if q and q.get("price"):
                price = safe_float(q["price"])
        except Exception:
            pass

        patterns = detect_patterns(df)
        result["entry"] = round(price, 2)

        s1, s2, fail = [], [], []
        stage1 = stage2 = False

        if side == "BUY":
            # ----- Stage 1: TREND (context) -----
            checks = [
                (price > ema20 > 0, f"Price ₹{price:,.2f} is above EMA20 ₹{ema20:,.2f} — short-term trend up."),
                (price > ema50 > 0, f"Price above EMA50 ₹{ema50:,.2f} — swing trend up."),
                (ema200 <= 0 or price > ema200 * 0.98,
                 f"Not fighting primary trend (EMA200 ₹{ema200:,.2f})."),
                (adx >= 22, f"ADX {adx:.1f} — trend strength adequate (≥22 required for sure)."),
                (ema20 >= ema50 * 0.995 if ema50 > 0 else True,
                 "EMA20 aligned with / above EMA50 — trend stack healthy."),
            ]
            ok = 0
            for passed, msg in checks:
                if passed:
                    s1.append("✓ " + msg)
                    ok += 1
                else:
                    s1.append("✗ " + msg)
                    fail.append(msg)
            stage1 = ok >= 4

            # ----- Stage 2: TRIGGER -----
            bull_pat = any(
                p in patterns for p in [
                    "Hammer", "Bullish Engulfing", "Higher High / Higher Low",
                    "Breakout High", "Bullish Marubozu",
                ]
            )
            tchecks = [
                (52 <= rsi <= 70, f"RSI {rsi:.1f} in buyable momentum band (52–70), not exhausted."),
                (vol_r >= 1.05, f"Volume ratio {vol_r:.2f} — participation OK."),
                (bull_pat or (price > ema20 and rsi >= 55),
                 f"Trigger: pattern {patterns[:3] if patterns else 'momentum hold above EMA20'}."),
                (True, "No same-day chase rule violated inside engine (prefer buy dips to EMA20)."),
            ]
            ok2 = 0
            for passed, msg in tchecks:
                if passed:
                    s2.append("✓ " + msg)
                    ok2 += 1
                else:
                    s2.append("✗ " + msg)
                    fail.append(msg)
            stage2 = ok2 >= 3

            # Swing geometry for 15–20 day hold
            stop = price - 2.2 * atr
            target = price + 1.8 * atr
            result["stop"] = round(max(stop, price * 0.94), 2)
            result["target"] = round(target, 2)
            result["hold_days"] = 18
            result["entry_plan"] = (
                "PREFERRED ENTRY: Buy on same day only if price is holding above signal day's low "
                "and preferably on a dip toward EMA20 (not a vertical chase into resistance). "
                "If you miss the day, buy next 1–2 sessions only while Stage-1 trend is intact "
                "and price has not closed below EMA50. Cancel if Nifty flips risk-off."
            )
            result["invalidation"] = (
                f"Call is wrong if daily close < ₹{result['stop']:,.2f} (stop) "
                f"or decisive close back below EMA50 (₹{ema50:,.2f})."
            )
        else:
            # SELL two-stage
            checks = [
                (price < ema20 or ema20 <= 0, f"Price below EMA20 — short-term weakness."),
                (price < ema50 or ema50 <= 0, f"Price below EMA50 — swing downtrend."),
                (adx >= 22, f"ADX {adx:.1f} supports a trending decline."),
                (rsi <= 48, f"RSI {rsi:.1f} not in strong oversold bounce zone only."),
            ]
            ok = 0
            for passed, msg in checks:
                if passed:
                    s1.append("✓ " + msg)
                    ok += 1
                else:
                    s1.append("✗ " + msg)
                    fail.append(msg)
            stage1 = ok >= 3
            bear_pat = any(
                p in patterns for p in [
                    "Shooting Star", "Bearish Engulfing", "Lower High / Lower Low",
                    "Breakdown Low", "Bearish Marubozu",
                ]
            )
            tchecks = [
                (vol_r >= 1.0, f"Volume ratio {vol_r:.2f}."),
                (bear_pat or rsi < 45, f"Trigger patterns/momentum: {patterns[:3] if patterns else 'soft tape'}."),
                (True, "Prefer sell rallies into EMA20 resistance, not panic lows only."),
            ]
            ok2 = 0
            for passed, msg in tchecks:
                if passed:
                    s2.append("✓ " + msg)
                    ok2 += 1
                else:
                    s2.append("✗ " + msg)
            stage2 = ok2 >= 2
            target = price - 1.8 * atr
            stop = price + 2.2 * atr
            result["target"] = round(max(target, price * 0.90), 2)
            result["stop"] = round(stop, 2)
            result["hold_days"] = 18
            result["entry_plan"] = (
                "PREFERRED ENTRY: Sell / exit longs on same day if weakness holds below EMA20. "
                "Next 1–2 days only if stage-1 downtrend intact. Cover if price reclaims EMA50."
            )
            result["invalidation"] = f"Invalid if close > stop ₹{result['stop']:,.2f} or reclaim EMA50."

        result["stage1"] = s1
        result["stage2"] = s2
        result["fail"] = fail
        result["stage1_pass"] = stage1
        result["stage2_pass"] = stage2
        result["sure"] = bool(stage1 and stage2)

        # Score
        pred = 55
        if stage1:
            pred += 15
        if stage2:
            pred += 12
        if result["sure"]:
            pred += 8
        result["prediction"] = float(min(92, pred))

        if result["sure"]:
            result["analyst_summary"] = (
                f"SURE {side} on {display}: Stage-1 trend and Stage-2 trigger both clear. "
                f"This is the subset an experienced analyst would prioritise for a 15–20 day swing — "
                f"not a guarantee, but maximum positive asymmetry under the checklist. "
                f"Entry ~₹{result['entry']:,.2f}, Target ₹{result['target']:,.2f}, "
                f"Stop ₹{result['stop']:,.2f}, hold ~{result['hold_days']} days."
            )
        else:
            result["analyst_summary"] = (
                f"NOT a sure call on {display}. "
                + ("Stage-1 trend incomplete. " if not stage1 else "")
                + ("Stage-2 trigger incomplete. " if not stage2 else "")
                + "Wait for alignment; forcing a trade here raises error rate."
            )
    except Exception as e:
        result["fail"].append(str(e))
        result["analyst_summary"] = f"Analysis failed: {e}"
    return result


def build_sure_calls_from_scan(results: pd.DataFrame, regime: dict, max_n: int = 3) -> pd.DataFrame:
    """From last scan, keep only names that pass two-stage sure logic + regime."""
    if results is None or results.empty:
        return pd.DataFrame()
    x = results.copy()
    # Candidate pool: BUY/SELL/WATCH high score
    x["Prediction"] = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
    pool = x[x["Prediction"] >= 60].copy()
    if pool.empty:
        pool = x.sort_values("Prediction", ascending=False).head(40)
    else:
        pool = pool.sort_values("Prediction", ascending=False).head(40)

    rows = []
    for _, r in pool.iterrows():
        stock = str(r.get("Stock", ""))
        call = str(r.get("Call", "BUY")).upper()
        side = "SELL" if call == "SELL" else "BUY"
        if side == "BUY" and not regime.get("trade_longs", True):
            continue
        if side == "SELL" and not regime.get("trade_shorts", True):
            continue
        v = two_stage_stock_verdict(stock, side=side)
        if not v.get("sure"):
            continue
        rows.append({
            "Stock": stock,
            "Side": side,
            "Sure": "YES",
            "Prediction": v["prediction"],
            "Entry": v["entry"],
            "Target": v["target"],
            "Stop Loss": v["stop"],
            "Hold Days": v["hold_days"],
            "Stage1": "PASS" if v["stage1_pass"] else "FAIL",
            "Stage2": "PASS" if v["stage2_pass"] else "FAIL",
            "Analyst summary": v["analyst_summary"],
            "Entry plan": v["entry_plan"],
            "Invalidation": v["invalidation"],
            "Sector": r.get("Sector", ""),
        })
        if len(rows) >= max_n:
            break
    return pd.DataFrame(rows)


def show_sure_calls_page(results: pd.DataFrame):
    """Two-stage Sure Call desk — analyst-style explanations."""
    st.title("✅ Sure Call Mode — Two-Stage Analyst Desk")
    st.caption(
        "Stage 0: Nifty regime · Stage 1: Trend · Stage 2: Trigger. "
        "Only when all align do we label **SURE**. Horizon **15–20 days**. "
        "This raises quality; it does **not** guarantee 100% wins."
    )

    regime = nifty_market_regime()
    st.subheader("📊 Stage 0 — Nifty market regime")
    a, b, c, d = st.columns(4)
    a.metric("Regime", regime.get("regime", "—"))
    b.metric("Regime score", regime.get("score", "—"))
    c.metric("Nifty", f"{regime.get('price', 0):,.0f}", f"{regime.get('pct', 0):+.2f}%")
    d.metric(
        "New BUY allowed?",
        "YES" if regime.get("trade_longs") else "NO — STAY DEFENSIVE",
    )

    if not regime.get("trade_longs"):
        st.error(
            "**Do not take fresh long / sure BUY calls today.** "
            "Nifty is weak under the model’s regime rules. "
            "Experienced desks stand aside or only manage open risk."
        )
    elif regime.get("regime", "").startswith("MIXED"):
        st.warning("**Selective only** — mixed Nifty; require full two-stage pass.")
    else:
        st.success("**Risk-on bias** — two-stage sure BUYs allowed.")

    for r in regime.get("reasons", []):
        st.markdown(f"- {r}")

    st.divider()
    st.subheader("🧠 How an experienced analyst splits the call")
    st.markdown(
        """
| Stage | Question | BUY needs | SELL needs |
|-------|----------|-----------|------------|
| **0 Regime** | Is the index friendly? | Nifty not risk-off | Nifty not strong melt-up only |
| **1 Trend** | Is the stock in the right trend? | Price > EMA20 & EMA50, ADX strong | Price < EMA20 & EMA50, ADX strong |
| **2 Trigger** | Is *now* a good moment? | RSI 52–70, volume, bullish pattern / hold | RSI soft, bearish pattern, volume |

**Only Stage 1 + Stage 2 (and friendly Stage 0 for longs) → SURE CALL.**
        """
    )

    st.subheader("⏰ When to buy after the model gives a call")
    st.markdown(
        """
1. **Same day (preferred if setup is fresh)**  
   - Buy if price **holds above the signal day’s low** (BUY).  
   - Prefer a **dip toward EMA20**, not a blind chase at the high.  
2. **Next 1–2 trading sessions**  
   - Still valid if **Stage-1 trend is intact** (still above EMA50) and stop not hit.  
3. **After 3+ days without entry**  
   - **Do not** chase; re-run Sure Call — structure may be stale.  
4. **Repeated calls**  
   - Scan can list the same stock again on later days if it still passes.  
   - Treat as **one position idea**, not a new pile-on, unless you scaled a plan.  
   - If the stock **stopped out recently**, Sure mode should demand a full reset (new structure).
        """
    )

    st.divider()
    tab1, tab2 = st.tabs(["✅ Sure calls from last scan", "🔍 Manual two-stage on one stock"])

    with tab1:
        if results is None or results.empty:
            st.info("Run **FULL MARKET SCAN** first so the desk has a universe to filter.")
        else:
            max_n = st.slider("Max sure names", 1, 5, 3, key="sure_max_n")
            if st.button("Generate sure calls", type="primary", key="sure_gen"):
                with st.spinner("Running two-stage filter on top candidates..."):
                    sure_df = build_sure_calls_from_scan(results, regime, max_n=max_n)
                    st.session_state["sure_df"] = sure_df
                    n_saved = 0
                    try:
                        special = []
                        if sure_df is not None and not sure_df.empty:
                            for _, sr in sure_df.iterrows():
                                special.append({
                                    "Stock": sr.get("Stock"),
                                    "Call": sr.get("Side", "BUY"),
                                    "Call Source": "SURE",
                                    "Strategy": "Sure Call",
                                    "Entry": sr.get("Entry"),
                                    "Target": sr.get("Target"),
                                    "Stop Loss": sr.get("Stop Loss"),
                                    "Hold Days": sr.get("Hold Days", 18),
                                    "Prediction": 85,
                                    "Reason": str(sr.get("Analyst summary", "Sure Call") or "Sure Call")[:300],
                                })
                            n_saved = save_special_calls(special)
                    except Exception as e:
                        st.warning(f"Save to history failed: {e}")
                    if n_saved:
                        st.success(
                            f"Saved **{n_saved}** Sure call(s) to Past Predictions "
                            f"(Call Source = **SURE**). Open History → BUY source → Strategy/Sure."
                        )
                    elif sure_df is not None and not sure_df.empty:
                        st.info("Sure calls generated (already saved today or nothing new to write).")
            sure_df = st.session_state.get("sure_df", pd.DataFrame())
            if sure_df is not None and isinstance(sure_df, pd.DataFrame) and not sure_df.empty:
                if st.button("💾 Save these Sure calls to Past Predictions again", key="sure_resave"):
                    special = []
                    for _, sr in sure_df.iterrows():
                        special.append({
                            "Stock": sr.get("Stock"),
                            "Call": sr.get("Side", "BUY"),
                            "Call Source": "SURE",
                            "Strategy": "Sure Call",
                            "Entry": sr.get("Entry"),
                            "Target": sr.get("Target"),
                            "Stop Loss": sr.get("Stop Loss"),
                            "Hold Days": sr.get("Hold Days", 18),
                            "Prediction": 85,
                        })
                    n_saved = save_special_calls(special)
                    st.success(f"Wrote {n_saved} Sure row(s) to history CSV.")
            if sure_df is None or (isinstance(sure_df, pd.DataFrame) and sure_df.empty):
                st.caption("Click **Generate sure calls** after a scan — they are saved to history automatically.")
            else:
                st.dataframe(
                    sure_df.drop(columns=["Analyst summary", "Entry plan", "Invalidation"], errors="ignore"),
                    use_container_width=True,
                    hide_index=True,
                )
                for _, row in sure_df.iterrows():
                    stock_name = str(row.get("Stock", ""))
                    match = strategies_matching_stock(stock_name, side_filter="BUY")
                    strat_count = match.get("count", 0)
                    strat_names = ", ".join(match.get("names") or []) or "— (Sure two-stage only; no named swing strategy hit)"
                    st.markdown(
                        f"""
                        <div class="buy-box" style="padding:14px;margin-bottom:12px;border-radius:10px;">
                        <h3 style="margin:0 0 8px 0;">✅ SURE {row.get('Side')} · {stock_name}
                        <span style="font-size:0.8em;color:#666;">({row.get('Sector','')})</span></h3>
                        <p style="margin:0;line-height:1.65;">
                        <b>Prediction / call:</b> SURE {row.get('Side')} (two-stage analyst desk)<br>
                        <b>Falls under strategies:</b> <b>{strat_count}</b> strategy match(es)<br>
                        <b>Strategy names:</b> {strat_names}<br>
                        <b>Entry:</b> ₹{safe_float(row.get('Entry')):,.2f} &nbsp;|&nbsp;
                        <b>Target:</b> ₹{safe_float(row.get('Target')):,.2f} &nbsp;|&nbsp;
                        <b>Stop:</b> ₹{safe_float(row.get('Stop Loss')):,.2f} &nbsp;|&nbsp;
                        <b>Hold:</b> {row.get('Hold Days')} days<br>
                        <b>Stage1 / Stage2:</b> {row.get('Stage1')} / {row.get('Stage2')}<br><br>
                        <b>Analyst brief:</b> {row.get('Analyst summary')}<br><br>
                        <b>Entry timing:</b> {row.get('Entry plan')}<br><br>
                        <b>Invalidation:</b> {row.get('Invalidation')}
                        </p></div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if match.get("detail"):
                        st.caption("Strategy-level targets for this stock:")
                        st.dataframe(pd.DataFrame(match["detail"]), use_container_width=True, hide_index=True)
                    if st.button(f"Full chart — {stock_name}", key=f"sure_open_{stock_name}"):
                        st.session_state.selected_stock = stock_name
                        st.session_state.page = "Stock Analysis"
                        st.rerun()

    with tab2:
        st.write("Run the full two-stage checklist on any symbol (independent of scan).")
        c1, c2 = st.columns(2)
        with c1:
            msym = st.text_input("Symbol", value="RELIANCE", key="sure_manual_sym").upper().strip()
        with c2:
            mside = st.selectbox("Side", ["BUY", "SELL"], key="sure_manual_side")
        if st.button("Run two-stage analysis", key="sure_manual_btn"):
            if mside == "BUY" and not regime.get("trade_longs"):
                st.error("Nifty regime blocks new long sure-calls today — analysis below is educational only.")
            v = two_stage_stock_verdict(msym, side=mside)
            if v.get("sure"):
                st.success(v.get("analyst_summary"))
            else:
                st.warning(v.get("analyst_summary"))
            st.write("**Stage 1 — Trend**")
            for line in v.get("stage1", []):
                st.markdown(f"- {line}")
            st.write("**Stage 2 — Trigger**")
            for line in v.get("stage2", []):
                st.markdown(f"- {line}")
            st.write(
                f"**Levels:** Entry ₹{v.get('entry'):,.2f} · Target ₹{v.get('target'):,.2f} · "
                f"Stop ₹{v.get('stop'):,.2f} · Hold {v.get('hold_days')}d"
            )
            match = strategies_matching_stock(msym, side_filter=mside if mside == "SELL" else "BUY")
            st.write(
                f"**Falls under strategies:** **{match.get('count', 0)}** — "
                f"{', '.join(match.get('names') or []) or 'none of the named swing strategies'}"
            )
            if match.get("detail"):
                st.dataframe(pd.DataFrame(match["detail"]), use_container_width=True, hide_index=True)
            st.info(v.get("entry_plan", ""))
            st.caption(v.get("invalidation", ""))

    st.divider()
    st.markdown(
        """
### Repeated calls — policy
- The **scanner may list the same stock on multiple days** if it still scores well.  
- **Sure Call page** should be treated as: *one active idea per stock* unless you deliberately scale.  
- After a **stop-out**, wait for a **new Stage-1 rebuild** (e.g. reclaim EMA50) before another sure BUY.  
- History + learning panels still show prior reasons when the same name reappears on BUY/Dashboard.
        """
    )


# ============================================================
# INDEX CHARTS + FULL ANALYSIS (NIFTY / BANK NIFTY)
# ============================================================

def show_index_page(index_key="NIFTY"):
    """
    Dedicated page for NIFTY 50 or BANK NIFTY:
    live quote + TradingView chart + full technical analysis.
    """
    index_key = str(index_key).upper()
    if index_key in ["BANKNIFTY", "BANK NIFTY", "BANK"]:
        title = "BANK NIFTY"
        yf_symbol = "^NSEBANK"
        tv_hint = "NSE:BANKNIFTY"
    else:
        title = "NIFTY 50"
        yf_symbol = "^NSEI"
        tv_hint = "NSE:NIFTY"

    st.title(f"📈 {title} — Chart & Full Analysis")
    st.caption(market_status_text() + f" • Symbol: {tv_hint}")

    # Live quote
    quote = live_quote(yf_symbol)
    if quote:
        a, b, c, d = st.columns(4)
        a.metric(quote["label"], f"{quote['price']:,.2f}", f"{quote['pct']:+.2f}%")
        b.metric("Change", f"{quote['change']:+,.2f}")
        c.metric("Updated", quote["updated"])
        d.metric("Status", market_status_text().replace("NSE MARKET ", ""))
    else:
        st.warning("Live quote unavailable right now.")

    st.subheader("📊 Interactive Chart")
    # Target / stop filled after analysis below; chart refreshes with analysis block
    show_tradingview_chart(yf_symbol, title, height=560)

    # Full technical analysis
    st.subheader("🤖 Full Technical Analysis")
    with st.spinner(f"Analysing {title}..."):
        df = stock_history(yf_symbol)
        if df is None or df.empty or len(df) < 60:
            st.error("Not enough historical data for analysis.")
            return

        # analyse_stock expects equity-style symbols; indices work with same OHLC
        result = analyse_stock(yf_symbol, df, fetch_news=False)

    if not result:
        st.error("Analysis could not be completed for this index.")
        return

    call = result["Call"]
    if call == "BUY":
        css = "buy-box"
    elif call == "SELL":
        css = "sell-box"
    elif call == "HOLD":
        css = "hold-box"
    else:
        css = "watch-box"

    st.markdown(
        f"""
        <div class="{css}">
        <h2>Recommendation: {call}</h2>
        <p>
        Prediction: <b>{result['Prediction']}%</b>
        &nbsp; | &nbsp;
        Risk: <b>{result['Risk %']}%</b>
        &nbsp; | &nbsp;
        Risk Level: <b>{result['Risk Level']}</b>
        &nbsp; | &nbsp;
        Priority: <b>{result['Priority']}</b>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"{result['Price']:,.2f}")
    m2.metric("Target", f"{result['Target']:,.2f}")
    m3.metric("Stop Loss", f"{result['Stop Loss']:,.2f}")
    m4.metric("Hold Days", str(result["Hold Days"]))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RSI", f"{result.get('RSI', 0):.1f}")
    m2.metric("ADX", f"{result.get('ADX', 0):.1f}")
    m3.metric("MACD", f"{result.get('MACD', 0):.4f}")
    m4.metric("Volume Ratio", f"{result.get('Volume Ratio', 0):.2f}")

    # Live target / stop status
    current = safe_float(quote["price"]) if quote else safe_float(result["Price"])
    level_status = check_price_vs_levels(
        current_price=current,
        entry=result.get("Price"),
        target=result.get("Target"),
        stop_loss=result.get("Stop Loss"),
        call=result.get("Call", "BUY"),
    )

    if level_status["status"] == "TARGET ACHIEVED":
        rec_html = str(level_status.get("recommendation", "")).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="success-box">
            <h3>🎯 TARGET ACHIEVED</h3>
            <p style="margin:0;line-height:1.6;">
            Previous Target: {result['Target']:,.2f}<br>
            Current Price: {current:,.2f}<br>
            🎯 TARGET ACHIEVED
            </p>
            <br>
            <b>NEW ANALYSIS:</b><br>
            New Target: {level_status['new_target']:,.2f}<br>
            New Stop Loss: {level_status['new_stop']:,.2f}<br>
            <br>
            <b>Recommendation:</b><br>{rec_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif level_status["status"] == "STOP LOSS HIT":
        st.markdown(
            f"""
            <div class="danger-box">
            <h3>🔴 STOP LOSS HIT</h3>
            <p style="margin:0;line-height:1.6;">
            Current Price: {current:,.2f}<br>
            Stop Loss: {result['Stop Loss']:,.2f}<br>
            🔴 STOP LOSS HIT
            </p>
            <br>
            <b>Recommendation:</b><br>🔴 SELL / EXIT
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Analysis Reasons")
    for reason in str(result.get("Technical Reasons", "")).split(" | "):
        if reason.strip():
            st.write("• " + reason.strip())

    st.write(f"**Patterns:** {result.get('Patterns', '')}")

    st.subheader("📈 Technical Chart (Candles + EMA + BB)")
    try:
        st.plotly_chart(make_chart(result), use_container_width=True)
    except Exception:
        st.caption("Technical chart unavailable.")

    # Quick switch
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📈 Open NIFTY 50", use_container_width=True, key="idx_to_nifty"):
            st.session_state.page = "Nifty Analysis"
            st.rerun()
    with c2:
        if st.button("🏦 Open BANK NIFTY", use_container_width=True, key="idx_to_bn"):
            st.session_state.page = "BankNifty Analysis"
            st.rerun()


# ============================================================
# HISTORY PAGE
# ============================================================


def save_special_calls(rows_list):
    """
    Save Sure / Strategy / Precision / High-conv into history with Call Source.
    Dedupe: same Stock + Call Source + Strategy name within 1 day (open only).
    Different strategies on same stock can all save.
    """
    if not rows_list:
        return 0
    try:
        history = load_history()
    except Exception:
        history = pd.DataFrame()
    now = datetime.now()
    new_rows = []
    for r in rows_list:
        stock = display_symbol(str(r.get("Stock", "") or "")).upper().strip()
        if not stock:
            continue
        source = str(r.get("Call Source", "STRATEGY") or "STRATEGY").upper().strip()
        strat_name = str(r.get("Strategy", "") or "").strip()
        call = str(r.get("Call", "BUY") or "BUY").upper().strip()
        if call not in ("BUY", "SELL", "HOLD"):
            # Side field from strategy tables
            call = str(r.get("Side", "BUY") or "BUY").upper().strip()
            if call not in ("BUY", "SELL", "HOLD"):
                call = "BUY"
        entry = safe_float(r.get("Entry", r.get("Price", 0)))
        target = safe_float(r.get("Target", 0))
        stop = safe_float(r.get("Stop Loss", r.get("Stop", 0)))
        hold_days = int(safe_float(r.get("Hold Days", DEFAULT_HOLD_DAYS), DEFAULT_HOLD_DAYS)) or DEFAULT_HOLD_DAYS
        expiry = (now + timedelta(days=hold_days)).strftime("%Y-%m-%d")
        skip = False
        if history is not None and not history.empty:
            try:
                h = history.copy()
                if "Call Source" not in h.columns:
                    h["Call Source"] = "SCAN"
                if "Strategy" not in h.columns:
                    h["Strategy"] = ""
                mask = (
                    h["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip() == stock
                ) & (
                    h["Call Source"].astype(str).str.upper().str.strip() == source
                ) & (
                    h["Strategy"].astype(str).str.strip() == strat_name
                )
                sub = h.loc[mask]
                if len(sub):
                    last = pd.to_datetime(sub["Prediction Date"], errors="coerce").max()
                    if pd.notna(last):
                        try:
                            last_naive = last.to_pydatetime().replace(tzinfo=None)
                        except Exception:
                            last_naive = now
                        if (now - last_naive).total_seconds() < 86400:
                            res = str(sub.sort_values("Prediction Date").iloc[-1].get("Result", "")).upper()
                            if not any(x in res for x in ("TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD")):
                                skip = True
            except Exception:
                pass
        if skip:
            continue
        new_rows.append({
            "Prediction Date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Stock": stock,
            "Symbol": clean_symbol(stock),
            "Call Source": source,
            "Strategy": strat_name,
            "Patterns": str(r.get("Patterns", "") or ""),
            "Call": call,
            "Prediction": r.get("Prediction", 80 if source in ("SURE", "PRECISION", "HIGH_CONV") else 75),
            "Entry": entry if entry else "",
            "Target": target if target else "",
            "Stop Loss": stop if stop else "",
            "Risk %": r.get("Risk %", ""),
            "Risk Level": r.get("Risk Level", "MEDIUM"),
            "Hold Days": hold_days,
            "Expiry Date": expiry,
            "Status": "OPEN",
            "Result": "PENDING",
            "Result Detail": "",
            "Days Taken": "",
            "Outcome Message": "",
            "Recommendation": "",
            "Suggestion": "",
            "Reason": str(r.get("Reason", f"{source} | {strat_name}") or f"{source} call"),
            "Current Price": entry if entry else "",
            "Evaluation Date": "",
            "Exit Price": "",
            "Return %": "",
        })
    if not new_rows:
        return 0
    add = pd.DataFrame(new_rows)
    if history is None or history.empty:
        out = add
    else:
        for c in add.columns:
            if c not in history.columns:
                history[c] = ""
        for c in history.columns:
            if c not in add.columns:
                add[c] = ""
        out = pd.concat([history, add], ignore_index=True)
    try:
        # Do NOT run ensure_result_columns (scan schema) — keeps Call Source
        out = normalize_history_df(out)
        # Force Call Source on new rows if normalize blanked anything
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(HISTORY_FILE, index=False)
        bump_sync_version("past_predictions")
        return len(new_rows)
    except Exception:
        try:
            out.to_csv(HISTORY_FILE, index=False)
            bump_sync_version("past_predictions")
            return len(new_rows)
        except Exception:
            return 0


def import_strategy_files_to_history() -> int:
    """
    Pull strategy_live_signals.csv / session into recommendation_history as STRATEGY.
    Fixes '40 stocks shown green but Past Predictions empty'.
    """
    n = 0
    frames = []
    try:
        if STRATEGY_LIVE_FILE.exists():
            frames.append(pd.read_csv(STRATEGY_LIVE_FILE))
    except Exception:
        pass
    for key in ("live_strat_df", "stocks_under_all_df", "stocks_under_df", "sure_df"):
        try:
            df = st.session_state.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
        except Exception:
            pass
    special = []
    for df in frames:
        for _, lr in df.iterrows():
            stock = lr.get("Stock") or lr.get("Symbol")
            if not stock:
                continue
            strat = str(lr.get("Strategy", "Strategy Lab") or "Strategy Lab")
            source = "SURE" if "sure" in strat.lower() else "STRATEGY"
            special.append({
                "Stock": stock,
                "Call": lr.get("Side", lr.get("Call", "BUY")),
                "Call Source": source,
                "Strategy": strat,
                "Entry": lr.get("Price", lr.get("Entry", lr.get("Current / Live Price"))),
                "Target": lr.get("Target"),
                "Stop Loss": lr.get("Stop Loss"),
                "Hold Days": lr.get("Hold Days", 15),
                "Prediction": 78,
            })
    try:
        sdf = st.session_state.get("sure_df")
        if isinstance(sdf, pd.DataFrame) and not sdf.empty:
            for _, sr in sdf.iterrows():
                special.append({
                    "Stock": sr.get("Stock"),
                    "Call": sr.get("Side", "BUY"),
                    "Call Source": "SURE",
                    "Strategy": "Sure Call",
                    "Entry": sr.get("Entry"),
                    "Target": sr.get("Target"),
                    "Stop Loss": sr.get("Stop Loss"),
                    "Hold Days": sr.get("Hold Days", 18),
                    "Prediction": 85,
                })
    except Exception:
        pass
    if special:
        n = save_special_calls(special)
    return n


def expert_trader_verdict(symbol: str, call_hint: str = "") -> dict:
    """
    25Y desk feedback: right time or not, BUY/SELL/HOLD/LT, one learning line.
    Uses technicals + LT fundamentals when available.
    """
    sym = display_symbol(symbol)
    out = {
        "symbol": sym,
        "timing": "WAIT",
        "action": "WATCH",
        "lt_action": "Review",
        "confidence": 50,
        "feedback": "",
        "learning": "",
        "index": "",
        "mcap_cr": "—",
        "book_value": "—",
        "revenue_cr": "—",
        "profit_cr": "—",
        "face_value": "—",
    }
    try:
        df = stock_history(clean_symbol(sym), interval="1d")
        if df is None or len(df) < 60:
            out["feedback"] = "Not enough price history — no trade until data is clean."
            return out
        df = calculate_indicators(df)
        row = df.iloc[-1]
        price = safe_float(row.get("Close"))
        rsi = safe_float(row.get("RSI"), 50)
        adx = safe_float(row.get("ADX"), 0)
        ema20 = safe_float(row.get("EMA20"))
        ema50 = safe_float(row.get("EMA50"))
        ema200 = safe_float(row.get("EMA200"))
        macd_h = safe_float(row.get("MACDHist"))
        atr = safe_float(row.get("ATR")) or price * 0.02

        try:
            prof = enrich_stock_profile(sym)
            out["index"] = prof.get("Index Tags", "")
            out["mcap_cr"] = prof.get("Market Cap Cr Text", "—")
            out["book_value"] = prof.get("Book Value Text", "—")
            out["revenue_cr"] = prof.get("Revenue Cr Text", "—")
            out["profit_cr"] = prof.get("Net Income Cr Text", "—")
            lt_score = safe_float(prof.get("LT Score"), 45)
            out["lt_action"] = (
                "CORE LONG-TERM HOLD/ACCUMULATE" if lt_score >= 72
                else ("SMALL CORE ONLY" if lt_score >= 58 else "NO LT CORE — trade only")
            )
        except Exception:
            lt_score = 45
            prof = {}

        try:
            fund = fetch_fundamentals(sym)
            fv = fund.get("face_value")
            if fv:
                out["face_value"] = f"₹{safe_float(fv):.2f}"
        except Exception:
            pass

        above20 = ema20 <= 0 or price > ema20
        above50 = ema50 <= 0 or price > ema50
        above200 = ema200 <= 0 or price > ema200
        structure_long = above20 and above50
        stretched = rsi >= 78
        washed = rsi <= 25
        trend_on = adx >= 18

        conf = 50
        action = "WATCH"
        timing = "WAIT"
        lines = []

        if structure_long and trend_on and 42 <= rsi <= 68 and macd_h >= 0:
            action = "BUY"
            timing = "GOOD TIME — structure + momentum aligned"
            conf = 72 + (5 if above200 else 0)
            lines.append("Price holds above short EMAs with usable ADX — classic swing long window.")
        elif structure_long and 40 <= rsi <= 72:
            action = "BUY"
            timing = "OK TIME — but size moderate"
            conf = 62
            lines.append("Trend soft-positive; take only if stop is respected.")
        elif not above50 and adx >= 22 and rsi < 45 and macd_h < 0:
            action = "SELL"
            timing = "GOOD TIME for trade-short / exit longs"
            conf = 68
            lines.append("Breakdown under EMA50 with momentum — prefer exit or short plan.")
        elif stretched and above50:
            action = "HOLD"
            timing = "WAIT — extended; don't chase"
            conf = 55
            lines.append("RSI stretched; pros wait for pullback toward EMA20/50.")
        elif washed and above200:
            action = "BUY"
            timing = "SELECTIVE dip-buy only if LT quality high"
            conf = 58 if lt_score >= 60 else 48
            lines.append("Oversold in higher timeframe uptrend — only with strong LT score.")
        else:
            action = "HOLD" if above200 else "WATCH"
            timing = "NOT ideal — stand aside or manage existing only"
            conf = 45
            lines.append("No clean edge; cash / hold is a position.")

        if call_hint:
            ch = call_hint.upper()
            if "BUY" in ch and action == "SELL":
                lines.append("Strategy/Sure says BUY but desk sees weak structure — skip or tiny size.")
                conf = min(conf, 48)
                timing = "CONFLICT — prefer WAIT"
            if "SELL" in ch and action == "BUY":
                lines.append("Desk sees support for long while list says SELL — don't force short.")

        out["action"] = action
        out["timing"] = timing
        out["confidence"] = int(min(90, conf))
        out["feedback"] = " ".join(lines)
        out["learning"] = (
            "Edge = timing + risk. If timing is WAIT, the best trade is no trade. "
            "Separate swing risk (honour SL) from LT core (only high LT score)."
        )
        out["swing_stop"] = round(price - 1.4 * atr, 2) if action == "BUY" else round(price + 1.4 * atr, 2)
        out["swing_target"] = round(price + 2.2 * atr, 2) if action == "BUY" else round(price - 2.2 * atr, 2)
        out["price"] = price
        out["rsi"] = rsi
        out["adx"] = adx
    except Exception as e:
        out["feedback"] = f"Desk check failed: {e}"
    return out


def show_expert_trader_box(symbol: str, call_hint: str = ""):
    """UI block: experienced trader feedback for any page."""
    v = expert_trader_verdict(symbol, call_hint)
    st.markdown(f"#### 🎓 Experienced trader desk — {v.get('symbol')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Timing", v.get("timing", "—")[:28])
    c2.metric("Action", v.get("action", "—"))
    c3.metric("Confidence", f"{v.get('confidence', 0)}%")
    c4.metric("LT stance", str(v.get("lt_action", "—"))[:24])
    st.write(v.get("feedback", ""))
    st.caption(
        f"Index: {v.get('index') or '—'} · Mcap {v.get('mcap_cr')} · "
        f"Face {v.get('face_value')} · BV {v.get('book_value')} · "
        f"Rev {v.get('revenue_cr')} · Profit {v.get('profit_cr')}"
    )
    st.info(f"**Learning:** {v.get('learning', '')}")
    return v


def history_success_breakdown(history: pd.DataFrame) -> pd.DataFrame:
    """Success rate by Call Source / Strategy / Patterns."""
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    if "Call Source" not in h.columns:
        h["Call Source"] = "SCAN"
    if "Strategy" not in h.columns:
        h["Strategy"] = ""
    if "Patterns" not in h.columns:
        h["Patterns"] = ""
    res = h.get("Result", pd.Series([""] * len(h))).astype(str).str.upper()
    h["_win"] = res.str.contains("TARGET ACHIEVED", na=False)
    h["_loss"] = res.str.contains("STOP LOSS HIT", na=False)
    h["_closed"] = h["_win"] | h["_loss"]
    rows = []
    for col in ["Call Source", "Strategy", "Patterns"]:
        for key, g in h.groupby(h[col].astype(str).fillna("")):
            if not str(key).strip() or str(key) == "nan":
                continue
            closed = int(g["_closed"].sum())
            wins = int(g["_win"].sum())
            losses = int(g["_loss"].sum())
            rate = round(wins / closed * 100, 1) if closed else 0.0
            rows.append({
                "Filter": col,
                "Value": key[:80],
                "Calls": len(g),
                "Closed": closed,
                "Wins": wins,
                "Losses": losses,
                "Success %": rate,
            })
    return pd.DataFrame(rows)



def dedupe_history_same_day(df: pd.DataFrame, one_stock_per_day: bool = True) -> pd.DataFrame:
    """
    Remove same-day duplicate trades.
    one_stock_per_day=True → exactly ONE row per stock per calendar day
    (merges sources/strategies into the kept row).
    Priority: SURE > HIGH_CONV > PRECISION > STRATEGY > MY_STRATEGY > SCAN.
    """
    if df is None or df.empty or "Stock" not in df.columns:
        return df if df is not None else pd.DataFrame()
    x = df.copy()
    x["Stock"] = (
        x["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()
    )
    if "Prediction Date" in x.columns:
        x["_pdt"] = pd.to_datetime(x["Prediction Date"], errors="coerce")
        x["_day"] = x["_pdt"].dt.strftime("%Y-%m-%d")
    else:
        x["_day"] = ""
        x["_pdt"] = pd.NaT
    if "Call Source" not in x.columns:
        x["Call Source"] = "SCAN"
    x["_src"] = x["Call Source"].astype(str).str.upper().str.strip()

    def _src_rank(s):
        u = str(s).upper()
        order = ["SURE", "HIGH_CONV", "PRECISION", "STRATEGY", "MY_STRATEGY", "SCAN"]
        for i, name in enumerate(order):
            if name in u:
                return len(order) - i
        return 0

    def _rank_result(s):
        u = str(s).upper()
        if "TARGET ACHIEVED" in u or u == "WIN":
            return 3
        if "STOP LOSS" in u or u == "LOSS":
            return 2
        if "HOLDING PERIOD" in u:
            return 1
        return 0

    x["_sr"] = x["_src"].map(_src_rank)
    x["_rr"] = x["Result"].map(_rank_result) if "Result" in x.columns else 0

    if one_stock_per_day:
        # Collect all sources/strategies for the day before drop (pandas-safe)
        def _join_unique(series):
            try:
                parts = []
                for v in list(series):
                    v = str(v).strip()
                    if v and v.lower() not in ("nan", "none", ""):
                        if v not in parts:
                            parts.append(v)
                return " · ".join(parts)
            except Exception:
                return ""

        try:
            agg_src = (
                x.groupby(["Stock", "_day"], dropna=False)["_src"]
                .apply(_join_unique)
                .rename("_all_sources")
            )
        except Exception:
            agg_src = (
                x.groupby(["Stock", "_day"], dropna=False)["_src"]
                .first()
                .rename("_all_sources")
            )
        if "Strategy" in x.columns:
            try:
                agg_st = (
                    x.groupby(["Stock", "_day"], dropna=False)["Strategy"]
                    .apply(_join_unique)
                    .rename("_all_strategies")
                )
            except Exception:
                agg_st = (
                    x.groupby(["Stock", "_day"], dropna=False)["Strategy"]
                    .first()
                    .rename("_all_strategies")
                )
        else:
            agg_st = None

        x = x.sort_values(
            by=[c for c in ["_day", "Stock", "_sr", "_rr", "_pdt"] if c in x.columns],
            ascending=[True, True, False, False, False],
        )
        x = x.drop_duplicates(subset=["Stock", "_day"], keep="first")
        x = x.merge(agg_src.reset_index(), on=["Stock", "_day"], how="left")
        if agg_st is not None:
            x = x.merge(agg_st.reset_index(), on=["Stock", "_day"], how="left")
            x["Strategy"] = x.apply(
                lambda r: r["_all_strategies"] if r.get("_all_strategies") else r.get("Strategy", ""),
                axis=1,
            )
        x["Call Source"] = x.apply(
            lambda r: r["_all_sources"] if r.get("_all_sources") else r.get("Call Source", ""),
            axis=1,
        )
        drop_extra = ["_all_sources", "_all_strategies"]
    else:
        x = x.sort_values(
            by=[c for c in ["_day", "Stock", "_src", "_rr", "_pdt"] if c in x.columns],
            ascending=[True, True, True, False, False],
        )
        x = x.drop_duplicates(subset=["Stock", "_day", "_src"], keep="first")
        drop_extra = []

    drop_cols = [c for c in ["_pdt", "_day", "_src", "_rr", "_sr"] + drop_extra if c in x.columns]
    return x.drop(columns=drop_cols).reset_index(drop=True)


def _row_reward_risk(row) -> float:
    """R:R from Entry/Target/Stop. 0 if invalid."""
    entry = safe_float(row.get("Entry", row.get("Price", row.get("BUY Price"))))
    tgt = safe_float(row.get("Target"))
    sl = safe_float(row.get("Stop Loss"))
    call = str(row.get("Call", row.get("Side", "BUY"))).upper()
    if entry <= 0 or tgt <= 0 or sl <= 0:
        return 0.0
    if "SELL" in call:
        risk = sl - entry
        reward = entry - tgt
    else:
        risk = entry - sl
        reward = tgt - entry
    if risk <= 0:
        return 0.0
    return round(reward / risk, 2)


def _success_from_df(df: pd.DataFrame) -> dict:
    """Targets / (Targets+Stops) + counts on a history/paper slice."""
    if df is None or df.empty or "Result" not in df.columns:
        return {"wins": 0, "losses": 0, "open": 0, "success": 0.0, "n": 0}
    ru = df["Result"].astype(str).str.upper()
    wins = int(ru.str.contains("TARGET ACHIEVED", na=False).sum() + ru.isin(["WIN"]).sum())
    losses = int(ru.str.contains("STOP LOSS", na=False).sum() + ru.isin(["LOSS"]).sum())
    open_n = int(
        (~ru.str.contains("TARGET|STOP|HOLDING|WIN|LOSS", na=False, regex=True)).sum()
        if len(ru) else 0
    )
    # recount open more carefully
    open_n = len(df) - wins - losses - int(ru.str.contains("HOLDING", na=False).sum())
    open_n = max(0, open_n)
    decided = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "open": open_n,
        "success": round(100.0 * wins / decided, 1) if decided else 0.0,
        "n": len(df),
        "decided": decided,
    }


def show_history():
    """Past Predictions — ALL stocks visible; success rates with Target/Stop + R:R filters."""

    st.title("🕐 PAST PREDICTIONS")
    st.caption(
        "**All predicted stocks** are listed (including R:R &lt; 1.5). "
        "Use filters for Target / Stop / Source / R:R. "
        "Success rate updates from the **filtered** table. Same stock same day → once."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        do_eval = st.button("⟳ Update results", key="hist_eval_btn")
    with c2:
        force_eval = st.button("⚡ Force re-check all", type="primary", key="hist_force_eval")
    with c3:
        if st.button("📥 Import Strategy/Sure into history", type="primary", key="hist_import_strat"):
            with st.spinner("Importing live strategy / sure rows into Past Predictions…"):
                n_imp = import_strategy_files_to_history()
            if n_imp:
                st.success(f"Imported **{n_imp}** quality rows (STRATEGY/SURE). Scroll metrics below.")
                st.rerun()
            else:
                st.warning(
                    "Nothing new to import. Open Strategy Lab → Generate live signals "
                    "(or Sure Call → Generate) first, then click Import again."
                )

    if not st.session_state.get("_auto_learned_hist"):
        try:
            learn_from_history(min_closed=3)
            st.session_state._auto_learned_hist = True
        except Exception:
            pass

    if do_eval or force_eval:
        with st.spinner("Updating prediction outcomes + paper book + learning..."):
            try:
                evaluate_history(force_all=bool(force_eval))
                refresh_history_current_prices(max_stocks=40)
                sync_outcomes_across_books(force_paper=True)
                learn_from_history(min_closed=3)
            except Exception as e:
                st.warning(f"Evaluation note: {e}")
        st.session_state._last_hist_eval = datetime.now()
        st.session_state._auto_learned_hist = True

    history_all = normalize_history_df(load_history())
    if history_all is None or history_all.empty:
        st.info("No prediction history yet. Run FULL MARKET SCAN or import Strategy/Sure.")
        return

    if "Call Source" in history_all.columns:
        vc = history_all["Call Source"].astype(str).str.upper().value_counts().head(8)
        st.caption("History Call Source counts: " + ", ".join(f"{k}:{v}" for k, v in vc.items()))

    # Default: ALL stocks (not quality-only). Dedupe same-day.
    history = dedupe_history_same_day(history_all.copy())
    quality = quality_history_only(history_all)
    try:
        quality = dedupe_history_same_day(quality)
    except Exception:
        pass

    # Attach R:R for filters (safe)
    try:
        history = history.copy()
        if history is None or history.empty:
            st.info("No rows after dedupe.")
            return
        history["R:R"] = history.apply(lambda r: _row_reward_risk(r), axis=1)
    except Exception as e:
        try:
            history["R:R"] = 0.0
        except Exception:
            pass
        st.caption(f"R:R column note: {e}")

    stats_q = overall_statistics()
    stats_all = _success_from_df(history)

    st.subheader("📊 Success rate")
    st.caption(
        "Formula: **Targets ÷ (Targets + Stops)**. "
        "Quality = Sure/Strategy/High-conv. All = every saved prediction including SCAN & weak R:R."
    )
    m1, m2 = st.columns(2)
    with m1:
        succ_q = float(stats_q.get("success") or 0)
        tw_q = int(stats_q.get("wins", 0) or stats_q.get("quality_wins", 0))
        tl_q = int(stats_q.get("losses", 0) or stats_q.get("quality_losses", 0))
        st.markdown(
            f"""
            <div style="border-radius:12px;padding:14px;background:#134e4a;border:1px solid #2dd4bf;">
              <div style="color:#99f6e4;font-size:0.85rem;">QUALITY (Sure / Strategy)</div>
              <div style="color:#f0fdfa;font-size:2rem;font-weight:800;">{succ_q:.1f}%</div>
              <div style="color:#ccfbf1;">🎯 {tw_q} · 🔴 {tl_q}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div style="border-radius:12px;padding:14px;background:#1e3a5f;border:1px solid #38bdf8;">
              <div style="color:#bae6fd;font-size:0.85rem;">ALL STOCKS (incl. SCAN & R:R&lt;1.5)</div>
              <div style="color:#f0fdfa;font-size:2rem;font-weight:800;">{stats_all.get('success', 0):.1f}%</div>
              <div style="color:#e0f2fe;">🎯 {stats_all.get('wins', 0)} · 🔴 {stats_all.get('losses', 0)} · rows {stats_all.get('n', 0)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    a, b, c, d, e, f = st.columns(6)
    a.metric("All unique rows", len(history))
    b.metric("🎯 All targets", stats_all.get("wins", 0))
    c.metric("🔴 All stops", stats_all.get("losses", 0))
    d.metric("Quality success", f"{succ_q}%")
    e.metric("Still open (all)", stats_all.get("open", 0))
    f.metric("Quality rows", stats_q.get("quality_n", 0) or stats_q.get("recommendations", 0))

    # ---- Filters: show ALL by default ----
    st.subheader("🔎 Filters (table + success on filtered set)")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        hist_src = st.selectbox(
            "Source",
            ["ALL stocks", "Strategy / Sure / Precision", "Scan only"],
            index=0,
            key="hist_src_all_v1",
        )
    with f2:
        hist_res = st.selectbox(
            "Result",
            ["ALL", "TARGET ACHIEVED", "STOP LOSS HIT", "HOLDING PERIOD COMPLETED", "OPEN / PENDING"],
            key="hist_res_all_v1",
        )
    with f3:
        hist_rr = st.selectbox(
            "R:R filter",
            ["ALL (incl. R:R < 1.5)", "R:R ≥ 1.5 only", "R:R < 1.5 only", "Invalid / zero R:R"],
            index=0,
            key="hist_rr_v1",
        )
    with f4:
        hist_call = st.selectbox("Call", ["ALL", "BUY", "SELL"], key="hist_call_v1")
    with f5:
        hist_rows = st.selectbox("Rows", [50, 100, 200, 500, "ALL"], index=2, key="hist_rows_v1")

    view = history.copy()
    src_u = view["Call Source"].astype(str).str.upper() if "Call Source" in view.columns else pd.Series([""] * len(view))
    qpat = QUALITY_SOURCE_PATTERN
    if hist_src.startswith("Strategy"):
        view = view[src_u.str.contains(qpat, regex=True, na=False)]
    elif hist_src.startswith("Scan"):
        view = view[~src_u.str.contains(qpat, regex=True, na=False)]

    if hist_call != "ALL" and "Call" in view.columns:
        view = view[view["Call"].astype(str).str.upper().str.contains(hist_call, na=False)]

    ru = view["Result"].astype(str).str.upper() if "Result" in view.columns else pd.Series([""] * len(view))
    if hist_res == "TARGET ACHIEVED":
        view = view[ru.str.contains("TARGET", na=False) | ru.isin(["WIN"])]
    elif hist_res == "STOP LOSS HIT":
        view = view[ru.str.contains("STOP", na=False) | ru.isin(["LOSS"])]
    elif hist_res == "HOLDING PERIOD COMPLETED":
        view = view[ru.str.contains("HOLDING", na=False)]
    elif hist_res == "OPEN / PENDING":
        view = view[
            ~ru.str.contains("TARGET|STOP|HOLDING|WIN|LOSS", na=False, regex=True)
            | ru.str.contains("PENDING", na=False)
        ]

    rr = pd.to_numeric(view.get("R:R"), errors="coerce").fillna(0)
    if hist_rr.startswith("R:R ≥"):
        view = view[rr >= 1.5]
    elif hist_rr.startswith("R:R <"):
        view = view[(rr > 0) & (rr < 1.5)]
    elif hist_rr.startswith("Invalid"):
        view = view[rr <= 0]

    filt_stats = _success_from_df(view)
    st.info(
        f"**Filtered success:** {filt_stats.get('success', 0):.1f}% · "
        f"🎯 {filt_stats.get('wins', 0)} · 🔴 {filt_stats.get('losses', 0)} · "
        f"rows **{len(view)}** (weak R:R included unless you filter them out)"
    )

    show = view
    if hist_rows != "ALL":
        show = view.head(int(hist_rows))

    show_cols = [
        c for c in [
            "Prediction Date", "Stock", "Call", "Call Source", "Entry", "Target", "Stop Loss",
            "R:R", "Current Price", "Result", "Return %", "Exit Date", "Exit Price", "Hold Days",
        ] if c in show.columns
    ]
    st.markdown("##### All predictions (filtered)")
    if show is None or show.empty:
        st.warning("No rows for this filter combination.")
    else:
        try:
            filterable_dataframe(show, key="hist_all_table_v1", default_cols=show_cols, height=380)
        except Exception as e:
            st.warning(f"Table view fallback: {e}")
            try:
                st.dataframe(show[show_cols] if show_cols else show, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(show, use_container_width=True, hide_index=True)

    # Downstream BUY sheet uses full history; metrics above already shown
    history = history_all.copy()
    try:
        history = dedupe_history_same_day(history)
    except Exception:
        pass
    stats = stats_q
    succ = succ_q
    decided = int(stats_q.get("quality_wins") or 0) + int(stats_q.get("quality_losses") or 0)
    tw, tl = tw_q, tl_q

    # Learning panel
    learning = load_learning()
    with st.expander("🧠 Model learning from past mistakes & chart patterns", expanded=True):
        st.caption(
            f"Learning file: `{LEARNING_FILE.name}` · "
            f"Closed samples: {learning.get('sample_closed', 0)} · "
            f"Wins: {learning.get('sample_wins', 0)} · Losses: {learning.get('sample_losses', 0)} · "
            f"Overall decided win rate: {learning.get('overall_win_rate', '—')}%"
        )
        if st.button("🔁 Re-train learning now", key="relearn_btn"):
            with st.spinner("Learning from closed trades..."):
                learning = learn_from_history(min_closed=3)
            st.success("Learning updated — next scan will use new pattern weights.")
            st.rerun()

        lessons = learning.get("lessons") or []
        if lessons:
            st.write("**Lessons the model is applying**")
            for L in lessons[:12]:
                st.markdown(f"- {L}")
        else:
            st.info(
                "Not enough closed target/stop outcomes yet. "
                "Run scans, then **Force re-check all** so the model can learn."
            )

        # Pattern importance table
        st.write("**Chart patterns — meaning & learned importance**")
        pat_rows = []
        stats_p = learning.get("pattern_stats", {})
        mults = learning.get("pattern_multipliers", {})
        for name, meta in PATTERN_IMPORTANCE.items():
            stt = stats_p.get(name, {})
            pat_rows.append({
                "Pattern": name,
                "Bias": meta.get("bias", ""),
                "Base weight": meta.get("base_weight", 0),
                "Learned mult": mults.get(name, 1.0),
                "Hist win %": stt.get("win_rate", "—"),
                "Samples": stt.get("n", 0),
                "Why it matters": meta.get("why", ""),
                "How to use": meta.get("use", ""),
            })
        st.dataframe(pd.DataFrame(pat_rows), use_container_width=True, hide_index=True)

        if learning.get("pred_bucket_stats"):
            st.write("**Prediction strength zones (learned)**")
            st.dataframe(
                pd.DataFrame([
                    {"Zone": k, **v} for k, v in learning["pred_bucket_stats"].items()
                ]),
                use_container_width=True,
                hide_index=True,
            )

        manual_feedback_trainer()

    if history is None or history.empty:
        st.info("No prediction history yet. Run **FULL MARKET SCAN** first.")
        return

    hist = history.copy()
    if "Call Source" not in hist.columns:
        hist["Call Source"] = "SCAN"
    else:
        hist["Call Source"] = hist["Call Source"].fillna("SCAN").astype(str)
    if "Strategy" not in hist.columns:
        hist["Strategy"] = ""
    if "Patterns" not in hist.columns:
        hist["Patterns"] = ""

    hist["_pred_dt"] = pd.to_datetime(hist.get("Prediction Date"), errors="coerce")

    # ============================================================
    # ALL BUY CALLS — your own backtest sheet
    # ============================================================
    st.subheader("📗 All BUY calls (backtest sheet)")
    st.caption(
        "Every historical **BUY** in `recommendation_history.csv`. "
        "Daily **FULL MARKET SCAN** saves source as **SCAN**. "
        "Sure / Strategy / Precision only appear if you saved them from those pages. "
        "Default view = **ALL sources** so your daily BUYs always show."
    )

    # Normalize Call + Call Source so older rows still match
    hist["Call"] = hist["Call"].astype(str).str.upper().str.strip()
    hist["Call Source"] = (
        hist["Call Source"].astype(str).str.upper().str.strip()
        .replace({"NAN": "SCAN", "NONE": "SCAN", "": "SCAN"})
    )

    buy_all = hist[hist["Call"].str.contains("BUY", na=False)].copy()

    # Quick diagnostic so empty filter is explained
    n_hist = len(hist)
    n_buy_total = len(buy_all)
    src_counts = (
        buy_all["Call Source"].value_counts().head(12)
        if n_buy_total else pd.Series(dtype=int)
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("History rows (all calls)", n_hist)
    d2.metric("BUY rows (all sources)", n_buy_total)
    if n_buy_total:
        top_src = ", ".join(f"{k}:{v}" for k, v in src_counts.items())
        d3.caption(f"**BUY by source:** {top_src}")
    else:
        d3.warning("No BUY in CSV yet — run FULL MARKET SCAN once.")

    bt1, bt2, bt3, bt4 = st.columns(4)
    with bt1:
        # Default index 0 = ALL sources (was Strategy-only which hid SCAN)
        bt_src = st.selectbox(
            "BUY source",
            [
                "Strategy / Sure / Precision only",
                "ALL sources",
                "Scan only (daily market scan)",
            ],
            index=1,
            key="buy_bt_src_v3",
            help="Priority: Sure/Strategy first when present. SCAN = full market scan bulk BUYs.",
        )
    with bt2:
        bt_result = st.selectbox(
            "BUY result",
            [
                "ALL",
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "HOLDING PERIOD COMPLETED",
                "OPEN / PENDING",
            ],
            key="buy_bt_result_v2",
        )
    with bt3:
        bt_date = st.selectbox(
            "BUY date range",
            ["All dates", "Last 7 days", "Last 30 days", "Last 90 days", "Today"],
            key="buy_bt_date_v2",
        )
    with bt4:
        bt_show = st.selectbox(
            "Rows to show",
            [50, 100, 200, 500, 1000, "ALL"],
            index=2,
            key="buy_bt_rows_v2",
        )

    buy_view = buy_all.copy()
    src_u0 = buy_view["Call Source"].astype(str).str.upper()
    quality_pat = "STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV"

    if bt_src.startswith("Scan only"):
        buy_view = buy_view[~src_u0.str.contains(quality_pat, regex=True, na=False)]
    elif bt_src.startswith("Strategy"):
        buy_view = buy_view[src_u0.str.contains(quality_pat, regex=True, na=False)]
        if buy_view.empty and n_buy_total > 0:
            st.warning(
                f"You have **{n_buy_total} BUY** rows, but **none** marked Strategy/Sure/Precision. "
                "Daily calls from **FULL MARKET SCAN** are saved as **SCAN**. "
                "Switch **BUY source → ALL sources** or **Scan only** to see them."
            )

    today = datetime.now().date()
    if bt_date == "Today":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date == today]
    elif bt_date == "Last 7 days":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date >= (today - timedelta(days=7))]
    elif bt_date == "Last 30 days":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date >= (today - timedelta(days=30))]
    elif bt_date == "Last 90 days":
        buy_view = buy_view[buy_view["_pred_dt"].dt.date >= (today - timedelta(days=90))]

    res_u0 = buy_view["Result"].astype(str).str.upper() if "Result" in buy_view.columns else pd.Series([""] * len(buy_view), index=buy_view.index)
    if bt_result == "TARGET ACHIEVED":
        buy_view = buy_view[res_u0.str.contains("TARGET ACHIEVED|WIN", na=False)]
    elif bt_result == "STOP LOSS HIT":
        buy_view = buy_view[res_u0.str.contains("STOP LOSS", na=False)]
    elif bt_result == "HOLDING PERIOD COMPLETED":
        buy_view = buy_view[res_u0.str.contains("HOLDING PERIOD", na=False)]
    elif bt_result == "OPEN / PENDING":
        buy_view = buy_view[
            ~res_u0.str.contains("TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD", na=False, regex=True)
        ]

    # One stock name per calendar day (CRESTO STRATEGY + SURE same day → 1 row)
    if not buy_view.empty:
        _before = len(buy_view)
        buy_view = dedupe_history_same_day(buy_view, one_stock_per_day=True)
        _after = len(buy_view)
        if _before > _after:
            st.caption(f"Unique stocks today/period: removed **{_before - _after}** same-day duplicates.")

    # BUY-only stats for this filtered sheet
    if not buy_view.empty:
        ru = buy_view["Result"].astype(str).str.upper()
        n_buy = len(buy_view)
        n_tgt = int(ru.str.contains("TARGET ACHIEVED", na=False).sum())
        n_sl = int(ru.str.contains("STOP LOSS HIT", na=False).sum())
        n_time = int(ru.str.contains("HOLDING PERIOD", na=False).sum())
        n_open = n_buy - n_tgt - n_sl - n_time
        closed = n_tgt + n_sl
        win_rate = round(100.0 * n_tgt / closed, 1) if closed else 0.0
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("BUY rows", n_buy)
        m2.metric("🎯 Target hit", n_tgt)
        m3.metric("🔴 Stop hit", n_sl)
        m4.metric("⏰ Time exit", n_time)
        m5.metric("⏳ Open", max(n_open, 0))
        m6.metric("Win rate (closed)", f"{win_rate}%")
    else:
        st.info("No BUY rows for this filter. Run scans / Sure / Strategy so BUYs are saved to history.")

    buy_cols = [
        c for c in [
            "Prediction Date",
            "Stock",
            "Call Source",
            "Strategy",
            "Call",
            "Prediction",
            "Entry",
            "Current Price",
            "Target",
            "Stop Loss",
            "Risk %",
            "Hold Days",
            "Days Taken",
            "Result",
            "Return %",
            "Exit Price",
            "Evaluation Date",
            "Outcome Message",
            "Recommendation",
            "Patterns",
            "Reason",
        ]
        if c in buy_view.columns
    ]

    if not buy_view.empty:
        buy_view = buy_view.sort_values("Prediction Date", ascending=False)
        if bt_show != "ALL":
            display_buy = buy_view.head(int(bt_show))
        else:
            display_buy = buy_view
        st.dataframe(
            display_buy[buy_cols],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Showing {len(display_buy):,} of {len(buy_view):,} BUY calls. "
            "Entry / Target / Stop are locked; Current Price updates on re-check."
        )
        # CSV download for your own backtest
        try:
            csv_bytes = buy_view[buy_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download all filtered BUY calls (CSV)",
                data=csv_bytes,
                file_name=f"buy_calls_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_buy_bt",
            )
        except Exception:
            pass
        if st.button("⚡ Force re-check outcomes on these BUYs", key="force_buy_bt"):
            with st.spinner("Re-checking target / stop on BUY history..."):
                try:
                    evaluate_history(force_all=True)
                    refresh_history_current_prices(max_stocks=50)
                except Exception as e:
                    st.warning(str(e))
            st.success("Updated — scroll metrics above for new target/stop counts.")
            st.rerun()

    st.divider()

    # --- Filters ---
    st.subheader("Prediction History (all call types)")
    page_tab = st.radio(
        "History page",
        [
            "Strategy + Sure + Precision only",
            "All sources",
            "Scan / general calls",
        ],
        horizontal=True,
        index=0,
        key="hist_page_src_v2",
        help="Default = quality calls first (Sure/Strategy). Generate+save them from Sure Call / Strategy Lab.",
    )

    br = history_success_breakdown(hist)
    if br is not None and not br.empty:
        with st.expander("📊 Success rate by Call Source / Strategy / Pattern", expanded=True):
            st.dataframe(br, use_container_width=True, hide_index=True)
            # Strategy-only success table
            st.write("**Strategy success rate**")
            sb = br[br["Filter"] == "Strategy"] if "Filter" in br.columns else br
            if sb is not None and not sb.empty:
                st.dataframe(sb, use_container_width=True, hide_index=True)

    # ============================================================
    # STRATEGY-WISE + HIGH CONVICTION PAST PREDICTIONS
    # ============================================================
    st.subheader("⭐ Strategy-wise & High Conviction past predictions")
    st.caption(
        "Filter by **source** (SURE / STRATEGY / HIGH_CONV) and by **strategy name**. "
        "Generate from Strategy Lab (Live BUY / High conviction) or Sure Call Desk so rows appear here."
    )
    src_u = hist["Call Source"].astype(str).str.upper()
    if "Strategy" not in hist.columns:
        hist["Strategy"] = ""
    rec = hist[src_u.str.contains("STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV", regex=True, na=False)].copy()
    _prio = {"SURE": 0, "HIGH_CONV": 1, "STRATEGY": 2, "MY_STRATEGY": 3, "PRECISION": 4}
    if not rec.empty:
        rec = rec.copy()
        rec["_prio"] = rec["Call Source"].astype(str).str.upper().map(lambda s: _prio.get(s, 9))
        rec = rec.sort_values(["_prio", "Prediction Date"], ascending=[True, False])

    # View mode
    view_mode = st.radio(
        "View",
        [
            "All quality (Sure + Strategy + High conv)",
            "High conviction only",
            "Strategy-wise table",
            "One strategy deep-dive",
        ],
        horizontal=True,
        key="hist_strat_view",
    )

    sf1, sf2, sf3 = st.columns(3)
    with sf1:
        src_opts = sorted(rec["Call Source"].astype(str).unique().tolist()) if len(rec) else [
            "SURE", "STRATEGY", "HIGH_CONV", "PRECISION", "MY_STRATEGY"
        ]
        src_pick = st.multiselect(
            "Call Source filter",
            options=src_opts,
            default=src_opts,
            key="hist_rec_src_v3",
        )
    with sf2:
        strat_names = sorted(
            [x for x in hist["Strategy"].astype(str).unique().tolist() if x and str(x).lower() not in ("nan", "none", "")]
        )
        # Also strategy names from SWING_STRATEGIES for empty history
        try:
            for _meta in SWING_STRATEGIES.values():
                n = str(_meta.get("name", ""))
                if n and n not in strat_names:
                    strat_names.append(n)
            strat_names = sorted(set(strat_names))
        except Exception:
            pass
        strat_pick = st.selectbox(
            "Strategy name filter",
            ["ALL strategies"] + strat_names,
            key="hist_one_strat",
        )
    with sf3:
        strat_result = st.selectbox(
            "Result (strategy view)",
            ["ALL", "TARGET ACHIEVED", "STOP LOSS HIT", "OPEN / PENDING"],
            key="hist_strat_res",
        )

    rec_f = rec.copy() if rec is not None else pd.DataFrame()
    if src_pick and not rec_f.empty:
        rec_f = rec_f[rec_f["Call Source"].astype(str).isin(src_pick)]
    if view_mode.startswith("High conviction"):
        rec_f = rec_f[rec_f["Call Source"].astype(str).str.upper() == "HIGH_CONV"] if not rec_f.empty else rec_f
    if strat_pick != "ALL strategies" and not rec_f.empty:
        rec_f = rec_f[
            rec_f["Strategy"].astype(str).str.contains(strat_pick, case=False, na=False)
            | (rec_f["Strategy"].astype(str) == strat_pick)
        ]
    if strat_result != "ALL" and not rec_f.empty and "Result" in rec_f.columns:
        ru = rec_f["Result"].astype(str).str.upper()
        if strat_result == "OPEN / PENDING":
            rec_f = rec_f[~ru.str.contains("TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD", na=False)]
        else:
            rec_f = rec_f[ru.str.contains(strat_result, na=False)]

    if rec_f is not None and not rec_f.empty:
        ru = rec_f["Result"].astype(str).str.upper()
        n_t = int(ru.str.contains("TARGET ACHIEVED", na=False).sum())
        n_s = int(ru.str.contains("STOP LOSS HIT", na=False).sum())
        closed = n_t + n_s
        wr = round(100.0 * n_t / closed, 1) if closed else 0.0
        a, b, c, d = st.columns(4)
        a.metric("Rows (filtered)", len(rec_f))
        b.metric("🎯 Targets", n_t)
        c.metric("🔴 Stops", n_s)
        d.metric("Win rate", f"{wr}%")

        # Strategy-wise success table
        if view_mode.startswith("Strategy-wise") or view_mode.startswith("All quality"):
            st.markdown("**📈 Success % by strategy name**")
            rows_bt = []
            for sname, g in rec_f.groupby(rec_f["Strategy"].astype(str)):
                if not sname or sname.lower() in ("nan", "none"):
                    sname = "(blank strategy)"
                gru = g["Result"].astype(str).str.upper()
                wt = int(gru.str.contains("TARGET ACHIEVED", na=False).sum())
                ls = int(gru.str.contains("STOP LOSS HIT", na=False).sum())
                cl = wt + ls
                rows_bt.append({
                    "Strategy": sname,
                    "Calls": len(g),
                    "Targets": wt,
                    "Stops": ls,
                    "Open/Other": len(g) - cl,
                    "Win % (closed)": round(100.0 * wt / cl, 1) if cl else 0.0,
                    "Sources": ", ".join(sorted(g["Call Source"].astype(str).unique().tolist())[:5]),
                })
            if rows_bt:
                st.dataframe(
                    pd.DataFrame(rows_bt).sort_values("Win % (closed)", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

        # By Call Source
        st.markdown("**Success % by Call Source**")
        rows_src = []
        for sname, g in rec_f.groupby(rec_f["Call Source"].astype(str)):
            gru = g["Result"].astype(str).str.upper()
            wt = int(gru.str.contains("TARGET ACHIEVED", na=False).sum())
            ls = int(gru.str.contains("STOP LOSS HIT", na=False).sum())
            cl = wt + ls
            rows_src.append({
                "Call Source": sname,
                "Calls": len(g),
                "Targets": wt,
                "Stops": ls,
                "Win %": round(100.0 * wt / cl, 1) if cl else 0.0,
            })
        if rows_src:
            st.dataframe(pd.DataFrame(rows_src), use_container_width=True, hide_index=True)

        show_cols = [c for c in [
            "Prediction Date", "Stock", "Call Source", "Strategy", "Call",
            "Entry", "Target", "Stop Loss", "Current Price", "Result", "Return %", "Days Taken", "Patterns",
        ] if c in rec_f.columns]
        st.markdown("**Detail rows — filterable table**")
        filterable_dataframe(rec_f[show_cols].head(200), key="hist_qual_table", default_cols=show_cols, height=420)
        st.markdown("**Quality call cards**")
        n_hc = st.slider("History cards", 3, 15, 6, key="hist_cards_n")
        for _, hr in rec_f.head(n_hc).iterrows():
            render_call_stock_card(hr.to_dict(), section_key="hist")
        try:
            st.download_button(
                "⬇️ Download filtered strategy / high-conv CSV",
                data=rec_f[show_cols].to_csv(index=False).encode("utf-8"),
                file_name=f"strategy_past_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_strat_hist_v2",
            )
        except Exception:
            pass
    else:
        st.warning(
            "No strategy / Sure / high-conviction rows in history yet.\n\n"
            "1. **Strategy Lab → 📡 Live BUY by strategy → Generate live signals**\n"
            "2. **Strategy Lab → ⭐ High conviction → Find high conviction**\n"
            "3. **Sure Call Desk → Generate sure calls**\n\n"
            "Wait for “Saved N…” then reopen Past Predictions."
        )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        call_filter = st.selectbox("Call", ["ALL", "BUY", "HOLD", "SELL"], key="hist_call")
    with f2:
        result_filter = st.selectbox(
            "Result",
            [
                "ALL",
                "TARGET ACHIEVED",
                "STOP LOSS HIT",
                "HOLDING PERIOD COMPLETED",
                "OPEN / PENDING",
            ],
            key="hist_result",
        )
    with f3:
        date_preset = st.selectbox(
            "Date range",
            ["All dates", "Last 7 days", "Last 30 days", "Today"],
            key="hist_date_preset",
        )
    with f4:
        strat_opts = ["ALL"] + sorted(
            [x for x in hist["Strategy"].astype(str).unique().tolist() if x and x != "nan"]
        )[:40]
        strat_filter = st.selectbox("Strategy", strat_opts, key="hist_strat_f")

    h = hist.copy()
    today = datetime.now().date()
    if date_preset == "Today":
        h = h[h["_pred_dt"].dt.date == today]
    elif date_preset == "Last 7 days":
        h = h[h["_pred_dt"].dt.date >= (today - timedelta(days=7))]
    elif date_preset == "Last 30 days":
        h = h[h["_pred_dt"].dt.date >= (today - timedelta(days=30))]

    # Source page split
    src = h["Call Source"].astype(str).str.upper()
    if page_tab.startswith("Strategy"):
        h = h[src.str.contains("STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV", regex=True, na=False)]
    elif page_tab.startswith("Scan"):
        h = h[~src.str.contains("STRATEGY|SURE|PRECISION|MY_STRATEGY|HIGH_CONV", regex=True, na=False)]

    if call_filter != "ALL" and "Call" in h.columns:
        h = h[h["Call"].astype(str).str.upper().str.strip() == call_filter]

    if strat_filter != "ALL" and "Strategy" in h.columns:
        h = h[h["Strategy"].astype(str) == strat_filter]

    if "Result" in h.columns:
        res_u = h["Result"].astype(str).str.upper().str.strip()
        if result_filter == "OPEN / PENDING":
            h = h[~res_u.str.contains(
                "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
                regex=True, na=True,
            )]
        elif result_filter == "TARGET ACHIEVED":
            h = h[res_u.str.contains("TARGET ACHIEVED", na=False)]
        elif result_filter == "STOP LOSS HIT":
            h = h[res_u.str.contains("STOP LOSS HIT", na=False)]
        elif result_filter == "HOLDING PERIOD COMPLETED":
            h = h[res_u.str.contains("HOLDING PERIOD COMPLETED", na=False)]

    # Days completed / remaining
    now_ts = pd.Timestamp(today)

    def _days_pair(pred_dt, hold_days):
        hold = safe_int(hold_days, DEFAULT_HOLD_DAYS)
        if hold <= 0:
            hold = DEFAULT_HOLD_DAYS
        try:
            if pred_dt is None or pd.isna(pred_dt):
                return 0, hold
            start = pd.Timestamp(pred_dt).normalize()
            completed = max(0, safe_int((now_ts - start).days, 0))
            return completed, max(0, hold - completed)
        except Exception:
            return 0, hold

    if len(h):
        pairs = [_days_pair(r.get("_pred_dt"), r.get("Hold Days")) for _, r in h.iterrows()]
        h = h.copy()
        h["Days Completed"] = [p[0] for p in pairs]
        h["Days Remaining"] = [p[1] for p in pairs]
        if "Prediction Date" in h.columns:
            h = h.sort_values("Prediction Date", ascending=False)

    st.write(f"Showing **{len(h)}** rows")

    display_cols = [
        c for c in [
            "Prediction Date", "Stock", "Call Source", "Strategy", "Call",
            "Entry", "Current Price", "Target", "Stop Loss",
            "Hold Days", "Days Completed", "Days Remaining",
            "Status", "Result", "Exit Price", "Return %", "Days Taken",
            "Patterns", "Recommendation",
        ] if c in h.columns
    ]
    st.markdown("##### Filterable prediction table")
    filterable_dataframe(h, key="hist_main_table", default_cols=display_cols, height=420)

    st.subheader("📋 Visual cards (mcap · BV · desk)")
    n_vis = st.slider("Visual cards", 3, 12, 5, key="hist_vis_n")
    for _, row in h.head(n_vis).iterrows():
        render_call_stock_card(row.to_dict(), section_key="hist_main")

    # --- Detail cards (top 25 of filtered) ---
    st.subheader("📋 Detailed outcomes")
    for _, row in h.head(25).iterrows():
        stock = str(row.get("Stock", ""))
        call = str(row.get("Call", "")).upper()
        entry = safe_float(row.get("Entry"))
        target = safe_float(row.get("Target"))
        stop = safe_float(row.get("Stop Loss"))
        cur = safe_float(row.get("Current Price"))
        hold_days = safe_int(row.get("Hold Days"), DEFAULT_HOLD_DAYS) or DEFAULT_HOLD_DAYS
        result = str(row.get("Result", "PENDING")).upper()
        result_detail = str(row.get("Result Detail", "") or "")
        recommendation = str(row.get("Recommendation", "") or "")
        return_pct = safe_float(row.get("Return %"))
        exit_price = safe_float(row.get("Exit Price"))
        days_taken = row.get("Days Taken", "")
        eval_date = row.get("Evaluation Date", "")
        days_completed = safe_int(row.get("Days Completed"), 0)
        days_remaining = safe_int(row.get("Days Remaining"), hold_days)

        try:
            pred_date_fmt = pd.to_datetime(row.get("Prediction Date")).strftime("%d %B %Y")
        except Exception:
            pred_date_fmt = str(row.get("Prediction Date", ""))

        # Always compute P&L for display
        pnl_pct = return_pct if return_pct else _pnl_from_row(row)
        if abs(pnl_pct) < 1e-9 and entry > 0:
            if "TARGET ACHIEVED" in result and target > 0:
                pnl_pct = ((target - entry) / entry * 100) if "SELL" not in call else ((entry - target) / entry * 100)
            elif "STOP LOSS" in result and stop > 0:
                pnl_pct = ((stop - entry) / entry * 100) if "SELL" not in call else ((entry - stop) / entry * 100)
        profit_rupees = entry * pnl_pct / 100.0 if entry else 0.0

        if "TARGET ACHIEVED" in result:
            badge, box = "🎯 TARGET ACHIEVED", "success-box"
            pnl_line = f"<b style='color:#16a34a;'>Profit: {pnl_pct:+.2f}% (≈ ₹{profit_rupees:+,.2f} per share)</b>"
        elif "STOP LOSS HIT" in result:
            badge, box = "🔴 STOP LOSS HIT", "danger-box"
            pnl_line = f"<b style='color:#dc2626;'>Loss: {pnl_pct:+.2f}% (≈ ₹{profit_rupees:+,.2f} per share)</b>"
        elif "HOLDING PERIOD" in result:
            badge, box = "⏰ HOLDING PERIOD COMPLETED", "watch-box"
            pnl_line = f"<b>P&L at exit: {pnl_pct:+.2f}% (≈ ₹{profit_rupees:+,.2f} / share)</b>"
        else:
            badge, box = "⏳ OPEN / PENDING", "hold-box"
            unreal = 0.0
            if entry > 0 and cur > 0:
                unreal = ((cur - entry) / entry * 100) if "SELL" not in call else ((entry - cur) / entry * 100)
            pnl_line = f"<b>Unrealized: {unreal:+.2f}%</b>"

        if not result_detail or result_detail in ("", "nan", "None"):
            if "TARGET ACHIEVED" in result:
                result_detail = (
                    f"🎯 TARGET ACHIEVED\n"
                    f"Target reached: ₹{target:,.2f}\n"
                    f"Days Taken: {days_taken}\n"
                    f"Profit: {pnl_pct:+.2f}%\n"
                    f"Profit (₹/share): {profit_rupees:+,.2f}"
                )
            elif "STOP LOSS HIT" in result:
                result_detail = (
                    f"🔴 STOP LOSS HIT\n"
                    f"Stop hit: ₹{stop:,.2f}\n"
                    f"Days Taken: {days_taken}\n"
                    f"Loss: {pnl_pct:+.2f}%\n"
                    f"Loss (₹/share): {profit_rupees:+,.2f}"
                )
            elif "HOLDING PERIOD" in result:
                result_detail = (
                    f"⏰ HOLDING PERIOD COMPLETED\nExit: ₹{exit_price:,.2f}\n"
                    f"Date: {eval_date}\nP&L: {pnl_pct:+.2f}%"
                )
            else:
                cur_txt = f"₹{cur:,.2f}" if cur else "—"
                result_detail = (
                    f"Still open\nCurrent: {cur_txt}\n"
                    f"Waiting for Target ₹{target:,.2f} or Stop ₹{stop:,.2f}"
                )

        detail_html = result_detail.replace("\n", "<br>")
        rec_html = recommendation.replace("\n", "<br>") if recommendation else ""
        cur_line = f"<b>Current Price:</b> ₹{cur:,.2f}<br>" if cur else ""

        st.markdown(
            f"""
            <div class="{box}" style="margin-bottom:12px;padding:14px;border-radius:10px;
                 transition: transform 0.2s ease, box-shadow 0.2s ease;">
            <h4 style="margin:0 0 8px 0;">{stock} — {call} &nbsp; {badge}</h4>
            <p style="margin:0 0 8px 0;">{pnl_line}</p>
            <p style="margin:0;line-height:1.6;">
            <b>Prediction Date:</b> {pred_date_fmt}<br>
            <b>Entry:</b> ₹{entry:,.2f}<br>
            {cur_line}
            <b>Target (locked):</b> ₹{target:,.2f}<br>
            <b>Stop Loss (locked):</b> ₹{stop:,.2f}<br>
            <b>Holding:</b> {hold_days} days |
            <b>Completed:</b> {days_completed} |
            <b>Remaining:</b> {days_remaining}<br><br>
            <b>RESULT:</b><br>{detail_html}
            {f'<br><br><b>Recommendation:</b><br>{rec_html}' if rec_html else ''}
            </p></div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HOLDING ADVISOR — user stock + buy price + shares
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    """Company fundamentals from Yahoo Finance (works for most NSE .NS symbols)."""
    out = {
        "name": "",
        "sector": "",
        "industry": "",
        "summary": "",
        "market_cap": None,
        "pe": None,
        "forward_pe": None,
        "pb": None,
        "ps": None,
        "eps": None,
        "book_value": None,
        "dividend_yield": None,
        "roe": None,
        "roa": None,
        "profit_margins": None,
        "operating_margins": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "target_mean": None,
        "recommendation": "",
        "fifty_two_high": None,
        "fifty_two_low": None,
        "beta": None,
        "employees": None,
        "currency": "INR",
        "error": "",
    }
    try:
        t = yf.Ticker(clean_symbol(symbol))
        info = {}
        try:
            info = t.info or {}
        except Exception:
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}

        def g(*keys):
            for k in keys:
                v = info.get(k)
                if v is not None and v == v:  # not NaN
                    return v
            return None

        out["name"] = g("longName", "shortName") or display_symbol(symbol)
        out["sector"] = g("sector") or ""
        out["industry"] = g("industry") or ""
        out["summary"] = (g("longBusinessSummary") or "")[:900]
        out["market_cap"] = g("marketCap")
        out["pe"] = g("trailingPE", "peRatio")
        out["forward_pe"] = g("forwardPE")
        out["pb"] = g("priceToBook")
        out["ps"] = g("priceToSalesTrailing12Months")
        out["eps"] = g("trailingEps")
        out["book_value"] = g("bookValue")
        out["face_value"] = g("faceValue")
        # Sales / revenue (absolute INR — convert to Cr in UI)
        out["total_revenue"] = g("totalRevenue", "revenue")
        out["revenue_latest"] = out.get("revenue_latest") or out["total_revenue"]
        dy = g("dividendYield")
        out["dividend_yield"] = (dy * 100 if dy is not None and dy < 1 else dy)
        out["roe"] = g("returnOnEquity")
        out["roa"] = g("returnOnAssets")
        out["profit_margins"] = g("profitMargins")
        out["operating_margins"] = g("operatingMargins")
        out["revenue_growth"] = g("revenueGrowth")
        out["earnings_growth"] = g("earningsGrowth")
        out["debt_to_equity"] = g("debtToEquity")
        out["current_ratio"] = g("currentRatio")
        out["target_mean"] = g("targetMeanPrice")
        out["recommendation"] = g("recommendationKey") or g("recommendationMean") or ""
        out["fifty_two_high"] = g("fiftyTwoWeekHigh")
        out["fifty_two_low"] = g("fiftyTwoWeekLow")
        out["beta"] = g("beta")
        out["employees"] = g("fullTimeEmployees")
        out["currency"] = g("currency") or "INR"

        # Simple financial statement snapshot (yearly)
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                out["financials_cols"] = [str(c)[:10] for c in list(fin.columns)[:4]]
                for label, keys in [
                    ("revenue", ["Total Revenue", "Operating Revenue"]),
                    ("net_income", ["Net Income", "Net Income Common Stockholders"]),
                ]:
                    for k in keys:
                        if k in fin.index:
                            series = fin.loc[k]
                            out[f"{label}_latest"] = safe_float(series.iloc[0]) if len(series) else None
                            # Year-wise in crore for dashboard
                            yearly = []
                            for col in list(series.index)[:5]:
                                yearly.append({
                                    "Year": str(col)[:10],
                                    f"{label}_cr": in_crore(series[col]),
                                })
                            out[f"{label}_yearly"] = yearly
                            break
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fundamental_yearly_table(symbol: str) -> pd.DataFrame:
    """Revenue / profit year-wise in ₹ Cr for dashboards."""
    try:
        fund = fetch_fundamentals(symbol)
    except Exception:
        return pd.DataFrame()
    rows = {}
    for item in fund.get("revenue_yearly") or []:
        y = item.get("Year")
        rows.setdefault(y, {"Year": y})
        rows[y]["Revenue (₹ Cr)"] = item.get("revenue_cr")
    for item in fund.get("net_income_yearly") or []:
        y = item.get("Year")
        rows.setdefault(y, {"Year": y})
        rows[y]["Profit (₹ Cr)"] = item.get("net_income_cr")
    if not rows:
        # single latest
        if fund.get("revenue_latest") or fund.get("net_income_latest"):
            return pd.DataFrame([{
                "Year": "Latest",
                "Revenue (₹ Cr)": in_crore(fund.get("revenue_latest")),
                "Profit (₹ Cr)": in_crore(fund.get("net_income_latest")),
            }])
        return pd.DataFrame()
    return pd.DataFrame(list(rows.values())).sort_values("Year", ascending=False)


def _fmt_num(v, pct=False, money=False):
    if v is None:
        return "—"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if pct:
        # Yahoo often gives 0.15 for 15%
        if abs(v) <= 1.5:
            v = v * 100
        return f"{v:.2f}%"
    if money:
        if abs(v) >= 1e7:
            return f"₹{v/1e7:.2f} Cr"
        if abs(v) >= 1e5:
            return f"₹{v/1e5:.2f} L"
        return f"₹{v:,.2f}"
    return f"{v:,.2f}"


def holding_horizon_advice(result, buy_price, shares, fund: dict) -> dict:
    """
    Short-term (days–weeks) and long-term (months+) advice
    for an existing position.
    """
    price = safe_float(result.get("Price"))
    call = str(result.get("Call", "")).upper()
    pred = safe_float(result.get("Prediction"))
    rsi = safe_float(result.get("RSI"))
    adx = safe_float(result.get("ADX"))
    target = safe_float(result.get("Target"))
    stop = safe_float(result.get("Stop Loss"))
    risk_pct = safe_float(result.get("Risk %"))
    patterns = str(result.get("Patterns", ""))

    pnl = ((price - buy_price) / buy_price * 100) if buy_price > 0 else 0
    invested = buy_price * shares
    mkt_value = price * shares
    pnl_rs = mkt_value - invested

    data = result.get("Data")
    above_ema50 = above_ema200 = None
    if data is not None and not getattr(data, "empty", True):
        row = data.iloc[-1]
        e50 = safe_float(row.get("EMA50"))
        e200 = safe_float(row.get("EMA200"))
        if e50:
            above_ema50 = price > e50
        if e200:
            above_ema200 = price > e200

    # Short-term score
    st_score = 0
    st_notes = []
    if call == "BUY":
        st_score += 2
        st_notes.append("Model short-term signal is BUY.")
    elif call == "HOLD":
        st_score += 1
        st_notes.append("Model short-term signal is HOLD.")
    elif call == "SELL":
        st_score -= 2
        st_notes.append("Model short-term signal is SELL.")
    if pred >= 70:
        st_score += 1
        st_notes.append(f"Prediction strength {pred:.0f}% is solid.")
    elif pred < 55:
        st_score -= 1
        st_notes.append(f"Prediction strength {pred:.0f}% is weak.")
    if 50 <= rsi <= 70:
        st_score += 1
        st_notes.append(f"RSI {rsi:.0f} is in a healthy bullish band.")
    elif rsi > 78:
        st_score -= 1
        st_notes.append(f"RSI {rsi:.0f} is overbought — short-term pullback risk.")
    elif rsi < 35:
        st_notes.append(f"RSI {rsi:.0f} is weak/oversold — bounce possible but trend may be soft.")
    if above_ema50 is True:
        st_score += 1
        st_notes.append("Price is above EMA50 (short/medium trend supportive).")
    elif above_ema50 is False:
        st_score -= 1
        st_notes.append("Price is below EMA50 (short/medium trend weak).")
    if "Bearish" in patterns or "Shooting Star" in patterns or "Lower High" in patterns:
        st_score -= 1
        st_notes.append(f"Caution pattern: {patterns}.")
    if target > 0 and price >= target * 0.98:
        st_score -= 1
        st_notes.append("Price is near/above model target — consider booking some profit short-term.")
    if stop > 0 and price <= stop * 1.02:
        st_score -= 2
        st_notes.append("Price is near model stop — short-term risk of further downside.")

    if st_score >= 3:
        st_action = "BUY MORE (short-term) / HOLD with confidence"
        st_color = "buy"
    elif st_score >= 1:
        st_action = "HOLD (short-term) — trail stop, avoid adding aggressively"
        st_color = "hold"
    elif st_score >= -1:
        st_action = "HOLD lightly or reduce — wait for clearer setup"
        st_color = "hold"
    else:
        st_action = "SELL / EXIT (short-term) — risk outweighs reward"
        st_color = "sell"

    # Long-term score
    lt_score = 0
    lt_notes = []
    if above_ema200 is True:
        lt_score += 2
        lt_notes.append("Price above EMA200 — long-term uptrend intact.")
    elif above_ema200 is False:
        lt_score -= 2
        lt_notes.append("Price below EMA200 — long-term structure is weak.")
    if above_ema50 is True:
        lt_score += 1
    pe = fund.get("pe")
    if pe is not None:
        if 0 < pe < 25:
            lt_score += 1
            lt_notes.append(f"Trailing P/E {pe:.1f} is not stretched.")
        elif pe >= 40:
            lt_score -= 1
            lt_notes.append(f"Trailing P/E {pe:.1f} is expensive — growth must justify it.")
    roe = fund.get("roe")
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) <= 1.5 else roe
        if roe_pct >= 15:
            lt_score += 1
            lt_notes.append(f"ROE ~{roe_pct:.1f}% supports long-term quality.")
        elif roe_pct < 8:
            lt_score -= 1
            lt_notes.append(f"ROE ~{roe_pct:.1f}% is modest.")
    pm = fund.get("profit_margins")
    if pm is not None:
        pm_pct = pm * 100 if abs(pm) <= 1.5 else pm
        if pm_pct >= 10:
            lt_score += 1
            lt_notes.append(f"Profit margin ~{pm_pct:.1f}% looks healthy.")
    de = fund.get("debt_to_equity")
    if de is not None:
        # Yahoo often scales D/E * 100
        de_v = de / 100 if de > 20 else de
        if de_v > 2:
            lt_score -= 1
            lt_notes.append(f"Debt/Equity looks elevated ({de}).")
        elif de_v < 1:
            lt_score += 1
            lt_notes.append("Balance sheet leverage looks controlled.")
    rg = fund.get("revenue_growth")
    if rg is not None:
        rg_pct = rg * 100 if abs(rg) <= 1.5 else rg
        if rg_pct > 10:
            lt_score += 1
            lt_notes.append(f"Revenue growth ~{rg_pct:.1f}% is supportive.")
        elif rg_pct < 0:
            lt_score -= 1
            lt_notes.append(f"Revenue growth ~{rg_pct:.1f}% is negative.")
    if call == "SELL" and above_ema200 is False:
        lt_score -= 1
        lt_notes.append("Technical SELL + below EMA200 — long-term accumulation is riskier now.")
    if call in ["BUY", "HOLD"] and above_ema200 is True:
        lt_score += 1
        lt_notes.append("Technical bias aligns with long-term uptrend.")

    if lt_score >= 3:
        lt_action = "HOLD / ACCUMULATE (long-term) on dips"
        lt_color = "buy"
    elif lt_score >= 1:
        lt_action = "HOLD (long-term) — review quarterly results"
        lt_color = "hold"
    elif lt_score >= -1:
        lt_action = "HOLD only with strict review — or reduce overweight"
        lt_color = "hold"
    else:
        lt_action = "Prefer EXIT / avoid fresh long-term capital"
        lt_color = "sell"

    return {
        "pnl_pct": pnl,
        "pnl_rs": pnl_rs,
        "invested": invested,
        "mkt_value": mkt_value,
        "st_action": st_action,
        "st_color": st_color,
        "st_notes": st_notes,
        "st_score": st_score,
        "lt_action": lt_action,
        "lt_color": lt_color,
        "lt_notes": lt_notes,
        "lt_score": lt_score,
    }


def show_holding_advisor(results=None):
    """
    Separate page: user enters stock, buy price, shares →
    detailed technical + fundamental analysis and ST/LT decision.
    """
    st.title("📥 My Holding — Hold / Sell / Buy?")
    st.caption(
        "Enter any stock you already own (or plan to buy). "
        "You get technical analysis, fundamental snapshot, "
        "and separate short-term vs long-term advice."
    )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sym_in = st.text_input(
            "NSE symbol",
            value=st.session_state.get("holding_sym", "RELIANCE"),
            placeholder="RELIANCE, TCS, INFY…",
            key="holding_sym_input",
        )
    with c2:
        buy_price = st.number_input(
            "Your buy price (₹)",
            min_value=0.05,
            value=float(st.session_state.get("holding_buy", 1000.0)),
            step=1.0,
            key="holding_buy_input",
        )
    with c3:
        shares = st.number_input(
            "No. of shares",
            min_value=1,
            value=int(st.session_state.get("holding_shares", 10)),
            step=1,
            key="holding_shares_input",
        )
    with c4:
        st.write("")
        st.write("")
        run = st.button("🔎 Analyse holding", type="primary", use_container_width=True)

    if not run and not st.session_state.get("holding_last_sym"):
        st.info("Fill symbol, buy price and shares, then click **Analyse holding**.")
        return

    if run:
        st.session_state.holding_last_sym = display_symbol(sym_in)
        st.session_state.holding_buy = buy_price
        st.session_state.holding_shares = shares

    symbol = st.session_state.get("holding_last_sym") or display_symbol(sym_in)
    buy_price = float(st.session_state.get("holding_buy", buy_price))
    shares = int(st.session_state.get("holding_shares", shares))

    with st.spinner(f"Loading {symbol} technicals + fundamentals…"):
        df = stock_history(symbol, interval="1d")
        if df is None or df.empty:
            st.error("Could not load price history for this symbol.")
            return
        result = analyse_stock(clean_symbol(symbol), df, fetch_news=True)
        if not result:
            st.error("Analysis failed for this symbol.")
            return
        # Prefer live/last quote if available
        q = live_quote(symbol)
        if q and q.get("price"):
            result["Price"] = q["price"]
        fund = fetch_fundamentals(symbol)
        advice = holding_horizon_advice(result, buy_price, shares, fund)

    # P&L strip
    st.subheader(f"{symbol} — {fund.get('name') or symbol}")
    a, b, c, d, e = st.columns(5)
    a.metric("LTP / Model price", f"₹{safe_float(result['Price']):,.2f}")
    b.metric("Your buy", f"₹{buy_price:,.2f}")
    c.metric("Shares", f"{shares}")
    d.metric("P&L %", f"{advice['pnl_pct']:+.2f}%")
    e.metric("P&L ₹", f"₹{advice['pnl_rs']:+,.0f}")
    st.caption(
        f"Invested ₹{advice['invested']:,.0f} · Market value ₹{advice['mkt_value']:,.0f} · "
        f"Sector: {fund.get('sector') or result.get('Sector', '')} · "
        f"{fund.get('industry', '')}"
    )

    # ST / LT recommendation boxes
    st.subheader("🧭 Decision — Short term vs Long term")
    left, right = st.columns(2)
    with left:
        box = "buy-box" if advice["st_color"] == "buy" else (
            "danger-box" if advice["st_color"] == "sell" else "success-box"
        )
        st.markdown(
            f"""
            <div class="{box}" style="padding:14px;border-radius:10px;">
            <h4 style="margin:0 0 8px 0;">Short term (days–few weeks)</h4>
            <p style="font-size:1.15em;margin:0;"><b>{advice['st_action']}</b></p>
            <p style="margin:8px 0 0 0;opacity:0.9;">Score {advice['st_score']} · Model call: {result.get('Call')} · Pred {safe_float(result.get('Prediction')):.0f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for n in advice["st_notes"]:
            st.write(f"• {n}")
        st.write(
            f"**Model target** ₹{safe_float(result.get('Target')):,.2f} · "
            f"**Stop** ₹{safe_float(result.get('Stop Loss')):,.2f} · "
            f"**Hold days** {result.get('Hold Days')}"
        )
    with right:
        box = "buy-box" if advice["lt_color"] == "buy" else (
            "danger-box" if advice["lt_color"] == "sell" else "success-box"
        )
        st.markdown(
            f"""
            <div class="{box}" style="padding:14px;border-radius:10px;">
            <h4 style="margin:0 0 8px 0;">Long term (months+)</h4>
            <p style="font-size:1.15em;margin:0;"><b>{advice['lt_action']}</b></p>
            <p style="margin:8px 0 0 0;opacity:0.9;">Score {advice['lt_score']} · Trend + fundamentals combined</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for n in advice["lt_notes"]:
            st.write(f"• {n}")
        if fund.get("target_mean"):
            st.write(f"**Analyst mean target (Yahoo):** ₹{safe_float(fund['target_mean']):,.2f}")
        if fund.get("recommendation"):
            st.write(f"**Street lean:** {fund['recommendation']}")

    st.divider()

    # Fundamentals report
    st.subheader("🏛️ Fundamental snapshot")
    if fund.get("error") and not fund.get("name"):
        st.warning(f"Fundamentals limited: {fund['error']}")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Market Cap", _fmt_num(fund.get("market_cap"), money=True))
    f2.metric("P/E (TTM)", _fmt_num(fund.get("pe")))
    f3.metric("Forward P/E", _fmt_num(fund.get("forward_pe")))
    f4.metric("P/B", _fmt_num(fund.get("pb")))
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("EPS", _fmt_num(fund.get("eps")))
    g2.metric("ROE", _fmt_num(fund.get("roe"), pct=True))
    g3.metric("Profit margin", _fmt_num(fund.get("profit_margins"), pct=True))
    g4.metric("Debt / Equity", _fmt_num(fund.get("debt_to_equity")))
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Revenue growth", _fmt_num(fund.get("revenue_growth"), pct=True))
    h2.metric("Earnings growth", _fmt_num(fund.get("earnings_growth"), pct=True))
    h3.metric("Dividend yield", _fmt_num(fund.get("dividend_yield"), pct=True))
    h4.metric("Beta", _fmt_num(fund.get("beta")))
    i1, i2, i3 = st.columns(3)
    i1.metric("52w high", _fmt_num(fund.get("fifty_two_high"), money=True))
    i2.metric("52w low", _fmt_num(fund.get("fifty_two_low"), money=True))
    i3.metric("Current ratio", _fmt_num(fund.get("current_ratio")))

    if fund.get("summary"):
        with st.expander("Business summary", expanded=False):
            st.write(fund["summary"])

    if fund.get("revenue_latest") or fund.get("net_income_latest"):
        st.caption(
            f"Latest reported · Revenue: {_fmt_num(fund.get('revenue_latest'), money=True)} · "
            f"Net income: {_fmt_num(fund.get('net_income_latest'), money=True)}"
        )

    st.divider()

    # Technical school + chart (reuse existing)
    st.subheader("📈 Technical position vs your entry")
    st.write(
        f"**Model call:** {result.get('Call')} · **Prediction:** {safe_float(result.get('Prediction')):.1f}% · "
        f"**Risk:** {safe_float(result.get('Risk %')):.2f}% ({result.get('Risk Level')}) · "
        f"**Patterns:** {result.get('Patterns')}"
    )
    st.write(f"**Reason:** {result.get('Reason', '')}")
    try:
        show_indicator_school(result)
    except Exception:
        pass
    try:
        show_interactive_chart_panel(symbol, result)
    except Exception as e:
        st.caption(f"Chart: {e}")

    # Optional: save into portfolio csv
    st.divider()
    st.subheader("💾 Save into My Portfolio file")
    if st.button("Add / update this holding in my_portfolio.csv"):
        try:
            ensure_files()
            port = load_portfolio()
            stock = display_symbol(symbol)
            row = {
                "Stock": stock,
                "Buy Price": buy_price,
                "Shares": shares,
                "Buy Date": datetime.now().strftime("%Y-%m-%d"),
                "Notes": f"Added from Holding Advisor · LTP {safe_float(result['Price']):.2f}",
            }
            # Flexible columns
            for col in row:
                if col not in port.columns:
                    port[col] = ""
            if "Stock" in port.columns and not port.empty:
                mask = port["Stock"].astype(str).str.upper() == stock
                if mask.any():
                    for k, v in row.items():
                        port.loc[mask, k] = v
                else:
                    port = pd.concat([port, pd.DataFrame([row])], ignore_index=True)
            else:
                port = pd.DataFrame([row])
            port.to_csv(PORTFOLIO_FILE, index=False)
            st.success(f"Saved {stock} to portfolio.")
        except Exception as e:
            st.error(f"Could not save: {e}")


# ============================================================
# PORTFOLIO PAGE
# ============================================================

def show_portfolio(
    results
):

    st.title(
        "💼 MY PORTFOLIO"
    )

    st.info(
        "Edit my_portfolio.csv to enter your holdings. "
        "The dashboard will automatically calculate HOLD / SELL / BUY MORE."
    )

    portfolio = load_portfolio()

    if portfolio.empty:

        st.warning(
            "Your portfolio is empty."
        )

        st.write(
            "Use these columns in my_portfolio.csv:"
        )

        st.code(
            """
Stock,Shares,Buy Price,Purchase Date,Maximum Holding Days
RELIANCE,10,1400,2026-08-01,15
TCS,5,3000,2026-08-05,15
            """.strip()
        )

        return

    analysis = portfolio_analysis(
        results
    )

    if analysis.empty:

        st.warning(
            "Run FULL MARKET SCAN first so the portfolio can be analysed."
        )

        st.dataframe(
            portfolio,
            use_container_width=True,
            hide_index=True
        )

        return

    st.subheader(
        "Portfolio Decision"
    )

    st.dataframe(
        analysis,
        use_container_width=True,
        hide_index=True
    )

    sells = analysis[
        analysis["Action"]
        .astype(str)
        .str.startswith("SELL")
    ]

    if not sells.empty:

        st.subheader(
            "🔴 Portfolio SELL Calls"
        )

        st.dataframe(
            sells,
            use_container_width=True,
            hide_index=True
        )

    buys = analysis[
        analysis["Action"]
        .astype(str)
        .str.contains(
            "BUY MORE"
        )
    ]

    if not buys.empty:

        st.subheader(
            "🟢 Portfolio BUY MORE Signals"
        )

        st.dataframe(
            buys,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SECTOR PAGE
# ============================================================

def show_sector_charts(sec: pd.DataFrame):
    """Bar / ranking charts for sector strength, BUY%, risk."""
    if sec is None or sec.empty:
        return

    st.subheader("📈 Sector charts")
    st.caption("Compare sectors by strength, BUY share, and average risk from the latest scan.")

    chart_kind = st.radio(
        "Chart type",
        ["Sector Strength", "BUY % by sector", "Avg Prediction", "Avg Risk %"],
        horizontal=True,
        key="sector_chart_kind",
    )

    plot_df = sec.copy()
    if chart_kind == "Sector Strength" and "Sector Strength" in plot_df.columns:
        ycol, color = "Sector Strength", "#4dabf7"
        title = "Sector Strength (higher = stronger)"
    elif chart_kind == "BUY % by sector" and "BUY %" in plot_df.columns:
        ycol, color = "BUY %", "#1a9b5f"
        title = "Share of BUY calls within each sector"
    elif chart_kind == "Avg Prediction" and "Avg_Prediction" in plot_df.columns:
        ycol, color = "Avg_Prediction", "#fcc419"
        title = "Average Prediction % by sector"
    elif "Avg_Risk" in plot_df.columns:
        ycol, color = "Avg_Risk", "#ff6b6b"
        title = "Average Risk % by sector (lower is safer)"
    else:
        st.caption("Not enough sector metrics to chart.")
        return

    plot_df = plot_df.sort_values(ycol, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=plot_df[ycol],
            y=plot_df["Sector"],
            orientation="h",
            marker_color=color,
            text=plot_df[ycol].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=max(360, 28 * len(plot_df)),
        template="plotly_dark",
        title=title,
        margin=dict(l=10, r=40, t=50, b=10),
        xaxis_title=ycol,
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Combined overview
    if {"Sector Strength", "BUY %", "Avg_Risk"}.issubset(set(plot_df.columns)):
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=plot_df["Avg_Risk"],
                y=plot_df["Sector Strength"],
                mode="markers+text",
                text=plot_df["Sector"],
                textposition="top center",
                marker=dict(
                    size=plot_df["BUY %"].clip(lower=5) / 2 + 8,
                    color=plot_df["BUY %"],
                    colorscale="Greens",
                    showscale=True,
                    colorbar=dict(title="BUY %"),
                ),
                name="Sectors",
            )
        )
        fig2.update_layout(
            height=480,
            template="plotly_dark",
            title="Sector map — Strength vs Risk (bubble size ~ BUY %)",
            xaxis_title="Avg Risk % →",
            yaxis_title="Sector Strength →",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)


def show_sector_page(
    results
):

    st.title(
        "🏭 SECTOR-WISE ANALYSIS"
    )

    results = ensure_result_columns(results)

    if results.empty:

        st.info(
            "Run FULL MARKET SCAN first."
        )

        return

    sec = sector_table(
        results
    )

    if sec.empty:
        return

    show_sector_charts(sec)

    st.subheader("Sector table")
    st.dataframe(
        sec,
        use_container_width=True,
        hide_index=True
    )

    selected_sector = st.selectbox(
        "Select Sector",
        sec["Sector"].tolist()
    )

    if st.button(
        "📂 OPEN SELECTED SECTOR",
        type="primary"
    ):

        st.session_state.selected_sector = (
            selected_sector
        )

    chosen = (
        st.session_state.selected_sector
        or
        selected_sector
    )

    x = results[
        results["Sector"]
        ==
        chosen
    ].copy()

    x = x.sort_values(
        [
            "Prediction",
            "Risk %",
        ],
        ascending=[
            False,
            True,
        ]
    )

    st.subheader(
        f"📊 {chosen} Stocks"
    )

    st.dataframe(
        x.drop(
            columns=[
                "Data",
                "News",
            ],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Open Individual Stock"
    )

    for stock in x[
        "Stock"
    ].tolist():

        if st.button(
            f"📈 OPEN {stock}",
            key="sector_"
            + chosen
            + "_"
            + stock
        ):

            st.session_state.selected_stock = (
                stock
            )

            st.session_state.page = (
                "Stock Analysis"
            )

            st.rerun()


# ============================================================
# FIND STOCK PAGE
# ============================================================

def recommend_stocks_from_words(results: pd.DataFrame, query: str, limit: int = 15) -> pd.DataFrame:
    """Natural-language-ish recommendations from a few words."""
    if results is None or results.empty or not str(query).strip():
        return pd.DataFrame()
    q = str(query).lower().strip()
    x = ensure_result_columns(results).copy()
    # Start from scan search
    try:
        found = find_stock_results(x, query)
    except Exception:
        found = x.copy()
    if found is None or found.empty:
        found = x.copy()

    # Intent filters
    call_u = found["Call"].astype(str).str.upper() if "Call" in found.columns else pd.Series([""] * len(found))
    if any(w in q for w in ("buy", "long", "accumulate", "upside")):
        found = found[call_u.str.contains("BUY|HOLD|WATCH", na=False)]
    if any(w in q for w in ("sell", "short", "exit", "weak")):
        found = found[call_u.str.contains("SELL", na=False)]
    if "low risk" in q or "safe" in q:
        if "Risk Level" in found.columns:
            found = found[found["Risk Level"].astype(str).str.upper().isin(["LOW", "MEDIUM"])]
    if "high risk" in q:
        if "Risk Level" in found.columns:
            found = found[found["Risk Level"].astype(str).str.upper().str.contains("HIGH", na=False)]

    # Sector keywords
    sector_map = {
        "bank": "Bank", "banking": "Bank", "it ": "IT", " software": "IT", "pharma": "Pharma",
        "auto": "Auto", "metal": "Metal", "fmcg": "FMCG", "energy": "Energy", "oil": "Energy",
        "realty": "Realty", "infra": "Infra", "finance": "Finance",
    }
    if "Sector" in found.columns:
        for kw, sec in sector_map.items():
            if kw in q:
                found = found[found["Sector"].astype(str).str.contains(sec, case=False, na=False)]

    # Price under X
    import re
    m = re.search(r"under\s+(\d+)", q)
    if m and "Price" in found.columns:
        found = found[pd.to_numeric(found["Price"], errors="coerce") <= float(m.group(1))]

    if "Prediction" in found.columns:
        found = found.sort_values("Prediction", ascending=False)
    return found.head(limit)


def show_find_stock(
    results
):

    st.title(
        "🔍 FIND STOCK"
    )

    st.write(
        "Type a few words — get stock **recommendations** (buy/sell, sector, risk)."
    )

    query = st.text_input(
        "Search / recommend",
        placeholder=(
            "Example: low risk banking stocks to buy"
        ),
        key="find_stock_q",
    )

    examples = st.multiselect(
        "Quick phrases",
        [
            "low risk bank stocks to buy",
            "IT stocks to buy",
            "strong pharma",
            "stocks under 500",
            "sell calls high risk",
            "metal stocks",
            "RELIANCE",
        ],
        key="find_quick",
    )
    if examples and not query.strip():
        query = examples[0]

    if results.empty:

        st.info(
            "Run FULL MARKET SCAN first for best recommendations."
        )

        return

    if not query.strip():

        st.info(
            "Examples: "
            "low risk bank stocks, "
            "IT stocks to buy, "
            "strong pharma stocks, "
            "stocks under 1000, "
            "sell calls, RELIANCE"
        )

        return

    found = recommend_stocks_from_words(results, query, limit=25)
    if found.empty:
        found = find_stock_results(results, query)

    if found.empty:

        st.warning(
            "No matching stocks found in the current scan."
        )

        return

    st.success(
        f"**{len(found)}** recommendations for: _{query}_"
    )

    filterable_dataframe(
        found.drop(columns=["Data", "News"], errors="ignore"),
        key="find_rec_table",
        default_cols=[c for c in [
            "Stock", "Sector", "Call", "Price", "Prediction", "Risk Level",
            "Target", "Stop Loss", "Patterns",
        ] if c in found.columns],
    )

    st.subheader("Cards — buy/sell executes to Paper Trading")
    for stock in found["Stock"].head(10).tolist():
        row = found[found["Stock"] == stock].iloc[0]
        render_call_stock_card(row.to_dict(), section_key="find")

    st.subheader(
        "Open Detailed Analysis"
    )

    for stock in found[
        "Stock"
    ].head(50).tolist():

        row = found[
            found["Stock"]
            ==
            stock
        ].iloc[0]

        if st.button(
            f"📈 {stock} | {row['Call']} | Prediction {row['Prediction']}%",
            key="find_"
            + stock
        ):

            st.session_state.selected_stock = (
                stock
            )

            st.session_state.page = (
                "Stock Analysis"
            )

            st.rerun()


# ============================================================
# EXPORT
# ============================================================

def create_excel(
    results
):

    filename = (
        "NSE_V12_Report_"
        +
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        +
        ".xlsx"
    )

    path = (
        APP_DIR /
        filename
    )

    portfolio = load_portfolio()

    history = load_history()

    with pd.ExcelWriter(
        path,
        engine="openpyxl"
    ) as writer:

        if (
            results is not None
            and
            not results.empty
        ):

            clean = results.drop(
                columns=[
                    "Data",
                    "News",
                ],
                errors="ignore"
            )

            clean.to_excel(
                writer,
                sheet_name="All Stocks",
                index=False
            )

            sells = clean[
                clean["Call"] == "SELL"
            ]

            sells.to_excel(
                writer,
                sheet_name="SELL Calls",
                index=False
            )

            buys = clean[
                clean["Call"] == "BUY"
            ]

            buys.to_excel(
                writer,
                sheet_name="BUY Calls",
                index=False
            )

            holds = clean[
                clean["Call"] == "HOLD"
            ]

            holds.to_excel(
                writer,
                sheet_name="HOLD Calls",
                index=False
            )

            sectors = sector_table(
                results
            )

            sectors.to_excel(
                writer,
                sheet_name="Sector Analysis",
                index=False
            )

            p_analysis = portfolio_analysis(
                results
            )

            if not p_analysis.empty:

                p_analysis.to_excel(
                    writer,
                    sheet_name="Portfolio Decision",
                    index=False
                )

        if not portfolio.empty:

            portfolio.to_excel(
                writer,
                sheet_name="My Portfolio",
                index=False
            )

        history.to_excel(
            writer,
            sheet_name="Prediction History",
            index=False
        )

    return path


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "results":
        pd.DataFrame(),

    "last_run":
        None,

    "page":
        "Dashboard",

    "selected_stock":
        "",

    "selected_sector":
        "",

    "auto_refresh":
        True,  # live prices auto-update on a timer
    "live_prices_on":
        False,  # refresh LTP on every page cycle (throttled)
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# LOAD SAVED RESULT (shared scan across phone / PC on same Cloud app)
# ============================================================

SCAN_META_FILE = APP_DIR / "latest_results_meta.json"


def bump_sync_version(reason: str = "") -> int:
    """
    Global version counter for cross-device near-real-time sync.
    Any device that sees a higher version reloads shared files.
    (Streamlit Cloud cannot host a custom WebSocket server; version+poll is the reliable pattern.)
    """
    ver = 1
    try:
        if SYNC_VERSION_FILE.exists():
            data = json.loads(SYNC_VERSION_FILE.read_text(encoding="utf-8"))
            ver = int(data.get("version", 0)) + 1
        payload = {
            "version": ver,
            "reason": reason or "update",
            "at": datetime.now().isoformat(timespec="seconds"),
            "at_ist": india_now().strftime("%Y-%m-%d %H:%M:%S IST"),
        }
        SYNC_VERSION_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    return ver


def read_sync_version() -> dict:
    try:
        if SYNC_VERSION_FILE.exists():
            return json.loads(SYNC_VERSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"version": 0, "reason": "", "at_ist": ""}


def _batch_last_prices(symbols: list, max_n: int = 120) -> dict:
    """
    One Yahoo batch download for many symbols → {SYM: last_price}.
    Faster than per-stock live_quote when refreshing whole books.
    """
    out = {}
    syms = []
    for s in symbols:
        try:
            c = clean_symbol(s)
            if c and c not in syms:
                syms.append(c)
        except Exception:
            continue
        if len(syms) >= max_n:
            break
    if not syms:
        return out
    try:
        period = "1d" if nse_market_open_now() else "5d"
        interval = "1m" if nse_market_open_now() else "1d"
        data = yf.download(
            tickers=" ".join(syms),
            period=period,
            interval=interval,
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
        if data is None or data.empty:
            return out
        # Multi-ticker: columns MultiIndex (ticker, field) or single ticker
        if isinstance(data.columns, pd.MultiIndex):
            for sym in syms:
                try:
                    if sym not in data.columns.get_level_values(0):
                        continue
                    sub = data[sym]
                    if "Close" in sub.columns:
                        s = sub["Close"].dropna()
                        if not s.empty:
                            out[sym] = float(s.iloc[-1])
                            out[display_symbol(sym)] = out[sym]
                except Exception:
                    continue
        else:
            # Single symbol frame
            if "Close" in data.columns:
                s = data["Close"].dropna()
                if not s.empty and len(syms) == 1:
                    out[syms[0]] = float(s.iloc[-1])
                    out[display_symbol(syms[0])] = out[syms[0]]
    except Exception:
        pass
    # Fill misses with single live_quote
    for sym in syms:
        if sym in out:
            continue
        try:
            q = live_quote(sym)
            if q and q.get("price"):
                out[sym] = float(q["price"])
                out[display_symbol(sym)] = out[sym]
        except Exception:
            continue
    return out


def refresh_live_prices_all_books(max_per_book: int = 80) -> dict:
    """
    Update Current / live price on Scan, Strategy, Past Predictions, Paper.
    Does NOT change Entry / Target / Stop Loss.
    Uses batch Yahoo download so many stocks update together.
    """
    stats = {"scan": 0, "strategy": 0, "history": 0, "paper": 0}
    all_syms = []

    # Collect symbols from every book
    try:
        if st.session_state.get("results") is not None and not st.session_state.results.empty:
            r = st.session_state.results
            col = "Symbol" if "Symbol" in r.columns else "Stock"
            all_syms.extend(r[col].astype(str).head(max_per_book).tolist())
    except Exception:
        pass
    try:
        live_df = st.session_state.get("live_strat_df")
        if live_df is None or (isinstance(live_df, pd.DataFrame) and live_df.empty):
            if STRATEGY_LIVE_FILE.exists():
                live_df = pd.read_csv(STRATEGY_LIVE_FILE)
        if live_df is not None and not live_df.empty:
            col = "Stock" if "Stock" in live_df.columns else "Symbol"
            all_syms.extend(live_df[col].astype(str).head(max_per_book).tolist())
            st.session_state["_tmp_live_df"] = live_df
    except Exception:
        pass
    try:
        hist = load_history()
        if hist is not None and not hist.empty and "Stock" in hist.columns:
            res_u = hist["Result"].astype(str).str.upper() if "Result" in hist.columns else pd.Series([""] * len(hist))
            open_h = hist[~res_u.str.contains("TARGET|STOP|HOLDING|WIN|LOSS", na=False, regex=True)]
            all_syms.extend(open_h["Stock"].astype(str).head(max_per_book).tolist())
    except Exception:
        pass
    try:
        paper = load_paper_portfolio()
        if paper is not None and not paper.empty and "Stock" in paper.columns:
            su = paper["Status"].astype(str).str.upper() if "Status" in paper.columns else pd.Series(["OPEN"] * len(paper))
            all_syms.extend(paper.loc[su != "CLOSED", "Stock"].astype(str).head(max_per_book).tolist())
    except Exception:
        pass

    price_map = _batch_last_prices(all_syms, max_n=min(150, max(40, max_per_book * 2)))

    def _px(sym):
        if not sym:
            return None
        s = str(sym).upper().replace(".NS", "").strip()
        v = price_map.get(s) or price_map.get(s + ".NS") or price_map.get(display_symbol(s))
        return round(float(v), 2) if v else None

    # --- Scan results ---
    try:
        if st.session_state.get("results") is not None and not st.session_state.results.empty:
            x = st.session_state.results.copy()
            if "Current Price" not in x.columns:
                x["Current Price"] = pd.to_numeric(x.get("Price"), errors="coerce")
            if "Locked Price" not in x.columns and "Price" in x.columns:
                x["Locked Price"] = pd.to_numeric(x["Price"], errors="coerce")
            col = "Symbol" if "Symbol" in x.columns else "Stock"
            for idx in list(x.index)[:max_per_book]:
                px = _px(x.at[idx, col])
                if px:
                    x.at[idx, "Current Price"] = px
                    x.at[idx, "Price"] = px
                    stats["scan"] += 1
            st.session_state.results = x
    except Exception:
        pass

    # --- Strategy ---
    try:
        live_df = st.session_state.get("_tmp_live_df") or st.session_state.get("live_strat_df")
        if live_df is not None and not live_df.empty:
            x = live_df.copy()
            if "Current Price" not in x.columns:
                x["Current Price"] = pd.to_numeric(x.get("Price"), errors="coerce")
            col = "Stock" if "Stock" in x.columns else "Symbol"
            for idx in list(x.index)[:max_per_book]:
                px = _px(x.at[idx, col])
                if px:
                    x.at[idx, "Current Price"] = px
                    if "Price" in x.columns:
                        x.at[idx, "Price"] = px
                    stats["strategy"] += 1
            st.session_state["live_strat_df"] = x
            try:
                x.to_csv(STRATEGY_LIVE_FILE, index=False)
            except Exception:
                pass
        st.session_state.pop("_tmp_live_df", None)
    except Exception:
        pass

    # --- History open ---
    try:
        history = normalize_history_df(load_history())
        if history is not None and not history.empty:
            if "Current Price" not in history.columns:
                history["Current Price"] = ""
            res_u = history["Result"].astype(str).str.upper().str.strip()
            open_mask = ~res_u.str.contains(
                "TARGET ACHIEVED|STOP LOSS HIT|HOLDING PERIOD COMPLETED|^WIN$|^LOSS$",
                regex=True, na=True,
            )
            for idx in history.index[open_mask].tolist()[:max_per_book]:
                px = _px(history.at[idx, "Stock"])
                if px:
                    history.at[idx, "Current Price"] = px
                    stats["history"] += 1
            try:
                history.to_csv(HISTORY_FILE, index=False)
            except Exception:
                pass
    except Exception:
        pass

    # --- Paper open ---
    try:
        paper = load_paper_portfolio()
        if paper is not None and not paper.empty:
            if "Current Price" not in paper.columns:
                paper["Current Price"] = ""
            status_u = paper["Status"].astype(str).str.upper() if "Status" in paper.columns else pd.Series(["OPEN"] * len(paper))
            open_idx = paper.index[status_u != "CLOSED"].tolist()[:max_per_book]
            for idx in open_idx:
                px = _px(paper.at[idx, "Stock"])
                if not px:
                    continue
                paper.at[idx, "Current Price"] = px
                entry = safe_float(paper.at[idx, "Entry"])
                shares = safe_int(paper.at[idx, "Shares"], 1) or 1
                side = str(paper.at[idx, "Side"] if "Side" in paper.columns else "BUY").upper()
                if entry > 0 and str(paper.at[idx, "Status"]).upper() != "CLOSED":
                    ret = ((entry - px) / entry * 100) if "SELL" in side else ((px - entry) / entry * 100)
                    paper.at[idx, "Return %"] = round(ret, 2)
                    paper.at[idx, "PnL ₹"] = round(shares * entry * ret / 100, 2)
                stats["paper"] += 1
            save_paper_portfolio(paper)
    except Exception:
        pass

    st.session_state["_last_price_refresh"] = india_now().strftime("%H:%M:%S IST")
    return stats


def apply_realtime_sync(force: bool = False, refresh_prices: bool = True) -> bool:
    """
    Reload shared files when version advances; optionally refresh live prices every time.
    """
    remote = read_sync_version()
    remote_v = int(remote.get("version") or 0)
    local_v = int(st.session_state.get("_sync_version_seen", 0) or 0)
    data_changed = force or remote_v > local_v
    changed = False
    if data_changed:
        try:
            n = load_scan_from_disk(force=True)
            if n:
                changed = True
        except Exception:
            pass
        try:
            df = load_strategy_snapshot(force=True)
            if df is not None and not getattr(df, "empty", True):
                changed = True
        except Exception:
            pass
        try:
            if PAPER_FILE.exists() or HISTORY_FILE.exists():
                changed = True
        except Exception:
            pass
        st.session_state["_sync_version_seen"] = remote_v
        st.session_state["_sync_last_reason"] = remote.get("reason", "")
        st.session_state["_sync_last_at"] = remote.get("at_ist", "")

    if refresh_prices:
        try:
            refresh_live_prices_all_books(max_per_book=25)
            changed = True
        except Exception:
            pass
    return changed


def save_strategy_snapshot(live_df: pd.DataFrame) -> dict:
    """
    Save strategy list for all devices. Compare to previous file → new stocks only.
    Returns {n, n_new, new_stocks, saved_at_ist}.
    """
    out = {"n": 0, "n_new": 0, "new_stocks": [], "saved_at_ist": ""}
    if live_df is None or live_df.empty:
        return out
    df = live_df.copy()
    if "Stock" in df.columns:
        df["Stock"] = df["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()
    prev_stocks = set()
    try:
        if STRATEGY_LIVE_FILE.exists():
            # current becomes previous
            try:
                old = pd.read_csv(STRATEGY_LIVE_FILE)
                if not old.empty and "Stock" in old.columns:
                    prev_stocks = set(
                        old["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()
                    )
                    old.to_csv(STRATEGY_PREV_FILE, index=False)
            except Exception:
                pass
        df.to_csv(STRATEGY_LIVE_FILE, index=False)
        now_stocks = set(df["Stock"].astype(str).tolist()) if "Stock" in df.columns else set()
        new_stocks = sorted(now_stocks - prev_stocks)
        out["n"] = len(df)
        out["n_new"] = len(new_stocks)
        out["new_stocks"] = new_stocks
        out["saved_at_ist"] = india_now().strftime("%Y-%m-%d %H:%M:%S IST")
        STRATEGY_META_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
        bump_sync_version("strategy_list")
    except Exception:
        pass
    return out


def load_strategy_snapshot(force: bool = True) -> pd.DataFrame:
    """Load shared strategy list from disk (all devices on same Cloud app)."""
    try:
        if STRATEGY_LIVE_FILE.exists():
            df = pd.read_csv(STRATEGY_LIVE_FILE)
            if df is not None and not df.empty:
                if force or st.session_state.get("live_strat_df") is None:
                    st.session_state["live_strat_df"] = df
                return df
    except Exception:
        pass
    return st.session_state.get("live_strat_df", pd.DataFrame())


def strategy_new_only(live_df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose stock was not in the previous shared strategy file."""
    if live_df is None or live_df.empty:
        return pd.DataFrame()
    prev = set()
    try:
        if STRATEGY_PREV_FILE.exists():
            old = pd.read_csv(STRATEGY_PREV_FILE)
            if not old.empty and "Stock" in old.columns:
                prev = set(
                    old["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()
                )
    except Exception:
        pass
    x = live_df.copy()
    x["_s"] = x["Stock"].astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()
    if not prev:
        return x.drop(columns=["_s"], errors="ignore")
    return x[~x["_s"].isin(prev)].drop(columns=["_s"], errors="ignore")


def save_scan_to_disk(results: pd.DataFrame) -> int:
    """Persist scan so other browser sessions on the same app can load it."""
    if results is None or results.empty:
        return 0
    out = results.drop(columns=["Data", "News"], errors="ignore").copy()
    try:
        out.to_csv(RESULT_FILE, index=False)
        # Force flush for Cloud multi-session reads
        try:
            import os
            with open(RESULT_FILE, "a", encoding="utf-8") as _:
                os.fsync(_.fileno()) if False else None
        except Exception:
            pass
        meta = {
            "rows": int(len(out)),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "saved_at_ist": india_now().strftime("%Y-%m-%d %H:%M:%S IST"),
        }
        SCAN_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        bump_sync_version("full_market_scan")
        return int(len(out))
    except Exception:
        return 0


def load_scan_from_disk(force: bool = False) -> int:
    """
    Load latest_results.csv into session.
    force=True → always replace session with disk (use after phone scan on PC).
    Returns number of rows loaded, or 0.
    """
    if not RESULT_FILE.exists():
        return 0
    try:
        saved = pd.read_csv(RESULT_FILE)
        if saved is None or saved.empty:
            return 0
        saved = ensure_result_columns(saved)
        n_disk = len(saved)
        n_mem = 0
        try:
            if st.session_state.results is not None and not st.session_state.results.empty:
                n_mem = len(st.session_state.results)
        except Exception:
            n_mem = 0
        if force or n_mem == 0 or n_disk > n_mem:
            st.session_state.results = saved
            st.session_state["_scan_loaded_from_disk"] = True
            try:
                if SCAN_META_FILE.exists():
                    st.session_state["_scan_meta"] = json.loads(
                        SCAN_META_FILE.read_text(encoding="utf-8")
                    )
            except Exception:
                st.session_state["_scan_meta"] = {"rows": n_disk}
            return n_disk
    except Exception:
        return 0
    return n_mem


# Always try to pick up a newer/shared scan from disk at session start
try:
    load_scan_from_disk(force=False)
except Exception:
    pass
try:
    load_strategy_snapshot(force=False)
except Exception:
    pass


# ============================================================
# SIDEBAR
# ============================================================

# Parent hub for each inner page → Back returns here
PAGE_PARENT = {
    "Nifty Analysis": "Swing Hub",
    "BankNifty Analysis": "Swing Hub",
    "BUY Calls": "Swing Hub",
    "SELL Calls": "Swing Hub",
    "Sure Calls": "Swing Hub",
    "Strategy Lab": "Swing Hub",
    "History": "Swing Hub",
    "Trade Tracker": "Swing Hub",
    "Paper Trading": "Swing Hub",
    "Holding Advisor": "Swing Hub",
    "Portfolio": "Swing Hub",
    "Performance Summary": "Swing Hub",
    "Chart Pattern Scanner": "Pattern Hub",
    "Multi Timeframe": "Pattern Hub",
    "Find Stock": "Tools Hub",
    "Sector Analysis": "Tools Hub",
    "Stock Analysis": "Tools Hub",
    "FII DII": "Dashboard",
    "NSE Tools": "Dashboard",
    "Intraday FO": "Dashboard",
    "Watchlist": "Dashboard",
    "Scanner Query": "Dashboard",
    "Intraday Backtest": "Intraday FO",
    "Tool Square9": "NSE Tools",
    "Tool IV": "NSE Tools",
    "Tool OI Chain": "NSE Tools",
    "Tool Option Calculator": "NSE Tools",
    "Tool Option History": "NSE Tools",
    "Tool Option Backtester": "NSE Tools",
    "Tool Candles": "NSE Tools",
    "Tool Momentum": "NSE Tools",
    "Swing Hub": "Dashboard",
    "Pattern Hub": "Dashboard",
    "Tools Hub": "Dashboard",
}

PAGE_LABELS = {
    "Dashboard": "Dashboard",
    "Swing Hub": "Swing Trading Setup",
    "Pattern Hub": "Chart Patterns",
    "Tools Hub": "Tools",
    "BUY Calls": "BUY Calls",
    "SELL Calls": "SELL Calls",
    "Sure Calls": "Sure Call Desk",
    "Strategy Lab": "Strategy Lab",
    "History": "Past Predictions",
    "Trade Tracker": "Auto Trade Tracker",
    "Paper Trading": "Paper / Dummy Trades",
    "Holding Advisor": "Holding Advisor",
    "Portfolio": "Portfolio",
    "Performance Summary": "Performance Summary",
    "Nifty Analysis": "NIFTY 50",
    "BankNifty Analysis": "BANK NIFTY",
    "Chart Pattern Scanner": "Pattern Scanner",
    "Multi Timeframe": "Multi-Timeframe",
    "Find Stock": "Find Stock",
    "Sector Analysis": "Sector Analysis",
    "Stock Analysis": "Analyze Stock",
    "FII DII": "FII & DII",
    "NSE Tools": "NSE Tools Workspace",
    "Intraday FO": "Intraday & F&O",
    "Watchlist": "Watchlist",
    "Scanner Query": "Scanner Query",
    "Intraday Backtest": "Intraday Backtest",
    "Tool Square9": "Square of 9",
    "Tool IV": "IV Tool",
    "Tool OI Chain": "Open Interest Chain",
    "Tool Option Calculator": "Option Calculator",
    "Tool Option History": "Option Historical Chart",
    "Tool Option Backtester": "Option Backtester",
    "Tool Candles": "Candlestick Patterns",
    "Tool Momentum": "Momentum Divergence",
}


def inject_mobile_sidebar_close():
    """
    After a sidebar / hub nav tap on phone: force-hide the drawer so the new page
    is visible without tapping the main area again.
    """
    if not st.session_state.get("_collapse_sidebar"):
        return
    st.session_state["_collapse_sidebar"] = False
    components.html(
        """
        <div id="nse-close-side" style="height:1px;overflow:hidden;opacity:0;">close</div>
        <script>
        (function () {
          function closeSide() {
            try {
              var doc = window.parent.document;
              var win = window.parent;
              var w = win.innerWidth || doc.documentElement.clientWidth || 0;
              if (w > 991) return;

              function clickEl(el) {
                if (!el) return false;
                try {
                  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: win }));
                  el.click();
                  return true;
                } catch (e) { return false; }
              }

              var sels = [
                '[data-testid="stSidebarCollapseButton"]',
                '[data-testid="stBaseButton-header"]',
                '[data-testid="baseButton-header"]',
                '[data-testid="stSidebarCollapsedControl"]',
                '[data-testid="collapsedControl"]',
                'button[kind="headerNoPadding"]',
                'button[kind="header"]',
                '[data-testid="stHeader"] button',
                '[data-testid="stExpandSidebarButton"]',
              ];
              for (var i = 0; i < sels.length; i++) {
                var nodes = doc.querySelectorAll(sels[i]);
                for (var j = 0; j < nodes.length; j++) {
                  if (clickEl(nodes[j])) break;
                }
              }

              var side = doc.querySelector('section[data-testid="stSidebar"]');
              if (side) {
                var buttons = side.querySelectorAll("button");
                if (buttons.length) clickEl(buttons[0]);
              }

              var main = doc.querySelector('[data-testid="stAppViewContainer"]')
                      || doc.querySelector(".main")
                      || doc.querySelector('[data-testid="stMain"]');
              if (main) {
                clickEl(main);
                try {
                  main.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
                  main.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
                } catch (e2) {}
              }

              // Force-hide overlay drawer on mobile
              if (side) {
                side.style.setProperty("transform", "translateX(-105%)", "important");
                side.style.setProperty("visibility", "hidden", "important");
                side.style.setProperty("pointer-events", "none", "important");
                side.setAttribute("aria-expanded", "false");
              }
              var backdrop = doc.querySelector('[data-testid="stSidebar"]')
                          || doc.querySelector('[data-testid="stAppViewContainer"] > div');
              doc.querySelectorAll('section[data-testid="stSidebar"]').forEach(function (el) {
                el.style.setProperty("transform", "translateX(-105%)", "important");
                el.style.setProperty("visibility", "hidden", "important");
              });
            } catch (err) {}
          }
          closeSide();
          setTimeout(closeSide, 30);
          setTimeout(closeSide, 100);
          setTimeout(closeSide, 250);
          setTimeout(closeSide, 500);
          setTimeout(closeSide, 900);
          setTimeout(closeSide, 1500);
        })();
        </script>
        """,
        height=1,
        width=1,
    )


def _nav_push(page_key: str):
    """Push current page to history then go to page_key."""
    cur = st.session_state.get("page", "Dashboard")
    hist = list(st.session_state.get("_nav_history") or [])
    if cur and cur != page_key:
        if not hist or hist[-1] != cur:
            hist.append(cur)
        # keep stack short
        st.session_state["_nav_history"] = hist[-20:]
    st.session_state.page = page_key
    st.session_state["_collapse_sidebar"] = True


def _sidebar_go(page_key: str):
    """Click sidebar item → open that page and hide mobile drawer."""
    _nav_push(page_key)
    st.rerun()


def nav_go(page_key: str):
    """Hub card / in-page link → open page, record history, close mobile menu."""
    _nav_push(page_key)
    st.rerun()


def nav_back():
    """Back: history stack first, else parent hub, else Dashboard."""
    hist = list(st.session_state.get("_nav_history") or [])
    if hist:
        prev = hist.pop()
        st.session_state["_nav_history"] = hist
        st.session_state.page = prev
    else:
        cur = st.session_state.get("page", "Dashboard")
        parent = PAGE_PARENT.get(cur, "Dashboard")
        st.session_state.page = parent
    st.session_state["_collapse_sidebar"] = True
    st.rerun()



def render_header_nav():
    """
    Fixed top header (RG-style):
    - Same browser tab only (no new tab) via parent location + target=_self
    - Live market chip + notifications in the header strip
    - Clear single-row formatting
    """
    do_scan = False
    cur = st.session_state.get("page", "Dashboard")

    # --- Query-param navigation ---
    nav = ""
    try:
        raw = st.query_params.get("jp_nav", "")
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else ""
        nav = str(raw or "").strip().lower()
    except Exception:
        nav = ""

    nav_map = {
        "home": "Dashboard",
        "swing": "Swing Hub",
        "tools": "NSE Tools",
        "intraday": "Intraday FO",
        "fo": "Intraday FO",
        "backtest": "Intraday Backtest",
        "paper": "Paper Trading",
        "history": "History",
        "scanner": "Scanner Query",
        "query": "Scanner Query",
        "watchlist": "Watchlist",
    }

    def _clear_jp_nav():
        try:
            q = dict(st.query_params)
            q.pop("jp_nav", None)
            st.query_params.clear()
            for k, v in q.items():
                st.query_params[k] = v
        except Exception:
            try:
                del st.query_params["jp_nav"]
            except Exception:
                pass

    if nav == "scan":
        do_scan = True
        _clear_jp_nav()
    elif nav in nav_map:
        target = nav_map[nav]
        prev = st.session_state.get("page", "Dashboard")
        if prev != target:
            try:
                hist = list(st.session_state.get("_nav_history") or [])
                hist.append(prev)
                st.session_state["_nav_history"] = hist[-20:]
            except Exception:
                pass
            st.session_state.page = target
        _clear_jp_nav()
        cur = st.session_state.get("page", "Dashboard")

    try:
        mkt = market_status_text()
    except Exception:
        mkt = "NSE"
    n_res = 0
    try:
        n_res = len(st.session_state.results) if st.session_state.results is not None else 0
    except Exception:
        n_res = 0
    scan_chip = f"{n_res} scanned" if n_res else "Scan ready"
    try:
        open_now = nse_market_open_now()
    except Exception:
        open_now = False
    live_dot = "LIVE" if open_now else "CLOSED"
    live_color = "#16a34a" if open_now else "#dc2626"

    # Notification prefs for header chip
    try:
        prefs = load_notify_prefs()
    except Exception:
        prefs = {"enabled": False}
    notif_on = bool(prefs.get("enabled"))
    notif_label = "🔔 ON" if notif_on else "🔔 OFF"

    def _pill(key: str, label: str) -> str:
        active = False
        if key == "home" and cur == "Dashboard":
            active = True
        elif key == "swing" and cur in (
            "BUY Calls", "SELL Calls", "Strategy Lab", "Sure Calls", "Swing Hub",
            "History", "Paper Trading", "Performance Summary", "Trade Tracker",
            "Holding Advisor", "Portfolio",
        ):
            active = True
        elif key == "tools" and (
            cur == "NSE Tools" or str(cur).startswith("Tool")
            or cur in ("FII DII", "Pattern Hub", "Chart Pattern Scanner", "Multi Timeframe", "Tools Hub", "Find Stock")
        ):
            active = True
        elif key == "intraday" and cur in ("Intraday FO", "Intraday Backtest"):
            active = True
        elif key == "scanner" and cur == "Scanner Query":
            active = True
        cls = "jp-pill jp-pill-on" if active else "jp-pill"
        # Same tab only: target=_self + JS sets parent search (Streamlit iframe-safe)
        return (
            f'<a class="{cls}" href="?jp_nav={key}" target="_self" rel="noopener" '
            f'onclick="try{{var p=window.parent;p.location.search=\'?jp_nav={key}\';return false;}}catch(e){{}}" '
            f'>{label}</a>'
        )

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        button[kind="header"] {{ display: none !important; }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0 !important; min-height: 0 !important;
        }}
        .main .block-container {{
            max-width: 1180px !important;
            padding-top: 4.8rem !important;
            padding-bottom: 2rem !important;
        }}
        .jp-top-fixed {{
            position: fixed !important;
            top: 0 !important; left: 0 !important; right: 0 !important;
            z-index: 999999 !important;
            background: #ffffff !important;
            border-bottom: 1px solid #e8edf5 !important;
            box-shadow: 0 2px 14px rgba(15,23,42,0.07) !important;
        }}
        .jp-top-inner {{
            max-width: 1180px;
            margin: 0 auto;
            padding: 0.45rem 0.9rem;
            display: flex;
            align-items: center;
            gap: 10px 14px;
            flex-wrap: nowrap;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }}
        .jp-brand {{
            display: flex; align-items: center; gap: 8px;
            text-decoration: none !important;
            flex-shrink: 0;
        }}
        .jp-logo {{
            width: 34px; height: 34px; border-radius: 10px;
            background: linear-gradient(135deg,#0ea5e9,#2563eb 55%,#7c3aed);
            color: #fff; font-weight: 900; font-size: 0.78rem;
            display: flex; align-items: center; justify-content: center;
        }}
        .jp-brand-title {{
            font-size: 0.88rem; font-weight: 800; color: #0f172a;
            white-space: nowrap; line-height: 1.15;
        }}
        .jp-brand-sub {{
            font-size: 0.62rem; color: #64748b; font-weight: 600;
            white-space: nowrap;
        }}
        .jp-nav-pills {{
            display: flex; align-items: center; gap: 5px;
            flex: 1; justify-content: center; flex-shrink: 0;
        }}
        .jp-pill {{
            text-decoration: none !important;
            padding: 0.32rem 0.78rem;
            border-radius: 999px;
            font-size: 0.82rem; font-weight: 700;
            color: #334155 !important;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            white-space: nowrap;
            cursor: pointer;
        }}
        .jp-pill:hover {{ background: #e2e8f0; color: #0f172a !important; }}
        .jp-pill-on {{
            background: linear-gradient(135deg,#0ea5e9,#2563eb) !important;
            color: #fff !important;
            border-color: transparent !important;
            box-shadow: 0 2px 8px rgba(37,99,235,0.28);
        }}
        .jp-pill-scan {{
            background: linear-gradient(135deg,#2563eb,#1d4ed8) !important;
            color: #fff !important;
            border-color: transparent !important;
        }}
        .jp-meta {{
            display: flex; align-items: center; gap: 6px;
            flex-shrink: 0; margin-left: auto;
        }}
        .jp-chip {{
            font-size: 0.68rem; font-weight: 700;
            border-radius: 999px; padding: 3px 9px;
            white-space: nowrap;
            border: 1px solid #e2e8f0;
            background: #f8fafc; color: #475569;
        }}
        .jp-chip-live {{
            border-color: {live_color}33;
            color: {live_color};
            background: {live_color}12;
        }}
        .jp-chip-mkt {{
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        @media (max-width: 820px) {{
            .main .block-container {{ padding-top: 5.2rem !important; }}
            .jp-brand-sub {{ display: none; }}
            .jp-chip-mkt {{ display: none; }}
            .jp-top-inner {{ flex-wrap: wrap; overflow-x: visible; }}
            .jp-nav-pills {{ justify-content: flex-start; width: 100%; order: 3; }}
        }}
        </style>
        <div class="jp-top-fixed">
          <div class="jp-top-inner">
            <a class="jp-brand" href="?jp_nav=home" target="_self"
               onclick="try{{window.parent.location.search='?jp_nav=home';return false;}}catch(e){{}}">
              <div class="jp-logo">JP</div>
              <div>
                <div class="jp-brand-title">JP Stock Market Model</div>
                <div class="jp-brand-sub">NSE · Swing · Tools · Intraday</div>
              </div>
            </a>
            <div class="jp-nav-pills">
              {_pill("home", "Home")}
              {_pill("swing", "Swing")}
              {_pill("tools", "Tools")}
              {_pill("intraday", "Intraday / F&amp;O")}
              {_pill("scanner", "Scanner")}
              <a class="jp-pill jp-pill-scan" href="?jp_nav=scan" target="_self"
                 onclick="try{{window.parent.location.search='?jp_nav=scan';return false;}}catch(e){{}}">Scan</a>
            </div>
            <div class="jp-meta">
              <span class="jp-chip jp-chip-live">● {live_dot}</span>
              <span class="jp-chip jp-chip-mkt">{mkt} · {scan_chip}</span>
              <span class="jp-chip">{notif_label}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Notifications + live controls in header zone (compact)
    with st.expander("⚙️ Live prices · notifications · sync", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("**Live market**")
            st.write(f"{'🟢' if open_now else '🔴'} **{live_dot}** · {mkt}")
            st.session_state.live_prices_on = st.checkbox(
                "Auto live prices (uses data quota — keep OFF on Cloud)",
                value=bool(st.session_state.get("live_prices_on", False)),
                key="hdr_live_on",
            )
            _opts = [30, 60, 120, 300, 600]
            _cur_sec = int(st.session_state.get("refresh_seconds") or 60)
            if _cur_sec not in _opts:
                _cur_sec = 60
            st.session_state.refresh_seconds = int(
                st.selectbox("Refresh (sec)", _opts, index=_opts.index(_cur_sec), key="refresh_secs_select")
            )
            st.markdown("**Scan type** (light = safe to run often)")
            _modes = {
                "Nifty 50 (light)": "nifty50",
                "Nifty 200 (light)": "nifty200",
                "Bank Nifty stocks (light)": "banknifty",
                "Sector-wise (light)": "sector",
                "Master — full market (heavy)": "master",
            }
            _labels = list(_modes.keys())
            _cur_mode = st.session_state.get("scan_mode", "nifty50")
            _cur_label = next((k for k, v in _modes.items() if v == _cur_mode), _labels[0])
            _pick = st.selectbox(
                "Universe",
                _labels,
                index=_labels.index(_cur_label) if _cur_label in _labels else 0,
                key="hdr_scan_mode_label",
                help="Nifty / Bank / Sector can be run multiple times. Master is heavy — use rarely.",
            )
            st.session_state.scan_mode = _modes[_pick]
            if st.session_state.scan_mode == "sector":
                _secs = sorted({str(v) for v in (SECTOR_MAP.values() if SECTOR_MAP else []) if v and str(v) != "Other"})
                if not _secs:
                    _secs = ["Banking", "IT", "Pharma", "Automobile", "FMCG", "Energy", "Metals", "Financial Services"]
                st.session_state.scan_sector = st.selectbox(
                    "Sector",
                    _secs,
                    index=0,
                    key="hdr_scan_sector",
                )
            else:
                st.session_state.scan_sector = ""
            st.caption(
                "Header **Scan** runs the selected type. "
                "Light scans ≈ low Cloud load. Master ≈ high load."
            )
        with c2:
            st.caption("**Notifications**")
            try:
                render_notification_controls("header")
            except Exception:
                st.caption("Notifications unavailable")
        with c3:
            st.caption("**Sync / reload**")
            st.session_state.auto_refresh = st.checkbox(
                "Auto refresh page",
                value=bool(st.session_state.get("auto_refresh", True)),
                key="hdr_auto_ref",
            )
            st.session_state.live_sync = st.checkbox(
                "Live sync (devices)",
                value=bool(st.session_state.get("live_sync", False)),
                key="hdr_live_sync",
            )
            if st.button("📥 Reload scan", use_container_width=True, key="hdr_reload"):
                n_load = load_scan_from_disk(force=True)
                if n_load > 0:
                    st.success(f"Loaded {n_load} stocks")
                    st.rerun()
                else:
                    st.error("No shared scan on server")
            if st.button("🔄 Refresh now", use_container_width=True, key="hdr_manual_ref"):
                try:
                    live_quote.clear()
                except Exception:
                    pass
                try:
                    apply_realtime_sync(force=True, refresh_prices=True)
                except Exception:
                    try:
                        apply_realtime_sync(force=True)
                    except Exception:
                        pass
                st.rerun()

    return do_scan



def render_back_bar():
    """
    Mobile-first fixed ← Back (always on screen without opening sidebar).
    Bottom-left so it does not cover charts/tables. Sidebar keeps a copy for PC.
    """
    cur = st.session_state.get("page", "Dashboard")
    if cur == "Dashboard":
        return
    parent = PAGE_PARENT.get(cur, "Dashboard")
    hist = st.session_state.get("_nav_history") or []
    back_label = PAGE_LABELS.get(hist[-1], None) if hist else PAGE_LABELS.get(parent, parent)
    if not back_label:
        back_label = PAGE_LABELS.get(parent, "Home")
    crumb = PAGE_LABELS.get(cur, cur)
    parent_name = PAGE_LABELS.get(parent, parent)

    st.caption(f"📍 **{parent_name}** › **{crumb}**")

    st.markdown(
        """
        <style>
        .block-container { padding-bottom: 5rem !important; }
        /* Streamlit 1.33+ assigns st-key-<widget_key> on the wrapper */
        div.st-key-nav_back_float,
        div[class*="st-key-nav_back_float"] {
            position: fixed !important;
            bottom: 16px !important;
            left: 12px !important;
            z-index: 10060 !important;
            width: auto !important;
            max-width: 48vw !important;
        }
        div.st-key-nav_back_float button,
        div[class*="st-key-nav_back_float"] button {
            background: linear-gradient(135deg, #0ea5e9, #0369a1) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 800 !important;
            font-size: 0.95rem !important;
            min-height: 2.75rem !important;
            padding: 0.45rem 1.1rem !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.45) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("← Back", key="nav_back_float", help=f"Back to {back_label}"):
        nav_back()


def render_sidebar_back():
    """Compact Back in sidebar (only when not on Dashboard)."""
    cur = st.session_state.get("page", "Dashboard")
    if cur == "Dashboard":
        return
    parent = PAGE_PARENT.get(cur, "Dashboard")
    hist = st.session_state.get("_nav_history") or []
    back_label = PAGE_LABELS.get(hist[-1], None) if hist else PAGE_LABELS.get(parent, parent)
    if not back_label:
        back_label = PAGE_LABELS.get(parent, "Home")
    crumb = PAGE_LABELS.get(cur, cur)
    st.caption(f"Now: **{crumb}**")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("← Back", key="nav_back_btn", use_container_width=True, type="primary",
                     help=f"Back to {back_label}"):
            nav_back()
    with b2:
        if parent != "Dashboard":
            if st.button("⌂ Menu", key="nav_hub_btn", use_container_width=True,
                         help=PAGE_LABELS.get(parent, parent)):
                nav_go(parent)


# Sidebar removed — navigation & scan live in fixed header
scan = False

# Fixed header (no sidebar)
try:
    _hdr_scan = render_header_nav()
    if _hdr_scan:
        scan = True
except Exception as _hdr_err:
    st.warning(f"Header menu issue: {_hdr_err}")
    scan = False

# Explicit scan from Dashboard buttons (Nifty 50 / Bank / 200)
if st.session_state.pop("_force_scan", None):
    scan = True

try:
    render_back_bar()
except Exception:
    pass

# ============================================================
# FULL MARKET SCAN
# ============================================================

if scan:

    with st.spinner(
        f"Scanning {len(NSE_STOCKS):,} NSE stocks..."
    ):

        # Clear stale empty caches so a new scan can succeed
        try:
            _stock_history_cached.clear()
        except Exception:
            pass
        try:
            stock_history.clear()
        except Exception:
            pass
        try:
            download_market_data.clear()
        except Exception:
            pass
        # Default mode nifty50 if unset
        if not st.session_state.get("scan_mode"):
            st.session_state.scan_mode = "nifty50"
        results = run_scanner(
            full_market=bool(st.session_state.get("scan_mode") == "master"),
            mode=st.session_state.get("scan_mode", "nifty50"),
            sector=st.session_state.get("scan_sector", ""),
        )

    if results is None or results.empty:

        # Silent soft retry path — no provider alarm to the user
        st.info(
            "Scan finished with no rows this run — **not** due to a missing Excel. "
            "Stock list is built-in / downloaded automatically. "
            "Price history (NSE/Yahoo) did not load on this server. "
            "Click **Scan** once more, or **Reload scan** if a previous scan is saved."
        )
        try:
            n_load = load_scan_from_disk(force=True)
            if n_load and n_load > 0:
                st.success(f"Loaded previous scan: **{n_load}** stocks")
                st.rerun()
        except Exception:
            pass

    else:

        results = ensure_result_columns(results)
        results = refresh_sector_for_results(results)

        st.session_state.results = results
        st.session_state.last_run = datetime.now()

        # Save CSV so phone/PC on same Cloud URL can share this scan
        n_saved = save_scan_to_disk(results)
        st.session_state["_scan_meta"] = {
            "rows": n_saved,
            "saved_at_ist": india_now().strftime("%Y-%m-%d %H:%M:%S IST"),
        }
        if n_saved:
            st.caption(f"Shared scan saved on server: **{n_saved}** stocks → use **Reload shared scan** on other device.")

        save_recommendations(results)
        evaluate_history()
        try:
            learn_from_history(min_closed=5)
        except Exception:
            pass
        try:
            sync_auto_trades_tracker(max_live=25)
        except Exception:
            pass

        # Sector coverage summary
        try:
            sec_counts = results["Sector"].value_counts()
            top_secs = ", ".join(
                f"{k}: {v}" for k, v in sec_counts.head(8).items()
            )
            st.success(
                f"Scan completed: {len(results):,} stocks. Sectors → {top_secs}"
            )
        except Exception:
            st.success(f"Scan completed: {len(results):,} stocks.")


# ============================================================
# RESULTS
# ============================================================

results = ensure_result_columns(st.session_state.results)
results = refresh_sector_for_results(results)
st.session_state.results = results


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.title(
        "📈 JP STOCK MARKET MODEL"
    )

    st.caption(
        "All stocks • Live market mode • Closing-price mode • "
        "Prediction history • Risk • Target • Stop Loss • Live prices"
    )

    live_market()

    # Quick access to index charts + BUY/SELL
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("📈 NIFTY Chart + Analysis", use_container_width=True, key="dash_nifty"):
            st.session_state.page = "Nifty Analysis"
            st.rerun()
    with q2:
        if st.button("🏦 BANK NIFTY Chart + Analysis", use_container_width=True, key="dash_bn"):
            st.session_state.page = "BankNifty Analysis"
            st.rerun()
    with q3:
        if st.button("🟢 BUY Calls", use_container_width=True, key="dash_buy"):
            st.session_state.page = "BUY Calls"
            st.rerun()
    with q4:
        if st.button("🔴 SELL Calls", use_container_width=True, key="dash_sell"):
            st.session_state.page = "SELL Calls"
            st.rerun()

    if st.button("⭐ Watchlist", use_container_width=True, key="dash_wl"):
        st.session_state.page = "Watchlist"
        st.rerun()

    try:
        render_index_lists_home()
    except Exception as _ix_err:
        st.caption(f"Index lists note: {_ix_err}")

    st.divider()

    # Precision 1–2 BUY picks on dashboard (fallback to high WATCH if no BUY)
    if not results.empty:
        _buys = results[results["Call"].astype(str).str.upper() == "BUY"]
        if _buys.empty:
            _buys = results[
                (results["Call"].astype(str).str.upper() == "WATCH")
                & (pd.to_numeric(results.get("Prediction"), errors="coerce").fillna(0) >= 65)
            ]
            if not _buys.empty:
                st.info(
                    "No strong-trend BUY in last scan — showing top near-buy **WATCH** names so the board is not empty."
                )
        if not _buys.empty:
            try:
                _picks = pick_precision_buys(_buys, n=2)
            except Exception as _pe:
                st.caption(f"Precision picks note: {_pe}")
                _picks = _buys.head(2)
            if _picks is not None and not _picks.empty:
                st.subheader("🎯 Today’s Precision BUY Picks (1–2 stocks)")
                st.caption(
                    "Shortlist — high prediction, controlled risk. Run a fresh FULL MARKET SCAN if list looks stale."
                )
                cols = st.columns(len(_picks))
                for i, (_, prow) in enumerate(_picks.iterrows()):
                    with cols[i]:
                        st.markdown(
                            f"**#{i+1} {prow['Stock']}** ({prow.get('Sector','')})\n\n"
                            f"Pred **{safe_float(prow['Prediction']):.0f}%** · "
                            f"Risk **{safe_float(prow['Risk %']):.1f}%** · "
                            f"Score **{safe_float(prow.get('Precision Score')):.0f}**\n\n"
                            f"₹{safe_float(prow['Price']):,.1f} → "
                            f"T ₹{safe_float(prow['Target']):,.1f} / "
                            f"SL ₹{safe_float(prow['Stop Loss']):,.1f}"
                        )
                        if st.button("Open", key=f"dash_prec_{prow['Stock']}"):
                            st.session_state.selected_stock = prow["Stock"]
                            st.session_state.page = "Stock Analysis"
                            st.rerun()
                if st.button("See all Precision Picks on BUY page →", key="dash_to_buy_prec"):
                    st.session_state.page = "BUY Calls"
                    st.rerun()
                for _, prow in _picks.iterrows():
                    with st.expander(f"🔁 Prior history & learning — {prow['Stock']}", expanded=False):
                        show_prior_call_learning_panel(
                            prow["Stock"], str(prow.get("Reason", ""))
                        )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    # Refresh current prices on scan results (Target/SL locked)
    if not results.empty:
        try:
            results = update_results_with_live_prices(results, max_stocks=30)
            st.session_state.results = results
        except Exception:
            pass
        try:
            refresh_history_current_prices(max_stocks=30)
        except Exception:
            pass

    st.divider()

    # --------------------------------------------------------
    # Scan information
    # --------------------------------------------------------

    if st.session_state.last_run:

        st.info(
            "Last full scan: "
            +
            st.session_state.last_run.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    if results.empty:

        st.warning("No scan results available.")
        st.caption("Use ⚙️ → set **Nifty 50** → header **Scan**. Or buttons below.")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("▶ Nifty 50 scan now", type="primary", key="dash_n50_scan"):
                st.session_state.scan_mode = "nifty50"
                st.session_state["_force_scan"] = True
                st.rerun()
        with b2:
            if st.button("▶ Bank Nifty scan", key="dash_bn_scan"):
                st.session_state.scan_mode = "banknifty"
                st.session_state["_force_scan"] = True
                st.rerun()
        with b3:
            if st.button("▶ Nifty 200 scan", key="dash_n200_scan"):
                st.session_state.scan_mode = "nifty200"
                st.session_state["_force_scan"] = True
                st.rerun()


        st.write(
            "Press **🚀 Scan market** in the top header."
        )

        st.write(
            "After the scan, all usable NSE stocks will appear here."
        )

    else:

        results = ensure_result_columns(results)

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        st.subheader(
            "📋 ALL SCANNED STOCKS"
        )

        f1, f2, f3, f4, f5 = st.columns(5)

        call_filter = f1.selectbox(
            "Call",
            [
                "ALL",
                "BUY",
                "HOLD",
                "SELL",
                "WATCH",
            ]
        )

        risk_filter = f2.selectbox(
            "Risk",
            [
                "ALL",
                "LOW",
                "MEDIUM",
                "HIGH",
                "VERY HIGH",
            ]
        )

        sector_options = [
            "ALL"
        ] + sorted(

results["Sector"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        sector_filter = f3.selectbox(
            "Sector",
            sector_options
        )

        min_prediction = f4.slider(
            "Minimum Prediction %",
            0,
            95,
            0
        )

        display_options = [
            10,
            25,
            50,
            100,
            200,
            500,
            1000,
            "ALL",
        ]

        display_count = f5.selectbox(
            "Stocks to Display",
            display_options,
            index=1
        )

        # ----------------------------------------------------
        # Apply filters
        # ----------------------------------------------------

        filtered = results.copy()

        if call_filter != "ALL":

            filtered = filtered[
                filtered["Call"]
                ==
                call_filter
            ]

        if risk_filter != "ALL":

            filtered = filtered[
                filtered["Risk Level"]
                ==
                risk_filter
            ]

        if sector_filter != "ALL":

            filtered = filtered[
                filtered["Sector"]
                ==
                sector_filter
            ]

        filtered = filtered[
            pd.to_numeric(
                filtered["Prediction"],
                errors="coerce"
            )
            >=
            min_prediction
        ]

        filtered = filtered.sort_values(
            [
                "Prediction",
                "Risk %",
            ],
            ascending=[
                False,
                True,
            ]
        )

        if display_count != "ALL":

            filtered = filtered.head(
                int(display_count)
            )

        st.write(
            f"Showing {len(filtered):,} stocks "
            f"from {len(results):,} scanned stocks."
        )

        # ----------------------------------------------------
        # Main table
        # ----------------------------------------------------

        table_columns = [
            "Rank",
            "Stock",
            "Sector",
            "Price",
            "Current Price",
            "Call",
            "Prediction",
            "Risk %",
            "Risk Level",
            "Target",
            "Stop Loss",
            "Hold Days",
            "Priority",
            "RSI",
            "ADX",
        ]
        st.caption(
            "**Price / Current Price** = live or last close. "
            "**Target & Stop Loss** stay as set at scan time (locked)."
        )

        available_columns = [
            c
            for c in table_columns
            if c in filtered.columns
        ]

        st.dataframe(
            filtered[
                available_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Individual stock buttons
        # ----------------------------------------------------

        st.subheader(
            "📈 Open Detailed Stock Analysis"
        )

        # Show buttons for selected filtered results.
        # This allows a separate analysis page.

        for _, row in filtered.head(
            100
        ).iterrows():

            stock = row[
                "Stock"
            ]

            call = row[
                "Call"
            ]

            prediction = row[
                "Prediction"
            ]

            if st.button(
                f"📈 {stock} | {call} | Prediction {prediction}%",
                key="dashboard_"
                + str(stock)
            ):

                st.session_state.selected_stock = (
                    stock
                )

                st.session_state.page = (
                    "Stock Analysis"
                )

                st.rerun()

        # ----------------------------------------------------
        # Quick SELL CALL
        # ----------------------------------------------------

        st.subheader(
            "🔴 Current SELL Calls"
        )

        sells = results[
            results["Call"]
            ==
            "SELL"
        ].copy()

        if sells.empty:

            st.success(
                "No SELL calls currently detected."
            )

        else:

            sells = sells.sort_values(
                [
                    "Prediction",
                    "Risk %",
                ],
                ascending=[
                    True,
                    False,
                ]
            )

            st.dataframe(
                sells[
                    [
                        "Rank",
                        "Stock",
                        "Sector",
                        "Price",
                        "Prediction",
                        "Risk %",
                        "Risk Level",
                        "Target",
                        "Stop Loss",
                        "Hold Days",
                        "Reason",
                    ]
                ].head(50),
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # Sector priority
        # ----------------------------------------------------

        st.subheader(
            "🏭 Sector Priority"
        )

        sec = sector_table(
            results
        )

        if not sec.empty:

            st.dataframe(
                sec.head(20),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FIND STOCK
# ============================================================

elif st.session_state.page == "Find Stock":

    show_find_stock(
        results
    )


# ============================================================
# BUY CALLS
# ============================================================

elif st.session_state.page == "BUY Calls":

    show_buy_calls(
        results
    )


# ============================================================
# SELL CALLS
# ============================================================

elif st.session_state.page == "SELL Calls":

    show_sell_calls(
        results
    )


elif st.session_state.page == "Sure Calls":

    show_sure_calls_page(results)


elif st.session_state.page == "Strategy Lab":

    show_strategy_lab(results)


# ============================================================
# NIFTY / BANK NIFTY
# ============================================================

elif st.session_state.page == "Nifty Analysis":

    show_index_page("NIFTY")


elif st.session_state.page == "BankNifty Analysis":

    show_index_page("BANKNIFTY")


# ============================================================
# SECTOR
# ============================================================

elif st.session_state.page == "Sector Analysis":

    show_sector_page(
        results
    )


# ============================================================
# PORTFOLIO
# ============================================================

elif st.session_state.page == "Holding Advisor":

    show_holding_advisor(results)


elif st.session_state.page == "Portfolio":

    show_portfolio(
        results
    )


# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "History":

    show_history()


elif st.session_state.page == "Trade Tracker":

    show_trade_tracker()


elif st.session_state.page == "Swing Hub":
    show_swing_hub()

elif st.session_state.page == "Performance Summary":
    show_performance_summary()

elif st.session_state.page == "Pattern Hub":
    show_pattern_hub()

elif st.session_state.page == "Tools Hub":
    show_tools_hub()

elif st.session_state.page == "FII DII":
    show_fii_dii_page()

elif st.session_state.page == "NSE Tools":
    show_nse_tools_workspace()

elif st.session_state.page == "Intraday FO":
    show_intraday_fo_desk()

elif st.session_state.page == "Watchlist":
    show_watchlist_page()

elif st.session_state.page == "Scanner Query":
    show_scanner_query_page(st.session_state.results if st.session_state.get("results") is not None else pd.DataFrame())

elif st.session_state.page == "Intraday Backtest":
    show_intraday_backtest_page()

elif st.session_state.page == "Tool Square9":
    show_tool_square9()

elif st.session_state.page == "Tool IV":
    show_tool_iv()

elif st.session_state.page == "Tool OI Chain":
    show_tool_oi_chain()

elif st.session_state.page == "Tool Option Calculator":
    show_tool_option_calculator()

elif st.session_state.page == "Tool Option History":
    show_tool_option_history()

elif st.session_state.page == "Tool Option Backtester":
    show_tool_option_backtester()

elif st.session_state.page == "Tool Candles":
    show_tool_candles()

elif st.session_state.page == "Tool Momentum":
    show_tool_momentum_divergence()



elif st.session_state.page == "Chart Pattern Scanner":
    show_chart_pattern_scanner(
        st.session_state.results if st.session_state.get("results") is not None else pd.DataFrame()
    )

elif st.session_state.page == "Multi Timeframe":
    show_multi_timeframe(
        st.session_state.results if st.session_state.get("results") is not None else pd.DataFrame()
    )

elif st.session_state.page == "Paper Trading":

    show_paper_trading()


# ============================================================
# STOCK ANALYSIS
# ============================================================

elif st.session_state.page == "Stock Analysis":

    st.title(
        "🔍 ANALYZE ANY NSE STOCK"
    )

    typed = st.text_input(
        "Type NSE stock symbol",
        value=st.session_state.selected_stock,
        placeholder=(
            "RELIANCE, TCS, SBIN, INFY, TATAMOTORS"
        )
    )

    if st.button(
        "🔎 ANALYZE NOW",
        type="primary"
    ):

        if typed.strip():

            st.session_state.selected_stock = (
                display_symbol(
                    typed
                )
            )

            st.rerun()

    st.caption(
        "This page works even when the market is closed. "
        "It uses the latest available closing data."
    )

    if st.session_state.selected_stock:

        show_stock(
            st.session_state.selected_stock
        )


# ============================================================
# LIVE PRICES + SYNC + AUTO REFRESH
# Essential for Intraday / F&O + swing books — auto LTP everywhere trading matters
# ============================================================

_LIVE_PRICE_PAGES = {
    "Dashboard", "Intraday FO", "Swing Hub",
    "BUY Calls", "SELL Calls", "Sure Calls", "Strategy Lab",
    "History", "Paper Trading", "Trade Tracker", "Portfolio",
    "Find Stock", "Stock Analysis", "Nifty Analysis", "BankNifty Analysis",
    "Chart Pattern Scanner", "NSE Tools", "Holding Advisor",
}
_cur_page = st.session_state.get("page", "Dashboard")
_on_live_page = _cur_page in _LIVE_PRICE_PAGES

_lp_on = st.session_state.get("live_prices_on", True)
_auto = st.session_state.get("auto_refresh", True)
_live_sync = st.session_state.get("live_sync", False)
_secs = int(st.session_state.get("refresh_seconds", LIVE_REFRESH_SECONDS) or 60)
if _secs < 30:
    _secs = 30
# Intraday / F&O: force minimum practical refresh (1 second — 0.1s not feasible)
try:
    if _cur_page == "Intraday FO":
        _secs = min(max(1, _secs), 2) if nse_market_open_now() else min(max(1, _secs), 5)
        st.session_state.refresh_seconds = _secs
except Exception:
    if _cur_page == "Intraday FO":
        _secs = 1
if _live_sync:
    _secs = min(int(st.session_state.get("live_sync_secs", 5) or 5), _secs)

_now_ts = time.time()
_last_ts = float(st.session_state.get("_last_price_refresh_ts") or 0)
# Allow 1s throttle on intraday; other pages keep a slightly softer floor
if _cur_page == "Intraday FO":
    _throttle = max(1, min(_secs, 2))
else:
    _throttle = max(2, min(_secs, 12))

if _on_live_page and _lp_on and ((_now_ts - _last_ts) >= _throttle):
    try:
        if _cur_page == "History":
            refresh_history_current_prices(max_stocks=120)
            st.session_state["_last_price_refresh"] = india_now().strftime("%H:%M:%S IST")
        else:
            # All books: scan results, strategy, paper, history open rows
            try:
                live_df = st.session_state.get("live_strat_df")
                if live_df is None or (isinstance(live_df, pd.DataFrame) and live_df.empty):
                    if STRATEGY_LIVE_FILE.exists():
                        live_df = pd.read_csv(STRATEGY_LIVE_FILE)
                if live_df is not None and not live_df.empty:
                    st.session_state["_tmp_live_df"] = live_df
            except Exception:
                pass
            refresh_live_prices_all_books(max_per_book=100)
            # Keep index quotes fresh on dashboard / F&O
            try:
                if st.session_state.get("results") is not None and not st.session_state.results.empty:
                    st.session_state.results = update_results_with_live_prices(
                        st.session_state.results, max_stocks=80
                    )
            except Exception:
                pass
        st.session_state["_last_price_refresh_ts"] = _now_ts
    except Exception:
        pass

if _live_sync and ((_now_ts - _last_ts) >= 3):
    try:
        apply_realtime_sync(force=False, refresh_prices=_on_live_page and _lp_on)
        if _on_live_page:
            st.session_state["_last_price_refresh_ts"] = _now_ts
    except Exception:
        pass

try:
    scan_and_notify_outcomes()
    flush_notifications()
except Exception:
    pass

_lp = st.session_state.get("_last_price_refresh", "")
if _on_live_page and _lp_on and _lp:
    st.caption(
    f"💹 Live prices · last **{_lp}** · every **{_secs}s**"
    + (" · **Intraday fast feed**" if _cur_page == "Intraday FO" else "")
)

# Auto page reload so LTP keeps moving (Intraday needs this)
if (_on_live_page and (_lp_on or _auto)) or _live_sync:
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {_secs * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "JP Stock Market Model • "
    +
    (
        "LIVE MARKET ANALYSIS"
        if nse_market_open_now()
        else
        "MARKET CLOSED • USING LATEST AVAILABLE CLOSING DATA"
    )
    +
    " • Research and decision-support only."
)