#!/usr/bin/env python3
"""
Headless market scan for GitHub Actions / any server without Streamlit UI.

Writes latest_results.csv next to this file so Streamlit Cloud can load it
after the file is committed to the repo (or uploaded).

Usage:
  python scan_job.py              # Nifty-like universe (~150–200 names)
  python scan_job.py --full       # wider list (slower, more rate-limit risk)
  python scan_job.py --limit 80

Schedule this on GitHub Actions so phones/PCs without Python still see results.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Install yfinance: pip install yfinance pandas numpy") from e

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "latest_results.csv"
OUT_META = ROOT / "latest_results_meta.json"

# Liquid universe — expand over time; keep free of Streamlit dependency
NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "BHARTIARTL",
    "LT", "AXISBANK", "KOTAKBANK", "BAJFINANCE", "ASIANPAINT", "HCLTECH", "MARUTI",
    "SUNPHARMA", "TITAN", "NTPC", "TATAMOTORS", "POWERGRID", "ULTRACEMCO", "M&M",
    "WIPRO", "ONGC", "TATASTEEL", "COALINDIA", "JSWSTEEL", "ADANIENT", "TECHM",
    "HINDALCO", "BAJAJFINSV", "NESTLEIND", "INDUSINDBK", "CIPLA", "DRREDDY",
    "BPCL", "EICHERMOT", "APOLLOHOSP", "HEROMOTOCO", "DIVISLAB", "BRITANNIA",
    "TATACONSUM", "BAJAJ-AUTO", "HINDUNILVR", "SBILIFE", "HDFCLIFE", "BEL", "TRENT",
    "ADANIPORTS", "GRASIM",
]
EXTRA = [
    "FEDERALBNK", "PNB", "BANKBARODA", "CANBK", "DMART", "ZOMATO", "PERSISTENT",
    "COFORGE", "LTIM", "PIDILITIND", "HAVELLS", "SIEMENS", "DLF", "IRCTC",
    "TATAPOWER", "RECLTD", "PFC", "TVSMOTOR", "ASHOKLEY", "VEDL", "GAIL", "IOC",
    "GODREJCP", "DABUR", "MARICO", "COLPAL", "BERGEPAINT", "PAGEIND", "DIXON",
    "POLYCAB", "CUMMINSIND", "ABB", "HAL", "BHEL", "NMDC", "SAIL", "HINDCOPPER",
    "INDIGO", "MOTHERSON", "BOSCHLTD", "MRF", "PIIND", "SRF", "AUBANK", "IDFCFIRSTB",
]


def yf_sym(s: str) -> str:
    s = str(s).strip().upper().replace(".NS", "")
    return f"{s}.NS"


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def analyse_one(symbol: str, df: pd.DataFrame) -> dict | None:
    if df is None or df.empty or len(df) < 60:
        return None
    d = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col not in d.columns:
            return None
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["Close"])
    if len(d) < 60:
        return None

    c = d["Close"]
    price = float(c.iloc[-1])
    if price <= 0:
        return None

    ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1]) if len(c) >= 200 else ema50
    r = float(rsi(c).iloc[-1] or 50)
    a = float(atr(d).iloc[-1] or price * 0.02)
    if a <= 0:
        a = price * 0.02

    score = 50
    reasons = []
    if price > ema20:
        score += 8
        reasons.append("Price > EMA20")
    else:
        score -= 5
        reasons.append("Price < EMA20")
    if price > ema50:
        score += 8
        reasons.append("Price > EMA50")
    if price > ema200:
        score += 6
        reasons.append("Price > EMA200")
    if 45 <= r <= 65:
        score += 8
        reasons.append(f"RSI healthy {r:.1f}")
    elif r > 70:
        score -= 4
        reasons.append(f"RSI high {r:.1f}")
    elif r < 35:
        score += 3
        reasons.append(f"RSI low {r:.1f}")

    # simple structure
    hi = float(d["High"].tail(30).max())
    lo = float(d["Low"].tail(30).min())
    if price >= hi * 0.98:
        score += 4
        reasons.append("Near 30d high")
    if price <= lo * 1.02:
        score -= 2
        reasons.append("Near 30d low")

    call = "WATCH"
    if score >= 72 and price > ema20 and price > ema50 and 42 <= r <= 68:
        call = "BUY"
        stop = price - 1.5 * a
        target = price + max(2.5 * a, 1.5 * (price - stop))
    elif score <= 38 and price < ema20 and r > 55:
        call = "SELL"
        stop = price + 1.5 * a
        target = price - 2.5 * a
    else:
        stop = price - 1.5 * a
        target = price + 2.5 * a

    risk_pct = abs(price - stop) / price * 100 if price else 0
    risk_level = "LOW" if risk_pct < 3 else "MEDIUM" if risk_pct < 6 else "HIGH" if risk_pct < 10 else "VERY HIGH"

    pa_bias = "BULLISH" if price > ema50 and ema20 > ema50 else "BEARISH" if price < ema50 and ema20 < ema50 else "NEUTRAL"

    return {
        "Stock": symbol.replace(".NS", ""),
        "Symbol": yf_sym(symbol),
        "Price": round(price, 2),
        "Call": call,
        "Prediction": int(max(0, min(99, score))),
        "Target": round(target, 2),
        "Stop Loss": round(max(stop, 0.05), 2),
        "Risk %": round(risk_pct, 2),
        "Risk Level": risk_level,
        "Hold Days": 15,
        "RSI": round(r, 1),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "PA Bias": pa_bias,
        "PA Support": round(lo, 2),
        "PA Resistance": round(hi, 2),
        "Price Action": f"{pa_bias} | S {lo:.1f} R {hi:.1f}",
        "Reason": "; ".join(reasons[:6]),
        "Technical Reasons": "; ".join(reasons),
        "Patterns": "",
        "Priority": "HIGH" if call == "BUY" and score >= 80 else "MEDIUM" if call == "BUY" else "LOW",
        "Sector": "Other",
        "Scan Source": "github_actions_scan_job",
        "Scan Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


def fetch_history(symbol: str) -> pd.DataFrame:
    sym = yf_sym(symbol)
    for period in ("2y", "1y", "6mo"):
        try:
            df = yf.download(sym, period=period, interval="1d", progress=False, threads=False, auto_adjust=True)
            if df is None or df.empty:
                t = yf.Ticker(sym)
                df = t.history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                # flatten multiindex if any
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                if "Close" in df.columns and len(df) >= 60:
                    return df
        except Exception:
            time.sleep(0.3)
    return pd.DataFrame()


def run(limit: int = 120, full: bool = False) -> pd.DataFrame:
    universe = list(dict.fromkeys(NIFTY50 + EXTRA))
    if full:
        # still capped — true full market needs paid data / local
        universe = universe  # same list unless you extend
    universe = universe[: max(20, int(limit))]
    rows = []
    for i, sym in enumerate(universe):
        try:
            df = fetch_history(sym)
            rec = analyse_one(sym, df)
            if rec:
                rows.append(rec)
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            print(f"… {i+1}/{len(universe)} · usable {len(rows)}")
            time.sleep(0.4)
        else:
            time.sleep(0.15)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    print("Starting headless scan…")
    df = run(limit=args.limit, full=args.full)
    if df is None or df.empty:
        print("ERROR: no rows — providers blocked or network issue")
        # still write empty meta so CI shows failure clearly
        OUT_META.write_text(json.dumps({"rows": 0, "ok": False, "ts": datetime.utcnow().isoformat()}), encoding="utf-8")
        raise SystemExit(1)
    df = df.sort_values(["Prediction", "Risk %"], ascending=[False, True])
    df.to_csv(OUT_CSV, index=False)
    meta = {
        "rows": int(len(df)),
        "ok": True,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "source": "scan_job.py",
        "limit": args.limit,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OK wrote {len(df)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
