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
    if df is None or df.empty or len(df) < 40:
        return None
    d = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col not in d.columns:
            return None
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["Close"])
    if len(d) < 40:
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
                if "Close" in df.columns and len(df) >= 40:
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



def _write_strategy_and_index(df: pd.DataFrame):
    """Auto strategy book + index snapshot for Streamlit (no manual Strategy Lab run)."""
    try:
        buys = df.copy()
        if "Call" in buys.columns:
            buys = buys[buys["Call"].astype(str).str.upper() == "BUY"]
        if "Prediction" in buys.columns:
            buys["Prediction"] = pd.to_numeric(buys["Prediction"], errors="coerce")
            buys = buys.sort_values("Prediction", ascending=False)
        rows = []
        for _, r in buys.head(60).iterrows():
            rows.append({
                "Stock": str(r.get("Stock") or r.get("Symbol") or "").upper().replace(".NS", ""),
                "Strategy": "GitHub Scan BUY",
                "Side": "BUY",
                "Price": float(r.get("Price") or 0) or None,
                "Entry": float(r.get("Price") or 0) or None,
                "Target": float(r.get("Target") or 0) or None,
                "Stop Loss": float(r.get("Stop Loss") or 0) or None,
                "Hold Days": int(float(r.get("Hold Days") or 15)),
                "Status": "AUTO_GITHUB",
                "Prediction": float(r.get("Prediction") or 0) if r.get("Prediction") == r.get("Prediction") else None,
            })
        if rows:
            sdf = pd.DataFrame(rows)
            sdf.to_csv(ROOT / "strategy_live_signals.csv", index=False)
            (ROOT / "strategy_live_meta.json").write_text(
                json.dumps({"n": len(sdf), "source": "scan_job", "saved_at": datetime.utcnow().isoformat() + "Z"}, indent=2),
                encoding="utf-8",
            )
            print(f"OK strategy_live_signals.csv · {len(sdf)} rows")
    except Exception as e:
        print(f"strategy write skip: {e}")
    # Index levels via yfinance
    try:
        import yfinance as yf
        idx = {}
        for name, sym in (("NIFTY", "^NSEI"), ("BANKNIFTY", "^NSEBANK"), ("SENSEX", "^BSESN")):
            try:
                h = yf.Ticker(sym).history(period="5d")
                if h is not None and len(h) >= 1:
                    last = float(h["Close"].iloc[-1])
                    prev = float(h["Close"].iloc[-2]) if len(h) >= 2 else last
                    idx[name] = {
                        "price": round(last, 2),
                        "pct": round((last / prev - 1) * 100, 2) if prev else 0,
                    }
            except Exception:
                continue
        (ROOT / "index_market_snap.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")
        print(f"OK index_market_snap.json · {list(idx.keys())}")
    except Exception as e:
        print(f"index snap skip: {e}")

def _write_sync_and_meta(n_rows: int):
    """So Streamlit app auto-picks new scan without manual Reload."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    meta = {
        "rows": n_rows,
        "saved_at_ist": now,
        "source": "scan_job.py / GitHub Actions",
        "note": "App auto-loads this file; set Streamlit secret GITHUB_RESULTS_RAW to raw.githubusercontent.com/.../latest_results.csv for live pull without redeploy",
    }
    try:
        OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        sync_p = ROOT / "realtime_sync_version.json"
        ver = 1
        if sync_p.exists():
            try:
                ver = int(json.loads(sync_p.read_text()).get("version") or 0) + 1
            except Exception:
                ver = 1
        sync_p.write_text(
            json.dumps({"version": ver, "reason": "github_scan", "at_ist": now}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass



def _big_candle_signal(o, h, l, c, min_body_pct=55.0, min_range_pct=0.15):
    """Big green → CALL at high; big red → PUT at low."""
    try:
        o, h, l, c = float(o), float(h), float(l), float(c)
    except Exception:
        return None
    rng = max(h - l, abs(o) * 1e-6)
    body = abs(c - o)
    body_pct = 100.0 * body / rng
    range_pct = 100.0 * (h - l) / max(abs(o), 1e-9)
    if body_pct < min_body_pct or range_pct < min_range_pct:
        return None
    if c > o:
        return {
            "Signal": "CE",
            "Option": "BUY CALL",
            "Entry (candle)": round(h, 2),
            "Stop": round(l, 2),
            "Target": round(h + (h - l), 2),
            "Candle": "GREEN",
            "Body %": round(body_pct, 1),
            "Reason": f"Big GREEN 1m · CALL at top ₹{h:.2f}",
        }
    if c < o:
        return {
            "Signal": "PE",
            "Option": "BUY PUT",
            "Entry (candle)": round(l, 2),
            "Stop": round(h, 2),
            "Target": round(l - (h - l), 2),
            "Candle": "RED",
            "Body %": round(body_pct, 1),
            "Reason": f"Big RED 1m · PUT at bottom ₹{l:.2f}",
        }
    return None


def run_915_open_strategy(limit: int = 20, list_mode: str = "both") -> pd.DataFrame:
    """
    9:15 open job: top gainers/losers + 1m big candle CE/PE.
    Runs on GitHub at ~09:15 IST so the app loads signals without manual scan.
    """
    import yfinance as yf
    from datetime import datetime, timezone, timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    print(f"9:15 strategy run · {now.strftime('%Y-%m-%d %H:%M:%S IST')} · mode={list_mode}")

    # Build liquid universe (Nifty50-like) as proxy if we cannot hit NSE gainers API
    syms_g, syms_l = [], []
    mode = str(list_mode).lower()
    try:
        # Yahoo day change on Nifty50+bank as free gainer/loser proxy
        base = NIFTY50 + EXTRA[:30]
        rows = []
        for s in base:
            try:
                h = yf.Ticker(yf_sym(s)).history(period="5d", interval="1d")
                if h is None or len(h) < 2:
                    continue
                last = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2])
                pct = (last / prev - 1) * 100 if prev else 0
                rows.append({"Stock": s, "LTP": last, "Move %": pct})
            except Exception:
                continue
            time.sleep(0.05)
        if rows:
            dfm = pd.DataFrame(rows)
            syms_g = dfm.sort_values("Move %", ascending=False).head(limit)["Stock"].tolist()
            syms_l = dfm.sort_values("Move %", ascending=True).head(limit)["Stock"].tolist()
    except Exception as e:
        print(f"mover list: {e}")
        syms_g = NIFTY50[:limit]
        syms_l = list(reversed(NIFTY50))[:limit]

    universe = []
    if mode in ("gainer", "top gainer", "gainers"):
        universe = [(s, "GAINER") for s in syms_g]
    elif mode in ("loser", "top loser", "losers"):
        universe = [(s, "LOSER") for s in syms_l]
    else:
        universe = [(s, "GAINER") for s in syms_g] + [(s, "LOSER") for s in syms_l]

    out_rows = []
    seen = set()
    for sym, tag in universe:
        if sym in seen:
            continue
        seen.add(sym)
        try:
            # 1-minute bars — critical for 9:15 candle
            h = yf.Ticker(yf_sym(sym)).history(period="1d", interval="1m")
            if h is None or len(h) < 1:
                h = yf.Ticker(yf_sym(sym)).history(period="5d", interval="1m")
            if h is None or len(h) < 1:
                continue
            # FIRST 1-min bar of session only (not a later green recovery bar)
            row = h.iloc[0]
            try:
                idx = h.index
                # prefer 09:15–09:20 IST bars if timestamps allow
                for i, ts in enumerate(idx):
                    try:
                        hh, mm = ts.hour, ts.minute
                        # assume already IST or naive market time
                        if hh == 9 and 15 <= mm <= 20:
                            row = h.iloc[i]
                            break
                        if hh == 3 and 45 <= mm <= 50:  # UTC≈IST-5:30 open window
                            row = h.iloc[i]
                            break
                    except Exception:
                        continue
            except Exception:
                pass
            sig = _big_candle_signal(row["Open"], row["High"], row["Low"], row["Close"])
            if not sig:
                continue
            # Losers → only PUT; Gainers → only CALL
            if tag == "LOSER" and sig.get("Signal") != "PE":
                continue
            if tag == "GAINER" and sig.get("Signal") != "CE":
                continue
            out_rows.append({
                "Stock": sym,
                "List": tag,
                "Day %": None,
                **sig,
                "LTP": round(float(h["Close"].iloc[-1]), 2),
                "Strategy": "9:15–9:30 · Big 1-min candle (Top 20 G/L)",
                "Call Source": "INTRADAY_915_CANDLE",
                "Prediction Date": now.strftime("%Y-%m-%d %H:%M"),
                "Result": "PENDING",
                "Fail Reason": "",
            })
        except Exception as e:
            print(f"  skip {sym}: {e}")
            continue
        time.sleep(0.08)

    df = pd.DataFrame(out_rows)
    live_path = ROOT / "strategy_915_candle_live.csv"
    past_path = ROOT / "strategy_915_candle_past.csv"
    meta_path = ROOT / "strategy_915_candle_meta.json"
    if not df.empty:
        df.to_csv(live_path, index=False)
        # append to past (dedupe same day stock+signal)
        try:
            if past_path.exists():
                past = pd.read_csv(past_path)
            else:
                past = pd.DataFrame()
            today = now.strftime("%Y-%m-%d")
            for _, r in df.iterrows():
                if not past.empty and "Stock" in past.columns and "Prediction Date" in past.columns:
                    m = (
                        (past["Stock"].astype(str).str.upper() == str(r["Stock"]).upper())
                        & past["Prediction Date"].astype(str).str.startswith(today)
                    )
                    if "Signal" in past.columns:
                        m = m & (past["Signal"].astype(str) == str(r.get("Signal")))
                    if m.any():
                        continue
                past = pd.concat([past, pd.DataFrame([r.to_dict()])], ignore_index=True)
            past.to_csv(past_path, index=False)
        except Exception as e:
            print(f"past append: {e}")
        meta_path.write_text(
            json.dumps({
                "rows": len(df),
                "saved_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
                "source": "scan_job.py --915",
                "list_mode": list_mode,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"OK 9:15 candle · {len(df)} signals → {live_path.name}")
        # Money tracker summary for app (how many making money) — without opening dashboard
        try:
            summary = {
                "signals_today": int(len(df)),
                "ce": int((df["Signal"] == "CE").sum()) if "Signal" in df.columns else 0,
                "pe": int((df["Signal"] == "PE").sum()) if "Signal" in df.columns else 0,
                "saved_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
                "note": "Auto signals from F&O-style movers; app evaluates target/stop → money count",
            }
            (ROOT / "strategy_915_money_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
    else:
        print("WARN: no 9:15 big-candle signals this run")
        live_path.write_text("Stock,List,Signal,Option,Result\n", encoding="utf-8")
    try:
        _write_sync_and_meta(len(df))
    except Exception:
        pass
    # index snap still useful at open
    try:
        _write_strategy_and_index(pd.DataFrame())  # may no-op on empty buys
    except Exception:
        pass
    try:
        import yfinance as yf
        idx = {}
        for name, sym in (("NIFTY", "^NSEI"), ("BANKNIFTY", "^NSEBANK"), ("SENSEX", "^BSESN")):
            try:
                hh = yf.Ticker(sym).history(period="5d")
                if hh is not None and len(hh) >= 1:
                    last = float(hh["Close"].iloc[-1])
                    prev = float(hh["Close"].iloc[-2]) if len(hh) >= 2 else last
                    idx[name] = {"price": round(last, 2), "pct": round((last / prev - 1) * 100, 2) if prev else 0}
            except Exception:
                continue
        (ROOT / "index_market_snap.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")
    except Exception:
        pass
    return df



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--915", dest="mode_915", action="store_true", help="9:15 open big-candle CE/PE on top movers")
    ap.add_argument("--915-list", dest="list_915", default="both", choices=["both", "gainer", "loser"],
                    help="Top gainer, top loser, or both")
    args = ap.parse_args()
    if args.mode_915:
        print("Starting 9:15–9:30 open strategy job…")
        df915 = run_915_open_strategy(limit=min(args.limit, 25), list_mode=args.list_915)
        if df915 is None or df915.empty:
            print("9:15 job finished with 0 signals (market may not be open yet on data feed)")
            # do not fail CI — still useful for schedule
            raise SystemExit(0)
        raise SystemExit(0)
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
    _write_strategy_and_index(df)
    try:
        _write_sync_and_meta(int(len(df)))
    except Exception:
        pass


if __name__ == "__main__":
    main()
