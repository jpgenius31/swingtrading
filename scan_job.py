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
            "Option": "BUY ATM CALL",
            "Entry (candle)": round(h, 2),  # stock ref; overwritten by ATM premium in run
            "Stop": round(l, 2),
            "Target": round(h + (h - l), 2),
            "Candle": "GREEN",
            "Body %": round(body_pct, 1),
            "spot_ref": round(c, 2),
            "Reason": f"Big GREEN 1m · ATM CE on option premium",
        }
    if c < o:
        return {
            "Signal": "PE",
            "Option": "BUY ATM PUT",
            "Entry (candle)": round(l, 2),
            "Stop": round(h, 2),
            "Target": round(l - (h - l), 2),
            "Candle": "RED",
            "Body %": round(body_pct, 1),
            "spot_ref": round(c, 2),
            "Reason": f"Big RED 1m · ATM PE on option premium",
        }
    return None



def near_atm_option_premium(symbol: str, side: str, spot: float = None) -> dict:
    """Near-ATM CE/PE premium for GitHub 9:15 job."""
    import yfinance as yf
    sym = str(symbol).upper().replace(".NS", "")
    out = {"premium": None, "strike": None, "expiry": None, "entry": None, "target": None, "stop": None}
    try:
        if not spot or spot <= 0:
            h = yf.Ticker(f"{sym}.NS").history(period="5d", interval="1d")
            if h is not None and not h.empty:
                spot = float(h["Close"].iloc[-1])
        if not spot:
            return out
        if spot < 250:
            step = 5.0
        elif spot < 1000:
            step = 10.0
        elif spot < 2500:
            step = 20.0
        else:
            step = 50.0
        atm = round(spot / step) * step
        t = yf.Ticker(f"{sym}.NS")
        exps = list(t.options or [])
        prem = None
        if exps:
            out["expiry"] = str(exps[0])
            chain = t.option_chain(exps[0])
            table = chain.calls if str(side).upper() in ("CE", "CALL") else chain.puts
            if table is not None and not table.empty:
                strikes = table["strike"].astype(float)
                i = (strikes - atm).abs().idxmin()
                row = table.loc[i]
                out["strike"] = float(row["strike"])
                last = float(row.get("lastPrice") or 0)
                bid = float(row.get("bid") or 0)
                ask = float(row.get("ask") or 0)
                if last > 0:
                    prem = last
                elif bid > 0 and ask > 0:
                    prem = (bid + ask) / 2
                elif ask > 0:
                    prem = ask
        if not prem or prem <= 0:
            prem = max(0.5, round(float(spot) * 0.006, 2))
        out["premium"] = round(float(prem), 2)
        out["entry"] = out["premium"]
        out["target"] = round(out["premium"] * 1.075, 2)  # +7.5% (5–10% band)
        out["stop"] = round(max(0.05, out["premium"] * 0.94), 2)  # −6%
    except Exception:
        if spot:
            prem = max(0.5, round(float(spot) * 0.006, 2))
            out.update({"premium": prem, "entry": prem, "target": round(prem * 1.075, 2), "stop": round(prem * 0.94, 2)})
    return out


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
            spot = float(h["Close"].iloc[-1])
            side = sig.get("Signal") or "CE"
            opt = near_atm_option_premium(sym, side, spot=spot)
            entry = opt.get("entry") or max(0.5, round(spot * 0.006, 2))
            tgt = opt.get("target") or round(entry * 1.075, 2)
            sl = opt.get("stop") or round(entry * 0.94, 2)
            out_rows.append({
                "Stock": sym,
                "List": tag,
                "Day %": None,
                "Signal": side,
                "Option": (
                    f"Buy {side}{int(opt['strike']) if opt.get('strike') and float(opt['strike'])==int(float(opt['strike'])) else opt.get('strike')} at ₹{entry}"
                    if opt.get("strike") else f"Buy ATM {side} at ₹{entry}"
                ),
                "Strike": opt.get("strike"),
                "Expiry": opt.get("expiry"),
                "Entry (candle)": entry,
                "Stop": sl,
                "Target": tgt,
                "Candle": sig.get("Candle"),
                "Body %": sig.get("Body %"),
                "Reason": f"{sig.get('Reason')} · option premium entry ₹{entry} · tgt ₹{tgt} · SL ₹{sl}",
                "Underlying LTP": round(spot, 2),
                "LTP": entry,
                "Instrument": "OPTION_ATM",
                "Strategy": "9:15–9:30 · Big 1-min · near-ATM option",
                "Call Source": "INTRADAY_915_CANDLE_OPT",
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
    
    # Auto chart-pattern snapshot (Double Top/Bottom etc.) for instant app load
    try:
        pat_rows = []
        # Reuse mover universe as pattern universe
        univ = list({s for s, _ in universe})[:40]
        for sym in univ:
            try:
                h = yf.Ticker(yf_sym(sym)).history(period="6mo", interval="1d")
                if h is None or len(h) < 50:
                    continue
                close = h["Close"].astype(float)
                low = h["Low"].astype(float)
                high = h["High"].astype(float)
                # Simple double bottom: two similar lows in last 40 bars
                window = low.tail(40)
                i1 = int(window.values.argmin())
                # second lowest away from first
                vals = list(window.values)
                idxs = sorted(range(len(vals)), key=lambda i: vals[i])
                if len(idxs) < 2:
                    continue
                a, b = idxs[0], idxs[1]
                if abs(a - b) < 5:
                    continue
                p1, p2 = float(vals[a]), float(vals[b])
                if abs(p1 - p2) / max(p1, 1e-9) > 0.03:
                    continue
                # neckline = max between the two lows
                lo, hi = min(a, b), max(a, b)
                neck = float(high.tail(40).iloc[lo:hi+1].max()) if hi > lo else float(high.iloc[-1])
                fit = 90 - abs(p1 - p2) / max(p1, 1e-9) * 100
                n = len(h)
                # map window indices to full
                base = n - 40
                pts = [("L1", base + min(a, b), min(p1, p2)), ("Neck", base + (lo + hi) // 2, neck), ("L2", base + max(a, b), max(p1, p2))]
                pat_rows.append({
                    "stock": sym,
                    "pattern": "Double Bottom",
                    "direction": "BULLISH",
                    "fit": round(fit, 0),
                    "points": pts,
                    "points_abs": pts,
                    "start_i": base + min(a, b),
                    "end_i": base + max(a, b),
                    "sessions": abs(a - b),
                    "sessions_ago": max(0, 39 - max(a, b)),
                    "date_range": f"{h.index[base+min(a,b)].strftime('%d-%b-%Y')} → {h.index[base+max(a,b)].strftime('%d-%b-%Y')}",
                    "entry": round(neck, 2),
                    "stop": round(min(p1, p2) * 0.98, 2),
                    "target": round(neck + (neck - min(p1, p2)), 2),
                    "explain": f"Double bottom near ₹{min(p1,p2):.1f}; neckline ₹{neck:.1f}",
                    "rules": "Two equal lows + neckline break bias",
                    "timeframe": "Daily",
                    "interval": "1d",
                    "live": round(float(close.iloc[-1]), 2),
                    "gap_pct": 0,
                    "extra_patterns": [],
                })
            except Exception:
                continue
            time.sleep(0.05)
        if pat_rows:
            (ROOT / "pattern_scan_results.json").write_text(
                json.dumps(pat_rows, indent=2, default=str), encoding="utf-8"
            )
            (ROOT / "pattern_scan_meta.json").write_text(
                json.dumps({
                    "rows": len(pat_rows),
                    "pattern": "Double Bottom (auto 9:15)",
                    "timeframe": "Daily",
                    "saved_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
                    "source": "scan_job --915",
                }, indent=2),
                encoding="utf-8",
            )
            print(f"OK pattern_scan_results.json · {len(pat_rows)} hits")
    except Exception as e:
        print(f"pattern auto skip: {e}")


    return df




def run_all_pattern_scan(
    limit_stocks: int = 200,
    min_fit: float = 55.0,
    completed_within: int = 45,
    timeframes: list = None,
) -> int:
    """
    Scan ALL structure patterns × ALL stocks (universe) × multiple timeframes.
    Writes pattern_scan_results.json for instant dashboard filter.
    Timeframes: Daily, Weekly, 1h, 30m, 15m (Yahoo-supported).
    """
    import yfinance as yf
    from datetime import datetime, timezone, timedelta
    try:
        from pattern_lib import detect_structure_pattern, CHART_STRUCTURE_PATTERNS
    except Exception as e:
        print(f"pattern_lib import failed: {e}")
        return 0

    # interval → (history period, max bars, label, sessions_ago max)
    TF_SPECS = timeframes or [
        ("1d", "1y", 180, "Daily", 45),
        ("1wk", "2y", 120, "Weekly", 26),
        ("1h", "60d", 160, "1 Hour", 40),
        ("30m", "30d", 160, "30 Min", 48),
        ("15m", "15d", 160, "15 Min", 48),
    ]

    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    # Full liquid universe — Nifty50 + EXTRA (cap by limit)
    symbols = list(dict.fromkeys(list(NIFTY50) + list(EXTRA)))
    if limit_stocks and limit_stocks > 0:
        symbols = symbols[: max(30, int(limit_stocks))]
    patterns = list(CHART_STRUCTURE_PATTERNS.keys())
    print(
        f"Pattern FULL scan · {len(symbols)} stocks × {len(patterns)} patterns × "
        f"{len(TF_SPECS)} TFs · {now.strftime('%Y-%m-%d %H:%M IST')}"
    )

    all_hits = []
    for si, sym in enumerate(symbols):
        ticker = None
        try:
            ticker = yf.Ticker(yf_sym(sym))
        except Exception as e:
            print(f"  skip {sym}: {e}")
            continue
        for interval, period, max_bars, tf_label, tf_within in TF_SPECS:
            try:
                h = ticker.history(period=period, interval=interval, auto_adjust=True)
                if h is None or len(h) < 40:
                    continue
                df = h.tail(int(max_bars)).copy()
                if not all(c in df.columns for c in ("Open", "High", "Low", "Close")):
                    continue
                live = float(df["Close"].iloc[-1])
                within = min(int(completed_within), int(tf_within))
                for pn in patterns:
                    try:
                        hit = detect_structure_pattern(
                            df, pn, swing_order=3, min_fit=float(min_fit)
                        )
                        if not hit:
                            continue
                        if int(hit.get("sessions_ago") or 99) > within:
                            continue
                        pts = hit.get("points") or []
                        all_hits.append({
                            "stock": sym,
                            "pattern": hit.get("pattern") or pn,
                            "direction": hit.get("direction"),
                            "fit": hit.get("fit"),
                            "points": pts,
                            "points_abs": pts,
                            "start_i": hit.get("start_i"),
                            "end_i": hit.get("end_i"),
                            "sessions": hit.get("sessions"),
                            "sessions_ago": hit.get("sessions_ago"),
                            "date_range": hit.get("date_range"),
                            "entry": hit.get("entry"),
                            "stop": hit.get("stop"),
                            "target": hit.get("target"),
                            "explain": hit.get("explain"),
                            "rules": hit.get("rules") or "",
                            "timeframe": tf_label,
                            "interval": interval,
                            "live": round(live, 2),
                            "gap_pct": 0,
                            "extra_patterns": [],
                        })
                    except Exception:
                        continue
            except Exception as e:
                print(f"  {sym} {interval}: {e}")
                continue
        if (si + 1) % 5 == 0:
            print(f"  … {si+1}/{len(symbols)} stocks · {len(all_hits)} hits so far")
        time.sleep(0.05)

    all_hits.sort(key=lambda x: (x.get("sessions_ago", 99), -float(x.get("fit") or 0)))
    out_p = ROOT / "pattern_scan_results.json"
    meta_p = ROOT / "pattern_scan_meta.json"
    out_p.write_text(json.dumps(all_hits, indent=2, default=str), encoding="utf-8")
    from collections import Counter
    counts = Counter(str(h.get("pattern")) for h in all_hits)
    tf_counts = Counter(str(h.get("timeframe")) for h in all_hits)
    meta_p.write_text(json.dumps({
        "rows": len(all_hits),
        "pattern": "ALL",
        "timeframe": "ALL",
        "timeframes": dict(tf_counts),
        "patterns_covered": len(counts),
        "counts": dict(counts),
        "stocks": len(symbols),
        "saved_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
        "source": "scan_job.run_all_pattern_scan_multi_tf",
    }, indent=2), encoding="utf-8")
    print(
        f"OK pattern_scan_results.json · {len(all_hits)} hits · "
        f"{len(counts)} pattern types · TFs={dict(tf_counts)}"
    )
    return len(all_hits)


def evaluate_915_past_on_github(max_rows: int = 120) -> None:
    """Close PENDING 915 rows to TARGET / STOP without Streamlit (GitHub Action)."""
    import yfinance as yf
    from datetime import datetime, timezone, timedelta, time as _time
    path = ROOT / "strategy_915_candle_past.csv"
    if not path.exists():
        return
    try:
        past = pd.read_csv(path)
    except Exception:
        return
    if past is None or past.empty:
        return
    for col in ("Result", "Fail Reason", "Signal", "Stock"):
        if col not in past.columns:
            past[col] = ""
        past[col] = past[col].astype(object)
    if "PnL %" not in past.columns:
        past["PnL %"] = pd.NA
    past["PnL %"] = pd.to_numeric(past["PnL %"], errors="coerce")
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    force = now.time() >= _time(9, 30)
    n = 0
    for idx in list(past.index)[:max_rows]:
        res = str(past.at[idx, "Result"] or "").upper()
        if "TARGET" in res:
            past.loc[idx, "Result"] = "TARGET ACHIEVED"
            continue
        if "STOP" in res:
            past.loc[idx, "Result"] = "STOP LOSS HIT"
            continue
        stock = str(past.at[idx, "Stock"]).upper()
        entry = None
        for c in ("Entry (candle)", "Entry", "LTP"):
            if c in past.columns:
                try:
                    entry = float(past.at[idx, c])
                    if entry and entry > 0:
                        break
                except Exception:
                    pass
        try:
            target = float(past.at[idx, "Target"]) if "Target" in past.columns else 0
        except Exception:
            target = 0
        try:
            stop = float(past.at[idx, "Stop"]) if "Stop" in past.columns else 0
        except Exception:
            stop = 0
        side = str(past.at[idx, "Signal"] if "Signal" in past.columns else "CE").upper()
        if not entry:
            if force:
                past.loc[idx, "Result"] = "STOP LOSS HIT"
                past.loc[idx, "Fail Reason"] = "No entry"
                past.loc[idx, "PnL %"] = 0.0
                n += 1
            continue
        try:
            h = yf.Ticker(yf_sym(stock)).history(period="1d", interval="1m")
            if h is None or len(h) < 2:
                h = yf.Ticker(yf_sym(stock)).history(period="5d", interval="5m")
            if h is None or len(h) < 1:
                if force:
                    past.loc[idx, "Result"] = "STOP LOSS HIT"
                    past.loc[idx, "Fail Reason"] = "No bars"
                    past.loc[idx, "PnL %"] = 0.0
                    n += 1
                continue
            hi = float(h["High"].max())
            lo = float(h["Low"].min())
            last = float(h["Close"].iloc[-1])
            if "CE" in side or "CALL" in side:
                hit_t = target and hi >= target
                hit_s = stop and lo <= stop
                pnl_t = ((target / entry) - 1) * 100 if target else 0
                pnl_s = ((stop / entry) - 1) * 100 if stop else 0
                pnl_x = ((last / entry) - 1) * 100
                won_x = last >= entry
            else:
                hit_t = target and lo <= target
                hit_s = stop and hi >= stop
                pnl_t = ((entry / target) - 1) * 100 if target else 0
                pnl_s = ((entry / stop) - 1) * 100 if stop else 0
                pnl_x = ((entry / last) - 1) * 100 if last else 0
                won_x = last <= entry
            if hit_t and not hit_s:
                past.loc[idx, "Result"] = "TARGET ACHIEVED"
                past.loc[idx, "Fail Reason"] = ""
                past.loc[idx, "PnL %"] = round(pnl_t, 2)
                n += 1
            elif hit_s:
                past.loc[idx, "Result"] = "STOP LOSS HIT"
                past.loc[idx, "Fail Reason"] = f"Stop {stop} before target {target}"
                past.loc[idx, "PnL %"] = round(pnl_s, 2)
                n += 1
            elif force:
                if won_x and pnl_x >= 0:
                    past.loc[idx, "Result"] = "TARGET ACHIEVED"
                    past.loc[idx, "Fail Reason"] = f"9:30 exit profit LTP {last:.2f}"
                else:
                    past.loc[idx, "Result"] = "STOP LOSS HIT"
                    past.loc[idx, "Fail Reason"] = f"9:30 exit LTP {last:.2f}"
                past.loc[idx, "PnL %"] = round(pnl_x, 2)
                n += 1
        except Exception as e:
            if force:
                past.loc[idx, "Result"] = "STOP LOSS HIT"
                past.loc[idx, "Fail Reason"] = type(e).__name__
                past.loc[idx, "PnL %"] = 0.0
                n += 1
    if n:
        past.to_csv(path, index=False)
        # money summary
        ru = past["Result"].astype(str).str.upper()
        wins = int(ru.str.contains("TARGET", na=False).sum())
        losses = int(ru.str.contains("STOP", na=False).sum())
        tot = wins + losses
        summary = {
            "wins": wins,
            "losses": losses,
            "total_closed": tot,
            "win_rate": round(100.0 * wins / tot, 1) if tot else 0,
            "updated_at_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
            "source": "github_eval",
        }
        (ROOT / "strategy_915_money_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"915 eval closed {n} rows · win {wins} / loss {losses} ({summary['win_rate']}%)")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--915", dest="mode_915", action="store_true", help="9:15 open big-candle CE/PE on top movers")
    ap.add_argument("--915-list", dest="list_915", default="both", choices=["both", "gainer", "loser"],
                    help="Top gainer, top loser, or both")
    ap.add_argument("--patterns-only", action="store_true", help="Only run full pattern library scan")
    args = ap.parse_args()
    if getattr(args, 'patterns_only', False):
        n = run_all_pattern_scan(limit_stocks=max(args.limit, 80), min_fit=55.0, completed_within=45)
        raise SystemExit(0 if n >= 0 else 1)
    if args.mode_915:
        print("Starting 9:15–9:30 open strategy job…")
        try:
            run_915_open_strategy(limit=min(args.limit, 25), list_mode=args.list_915)
        except Exception as e:
            print(f"915 candle: {e}")
        try:
            evaluate_915_past_on_github(max_rows=150)
        except Exception as _ev:
            print(f"915 eval: {_ev}")
        try:
            run_all_pattern_scan(limit_stocks=max(args.limit, 80), min_fit=55.0, completed_within=45)
        except Exception as e:
            print(f"pattern scan: {e}")
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
        run_all_pattern_scan(limit_stocks=max(args.limit, 80), min_fit=55.0, completed_within=45)
    except Exception as e:
        print(f"pattern scan after market: {e}")
    try:
        _write_sync_and_meta(int(len(df)))
    except Exception:
        pass


if __name__ == "__main__":
    main()
