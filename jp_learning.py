"""
jp_learning.py - the "learn from mistakes" engine for the JP Stock Market Model.

Pure pandas/numpy (no Streamlit) so it can be tested on its own.

What it fixes in the old learner
--------------------------------
1. NOT repeatable: every run re-added +0.15 / -0.2 to pattern weights for the
   SAME closed trades (and learn_from_history ran a second time on top), so
   weights drifted to the limits just by pressing a button.  Here everything is
   recomputed from scratch from the closed-trade history -> same data, same result.
2. Pattern weights started from 2.0 instead of each pattern's real base weight
   (a bearish "Shooting Star" at -5 could be turned into a positive weight after
   one stop-loss).  Here learning is a bounded SHIFT on top of the base weight.
3. Small samples: 2 trades could move a weight.  Here every estimate is shrunk
   toward the overall average (prior strength PRIOR_K) so it takes real evidence.
4. Market drift: in a rising market every pattern "wins".  Patterns are judged
   against the average outcome, not against zero.
5. Old trades counted the same as last week's.  Recent trades count more
   (half-life HALF_LIFE_DAYS).
6. call_adjustments were stored per call *source* but looked up per BUY/SELL
   key, so they were never used.  Now looked up by source (and sector).
7. Manual "pattern_multipliers" from the feedback trainer were saved but never
   read by scoring.  Now they feed the pattern shift.
8. The same stock stopped out again and again was never remembered.  Now a
   cool-off list penalises repeat losers for a while.
9. Adds: post-mortem tags for every stop-loss ("why we were wrong"), data-driven
   mistake rules (e.g. "RSI >= 70 entries stop out 70% of the time"), and a
   confidence calibration table (what score actually hits target how often).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PRIOR_K = 5.0            # pseudo-trades pulling every estimate toward the average
HALF_LIFE_DAYS = 90.0    # a trade this old counts half as much
MIN_RULE_N = 6           # minimum (weighted) trades before a mistake rule is created
MIN_GROUP_N = 5          # minimum trades before a source/sector adjustment applies
MAX_PATTERN_SHIFT = 3.0
COOLDOWN_LOOKBACK_DAYS = 45
COOLDOWN_DAYS = 21
CALIBRATION_EDGES = [0, 55, 65, 72, 80, 101]

SCHEMA_VERSION = 2


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _num(v, default=np.nan):
    try:
        if v is None:
            return default
        f = float(str(v).replace("%", "").replace(",", "").strip())
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def side_sign(call) -> int:
    """+1 long idea (BUY/HOLD), -1 short idea (SELL), 0 = not a directional call."""
    c = str(call or "").upper().strip()
    if c in ("BUY", "HOLD", "STRONG BUY"):
        return 1
    if c in ("SELL", "STRONG SELL"):
        return -1
    return 0


def outcome_of(result, ret=None):
    """(win 0/1, weight) for a closed trade, or None while still open/unknown.

    Target = clean win, stop = clean loss, holding-period exit counts half as
    much and is judged by its return sign."""
    s = str(result or "").upper()
    if "TARGET ACHIEVED" in s or s == "WIN":
        return 1, 1.0
    if "STOP LOSS" in s or s == "LOSS":
        return 0, 1.0
    if "HOLDING PERIOD" in s:
        r = _num(ret)
        if not math.isnan(r) and r > 0:
            return 1, 0.5
        if not math.isnan(r) and r < 0:
            return 0, 0.5
    return None


def _shrunk(num, den, prior_mean, k=PRIOR_K):
    return (num + k * prior_mean) / (den + k) if (den + k) > 0 else prior_mean


def _split_patterns(cell):
    out = []
    for p in str(cell or "").replace(";", ",").split(","):
        p = p.strip()
        if p and p.lower() not in ("nan", "none") and not p.lower().startswith("no major"):
            out.append(p)
    return out


# ----------------------------------------------------------------------------
# closed trades -> tidy table
# ----------------------------------------------------------------------------
def closed_trades(hist: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """One row per *distinct* closed outcome with features at entry.

    Duplicates (same stock + call + exit day, produced by repeated scans of the
    same idea) are collapsed so one trade is not counted ten times."""
    now = now or datetime.now()
    cols = ["stock", "call", "side", "win", "w_out", "w_time", "weight", "src", "strategy",
            "sector", "patterns", "pred", "risk_level", "risk_pct", "rsi", "adx", "vol",
            "days_taken", "ret", "when", "result"]
    if hist is None or len(hist) == 0 or "Result" not in hist.columns:
        return pd.DataFrame(columns=cols)

    rows = []
    seen = set()
    for _, r in hist.iterrows():
        oc = outcome_of(r.get("Result"), r.get("Return %"))
        if oc is None:
            continue
        side = side_sign(r.get("Call"))
        if side == 0:
            continue
        stock = str(r.get("Stock", "")).upper().replace(".NS", "").strip()
        when = pd.to_datetime(r.get("Evaluation Date") or r.get("Prediction Date"), errors="coerce")
        if pd.isna(when):
            when = pd.to_datetime(r.get("Prediction Date"), errors="coerce")
        key = (stock, str(r.get("Call", "")).upper(),
               "" if pd.isna(when) else when.strftime("%Y-%m-%d"), oc[0])
        if key in seen:
            continue
        seen.add(key)
        if pd.isna(when):
            w_time = 1.0
        else:
            age = max((now - when.to_pydatetime().replace(tzinfo=None)).days, 0)
            w_time = 0.5 ** (age / HALF_LIFE_DAYS)
        src = str(r.get("Call Source", "SCAN") or "SCAN").upper().strip()
        if src in ("", "NAN", "NONE"):
            src = "SCAN"
        sec = str(r.get("Sector", "") or "").strip()
        rows.append({
            "stock": stock,
            "call": str(r.get("Call", "")).upper(),
            "side": side,
            "win": oc[0],
            "w_out": oc[1],
            "w_time": w_time,
            "weight": oc[1] * w_time,
            "src": src,
            "strategy": str(r.get("Strategy", "") or "").strip(),
            "sector": "" if sec.lower() in ("nan", "none") else sec,
            "patterns": _split_patterns(r.get("Patterns")),
            "pred": _num(r.get("Prediction")),
            "risk_level": str(r.get("Risk Level", "") or "").upper().strip(),
            "risk_pct": _num(r.get("Risk %")),
            "rsi": _num(r.get("RSI")),
            "adx": _num(r.get("ADX")),
            "vol": _num(r.get("Volume Ratio")),
            "days_taken": _num(r.get("Days Taken")),
            "ret": _num(r.get("Return %")),
            "when": when,
            "result": str(r.get("Result", "")).upper(),
        })
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------
# feature buckets used for mistake rules
# ----------------------------------------------------------------------------
FEATURES = [
    ("rsi_hot",     "RSI >= 70 at entry (chasing)",         lambda t: t["rsi"] >= 70),
    ("rsi_cold",    "RSI < 40 at entry (catching a knife)", lambda t: t["rsi"] < 40),
    ("adx_weak",    "ADX < 20 (no real trend)",             lambda t: t["adx"] < 20),
    ("adx_strong",  "ADX >= 30 (strong trend)",             lambda t: t["adx"] >= 30),
    ("vol_weak",    "Volume below average (< 1.0x)",        lambda t: t["vol"] < 1.0),
    ("vol_surge",   "Volume surge (>= 2.0x)",               lambda t: t["vol"] >= 2.0),
    ("risk_high",   "HIGH / VERY HIGH risk setup",          lambda t: t["risk_level"] in ("HIGH", "VERY HIGH")),
    ("conv_low",    "Low-conviction score (< 65)",          lambda t: t["pred"] < 65),
]


def _feature_flags(ctx: dict) -> dict:
    t = {
        "rsi": _num(ctx.get("rsi")),
        "adx": _num(ctx.get("adx")),
        "vol": _num(ctx.get("vol")),
        "risk_level": str(ctx.get("risk_level", "") or "").upper().strip(),
        "pred": _num(ctx.get("pred")),
    }
    flags = {}
    for fid, _label, fn in FEATURES:
        try:
            if fid.startswith("rsi") and math.isnan(t["rsi"]):
                continue
            if fid.startswith("adx") and math.isnan(t["adx"]):
                continue
            if fid.startswith("vol") and math.isnan(t["vol"]):
                continue
            if fid == "conv_low" and math.isnan(t["pred"]):
                continue
            if fid.startswith("risk") and not t["risk_level"]:
                continue
            flags[fid] = bool(fn(t))
        except Exception:
            continue
    return flags


# ----------------------------------------------------------------------------
# the one-shot recompute
# ----------------------------------------------------------------------------
def recompute(hist: pd.DataFrame, old: dict | None, base_weights: dict | None = None,
              now: datetime | None = None) -> dict:
    """Rebuild every learned quantity from the closed-trade history.

    Idempotent: the result depends only on (history, manual feedback, now).
    Keys not owned by this engine (manual_feedback, strategy_* ...) are kept."""
    now = now or datetime.now()
    base_weights = base_weights or {}
    learning = dict(old or {})
    tr = closed_trades(hist, now)

    if tr.empty:
        learning.setdefault("history_stats", {})
        learning["schema_version"] = SCHEMA_VERSION
        return learning

    n = len(tr)
    wsum = float(tr["weight"].sum())
    wins_w = float((tr["weight"] * tr["win"]).sum())
    overall_wr = wins_w / wsum if wsum else 0.5
    # side-adjusted outcome: +1 = idea was right, -1 = idea was wrong
    tr["s"] = tr["side"] * (2 * tr["win"] - 1)
    baseline = float((tr["weight"] * tr["s"]).sum() / wsum) if wsum else 0.0

    recent_cut = now - timedelta(days=30)
    rc = tr[tr["when"] >= recent_cut]
    recent_wr = round(float(100.0 * rc["win"].mean()), 1) if len(rc) >= 3 else None

    learning["history_stats"] = {
        "closed": int(n),
        "wins": int(tr["win"].sum()),
        "losses": int((1 - tr["win"]).sum()),
        "target_hits": int(tr["result"].str.contains("TARGET").sum()),
        "stop_hits": int(tr["result"].str.contains("STOP").sum()),
        "win_rate": round(float(100.0 * tr["win"].mean()), 1),
        "weighted_win_rate": round(100.0 * overall_wr, 1),
        "recent_30d_win_rate": recent_wr,
        "recent_30d_closed": int(len(rc)),
    }

    # ---- 1. pattern weights = base + bounded learned shift (+ manual feedback) ----
    manual_mult = (old or {}).get("pattern_multipliers") or {}
    ev = {}
    for _, t in tr.iterrows():
        for p in t["patterns"]:
            e = ev.setdefault(p, [0.0, 0.0, 0, 0])  # evidence, weight, n, wins
            e[0] += t["weight"] * (t["s"] - baseline)
            e[1] += t["weight"]
            e[2] += 1
            e[3] += int(t["win"])
    pw, pstats = {}, {}
    names = set(ev) | set(manual_mult)
    for p in names:
        e, w, k, wn = ev.get(p, [0.0, 0.0, 0, 0])
        shift = _clip(MAX_PATTERN_SHIFT * e / (w + PRIOR_K), -MAX_PATTERN_SHIFT, MAX_PATTERN_SHIFT) if w else 0.0
        try:
            man = _clip((float(manual_mult.get(p, 1.0)) - 1.0) * 5.0, -MAX_PATTERN_SHIFT, MAX_PATTERN_SHIFT)
        except Exception:
            man = 0.0
        base = float(base_weights.get(p, 2.0))
        if abs(shift) < 1e-9 and abs(man) < 1e-9:
            continue
        pw[p] = round(base + shift + man, 2)
        pstats[p] = {"n": k, "wins": wn, "shift": round(shift, 2), "manual_shift": round(man, 2),
                     "base": base, "weight": pw[p]}
    learning["pattern_weights"] = pw
    learning["pattern_stats"] = pstats

    # ---- 2. source + sector adjustments ----
    def group_adj(col, scale, cap, min_n):
        out, stats = {}, {}
        for g, sub in tr.groupby(col):
            if not str(g).strip():
                continue
            sw = float(sub["weight"].sum())
            sn = len(sub)
            post = _shrunk(float((sub["weight"] * sub["win"]).sum()), sw, overall_wr)
            stats[str(g)] = {"n": int(sn), "wins": int(sub["win"].sum()),
                             "win_rate": round(100 * float(sub["win"].mean()), 1),
                             "adj": 0.0}
            if sn >= min_n:
                a = round(_clip(scale * (post - overall_wr), -cap, cap), 2)
                out[str(g)] = a
                stats[str(g)]["adj"] = a
        return out, stats

    learning["call_adjustments"], learning["source_adj_stats"] = group_adj("src", 10.0, 5.0, MIN_GROUP_N)
    learning["sector_adjustments"], learning["sector_adj_stats"] = group_adj("sector", 8.0, 4.0, MIN_GROUP_N)

    # ---- 3. mistake rules (data-driven) ----
    rules = []
    for side, side_name in ((1, "BUY"), (-1, "SELL")):
        sub_all = tr[tr["side"] == side]
        if sub_all.empty:
            continue
        sw_all = float(sub_all["weight"].sum())
        loss_all = 1.0 - float((sub_all["weight"] * sub_all["win"]).sum()) / sw_all if sw_all else 0.5
        for fid, label, fn in FEATURES:
            mask = []
            for _, t in sub_all.iterrows():
                try:
                    col = {"rsi": "rsi", "adx": "adx", "vol": "vol"}.get(fid.split("_")[0])
                    if col and math.isnan(t[col]):
                        mask.append(None)
                    elif fid.startswith("risk") and not t["risk_level"]:
                        mask.append(None)
                    elif fid == "conv_low" and math.isnan(t["pred"]):
                        mask.append(None)
                    else:
                        mask.append(bool(fn(t)))
                except Exception:
                    mask.append(None)
            m = pd.Series(mask, index=sub_all.index, dtype="object")
            hit = sub_all[m == True]  # noqa: E712
            if hit.empty:
                continue
            hw = float(hit["weight"].sum())
            if hw < MIN_RULE_N * 0.5:   # weighted floor (recent trades count more)
                continue
            if len(hit) < MIN_RULE_N:
                continue
            loss_b = _shrunk(float((hit["weight"] * (1 - hit["win"])).sum()), hw, loss_all)
            diff = loss_b - loss_all
            if abs(diff) < 0.10:
                continue
            delta = round(_clip(-30.0 * diff, -6.0, 3.0), 1)
            rules.append({
                "id": f"{side_name}:{fid}", "side": side, "feature": fid, "label": label,
                "n": int(len(hit)), "stop_rate": round(100 * loss_b, 0),
                "baseline_stop_rate": round(100 * loss_all, 0), "delta": delta,
                "text": (f"{side_name} calls with {label}: {int(round(100*loss_b))}% failed "
                         f"(vs {int(round(100*loss_all))}% overall, {len(hit)} trades) -> "
                         f"{'penalty' if delta < 0 else 'bonus'} {delta:+.1f} pts"),
            })
    rules.sort(key=lambda r: r["delta"])
    learning["mistake_rules"] = rules

    # ---- 4. confidence calibration ----
    cal = []
    longs = tr[(tr["side"] == 1) & tr["pred"].notna()]
    for lo, hi in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
        b = longs[(longs["pred"] >= lo) & (longs["pred"] < hi)]
        bw = float(b["weight"].sum())
        cal.append({
            "lo": lo, "hi": hi, "label": f"{lo}-{min(hi, 100)}", "n": int(len(b)),
            "wins": int(b["win"].sum()) if len(b) else 0,
            "hit_rate": round(100 * _shrunk(float((b["weight"] * b["win"]).sum()), bw, overall_wr), 1) if len(b) else None,
        })
    learning["calibration"] = cal

    # ---- 5. post-mortems: why were we wrong? ----
    losses = tr[(tr["win"] == 0) & (tr["result"].str.contains("STOP"))].sort_values("when", ascending=False)
    seen_loss = {}
    pm, tag_counts = [], {}
    for _, t in losses.sort_values("when").iterrows():          # chronological for "repeat offender"
        tags = []
        if not math.isnan(t["days_taken"]) and t["days_taken"] <= 2:
            tags.append("Stopped within 2 days (chased entry / stop too tight)")
        if t["side"] == 1 and not math.isnan(t["rsi"]) and t["rsi"] >= 70:
            tags.append(f"Overbought entry (RSI {t['rsi']:.0f})")
        if t["side"] == -1 and not math.isnan(t["rsi"]) and t["rsi"] <= 30:
            tags.append(f"Oversold short (RSI {t['rsi']:.0f})")
        if not math.isnan(t["adx"]) and t["adx"] < 20:
            tags.append(f"No trend (ADX {t['adx']:.0f})")
        if not math.isnan(t["vol"]) and t["vol"] < 1.0:
            tags.append(f"Weak volume ({t['vol']:.1f}x)")
        if t["risk_level"] in ("HIGH", "VERY HIGH"):
            tags.append(f"{t['risk_level'].title()} risk setup")
        if t["side"] == 1 and not math.isnan(t["pred"]) and t["pred"] < 65:
            tags.append(f"Low-conviction call (score {t['pred']:.0f})")
        if seen_loss.get(t["stock"], 0) >= 1:
            tags.append("Repeat loser (stopped out here before)")
        seen_loss[t["stock"]] = seen_loss.get(t["stock"], 0) + 1
        if not tags:
            tags.append("Market moved against a clean setup")
        for g in tags:
            key = g.split(" (")[0]
            tag_counts[key] = tag_counts.get(key, 0) + 1
        pm.append({
            "stock": t["stock"], "call": t["call"],
            "date": "" if pd.isna(t["when"]) else t["when"].strftime("%Y-%m-%d"),
            "return_pct": None if math.isnan(t["ret"]) else round(t["ret"], 2),
            "tags": tags,
        })
    learning["postmortems"] = list(reversed(pm))[:25]
    learning["mistake_tag_counts"] = dict(sorted(tag_counts.items(), key=lambda kv: -kv[1]))

    # ---- 6. cool-off list for repeat losers ----
    cool = {}
    cutoff = now - timedelta(days=COOLDOWN_LOOKBACK_DAYS)
    recent = tr[tr["when"] >= cutoff]
    manual_bad = {"STOP LOSS HIT", "SHOULD NOT HAVE BEEN BUY"}
    mf_loss = {}
    for fb in (old or {}).get("manual_feedback", []) or []:
        try:
            ts = pd.to_datetime(fb.get("time"), errors="coerce")
            if pd.notna(ts) and ts >= cutoff and fb.get("outcome") in manual_bad:
                s = str(fb.get("stock", "")).upper().replace(".NS", "")
                mf_loss[s] = mf_loss.get(s, 0) + 1
        except Exception:
            pass
    for stock, sub in recent.groupby("stock"):
        L = int((sub["win"] == 0).sum()) + mf_loss.get(stock, 0)
        W = int(sub["win"].sum())
        if L >= 2 and W == 0:
            last = sub[sub["win"] == 0]["when"].max()
            until = last + timedelta(days=COOLDOWN_DAYS)
            if until >= now:
                cool[stock] = {"losses": L, "last_loss": last.strftime("%Y-%m-%d"),
                               "until": until.strftime("%Y-%m-%d")}
    learning["stock_cooldown"] = cool

    # ---- human lessons ----
    lessons = []
    for r in rules[:4]:
        lessons.append("MISTAKE RULE: " + r["text"])
    for tag, c in list(learning["mistake_tag_counts"].items())[:3]:
        lessons.append(f"MOST COMMON MISTAKE: {tag} - {c} stop-losses")
    if cool:
        lessons.append("COOL-OFF: " + ", ".join(sorted(cool)[:8]) + " (repeat stop-losses)")
    learning["mistake_lessons"] = lessons
    learning["schema_version"] = SCHEMA_VERSION
    return learning


# ----------------------------------------------------------------------------
# applying what was learned to a new call
# ----------------------------------------------------------------------------
def adjustment(ctx: dict, learning: dict | None, base_weights: dict | None = None,
               now: datetime | None = None):
    """Score delta (in prediction points) and notes for one stock.

    ctx keys (all optional): patterns[list], call, source, sector, stock, rsi, adx,
    vol, risk_level, pred.  Pattern base weights are NOT added here (the scorer
    already adds them) - only the learned part."""
    learning = learning or {}
    base_weights = base_weights or {}
    now = now or datetime.now()
    notes, delta = [], 0.0
    side = side_sign(ctx.get("call", "BUY")) or 1

    # patterns: only the learned shift
    pw = learning.get("pattern_weights") or {}
    for p in ctx.get("patterns") or []:
        if p in pw:
            shift = float(pw[p]) - float(base_weights.get(p, 2.0))
            if abs(shift) >= 0.3:
                delta += 0.5 * shift
                notes.append(f"{'+' if shift > 0 else '-'} learned {'trust' if shift > 0 else 'distrust'} in {p} ({shift:+.1f})")

    adj = learning.get("call_adjustments") or {}
    src = str(ctx.get("source", "SCAN") or "SCAN").upper()
    if src in adj and abs(adj[src]) >= 0.3:
        delta += float(adj[src])
        notes.append(f"{src} source record {adj[src]:+.1f}")

    sec = str(ctx.get("sector", "") or "").strip()
    sadj = learning.get("sector_adjustments") or {}
    if sec and sec in sadj and abs(sadj[sec]) >= 0.3:
        delta += float(sadj[sec])
        notes.append(f"{sec} sector record {sadj[sec]:+.1f}")

    flags = _feature_flags(ctx)
    for r in learning.get("mistake_rules") or []:
        if r.get("side") == side and flags.get(r.get("feature")):
            delta += float(r["delta"])
            notes.append(("WATCH-OUT: " if r["delta"] < 0 else "EDGE: ") + r["label"] +
                         f" ({int(r['stop_rate'])}% failed, {r['delta']:+.1f})")

    stock = str(ctx.get("stock", "") or "").upper().replace(".NS", "")
    cd = (learning.get("stock_cooldown") or {}).get(stock)
    if cd:
        try:
            if pd.to_datetime(cd["until"]) >= pd.Timestamp(now.date()):
                delta -= 6.0
                notes.append(f"COOL-OFF: stopped out {cd['losses']}x recently, wait until {cd['until']} (-6.0)")
        except Exception:
            pass

    hs = learning.get("history_stats") or {}
    if (hs.get("closed") or 0) >= 15:
        wwr = hs.get("weighted_win_rate")
        if wwr is not None:
            if wwr < 40:
                delta -= 2.0
                notes.append("overall hit rate low -> stricter (-2.0)")
            elif wwr > 60:
                delta += 1.0

    return round(_clip(delta, -12.0, 8.0), 2), notes


def calibrated_hit_rate(pred, learning: dict | None, min_n: int = MIN_GROUP_N):
    """(hit_rate_pct, n) historically seen for long calls with this score, or None."""
    p = _num(pred)
    if math.isnan(p):
        return None
    for b in (learning or {}).get("calibration") or []:
        if b["lo"] <= p < b["hi"] and b.get("hit_rate") is not None and b["n"] >= min_n:
            return b["hit_rate"], b["n"]
    return None


def watchouts(ctx: dict, learning: dict | None, now: datetime | None = None) -> list:
    """Short human warnings for a stock row (for the finder table)."""
    _, notes = adjustment(ctx, learning, None, now)
    return [n for n in notes if n.startswith(("WATCH-OUT", "COOL-OFF"))]
