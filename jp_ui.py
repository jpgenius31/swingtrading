"""
jp_ui.py - quick stock finder + "what the bot learned" panel for the JP Stock Market Model.

Streamlit is passed in as `st` so the filtering/ranking logic stays testable
without a browser.  Nothing here changes or deletes any data - it only reads
the scan results and model_learning.json that the app already produces.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

import jp_learning as JL

CALL_ICON = {"BUY": "🟢", "HOLD": "🟡", "WATCH": "🔵", "SELL": "🔴"}
CALL_ORDER = ["BUY", "HOLD", "WATCH", "SELL"]

SECTOR_WORDS = {
    "bank": "Bank", "banking": "Bank", "software": "IT", "pharma": "Pharma", "auto": "Auto",
    "metal": "Metal", "metals": "Metal", "fmcg": "FMCG", "energy": "Energy", "oil": "Energy",
    "realty": "Realty", "infra": "Infra", "finance": "Finance", "power": "Power",
}
STOPWORDS = {"stock", "stocks", "to", "the", "a", "an", "for", "in", "and", "with", "me", "show",
             "find", "good", "best", "strong", "now", "today", "calls", "call", "only", "rs", "₹"}

SORTS = [
    "Best first (score, then lower risk)",
    "Highest score",
    "Biggest upside to target",
    "Best reward : risk",
    "Lowest risk",
    "A to Z",
]


# ----------------------------------------------------------------------------
# pure logic
# ----------------------------------------------------------------------------
def _f(v, d=np.nan):
    try:
        x = float(v)
        return d if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return d


def parse_query(q: str) -> dict:
    """Turn 'low risk bank stocks to buy under 500' into structured filters."""
    s = str(q or "").lower().strip()
    out = {"terms": [], "calls": [], "risk": [], "max_price": None, "min_price": None, "sectors": []}
    if not s:
        return out
    m = re.search(r"(?:under|below|less than|<=?)\s*(\d+(?:\.\d+)?)", s)
    if m:
        out["max_price"] = float(m.group(1))
        s = s.replace(m.group(0), " ")
    m = re.search(r"(?:above|over|more than|>=?)\s*(\d+(?:\.\d+)?)", s)
    if m:
        out["min_price"] = float(m.group(1))
        s = s.replace(m.group(0), " ")
    if "low risk" in s or "safe" in s:
        out["risk"] = ["LOW", "MEDIUM"]
        s = s.replace("low risk", " ").replace("safe", " ")
    if "high risk" in s:
        out["risk"] = ["HIGH", "VERY HIGH"]
        s = s.replace("high risk", " ")
    words = re.findall(r"[a-z0-9&\-\.]+", s)
    for w in words:
        if w in ("buy", "long", "accumulate", "purchase"):
            out["calls"].append("BUY")
        elif w in ("sell", "short", "exit"):
            out["calls"].append("SELL")
        elif w == "hold":
            out["calls"].append("HOLD")
        elif w == "watch":
            out["calls"].append("WATCH")
        elif w in SECTOR_WORDS:
            out["sectors"].append(SECTOR_WORDS[w])
        elif w in STOPWORDS:
            continue
        else:
            out["terms"].append(w)
    return out


def _first_sentence(text, n=110):
    t = str(text or "").strip()
    if not t or t.lower() == "nan":
        return ""
    t = re.split(r"(?<=[.!?])\s|\n|;", t)[0]
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def enrich(df: pd.DataFrame, learning: dict | None, base_weights: dict | None = None) -> pd.DataFrame:
    """Add easy-to-scan columns: Upside %, R:R, Hit rate, Watch-outs, Why."""
    x = df.copy()
    price = pd.to_numeric(x.get("Price"), errors="coerce")
    tgt = pd.to_numeric(x.get("Target"), errors="coerce")
    sl = pd.to_numeric(x.get("Stop Loss"), errors="coerce")
    x["Upside %"] = ((tgt - price) / price * 100).where(price > 0)
    risk = (price - sl).abs()
    reward = (tgt - price).abs()
    x["R:R"] = (reward / risk).where(risk > 0)
    hit, warn = [], []
    for _, r in x.iterrows():
        h = JL.calibrated_hit_rate(r.get("Prediction"), learning)
        hit.append(np.nan if h is None else h[0])
        ctx = {
            "patterns": JL._split_patterns(r.get("Patterns")),
            "call": r.get("Call", "BUY"), "source": r.get("Call Source", "SCAN"),
            "sector": r.get("Sector", ""), "stock": r.get("Stock", ""),
            "rsi": r.get("RSI"), "adx": r.get("ADX"), "vol": r.get("Volume Ratio"),
            "risk_level": r.get("Risk Level", ""), "pred": r.get("Prediction"),
        }
        try:
            w = JL.watchouts(ctx, learning)
        except Exception:
            w = []
        warn.append(" | ".join(n.replace("WATCH-OUT: ", "⚠ ").replace("COOL-OFF: ", "⏸ ") for n in w))
    x["Hit rate %"] = hit
    x["Watch-outs"] = warn
    x["Why"] = x["Reason"].map(_first_sentence) if "Reason" in x.columns else ""
    return x


def filter_results(df, query="", calls=None, risks=None, sectors=None, min_pred=0.0, max_price=None):
    """Apply typed query + dropdown filters.  Never raises on missing columns."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    x = df.copy()
    pq = parse_query(query)
    calls = list(calls or []) or pq["calls"]
    risks = list(risks or []) or pq["risk"]
    sectors = list(sectors or [])
    if calls and "Call" in x:
        x = x[x["Call"].astype(str).str.upper().isin([c.upper() for c in calls])]
    if risks and "Risk Level" in x:
        x = x[x["Risk Level"].astype(str).str.upper().isin([r.upper() for r in risks])]
    if "Sector" in x:
        sec = x["Sector"].astype(str)
        if sectors:
            x = x[sec.isin(sectors)]
        if pq["sectors"]:
            pat = "|".join(re.escape(s) for s in pq["sectors"])
            x = x[x["Sector"].astype(str).str.contains(pat, case=False, na=False)]
    if min_pred and "Prediction" in x:
        x = x[pd.to_numeric(x["Prediction"], errors="coerce").fillna(0) >= float(min_pred)]
    mp = max_price if max_price else pq["max_price"]
    if mp and "Price" in x:
        x = x[pd.to_numeric(x["Price"], errors="coerce") <= float(mp)]
    if pq["min_price"] and "Price" in x:
        x = x[pd.to_numeric(x["Price"], errors="coerce") >= pq["min_price"]]
    for term in pq["terms"]:
        t = term.upper().replace(".NS", "")
        hay = x["Stock"].astype(str).str.upper() if "Stock" in x else pd.Series("", index=x.index)
        m = hay.str.contains(re.escape(t), na=False)
        for col in ("Symbol", "Sector", "Patterns", "Reason", "Strategy"):
            if col in x:
                m = m | x[col].astype(str).str.upper().str.contains(re.escape(t), na=False)
        x = x[m]
    return x


def sort_results(df, how, query=""):
    if df is None or len(df) == 0:
        return df
    x = df.copy()
    pred = pd.to_numeric(x.get("Prediction"), errors="coerce").fillna(0)
    risk = pd.to_numeric(x.get("Risk %"), errors="coerce").fillna(99)
    if how == "Highest score":
        x = x.assign(_k=-pred).sort_values("_k")
    elif how == "Biggest upside to target":
        x = x.assign(_k=-x["Upside %"].fillna(-999)).sort_values("_k")
    elif how == "Best reward : risk":
        x = x.assign(_k=-x["R:R"].fillna(-999)).sort_values("_k")
    elif how == "Lowest risk":
        x = x.assign(_k=risk).sort_values("_k")
    elif how == "A to Z":
        x = x.sort_values("Stock")
    else:  # best first: score desc, then lower risk, then higher learned hit-rate
        hr = pd.to_numeric(x.get("Hit rate %"), errors="coerce").fillna(0)
        x = x.assign(_a=-pred.round(0), _b=risk, _c=-hr).sort_values(["_a", "_b", "_c"])
    # exact symbol typed -> always first
    t = str(query or "").strip().upper().replace(".NS", "")
    if t and "Stock" in x:
        exact = x["Stock"].astype(str).str.upper() == t
        x = pd.concat([x[exact], x[~exact]])
    return x.drop(columns=[c for c in ("_k", "_a", "_b", "_c") if c in x.columns])


# ----------------------------------------------------------------------------
# Streamlit renderers
# ----------------------------------------------------------------------------
def _fmt_money(v):
    v = _f(v)
    return "" if math.isnan(v) else f"₹{v:,.2f}"


def render_quick_finder(st, results, learning, open_stock, base_weights=None, key="qf", top_n_default=25):
    """One compact block: jump-to-stock, search box, filters, sorted table, top picks."""
    st.subheader("🔎 Find a stock")
    if results is None or len(results) == 0:
        st.info("No scan loaded yet — run a scan (or press Reload scan) and your stocks will appear here.")
        return

    df = enrich(results, learning, base_weights)
    if "Stock" not in df.columns:
        st.warning("Scan has no Stock column.")
        return

    # --- pending open from the jump box (selectbox callback sets this) ---
    pending = st.session_state.pop(f"{key}_open", None)
    if pending:
        open_stock(pending)
        return

    # --- 1. jump straight to one stock by typing a few letters ---
    names = df.sort_values("Stock")
    labels = {r["Stock"]: f"{r['Stock']}  ·  {CALL_ICON.get(str(r['Call']).upper(), '')} {r['Call']}  ·  {r['Sector']}"
              for _, r in names.iterrows()}

    def _jump_cb():
        v = st.session_state.get(f"{key}_jump")
        if v:
            st.session_state[f"{key}_open"] = v
            st.session_state[f"{key}_jump"] = ""

    c1, c2 = st.columns([1, 2])
    with c1:
        st.selectbox("Jump to stock (type to search)", [""] + list(labels.keys()),
                     format_func=lambda s: "— pick a stock —" if s == "" else labels[s],
                     key=f"{key}_jump", on_change=_jump_cb)
    with c2:
        query = st.text_input("Search by name, sector, pattern or plain words",
                              placeholder="e.g.  low risk bank buy   ·   stocks under 500   ·   hammer   ·   RELIANCE",
                              key=f"{key}_q")

    # --- 2. filters ---
    f1, f2, f3, f4 = st.columns([1.3, 1.1, 1.4, 1])
    with f1:
        calls = st.multiselect("Call", CALL_ORDER, key=f"{key}_calls", placeholder="All calls")
    with f2:
        risk_opts = [r for r in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"] if r in set(df["Risk Level"].astype(str).str.upper())]
        risks = st.multiselect("Risk", risk_opts, key=f"{key}_risks", placeholder="Any risk")
    with f3:
        sec_opts = sorted({s for s in df["Sector"].astype(str) if s and s.lower() != "nan"})
        sectors = st.multiselect("Sector", sec_opts, key=f"{key}_sectors", placeholder="All sectors")
    with f4:
        min_pred = st.slider("Min score", 0, 100, 0, 5, key=f"{key}_minpred")
    g1, g2, g3 = st.columns([1.6, 1, 1])
    with g1:
        how = st.selectbox("Sort by", SORTS, key=f"{key}_sort")
    with g2:
        max_price = st.number_input("Max price (₹, 0 = any)", min_value=0.0, value=0.0, step=50.0, key=f"{key}_maxp")
    with g3:
        top_n = st.selectbox("Show", [10, 25, 50, 100, 300], index=[10, 25, 50, 100, 300].index(top_n_default)
                             if top_n_default in (10, 25, 50, 100, 300) else 1, key=f"{key}_n")

    found = filter_results(df, query, calls, risks, sectors, min_pred, max_price or None)
    found = sort_results(found, how, query)

    # --- 3. summary chips ---
    if found is None or len(found) == 0:
        st.warning("No stock matches these filters. Clear a filter or try fewer words.")
        return
    cc = found["Call"].astype(str).str.upper().value_counts()
    chips = "  ·  ".join(f"{CALL_ICON[c]} **{int(cc.get(c, 0))}** {c}" for c in CALL_ORDER if cc.get(c, 0))
    st.caption(f"**{len(found)}** of {len(df)} stocks match   |   {chips}")

    # --- 4. top picks strip ---
    buys = found[found["Call"].astype(str).str.upper() == "BUY"].head(3)
    if len(buys):
        cols = st.columns(len(buys))
        for col, (_, r) in zip(cols, buys.iterrows()):
            with col:
                st.markdown(f"**🟢 {r['Stock']}** · {r['Sector']}")
                up = _f(r.get("Upside %"))
                st.caption(f"Score **{_f(r['Prediction']):.0f}**  ·  {_fmt_money(r['Price'])} → {_fmt_money(r['Target'])}"
                           + ("" if math.isnan(up) else f"  ({up:+.1f}%)"))
                if r.get("Watch-outs"):
                    st.caption(r["Watch-outs"])
                if st.button("Open analysis", key=f"{key}_top_{r['Stock']}", use_container_width=True):
                    open_stock(r["Stock"])
                    return

    # --- 5. the table ---
    view = found.head(int(top_n)).copy()
    view["Call"] = view["Call"].astype(str).str.upper().map(lambda c: f"{CALL_ICON.get(c, '')} {c}")
    cols = [c for c in ["Stock", "Call", "Prediction", "Hit rate %", "Price", "Target", "Stop Loss", "Upside %", "R:R",
                        "Risk Level", "Sector", "Watch-outs", "Why"] if c in view.columns]
    view = view[cols].reset_index(drop=True)
    cfg = {}
    try:
        cc_ = st.column_config
        cfg = {
            "Stock": cc_.TextColumn("Stock", width="small"),
            "Call": cc_.TextColumn("Call", width="small"),
            "Prediction": cc_.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "Hit rate %": cc_.NumberColumn("Past hit rate", help="How often long calls with this score hit target in YOUR closed history (blank = not enough data yet).", format="%.0f%%"),
            "Price": cc_.NumberColumn("Price", format="₹%.2f"),
            "Target": cc_.NumberColumn("Target", format="₹%.2f"),
            "Stop Loss": cc_.NumberColumn("Stop", format="₹%.2f"),
            "Upside %": cc_.NumberColumn("Upside", format="%+.1f%%"),
            "R:R": cc_.NumberColumn("Reward:Risk", format="%.1f"),
            "Watch-outs": cc_.TextColumn("Watch-outs", help="Mistakes the bot learned from past stop-losses that match this stock.", width="medium"),
            "Why": cc_.TextColumn("Why", width="large"),
        }
        cfg = {k: v for k, v in cfg.items() if k in view.columns}
    except Exception:
        cfg = {}

    st.caption("Click a row's left edge to open that stock's full analysis.")
    try:
        ev = st.dataframe(view, column_config=cfg, hide_index=True, use_container_width=True,
                          height=min(60 + 35 * len(view), 560), on_select="rerun",
                          selection_mode="single-row", key=f"{key}_tbl")
        sel = list(getattr(getattr(ev, "selection", None), "rows", []) or [])
        if sel:
            open_stock(str(found.head(int(top_n)).iloc[sel[0]]["Stock"]))
            return
    except TypeError:
        st.dataframe(view, column_config=cfg, hide_index=True, use_container_width=True)
        pick = st.selectbox("Open", [""] + view["Stock"].tolist(), key=f"{key}_openpick")
        if pick and st.button("Open analysis", key=f"{key}_openbtn"):
            open_stock(pick)
            return


def render_learning_report(st, learning, on_relearn=None, key="lr"):
    """Plain-language view of what the bot has learned from its mistakes."""
    learning = learning or {}
    hs = learning.get("history_stats") or {}
    closed = int(hs.get("closed") or 0)
    st.subheader("🧠 What the bot learned from its mistakes")
    if closed < 3:
        st.info("Not enough closed calls yet (needs a few TARGET / STOP-LOSS results). "
                "Keep scanning — the bot starts adjusting once real outcomes arrive. "
                "Until then it uses its base pattern weights only.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Closed calls learned from", closed)
    m2.metric("Overall hit rate", f"{hs.get('win_rate', '–')}%" if closed else "–")
    rw = hs.get("recent_30d_win_rate")
    m3.metric("Last 30 days", f"{rw}%" if rw is not None else "–",
              help="Recent calls count more than old ones when the bot adjusts.")
    m4.metric("Cool-off stocks", len(learning.get("stock_cooldown") or {}))
    if learning.get("updated_at"):
        st.caption(f"Last learned: {learning['updated_at']}")

    rules = learning.get("mistake_rules") or []
    if rules:
        st.markdown("**Rules it now applies to every new call**")
        for r in rules[:8]:
            st.markdown(f"- {'⚠' if r['delta'] < 0 else '✅'} {r.get('text') or r.get('label', '')}")

    pm = learning.get("postmortems") or []
    if pm:
        st.markdown("**Recent stop-losses — why we were wrong**")
        tbl = pd.DataFrame([{"Date": p["date"], "Stock": p["stock"], "Call": p["call"],
                             "Return %": p["return_pct"], "What went wrong": " · ".join(p["tags"])} for p in pm[:12]])
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        tags = learning.get("mistake_tag_counts") or {}
        if tags:
            st.caption("Most common mistakes")
            st.bar_chart(pd.Series(tags).head(8))

    cal = [c for c in (learning.get("calibration") or []) if c["n"]]
    if cal:
        st.markdown("**Is the score trustworthy?** (what each score band actually delivered)")
        st.dataframe(pd.DataFrame([{"Score band": c["label"], "Calls": c["n"], "Hit target %": c["hit_rate"]} for c in cal]),
                     hide_index=True, use_container_width=True)

    cd = learning.get("stock_cooldown") or {}
    if cd:
        st.markdown("**Cool-off list** (stopped out repeatedly — score is reduced until the date)")
        st.dataframe(pd.DataFrame([{"Stock": s, "Stops": v["losses"], "Last stop": v["last_loss"], "Until": v["until"]}
                                   for s, v in sorted(cd.items())]), hide_index=True, use_container_width=True)

    ps = learning.get("pattern_stats") or {}
    if ps:
        top = sorted(ps.items(), key=lambda kv: kv[1]["shift"], reverse=True)
        best = [(p, s) for p, s in top if s["shift"] > 0.2][:4]
        worst = [(p, s) for p, s in reversed(top) if s["shift"] < -0.2][:4]
        a, b = st.columns(2)
        with a:
            st.markdown("**Patterns it trusts more**")
            for p, s in best:
                st.markdown(f"- {p}: {s['shift']:+.1f} ({s['wins']}/{s['n']} worked)")
            if not best:
                st.caption("None yet")
        with b:
            st.markdown("**Patterns it trusts less**")
            for p, s in worst:
                st.markdown(f"- {p}: {s['shift']:+.1f} ({s['wins']}/{s['n']} worked)")
            if not worst:
                st.caption("None yet")

    if on_relearn is not None and st.button("🔄 Re-learn from history now", key=f"{key}_relearn"):
        on_relearn()
        st.success("Learning rebuilt from your closed calls.")
        st.rerun()
