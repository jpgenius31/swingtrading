"""Headless chart structure patterns for GitHub auto-scan."""
from __future__ import annotations
import pandas as pd
import numpy as np

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


