"""
jp_symbols.py - fuzzy NSE symbol / company-name search (pure Python, no Streamlit).

Typo-tolerant: "relience" -> RELIANCE, "hdfc bnk" -> HDFCBANK, "rlnc" -> RELIANCE,
"360" -> 360ONE, "tata mot" -> TATAMOTORS, company words ("infosys") -> INFY if the
company name is known.
"""

from __future__ import annotations

import difflib
import re


def key(s) -> str:
    """Comparison key: upper-case, no .NS, only letters/digits (M&M -> MM, BAJAJ-AUTO -> BAJAJAUTO)."""
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper().replace(".NS", ""))


def _subseq(q: str, s: str) -> bool:
    it = iter(s)
    return all(ch in it for ch in q)


class SymbolIndex:
    def __init__(self, symbols, names: dict | None = None):
        names = {key(k): v for k, v in (names or {}).items()}
        self.items, self.keys, self.by_sym = [], set(), {}
        for s in symbols or []:
            sym = str(s or "").upper().replace(".NS", "").strip()
            if not sym or sym == "NAN" or sym in self.by_sym:
                continue
            nm = str(names.get(key(sym), "") or "").strip()
            if nm.lower() == "nan":
                nm = ""
            e = {"sym": sym, "name": nm, "ks": key(sym), "kn": key(nm),
                 "words": [key(w) for w in re.split(r"[\s&\-\.,]+", nm) if len(key(w)) >= 2]}
            self.items.append(e)
            self.keys.add(e["ks"])
            self.by_sym[sym] = e
        self.items.sort(key=lambda e: e["sym"])

    def __len__(self):
        return len(self.items)

    def is_exact(self, text) -> bool:
        return key(text) in self.keys

    def name_of(self, sym) -> str:
        return (self.by_sym.get(str(sym).upper().replace(".NS", "")) or {}).get("name", "")

    # ------------------------------------------------------------------
    @staticmethod
    def _score(qk: str, e: dict) -> float:
        ks, kn = e["ks"], e["kn"]
        if ks == qk:
            return 1000.0
        if ks.startswith(qk):
            return 900.0 - (len(ks) - len(qk)) * 2
        if kn and kn.startswith(qk):
            return 780.0
        if any(w.startswith(qk) for w in e["words"]):
            return 750.0
        best = 0.0
        if qk in ks:
            best = 650.0 - ks.index(qk)
        elif kn and qk in kn:
            best = 600.0
        if len(qk) >= 3:
            sm = difflib.SequenceMatcher(None, qk, ks)
            if sm.real_quick_ratio() >= 0.6 and sm.quick_ratio() >= 0.6:
                r = sm.ratio()
                if r >= 0.6:
                    best = max(best, 300.0 + 400.0 * r)
            for w in e["words"]:
                sw = difflib.SequenceMatcher(None, qk, w)
                if sw.real_quick_ratio() >= 0.75 and sw.quick_ratio() >= 0.75:
                    r = sw.ratio()
                    if r >= 0.75:
                        best = max(best, 280.0 + 400.0 * r)
            if _subseq(qk, ks):
                best = max(best, 350.0 + 100.0 * len(qk) / max(len(ks), 1))
        return best

    def search(self, query, limit: int = 8, min_score: float = 340.0) -> list:
        qk = key(query)
        if not qk:
            return []
        scored = []
        for e in self.items:
            sc = self._score(qk, e)
            if sc >= min_score:
                scored.append((sc, e))
        scored.sort(key=lambda t: (-t[0], len(t[1]["sym"]), t[1]["sym"]))
        return [e for _, e in scored[:limit]]

    def label(self, sym) -> str:
        n = self.name_of(sym)
        return f"{sym} - {n}" if n else str(sym)
