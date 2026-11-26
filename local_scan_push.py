#!/usr/bin/env python3
"""
Run market scan on YOUR PC (home IP) then push results to GitHub.
NSE and Yahoo almost never block home broadband.

Setup once:
  1) pip install pandas numpy yfinance requests
  2) git clone your repo OR open the folder that already has scan_job.py
  3) git remote must point to jpgenius31/swingtrading

Every market day (or after 9:30 / 15:30):
  python local_scan_push.py
  python local_scan_push.py --limit 80
  python local_scan_push.py --915

This updates latest_results.csv on main → Streamlit/dashboard loads fresh data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--915", dest="m915", action="store_true")
    ap.add_argument("--patterns", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    if args.m915:
        code = run([sys.executable, "scan_job.py", "--915", "--limit", str(min(args.limit, 25))])
    elif args.patterns:
        code = run([sys.executable, "scan_job.py", "--patterns-only", "--limit", str(args.limit)])
    else:
        code = run([sys.executable, "scan_job.py", "--limit", str(args.limit)])

    print("scan exit", code)
    meta = ROOT / "latest_results_meta.json"
    if meta.exists():
        print("meta:", meta.read_text(encoding="utf-8")[:300])
    csv = ROOT / "latest_results.csv"
    if csv.exists():
        print("csv lines:", sum(1 for _ in csv.open(encoding="utf-8", errors="replace")))

    if args.no_push:
        return

    run(["git", "add",
         "latest_results.csv", "latest_results_meta.json", "last_scan_heartbeat.txt",
         "strategy_915_candle_live.csv", "strategy_915_candle_past.csv", "strategy_915_candle_meta.json",
         "pattern_scan_results.json", "pattern_scan_meta.json",
         "strategy_live_signals.csv", "strategy_live_meta.json", "index_market_snap.json"])
    run(["git", "status", "--short"])
    run(["git", "commit", "-m", "chore: local scan (home IP)"])
    # push with rebase retry
    for i in range(1, 6):
        if run(["git", "pull", "--rebase", "origin", "main"]) == 0:
            if run(["git", "push", "origin", "HEAD:main"]) == 0:
                print("Push OK")
                return
        run(["git", "rebase", "--abort"])
        print(f"retry push {i}")
    print("Push failed — commit is local; push manually when online")


if __name__ == "__main__":
    main()
