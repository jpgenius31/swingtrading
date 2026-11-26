# Stop NSE / Yahoo blocks + fix dashboard data

## Why dashboard still shows 20 Nov

GitHub Actions **runs green** but:

1. Cloud runner IP is often **blocked** by NSE (403) and sometimes Yahoo
2. Scan writes **0 new rows** or same file → commit step says **No changes** or push fails
3. `latest_results_meta.json` never moves → app banner: not from today

**Green job ≠ new prices on the dashboard.**

---

## Best fix (recommended): scan on your PC, push to GitHub

Home broadband is rarely blocked.

```bash
cd your-swingtrading-folder
pip install pandas numpy yfinance requests
python local_scan_push.py --limit 80
```

Then in the app: **Reload scan**.

Optional daily:
```bash
python local_scan_push.py --915
python local_scan_push.py --limit 80
```

---

## Option B: Self-hosted GitHub runner (automatic on your PC)

1. Repo → **Settings → Actions → Runners → New self-hosted runner**
2. Follow install for Windows/Mac/Linux
3. Leave the runner app open during market hours
4. Actions → Full Market Scan → Run workflow → set runner to **self-hosted** if prompted  
   (or add label `self-hosted` and use the workflow input)

Your PC does the download; GitHub only stores the CSV.

---

## Option C: Stay on GitHub cloud (weaker)

| Tactic | Why |
|--------|-----|
| `--limit 40` or `60` | Fewer requests → fewer bans |
| Sleep between symbols (already in scan_job) | Rate-limit friendly |
| Yahoo **chart API** primary (new scan_job) | Better than pure yfinance on cloud |
| Re-run later if empty | Temporary IP ban clears |

Still not as reliable as home IP.

---

## Checklist after any scan

1. Open: `https://raw.githubusercontent.com/jpgenius31/swingtrading/main/latest_results_meta.json`
2. `saved_at_ist` must be **today**
3. `rows` > 0
4. App header → **Reload scan**

If meta is still 20 Nov, **files never reached main** — upload new `scan_job.py` + run `local_scan_push.py`.

---

## What does NOT work long-term

- Hoping NSE allows GitHub datacenter IPs
- Only looking at green Actions without checking `saved_at_ist`
- Scanning 200 stocks quickly from cloud every few minutes
