# Strict schedule for 9:15–9:30 strategy

## Why GitHub alone failed

GitHub Actions `schedule:` (cron) is **not reliable to the minute**.

Your repo history showed:
- **Zero** runs near **9:15 AM IST** (03:45 UTC)
- Runs only in the **afternoon / night IST** (often 5:42 PM, 9:44 PM, 11:59 PM)

So the 9:15–9:30 strategy **cannot** depend on GitHub’s free scheduler alone.

## What we changed

| Workflow | Purpose | When (IST, Mon–Fri) |
|----------|---------|---------------------|
| `strategy_915.yml` | Only 9:15 CE/PE | **9:12** and **9:16** (starts early for delay) |
| `full_scan.yml` | Market scan | **9:22**, **12:30**, **15:00**, **15:35** |
| `market_scan.yml` | Manual only | You press Run workflow |

## REQUIRED for true 9:15 precision (free)

Use **cron-job.org** (or similar) to call GitHub **exactly** at 9:14 IST.

### 1. Create a GitHub token
1. GitHub → Settings → Developer settings → Personal access tokens  
2. Fine-grained or classic token with **`workflow` + `repo`** (or Contents: Read/Write + Actions: Write)  
3. Copy the token once

### 2. Create cron job (cron-job.org)
- URL (POST):
```
https://api.github.com/repos/jpgenius31/swingtrading/actions/workflows/strategy_915.yml/dispatches
```
- Schedule: **09:14** India time, Mon–Fri  
- Headers:
```
Authorization: Bearer YOUR_GITHUB_TOKEN
Accept: application/vnd.github+json
Content-Type: application/json
```
- Body:
```json
{"ref":"main"}
```

### 3. Optional second job
- Same URL at **09:17 IST** as backup if the first is delayed

### 4. Full scan backup (optional)
```
https://api.github.com/repos/jpgenius31/swingtrading/actions/workflows/full_scan.yml/dispatches
```
Body: `{"ref":"main"}` at 09:25, 12:35, 15:05 IST

## After push
1. Commit these workflow files to **main**
2. Actions → enable workflows if asked  
3. Next weekday check **Strategy 915** runs near **9:12–9:20 IST**  
4. App → Reload scan after 9:25

## Honest limit
Without external cron, GitHub may still skip or delay.  
With external cron → **workflow_dispatch**, timing is under **your** control.
