# All automatic jobs (external cron + GitHub)

GitHub native schedule is unreliable. Use **cron-job.org** (free) for each workflow.

## Workflows (must be under `.github/workflows/`)

| File | Purpose |
|------|---------|
| `strategy_915.yml` | 9:15–9:30 CE/PE |
| `full_scan.yml` | Full market / strategy scan |
| `patterns_scan.yml` | All chart patterns library |

## cron-job.org jobs (Asia/Kolkata, Mon–Fri)

Same headers for all:
```
Authorization: Bearer YOUR_GITHUB_TOKEN
Accept: application/vnd.github+json
Content-Type: application/json
```
Body for all:
```json
{"ref":"main"}
```
Method: **POST**

### 1) JP Strategy 915
- URL: `https://api.github.com/repos/jpgenius31/swingtrading/actions/workflows/strategy_915.yml/dispatches`
- Time: **09:14** (optional backup **09:17**)

### 2) JP Full Scan
- URL: `https://api.github.com/repos/jpgenius31/swingtrading/actions/workflows/full_scan.yml/dispatches`
- Times: **09:25**, **12:35**, **15:05**

### 3) JP Chart Patterns
- URL: `https://api.github.com/repos/jpgenius31/swingtrading/actions/workflows/patterns_scan.yml/dispatches`
- Times: **09:40**, **12:45**, **15:40**

## After each run
App → **Reload scan** (or wait for auto-load from GITHUB_RESULTS_RAW).

## Test
Each job → **Run now** once → GitHub Actions must show green for that workflow name.
