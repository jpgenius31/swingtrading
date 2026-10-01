JP Stock Dashboard V15 FIXED

Main fix:
- Official NSE MCP is now the primary source for daily NSE history.
- Scanner no longer depends on list_tools discovery succeeding before trying history.
- Reduced MCP concurrency from 12 to 4 for reliability.
- Direct NSE archive scraping is no longer the primary history path.
- Live equity quotes use one cached NSE MCP market-wide snapshot before per-stock fallbacks.
- Yahoo remains a controlled fallback.
- Existing dashboard/analysis logic is retained.

Streamlit Cloud:
1. Replace the deployed app.py with this app.py.
2. Use the included requirements.txt.
3. Reboot/redeploy the app.

NSE MCP provides daily EOD history and a live market snapshot; it is not a 0.1-second tick feed.
