V13 FIXED — NSE stock dashboard

Main fixes:
- Full/master scan uses NSE daily bhav history first instead of depending on Yahoo for every stock.
- Master universe uses all loaded NSE_STOCKS (up to MAX_SCAN_STOCKS), not an 800-stock cap.
- Yahoo is only a controlled fallback for missing symbols.
- Partial scan results are saved while scanning.
- A temporary provider failure no longer replaces the dashboard with an empty result when a previous successful scan exists.
- Live quote has a last-good fallback.
- Batch live quote no longer falls back to one Yahoo request for every missing symbol.
- UI refresh is 5 seconds; this is not a 0.1-second Yahoo request loop.

Deploy app.py as the Streamlit entrypoint.
Run locally:
  py -m pip install -r requirements.txt
  py -m py_compile app.py
  py -m streamlit run app.py
