# Quick standalone test — calls ONLY the Twelve Data functions directly,
# bypassing yfinance and Finnhub entirely. Confirms your Twelve Data key
# and the candle-parsing logic work before trusting it as a fallback.
#
# RUN (from your StockAssistant folder, with TWELVE_DATA_API_KEY set):
#     python3 test_twelvedata.py

from app import _td_get_price_history, _td_get_price_chart_data, _td_get_forex_rate

TESTS = [
    ("Price history (AAPL, 1mo)", lambda: _td_get_price_history("AAPL", "1mo")),
    ("Price history (AAPL, 1y)", lambda: _td_get_price_history("AAPL", "1y")),
    ("Chart data (TSLA, 3mo)", lambda: _td_get_price_chart_data("TSLA", "3mo")),
    ("Forex (EURUSD=X)", lambda: _td_get_forex_rate("EURUSD=X")),
]

for name, fn in TESTS:
    print(f"\n--- {name} ---")
    try:
        result = fn()
        # Chart data returns a (points, summary) tuple; price history returns a plain dict.
        if isinstance(result, tuple):
            points, summary = result
            if points is None:
                print(f"  RETURNED ERROR: {summary.get('error')}")
            else:
                print(f"  OK: {len(points)} points, summary: {summary}")
        elif isinstance(result, dict) and "error" in result:
            print(f"  RETURNED ERROR: {result['error']}")
        else:
            print(f"  OK: {result}")
    except Exception as e:
        print(f"  CRASHED: {type(e).__name__}: {e}")