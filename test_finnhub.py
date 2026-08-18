# Quick standalone test — calls ONLY the Finnhub functions directly,
# bypassing yfinance and the fallback logic entirely. This confirms
# your Finnhub key and endpoints work before trusting them as a fallback.
#
# RUN (from your StockAssistant folder, with both keys set):
#     python3 test_finnhub.py

from app import (
    _fh_get_stock_data,
    _fh_get_news,
    _fh_get_institutional_holders,
    _fh_get_crypto_data,
    _fh_get_market_overview,
    _fh_get_sector_top_stocks,
    _fh_get_earnings_calendar,
)

# Note: forex isn't tested here anymore — that fallback now uses Twelve
# Data instead of Finnhub (see test_twelvedata.py), since Finnhub's
# quote endpoint also came back 403 for forex symbols on the free tier.

TESTS = [
    ("Stock data (AAPL)", lambda: _fh_get_stock_data("AAPL")),
    ("News (AAPL)", lambda: _fh_get_news("AAPL", limit=3)),
    ("Institutional holders (AAPL)", lambda: _fh_get_institutional_holders("AAPL", limit=3)),
    ("Crypto (BTC-USD)", lambda: _fh_get_crypto_data("BTC-USD")),
    ("Market overview", lambda: _fh_get_market_overview()),
    ("Sector top stocks (technology)", lambda: _fh_get_sector_top_stocks("technology", limit=3)),
    ("Earnings calendar (AAPL)", lambda: _fh_get_earnings_calendar("AAPL")),
]

for name, fn in TESTS:
    print(f"\n--- {name} ---")
    try:
        result = fn()
        if isinstance(result, dict) and "error" in result:
            print(f"  RETURNED ERROR: {result['error']}")
        else:
            print(f"  OK: {result}")
    except Exception as e:
        print(f"  CRASHED: {type(e).__name__}: {e}")