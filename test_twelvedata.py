# Quick PROBE — not a permanent test, just checking whether Twelve Data
# covers international stocks at all, and what symbol format it wants,
# before we build any real fallback around it.
#
# RUN: python3 probe_twelvedata_international.py

from app import twelvedata_get

ATTEMPTS = [
    ("RELIANCE.NS (Yahoo-style, direct)", {"symbol": "RELIANCE.NS"}),
    ("RELIANCE + exchange=NSE", {"symbol": "RELIANCE", "exchange": "NSE"}),
    ("RELIANCE + mic_code=XNSE", {"symbol": "RELIANCE", "mic_code": "XNSE"}),
]

for label, params in ATTEMPTS:
    print(f"\n--- {label} ---")
    try:
        result = twelvedata_get("/quote", params)
        print(f"  OK: {result}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")