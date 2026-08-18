# ==================================================================
# STOCK RESEARCH ASSISTANT - Stage 7: portable web app
# ==================================================================
#
# SETUP (run once in your terminal):
#     pip install anthropic yfinance pandas fastapi uvicorn requests
#     export ANTHROPIC_API_KEY="your-key-here"
#     export FINNHUB_API_KEY="your-finnhub-key-here"       (fallback: quotes, news)
#     export TWELVE_DATA_API_KEY="your-twelvedata-key-here" (fallback: price history/charts)
#
# RUN:
#     uvicorn app:app --reload
#     then open http://localhost:8000 in your browser
# ------------------------------------------------------------------

import json
import os
import time
import uuid
import requests
import yfinance as yf
from anthropic import Anthropic
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

client = Anthropic()

MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a stock research assistant. You have access to tools for
fetching stock prices, news, institutional holders, price history,
and top companies within a market sector. Use them whenever a
question requires current data — never guess or make up numbers.

When reporting on a stock's current standing, always include its
P/E ratio (trailing) alongside price and day range — say clearly if
it's unavailable (common for unprofitable companies) rather than
omitting it silently.

The person you're talking to may not know financial terminology. The
first time you use a term like P/E ratio, market cap, institutional
ownership, or day range in a conversation, briefly define it in one
plain-language clause (e.g. "a P/E ratio of 32 — meaning investors
are paying $32 for every $1 of annual profit —"). Don't re-explain a
term you've already defined earlier in the same conversation.

This tool covers whatever Yahoo Finance covers, primarily reliable
for US-listed companies. If a company is listed on a US exchange (or
dual-listed, trading on both a US and foreign exchange), just use its
plain US ticker as normal — don't mention exchanges or suffixes at
all. Only if a company trades EXCLUSIVELY outside the US (e.g. Toyota,
BP, a company with no US listing) should you use the correct
Yahoo-style suffix (e.g. 7203.T for Tokyo, BP.L for London) and briefly
note that the data may be delayed and less complete than for US
stocks. If unsure whether a company has a US listing, try the plain
ticker first. If that tool call returns an error AND the company is
plausibly foreign, don't guess a random suffix — instead, ask the user
which country or exchange the company is listed on, so you use the
correct one rather than risk showing the wrong company's data.

If asked questions about the app itself (not a specific stock), use
these facts rather than guessing:
- Data comes primarily from Yahoo Finance (via the yfinance library),
  with Finnhub as an automatic backup source if Yahoo is temporarily
  unavailable or rate-limiting. You don't need to mention which
  source answered a given question unless asked directly.
- Price data is close to real-time for US markets; international
  quotes may run 15-20 minutes delayed.
- News headlines are aggregated from various outlets (Reuters,
  Bloomberg, Benzinga, and others) — cite the specific outlet when
  referencing a headline, not the aggregator itself.
- Institutional holder and sector data is most reliable for US-listed
  companies and may be sparse or unavailable for smaller or non-US names.
- This tool has no access to real-time trade execution, brokerage
  accounts, or personal portfolio data — it's read-only research.
- You can look up major cryptocurrencies (e.g. BTC-USD) and major
  currency pairs (e.g. EURUSD=X). You do NOT have access to options
  data (strikes, expirations, greeks) or futures contracts.
- Crypto trades 24/7, so its day_low/day_high figures are a rolling
  window that can lag slightly behind the current price (occasionally
  the current price sits just outside that range). This is a normal
  data quirk, not an error — mention it briefly and matter-of-factly
  if it comes up, rather than treating it as unusual.
- Institutional holders and international (non-US) stocks have NO
  backup data source — only the primary source covers them. If that
  primary lookup fails for one of these, be honest that this specific
  type of data has no fallback right now, rather than suggesting a
  simple retry will likely fix it (retrying only helps if the primary
  source's issue happens to be temporary, which isn't guaranteed).

For further reading, you can point users to these resources when
relevant (only when it adds value, not in every response):
- Yahoo Finance's own list of covered exchanges and data providers:
  https://help.yahoo.com/kb/finance/SLN2310.html
- Investopedia (https://www.investopedia.com) for deeper explanations
  of any financial term or concept beyond what you cover inline.

When asked to research good stocks in a sector, first call
get_sector_top_stocks to find leading companies, then pull
get_stock_data (and get_news if useful) for a few of the most
relevant ones so you can give real comparative data, not just a bare
list of names.

Keep responses concise. Structure them as short paragraphs — one topic
per paragraph (e.g. price action, then news, then ownership), separated
by a blank line — so they're easy to scan. Do not use markdown headers,
bold text, bullet points, tables, or emoji. Just clean, well-organized
prose broken into digestible paragraphs.

When referencing news, mention which outlet reported it. If a tool
returns an error, tell the user clearly what went wrong.

You are a research tool, not a financial advisor. Never tell a user
what to buy, sell, or how to allocate their money. If asked for
recommendations, redirect to research you can actually provide —
data on specific tickers, comparisons, or sector information."""


# --- Caching -----------------------------------------------------
# A DECORATOR: a function that wraps another function to add behavior
# without changing its code. You've already used one of these —
# @app.post("/chat") is a decorator FastAPI provides. This is us
# writing our own, for the same underlying reason: add one piece of
# behavior (caching) to many functions without repeating ourselves.
CACHE = {}
CACHE_TTL_SECONDS = 60  # how long a cached result stays valid


def with_cache(fn):
    def wrapper(*args, **kwargs):
        # Build a unique key from the function name + its exact arguments,
        # so get_stock_data("AAPL") and get_stock_data("TSLA") are cached separately.
        key = fn.__name__ + str(args) + str(sorted(kwargs.items()))
        now = time.time()

        if key in CACHE:
            cached_time, cached_value = CACHE[key]
            if now - cached_time < CACHE_TTL_SECONDS:
                print(f"  [cache hit: {key}]")
                return cached_value

        value = fn(*args, **kwargs)

        # Never cache a failure — an error should always be free to retry
        # on the very next call, not stuck repeating for the full TTL.
        # Two error shapes to check: a plain {"error": ...} dict (most
        # tools), or a (None, {"error": ...}) tuple (get_price_chart_data).
        is_error = (isinstance(value, dict) and "error" in value) or \
                   (isinstance(value, tuple) and value[0] is None)

        if not is_error:
            CACHE[key] = (now, value)

        return value

    return wrapper


# --- Hybrid data sourcing: yfinance first, Finnhub as fallback ------
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
FINNHUB_BASE = "https://finnhub.io/api/v1"


def finnhub_get(path, params):
    params = {**params, "token": FINNHUB_API_KEY}
    resp = requests.get(f"{FINNHUB_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()  # raises an exception on 4xx/5xx, e.g. Finnhub's own rate limit
    return resp.json()


# Twelve Data — used specifically for historical price data, since
# Finnhub's candle endpoint is confirmed paid-tier-only (we tested it
# directly and got a 403). Twelve Data's free tier (800 calls/day)
# includes real historical daily bars at no cost.
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
TWELVE_DATA_BASE = "https://api.twelvedata.com"


def twelvedata_get(path, params):
    params = {**params, "apikey": TWELVE_DATA_API_KEY}
    resp = requests.get(f"{TWELVE_DATA_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Twelve Data returns HTTP 200 even for errors — the actual error
    # shows up inside the JSON body instead, so we check for it explicitly.
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(data.get("message", "Twelve Data returned an error."))
    return data


def try_then_fallback(primary_fn, fallback_fn, *args, **kwargs):
    """Runs primary_fn (yfinance). If it raises OR returns an error dict,
    logs that failure and tries fallback_fn (Finnhub) instead. Only
    returns a real error if BOTH sources fail — this is what makes the
    whole app resilient to either single provider having a bad day."""
    try:
        result = primary_fn(*args, **kwargs)
        is_error = (isinstance(result, dict) and "error" in result) or \
                   (isinstance(result, tuple) and result[0] is None)
        if is_error:
            raise RuntimeError(str(result))
        return result
    except Exception as primary_error:
        print(f"  [PRIMARY (yfinance) failed: {primary_error} — falling back to Finnhub]")
        if not FINNHUB_API_KEY:
            return {"error": f"yfinance failed ({primary_error}) and no FINNHUB_API_KEY is set for fallback."}
        try:
            return fallback_fn(*args, **kwargs)
        except Exception as fallback_error:
            print(f"  [FALLBACK (Finnhub) also failed: {fallback_error}]")
            return {"error": f"Both data sources failed. yfinance: {primary_error} | Finnhub: {fallback_error}"}


# --- Tools -----------------------------------------------------------
@with_cache
def get_stock_data(ticker):
    return try_then_fallback(_yf_get_stock_data, _fh_get_stock_data, ticker)


def _yf_get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice")
        if price is None:
            return {"error": f"No price data found for '{ticker}'."}
        return {
            "ticker": ticker,
            "company": info.get("longName", ticker),
            "price": price,
            "previous_close": info.get("previousClose"),
            "day_low": info.get("dayLow"),
            "day_high": info.get("dayHigh"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe_ratio": info.get("forwardPE"),
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for '{ticker}': {str(e)}"}


def _fh_get_stock_data(ticker):
    quote = finnhub_get("/quote", {"symbol": ticker})
    if not quote or quote.get("c") in (None, 0):
        return {"error": f"No price data found for '{ticker}' on Finnhub either."}

    profile = finnhub_get("/stock/profile2", {"symbol": ticker})
    metrics = finnhub_get("/stock/metric", {"symbol": ticker, "metric": "all"}).get("metric", {})

    return {
        "ticker": ticker,
        "company": profile.get("name", ticker),
        "price": quote.get("c"),
        "previous_close": quote.get("pc"),
        "day_low": quote.get("l"),
        "day_high": quote.get("h"),
        "pe_ratio": metrics.get("peTTM"),
        "forward_pe_ratio": metrics.get("peForward"),
    }


@with_cache
def get_news(ticker, limit=5):
    return try_then_fallback(_yf_get_news, _fh_get_news, ticker, limit=limit)


def _yf_get_news(ticker, limit=5):
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        headlines = []
        for item in raw_news[:limit]:
            content = item.get("content", {})
            provider = content.get("provider", {})
            url = content.get("canonicalUrl", {})
            headlines.append({
                "title": content.get("title", "Untitled"),
                "date": content.get("pubDate", "unknown date"),
                "source": provider.get("displayName", "unknown source"),
                "url": url.get("url", ""),
            })
        return headlines if headlines else {"error": f"No recent news found for '{ticker}'."}
    except Exception as e:
        return {"error": f"Failed to fetch news for '{ticker}': {str(e)}"}


def _fh_get_news(ticker, limit=5):
    import datetime
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)

    items = finnhub_get("/company-news", {
        "symbol": ticker,
        "from": week_ago.isoformat(),
        "to": today.isoformat(),
    })

    if not items:
        return {"error": f"No recent news found for '{ticker}' on Finnhub either."}

    headlines = []
    for item in items[:limit]:
        headlines.append({
            "title": item.get("headline", "Untitled"),
            "date": datetime.datetime.fromtimestamp(item.get("datetime", 0)).isoformat() if item.get("datetime") else "unknown date",
            "source": item.get("source", "unknown source"),
            "url": item.get("url", ""),
        })
    return headlines


@with_cache
def get_institutional_holders(ticker, limit=5):
    return try_then_fallback(_yf_get_institutional_holders, _fh_get_institutional_holders, ticker, limit=limit)


def _yf_get_institutional_holders(ticker, limit=5):
    try:
        stock = yf.Ticker(ticker)
        holders_df = stock.institutional_holders
        if holders_df is None or holders_df.empty:
            return {"error": f"No institutional holder data found for '{ticker}'."}
        holders_df = holders_df.head(limit).copy()
        if "Date Reported" in holders_df.columns:
            holders_df["Date Reported"] = holders_df["Date Reported"].astype(str)
        return holders_df.to_dict(orient="records")
    except Exception as e:
        return {"error": f"Failed to fetch institutional holders for '{ticker}': {str(e)}"}


def _fh_get_institutional_holders(ticker, limit=5):
    data = finnhub_get("/stock/ownership", {"symbol": ticker, "limit": limit})
    ownership = data.get("ownership", [])
    if not ownership:
        return {"error": f"No institutional holder data found for '{ticker}' on Finnhub either."}

    holders = []
    for h in ownership[:limit]:
        holders.append({
            "Holder": h.get("name", "unknown"),
            "Shares": h.get("share"),
            "Date Reported": h.get("filingDate", "unknown"),
            "% Change": h.get("change"),
        })
    return holders


@with_cache
def get_price_history(ticker, period="6mo"):
    return try_then_fallback(_yf_get_price_history, _td_get_price_history, ticker, period=period)


def _yf_get_price_history(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": f"No historical price data found for '{ticker}'."}
        hist = hist.reset_index()
        start_price = round(float(hist.iloc[0]["Close"]), 2)
        end_price = round(float(hist.iloc[-1]["Close"]), 2)
        percent_change = round((end_price - start_price) / start_price * 100, 2)
        high_idx = hist["High"].idxmax()
        low_idx = hist["Low"].idxmin()
        return {
            "ticker": ticker,
            "period": period,
            "start_date": str(hist.iloc[0]["Date"].date()),
            "start_price": start_price,
            "end_date": str(hist.iloc[-1]["Date"].date()),
            "end_price": end_price,
            "percent_change": percent_change,
            "period_high": round(float(hist.loc[high_idx, "High"]), 2),
            "period_high_date": str(hist.loc[high_idx, "Date"].date()),
            "period_low": round(float(hist.loc[low_idx, "Low"]), 2),
            "period_low_date": str(hist.loc[low_idx, "Date"].date()),
        }
    except Exception as e:
        return {"error": f"Failed to fetch price history for '{ticker}': {str(e)}"}


def _period_to_days(period):
    # Twelve Data's outputsize parameter wants a number of trading days,
    # not calendar days — using calendar days here is a slight
    # overestimate, which is fine, since we just get a few extra points.
    return {"5d": 5, "1mo": 22, "3mo": 65, "6mo": 130, "1y": 260,
            "2y": 520, "5y": 1300, "max": 1300}.get(period, 130)


def _td_get_candles(ticker, period):
    """Shared by both get_price_history and get_price_chart_data —
    fetches daily OHLC candles from Twelve Data for the given period."""
    days = _period_to_days(period)

    data = twelvedata_get("/time_series", {
        "symbol": ticker, "interval": "1day", "outputsize": days
    })

    values = data.get("values")
    if not values:
        return None

    # Twelve Data returns newest-first; we want chronological order,
    # and every field comes back as a STRING, not a number — float()
    # conversion is required here, unlike yfinance which gives native numbers.
    points = []
    for v in reversed(values):
        points.append({
            "date": v["datetime"],
            "close": round(float(v["close"]), 2),
            "high": round(float(v["high"]), 2),
            "low": round(float(v["low"]), 2),
        })
    return points


def _td_get_price_history(ticker, period="6mo"):
    points = _td_get_candles(ticker, period)
    if not points:
        return {"error": f"No historical price data found for '{ticker}' on Twelve Data either."}

    start, end = points[0], points[-1]
    percent_change = round((end["close"] - start["close"]) / start["close"] * 100, 2)
    high_point = max(points, key=lambda p: p["high"])
    low_point = min(points, key=lambda p: p["low"])

    return {
        "ticker": ticker,
        "period": period,
        "start_date": start["date"],
        "start_price": start["close"],
        "end_date": end["date"],
        "end_price": end["close"],
        "percent_change": percent_change,
        "period_high": high_point["high"],
        "period_high_date": high_point["date"],
        "period_low": low_point["low"],
        "period_low_date": low_point["date"],
    }


# Valid sector keys yfinance understands — the AI must pick from this
# exact list (enforced via "enum" in the tool definition below).
VALID_SECTORS = [
    "technology", "healthcare", "financial-services", "consumer-cyclical",
    "industrials", "communication-services", "consumer-defensive",
    "energy", "basic-materials", "real-estate", "utilities"
]


@with_cache
def get_sector_top_stocks(sector, limit=10):
    return try_then_fallback(_yf_get_sector_top_stocks, _fh_get_sector_top_stocks, sector, limit=limit)


def _yf_get_sector_top_stocks(sector, limit=10):
    try:
        sector_data = yf.Sector(sector)
        top = sector_data.top_companies
        if top is None or top.empty:
            return {"error": f"No company data found for sector '{sector}'."}
        top = top.head(limit).reset_index()
        keep_cols = [c for c in ["symbol", "name", "market weight", "rating"] if c in top.columns]
        top = top[keep_cols] if keep_cols else top
        return top.to_dict(orient="records")
    except Exception as e:
        return {"error": f"Failed to fetch top companies for sector '{sector}': {str(e)}. Valid sectors are: {', '.join(VALID_SECTORS)}."}


# Finnhub's free tier has no "top companies in a sector" endpoint, so the
# fallback uses a small hardcoded list of major tickers per sector, then
# pulls live quotes for each — less elegant than a live lookup, but fully
# functional, and we control exactly which companies appear.
SECTOR_TICKERS = {
    "technology": ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL"],
    "healthcare": ["LLY", "UNH", "JNJ", "ABBV", "MRK"],
    "financial-services": ["JPM", "V", "MA", "BAC", "WFC"],
    "consumer-cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "industrials": ["GE", "CAT", "RTX", "UNP", "HON"],
    "communication-services": ["GOOGL", "META", "NFLX", "DIS", "TMUS"],
    "consumer-defensive": ["WMT", "PG", "KO", "PEP", "COST"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "basic-materials": ["LIN", "SHW", "FCX", "ECL", "NEM"],
    "real-estate": ["PLD", "AMT", "EQIX", "SPG", "O"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
}


def _fh_get_sector_top_stocks(sector, limit=10):
    tickers = SECTOR_TICKERS.get(sector)
    if not tickers:
        return {"error": f"Unknown sector '{sector}'. Valid sectors are: {', '.join(VALID_SECTORS)}."}

    results = []
    for t in tickers[:limit]:
        try:
            profile = finnhub_get("/stock/profile2", {"symbol": t})
            quote = finnhub_get("/quote", {"symbol": t})
            results.append({"symbol": t, "name": profile.get("name", t), "price": quote.get("c")})
        except Exception:
            continue  # skip a single bad ticker rather than failing the whole list

    if not results:
        return {"error": f"Could not fetch sector data for '{sector}' from Finnhub either."}
    return results


@with_cache
def get_price_chart_data(ticker, period="6mo"):
    """Unlike our other tools, this one returns TWO things:
    - points: the full list of {date, close} — goes straight to the browser to draw, never touches the AI
    - summary: a compact digest — goes to the AI, so it can talk about the chart without needing every data point
    This keeps token usage low even for a 5-year daily chart with 1000+ points.

    Kept OUT of try_then_fallback deliberately: that helper's final failure
    case returns a plain {"error": ...} dict, which would break the
    `points, summary = get_price_chart_data(...)` unpacking in run_agent_turn
    if both sources failed at once. This keeps the tuple shape consistent
    on every path — success, fallback, or total failure."""
    points, summary = _yf_get_price_chart_data(ticker, period)
    if points is not None:
        return points, summary

    print(f"  [PRIMARY (yfinance) failed for chart: {summary.get('error')} — falling back to Twelve Data]")
    if not TWELVE_DATA_API_KEY:
        return None, {"error": f"yfinance failed ({summary.get('error')}) and no TWELVE_DATA_API_KEY is set for fallback."}

    try:
        return _td_get_price_chart_data(ticker, period)
    except Exception as fallback_error:
        print(f"  [FALLBACK (Twelve Data) also failed for chart: {fallback_error}]")
        return None, {"error": f"Both data sources failed. yfinance: {summary.get('error')} | Twelve Data: {fallback_error}"}


def _yf_get_price_chart_data(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return None, {"error": f"No historical price data found for '{ticker}'."}

        hist = hist.reset_index()
        points = [
            {"date": str(row["Date"].date()), "close": round(float(row["Close"]), 2)}
            for _, row in hist.iterrows()
        ]

        closes = [p["close"] for p in points]
        summary = {
            "ticker": ticker,
            "period": period,
            "point_count": len(points),
            "start_date": points[0]["date"],
            "end_date": points[-1]["date"],
            "min_close": min(closes),
            "max_close": max(closes),
            "note": "A chart has been displayed to the user showing this data. Do not repeat all the numbers — just briefly reference the trend."
        }
        return points, summary
    except Exception as e:
        return None, {"error": f"Failed to fetch chart data for '{ticker}': {str(e)}"}


def _td_get_price_chart_data(ticker, period="6mo"):
    candles = _td_get_candles(ticker, period)
    if not candles:
        return None, {"error": f"No historical price data found for '{ticker}' on Twelve Data either."}

    points = [{"date": c["date"], "close": c["close"]} for c in candles]
    closes = [p["close"] for p in points]
    summary = {
        "ticker": ticker,
        "period": period,
        "point_count": len(points),
        "start_date": points[0]["date"],
        "end_date": points[-1]["date"],
        "min_close": min(closes),
        "max_close": max(closes),
        "note": "A chart has been displayed to the user showing this data. Do not repeat all the numbers — just briefly reference the trend."
    }
    return points, summary


@with_cache
def get_crypto_data(symbol):
    """symbol should be Yahoo-style, e.g. 'BTC-USD', 'ETH-USD'."""
    return try_then_fallback(_yf_get_crypto_data, _fh_get_crypto_data, symbol)


def _yf_get_crypto_data(symbol):
    try:
        coin = yf.Ticker(symbol)
        info = coin.info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price is None:
            return {"error": f"No price data found for '{symbol}'. Use format like BTC-USD."}
        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "price": price,
            "previous_close": info.get("previousClose"),
            "day_low": info.get("dayLow"),
            "day_high": info.get("dayHigh"),
        }
    except Exception as e:
        return {"error": f"Failed to fetch crypto data for '{symbol}': {str(e)}"}


def _fh_get_crypto_data(symbol):
    # Switched from /crypto/candle (paid-tier only, confirmed via testing)
    # to /quote, which works on the free tier — same endpoint used for
    # stocks, just pointed at a crypto symbol instead.
    base = symbol.replace("-USD", "").replace("-USDT", "")
    fh_symbol = f"BINANCE:{base}USDT"

    quote = finnhub_get("/quote", {"symbol": fh_symbol})
    price = quote.get("c")

    if not price:
        return {"error": f"No crypto data found for '{symbol}' on Finnhub either."}

    return {
        "symbol": symbol,
        "name": base,
        "price": price,
        "previous_close": quote.get("pc"),
        "day_low": quote.get("l"),
        "day_high": quote.get("h"),
    }


@with_cache
def get_forex_rate(pair):
    """pair should be Yahoo-style, e.g. 'EURUSD=X', 'GBPUSD=X'."""
    return try_then_fallback(_yf_get_forex_rate, _td_get_forex_rate, pair)


def _yf_get_forex_rate(pair):
    try:
        fx = yf.Ticker(pair)
        info = fx.info
        rate = info.get("regularMarketPrice") or info.get("currentPrice")
        if rate is None:
            return {"error": f"No rate found for '{pair}'. Use format like EURUSD=X."}
        return {
            "pair": pair,
            "rate": rate,
            "previous_close": info.get("previousClose"),
            "day_low": info.get("dayLow"),
            "day_high": info.get("dayHigh"),
        }
    except Exception as e:
        return {"error": f"Failed to fetch forex rate for '{pair}': {str(e)}"}


def _td_get_forex_rate(pair):
    # Switched from Finnhub's /quote with OANDA-style symbols (confirmed
    # paid-tier only via testing) to Twelve Data, reusing the key we
    # already have working for price history. Twelve Data's forex symbol
    # format uses a slash: "EUR/USD" rather than "EURUSD".
    raw = pair.replace("=X", "")
    base, quote_ccy = raw[:3], raw[3:6]
    td_symbol = f"{base}/{quote_ccy}"

    data = twelvedata_get("/quote", {"symbol": td_symbol})
    rate = data.get("close")

    if rate is None:
        return {"error": f"No rate found for '{pair}' on Twelve Data either."}

    return {
        "pair": pair,
        "rate": round(float(rate), 4),
        "previous_close": round(float(data["previous_close"]), 4) if data.get("previous_close") else None,
        "day_low": round(float(data["low"]), 4) if data.get("low") else None,
        "day_high": round(float(data["high"]), 4) if data.get("high") else None,
    }


# Major US indices, used as a proxy for "the overall market" when the
# user isn't asking about one specific stock.
MARKET_INDICES = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC"}
# Finnhub free tier can't quote raw index symbols reliably — liquid ETFs
# tracking the same indices work as a close, reliable proxy instead.
MARKET_INDEX_ETF_PROXIES = {"S&P 500": "SPY", "Dow Jones": "DIA", "Nasdaq": "QQQ"}


@with_cache
def get_market_overview():
    return try_then_fallback(_yf_get_market_overview, _fh_get_market_overview)


def _yf_get_market_overview():
    try:
        indices = []
        for name, symbol in MARKET_INDICES.items():
            idx = yf.Ticker(symbol)
            info = idx.info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev = info.get("previousClose")
            if price is None or prev is None:
                continue
            indices.append({
                "name": name,
                "level": round(price, 2),
                "percent_change": round((price - prev) / prev * 100, 2),
            })

        # get_news is itself hybrid, so this call already self-heals to
        # Finnhub if yfinance is down — no separate fallback needed here.
        market_news = get_news("SPY", limit=5)

        if not indices:
            return {"error": "Could not fetch market index data."}

        return {"indices": indices, "market_news": market_news}
    except Exception as e:
        return {"error": f"Failed to fetch market overview: {str(e)}"}


def _fh_get_market_overview():
    indices = []
    for name, symbol in MARKET_INDEX_ETF_PROXIES.items():
        quote = finnhub_get("/quote", {"symbol": symbol})
        price, prev = quote.get("c"), quote.get("pc")
        if not price or not prev:
            continue
        indices.append({
            "name": f"{name} (via {symbol} ETF)",
            "level": round(price, 2),
            "percent_change": round((price - prev) / prev * 100, 2),
        })

    if not indices:
        return {"error": "Could not fetch market index data from Finnhub either."}

    market_news = get_news("SPY", limit=5)
    return {"indices": indices, "market_news": market_news}


@with_cache
def get_earnings_calendar(ticker):
    return try_then_fallback(_yf_get_earnings_calendar, _fh_get_earnings_calendar, ticker)


def _yf_get_earnings_calendar(ticker):
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar  # dict-like: {"Earnings Date": [date, ...], "Earnings Average": ..., ...}

        if not cal or not cal.get("Earnings Date"):
            return {"error": f"No upcoming earnings date found for '{ticker}'."}

        return {
            "ticker": ticker,
            "next_earnings_dates": [str(d) for d in cal["Earnings Date"]],
            "eps_estimate_avg": cal.get("Earnings Average"),
            "revenue_estimate_avg": cal.get("Revenue Average"),
        }
    except Exception as e:
        return {"error": f"Failed to fetch earnings calendar for '{ticker}': {str(e)}"}


def _fh_get_earnings_calendar(ticker):
    import datetime
    today = datetime.date.today()
    # Requesting a wide window, but per Finnhub's own docs the free tier
    # realistically only confirms about a month out — anything further
    # simply won't be in the results yet, which is fine, not an error.
    horizon = today + datetime.timedelta(days=180)

    data = finnhub_get("/calendar/earnings", {
        "symbol": ticker, "from": today.isoformat(), "to": horizon.isoformat()
    })
    items = data.get("earningsCalendar", [])

    if not items:
        return {"error": f"No upcoming earnings date found for '{ticker}' on Finnhub either."}

    return {
        "ticker": ticker,
        "next_earnings_dates": [item.get("date") for item in items],
        "eps_estimate_avg": items[0].get("epsEstimate"),
        "revenue_estimate_avg": items[0].get("revenueEstimate"),
    }


AVAILABLE_TOOLS = {
    "get_stock_data": get_stock_data,
    "get_news": get_news,
    "get_institutional_holders": get_institutional_holders,
    "get_price_history": get_price_history,
    "get_sector_top_stocks": get_sector_top_stocks,
    "get_crypto_data": get_crypto_data,
    "get_forex_rate": get_forex_rate,
    "get_market_overview": get_market_overview,
    "get_earnings_calendar": get_earnings_calendar,
}

tool_definitions = [
    {"name": "get_stock_data", "description": "Get the current price and day range for a stock ticker.",
     "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "get_news", "description": "Get recent news headlines for a stock ticker.",
     "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "get_institutional_holders", "description": "Get the largest institutional shareholders of a stock.",
     "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "get_price_history",
     "description": "Get a summary of a stock's price performance over a past time period. Use for 'how has it done over time' type questions.",
     "input_schema": {"type": "object", "properties": {
         "ticker": {"type": "string"},
         "period": {"type": "string", "enum": ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]}
     }, "required": ["ticker"]}},
    {"name": "get_sector_top_stocks",
     "description": "Get the top/leading companies within a market sector, useful when the user wants to research or discover stocks in a particular sector or industry (e.g. 'good tech stocks', 'top energy companies').",
     "input_schema": {"type": "object", "properties": {
         "sector": {"type": "string", "enum": [
             "technology", "healthcare", "financial-services", "consumer-cyclical",
             "industrials", "communication-services", "consumer-defensive",
             "energy", "basic-materials", "real-estate", "utilities"
         ]},
         "limit": {"type": "integer", "description": "How many companies to return, default 10"}
     }, "required": ["sector"]}},
    {"name": "show_price_chart",
     "description": "Display a visual price trend chart to the user. Use this whenever they ask to see a chart, graph, or visualize a trend — do not use get_price_history for this, use this instead when a visual is wanted.",
     "input_schema": {"type": "object", "properties": {
         "ticker": {"type": "string"},
         "period": {"type": "string", "enum": ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]}
     }, "required": ["ticker"]}},
    {"name": "get_crypto_data",
     "description": "Get the current price for a cryptocurrency. Symbol must be Yahoo-style with a -USD suffix, e.g. BTC-USD, ETH-USD, SOL-USD.",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string", "description": "e.g. BTC-USD"}
     }, "required": ["symbol"]}},
    {"name": "get_forex_rate",
     "description": "Get the current exchange rate for a currency pair. Pair must be Yahoo-style ending in =X, e.g. EURUSD=X, GBPUSD=X, USDJPY=X.",
     "input_schema": {"type": "object", "properties": {
         "pair": {"type": "string", "description": "e.g. EURUSD=X"}
     }, "required": ["pair"]}},
    {"name": "get_market_overview",
     "description": "Get a snapshot of overall US market conditions: major index levels (S&P 500, Dow Jones, Nasdaq) plus general market-moving news. Use this when the user asks broadly about 'the market' or 'markets today' rather than a specific stock, sector, or ticker.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_earnings_calendar",
     "description": "Get the next confirmed or estimated earnings report date(s) for a stock, plus EPS/revenue estimates if available. IMPORTANT: this returns only the next known date(s), not a full multi-month calendar — companies rarely confirm earnings dates more than a few weeks ahead. If asked for dates further out, use this tool anyway and be honest in your answer that only the near-term date is actually known yet.",
     "input_schema": {"type": "object", "properties": {
         "ticker": {"type": "string"}
     }, "required": ["ticker"]}},
]


def run_agent_turn(messages):
    iterations = 0
    charts_to_send = []  # collects any charts requested during this turn

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            fallback_text = ("This is taking more steps than expected to answer — "
                              "try breaking your question into smaller parts.")
            # IMPORTANT: append this as an assistant message before returning.
            # Without this, the conversation history would end on a "user"
            # (tool result) message, and the next real user message would
            # break the required strict user/assistant alternation — silently
            # forcing a chat restart. This keeps the history valid so the
            # conversation can continue normally.
            messages.append({"role": "assistant", "content": fallback_text})
            return fallback_text, charts_to_send

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=tool_definitions,
            messages=messages
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            messages.append({"role": "assistant", "content": response.content})
            return final_text, charts_to_send

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "show_price_chart":
                # Special case: two different results for two different audiences.
                points, summary = get_price_chart_data(**block.input)
                if points is not None:
                    charts_to_send.append({
                        "ticker": block.input["ticker"],
                        "period": block.input.get("period", "6mo"),
                        "data": points
                    })
                result_for_ai = summary
                print(f"  [agent called show_price_chart({block.input})]")
            else:
                result_for_ai = AVAILABLE_TOOLS[block.name](**block.input)
                print(f"  [agent called {block.name}({block.input})]")

            # Log the RAW error to the server console whenever a tool fails —
            # separate from whatever paraphrased explanation the AI gives the
            # user. This is what lets us actually diagnose a failure later,
            # instead of just guessing based on the AI's summary of it.
            if isinstance(result_for_ai, dict) and "error" in result_for_ai:
                print(f"  [TOOL ERROR] {block.name}({block.input}) -> {result_for_ai['error']}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result_for_ai)
            })
        messages.append({"role": "user", "content": tool_results})


# ==================================================================
# THE WEB SERVER PART — everything below is new in Stage 7
# ==================================================================

app = FastAPI()

# In-memory session storage: session_id -> conversation history.
# NOTE: this resets if the server restarts, and only works with a
# single server instance. Fine for now — a real production app
# would use a database instead. We'll flag this as a known limit.
SESSIONS = {}

# --- Rate limiting -------------------------------------------------
# Sliding-window limiter: tracks request TIMESTAMPS per visitor (by IP),
# same "dictionary as memory" pattern as SESSIONS above.
RATE_LIMIT_WINDOW_SECONDS = 600   # 10 minutes
RATE_LIMIT_MAX_REQUESTS = 20      # per visitor, per window

request_log = {}  # ip -> list of timestamps


def get_client_ip(request: Request):
    # Render (and most hosts) sit behind a proxy, which means the "real"
    # visitor IP arrives in this header rather than request.client.host.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


def check_rate_limit(ip):
    now = time.time()
    timestamps = request_log.get(ip, [])
    # Keep only timestamps within the current window — this is what
    # makes it a SLIDING window rather than a hard reset every 10 minutes.
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    timestamps.append(now)
    request_log[ip] = timestamps
    return True


# --- Simple shared-password gate ------------------------------------
# Set APP_PASSWORD as an environment variable to require it. If it's
# not set (e.g. during local development), the gate is skipped entirely.
APP_PASSWORD = os.environ.get("APP_PASSWORD")


def check_password(provided):
    if not APP_PASSWORD:
        return True  # no password configured — gate is off
    return provided == APP_PASSWORD


# Defines the SHAPE of data we expect the frontend to send us.
# FastAPI uses this to validate incoming requests automatically.
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    password: Optional[str] = None


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    # 1. Password check (skipped automatically if APP_PASSWORD isn't set)
    if not check_password(req.password):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    # 2. Rate limit check, by visitor IP
    ip = get_client_ip(request)
    if not check_rate_limit(ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({RATE_LIMIT_MAX_REQUESTS} messages per {RATE_LIMIT_WINDOW_SECONDS // 60} minutes). Please wait a bit and try again."
        )

    # If this is a brand new visitor, create them a session.
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in SESSIONS:
        SESSIONS[session_id] = []

    messages = SESSIONS[session_id]
    messages.append({"role": "user", "content": req.message})

    reply, charts = run_agent_turn(messages)

    return {"session_id": session_id, "reply": reply, "charts": charts}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tape — Stock Research Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
        <style>
            :root {
                --ink: #1B2735;
                --paper: #EDF1F5;
                --paper-raised: #FFFFFF;
                --line: #D3DCE6;
                --teal: #2F8F72;
                --teal-dim: #E4F0EC;
                --coral: #C4573E;
                --muted: #5C6B7A;
            }
            * { box-sizing: border-box; }
            body {
                font-family: 'Inter', sans-serif;
                background: var(--paper);
                color: var(--ink);
                margin: 0;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
            header {
                padding: 20px 24px 16px;
                border-bottom: 1px solid var(--line);
                display: flex;
                align-items: baseline;
                gap: 10px;
            }
            header h1 {
                font-family: 'Fraunces', serif;
                font-weight: 600;
                font-size: 22px;
                margin: 0;
                letter-spacing: -0.01em;
            }
            header .tagline {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 11px;
                color: var(--muted);
                letter-spacing: 0.02em;
                padding-left: 2px;
                border-left: 1px solid var(--line);
                margin-left: 2px;
                padding-left: 10px;
            }
            header .status {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 11px;
                color: var(--muted);
                margin-left: auto;
            }
            header .dot {
                width: 7px; height: 7px; border-radius: 50%;
                background: var(--teal);
                box-shadow: 0 0 0 3px var(--teal-dim);
            }

            #chat {
                flex: 1;
                overflow-y: auto;
                padding: 24px;
                max-width: 720px;
                width: 100%;
                margin: 0 auto;
            }
            .row { display: flex; margin-bottom: 18px; }
            .row.user { justify-content: flex-end; }
            .bubble {
                max-width: 78%;
                padding: 12px 16px;
                border-radius: 10px;
                line-height: 1.55;
                font-size: 15px;
            }
            .row.user .bubble {
                background: var(--ink);
                color: var(--paper);
                border-bottom-right-radius: 3px;
            }
            .row.assistant .bubble {
                background: var(--paper-raised);
                border: 1px solid var(--line);
                border-bottom-left-radius: 3px;
            }
            .row.assistant .bubble p {
                margin: 0 0 10px 0;
            }
            .row.assistant .bubble p:last-child {
                margin-bottom: 0;
            }
            .chart-container {
                margin-top: 10px;
                background: var(--paper);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 12px;
            }
            .chart-label {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 11px;
                color: var(--muted);
                margin-bottom: 8px;
            }
            .label {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: var(--muted);
                margin-bottom: 4px;
            }
            .row.user .label { text-align: right; }

            .thinking { color: var(--muted); font-style: italic; font-size: 14px; }

            #input-bar {
                border-top: 1px solid var(--line);
                padding: 16px 24px 22px;
            }
            #input-row {
                max-width: 720px;
                margin: 0 auto;
                display: flex;
                gap: 10px;
                background: var(--paper-raised);
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 6px 6px 6px 18px;
            }
            #question {
                flex: 1;
                border: none;
                outline: none;
                background: transparent;
                font-family: 'Inter', sans-serif;
                font-size: 15px;
                color: var(--ink);
            }
            #question::placeholder { color: var(--muted); }
            button {
                border: none;
                background: var(--teal);
                color: white;
                font-family: 'Inter', sans-serif;
                font-weight: 500;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 999px;
                cursor: pointer;
            }
            button:hover { opacity: 0.92; }
            button:disabled { opacity: 0.5; cursor: default; }

            #chat::-webkit-scrollbar { width: 8px; }
            #chat::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
        </style>
    </head>
    <body>
        <header>
            <h1>Tape</h1>
            <span class="tagline">read between the ticks</span>
            <span class="status"><span class="dot"></span>connected</span>
        </header>

        <div id="chat"></div>

        <div id="input-bar">
            <div id="input-row">
                <input type="text" id="question" placeholder="Ask about a ticker, e.g. how is NVDA doing today?" autofocus />
                <button id="send-btn" onclick="send()">Send</button>
            </div>
        </div>

        <script>
            let sessionId = null;
            let appPassword = sessionStorage.getItem("tape_password") || "";

            async function ensurePassword() {
                // Only actually prompts if the server tells us a password
                // is required (via a 401 on first try) — see send()'s retry.
                const entered = prompt("This app requires a password to use. Enter it:");
                appPassword = entered || "";
                sessionStorage.setItem("tape_password", appPassword);
            }

            async function send() {
                const input = document.getElementById("question");
                const btn = document.getElementById("send-btn");
                const question = input.value.trim();
                if (!question) return;

                addMessage("user", "You", question);
                input.value = "";
                btn.disabled = true;

                const thinkingId = addThinking();

                try {
                    let res = await postChat(question);

                    if (res.status === 401) {
                        // Wrong or missing password — ask, then retry once.
                        await ensurePassword();
                        res = await postChat(question);
                    }

                    if (res.status === 429) {
                        const err = await res.json();
                        removeThinking(thinkingId);
                        addMessage("assistant", "Tape", err.detail);
                        btn.disabled = false;
                        input.focus();
                        return;
                    }

                    if (!res.ok) throw new Error("Request failed");

                    const data = await res.json();
                    sessionId = data.session_id;
                    removeThinking(thinkingId);
                    addMessage("assistant", "Tape", data.reply, data.charts);
                } catch (err) {
                    removeThinking(thinkingId);
                    addMessage("assistant", "Tape", "Something went wrong reaching the server. Is it still running?");
                }
                btn.disabled = false;
                input.focus();
            }

            function postChat(question) {
                return fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, message: question, password: appPassword })
                });
            }

            function addMessage(role, sender, text, charts) {
                const chat = document.getElementById("chat");
                const row = document.createElement("div");
                row.className = "row " + role;

                const bubble = document.createElement("div");
                bubble.className = "bubble";

                const label = document.createElement("div");
                label.className = "label";
                label.textContent = sender;
                bubble.appendChild(label);

                // Split into paragraphs on blank lines, render each as its own <p>
                // (textContent, not innerHTML, so nothing in the reply can inject HTML)
                const paragraphs = text.split(/\\n\\s*\\n/).filter(p => p.trim());
                if (paragraphs.length === 0) paragraphs.push(text);
                paragraphs.forEach(p => {
                    const pEl = document.createElement("p");
                    pEl.textContent = p.trim();
                    bubble.appendChild(pEl);
                });

                if (charts && charts.length > 0) {
                    charts.forEach(chart => bubble.appendChild(buildChart(chart)));
                }

                row.appendChild(bubble);
                chat.appendChild(row);
                chat.scrollTop = chat.scrollHeight;
            }

            function buildChart(chart) {
                const container = document.createElement("div");
                container.className = "chart-container";

                const label = document.createElement("div");
                label.className = "chart-label";
                label.textContent = chart.ticker + " · " + chart.period;
                container.appendChild(label);

                const canvas = document.createElement("canvas");
                canvas.height = 180;
                container.appendChild(canvas);

                // Chart.js needs the canvas actually in the page before it can draw,
                // so we build it AFTER appending — a tiny setTimeout ensures the
                // browser has laid it out first.
                setTimeout(() => {
                    new Chart(canvas.getContext("2d"), {
                        type: "line",
                        data: {
                            labels: chart.data.map(p => p.date),
                            datasets: [{
                                data: chart.data.map(p => p.close),
                                borderColor: "#2F8F72",
                                backgroundColor: "rgba(47, 143, 114, 0.08)",
                                fill: true,
                                pointRadius: 0,
                                borderWidth: 1.5,
                                tension: 0.15
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: { ticks: { maxTicksLimit: 6, font: { family: "IBM Plex Mono", size: 10 } }, grid: { display: false } },
                                y: { ticks: { font: { family: "IBM Plex Mono", size: 10 } }, grid: { color: "#D3DCE6" } }
                            }
                        }
                    });
                }, 0);

                return container;
            }

            function addThinking() {
                const chat = document.getElementById("chat");
                const row = document.createElement("div");
                row.className = "row assistant";
                row.id = "thinking-row";
                row.innerHTML = '<div class="bubble thinking">researching…</div>';
                chat.appendChild(row);
                chat.scrollTop = chat.scrollHeight;
                return "thinking-row";
            }

            function removeThinking(id) {
                const el = document.getElementById(id);
                if (el) el.remove();
            }

            document.getElementById("question").addEventListener("keypress", (e) => {
                if (e.key === "Enter") send();
            });
        </script>
    </body>
    </html>
    """