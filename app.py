# ==================================================================
# STOCK RESEARCH ASSISTANT - Stage 7: portable web app
# ==================================================================
#
# SETUP (run once in your terminal):
#     pip install anthropic yfinance pandas fastapi uvicorn
#     export ANTHROPIC_API_KEY="your-key-here"
#
# RUN:
#     uvicorn app:app --reload
#     then open http://localhost:8000 in your browser
# ------------------------------------------------------------------

import json
import os
import time
import uuid
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
- Data comes from Yahoo Finance (via the yfinance library), an
  unofficial source. It's not a licensed, real-time market data feed.
- Price data is close to real-time for US markets; international
  quotes may run 15-20 minutes delayed.
- News headlines are aggregated by Yahoo from various outlets
  (Reuters, Bloomberg, Benzinga, and others) — you cite the specific
  outlet when referencing a headline, not "Yahoo" itself.
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
        CACHE[key] = (now, value)
        return value

    return wrapper


# --- Tools (unchanged from Stage 6) ---------------------------------
@with_cache
def get_stock_data(ticker):
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


@with_cache
def get_news(ticker, limit=5):
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


@with_cache
def get_institutional_holders(ticker, limit=5):
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


@with_cache
def get_price_history(ticker, period="6mo"):
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


# Valid sector keys yfinance understands — the AI must pick from this
# exact list (enforced via "enum" in the tool definition below).
VALID_SECTORS = [
    "technology", "healthcare", "financial-services", "consumer-cyclical",
    "industrials", "communication-services", "consumer-defensive",
    "energy", "basic-materials", "real-estate", "utilities"
]


@with_cache
def get_sector_top_stocks(sector, limit=10):
    try:
        sector_data = yf.Sector(sector)
        top = sector_data.top_companies

        if top is None or top.empty:
            return {"error": f"No company data found for sector '{sector}'."}

        top = top.head(limit).reset_index()  # ticker symbol is the index -> becomes a column
        # Keep only the columns useful for a quick research overview
        keep_cols = [c for c in ["symbol", "name", "market weight", "rating"] if c in top.columns]
        top = top[keep_cols] if keep_cols else top

        return top.to_dict(orient="records")
    except Exception as e:
        return {"error": f"Failed to fetch top companies for sector '{sector}': {str(e)}. Valid sectors are: {', '.join(VALID_SECTORS)}."}


@with_cache
def get_price_chart_data(ticker, period="6mo"):
    """Unlike our other tools, this one returns TWO things:
    - points: the full list of {date, close} — goes straight to the browser to draw, never touches the AI
    - summary: a compact digest — goes to the AI, so it can talk about the chart without needing every data point
    This keeps token usage low even for a 5-year daily chart with 1000+ points."""
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


@with_cache
def get_crypto_data(symbol):
    """symbol should be Yahoo-style, e.g. 'BTC-USD', 'ETH-USD'."""
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


@with_cache
def get_forex_rate(pair):
    """pair should be Yahoo-style, e.g. 'EURUSD=X', 'GBPUSD=X'."""
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


# Major US indices, used as a proxy for "the overall market" when the
# user isn't asking about one specific stock.
MARKET_INDICES = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC"}


@with_cache
def get_market_overview():
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

        # Reuse get_news, pointed at the S&P 500 index, as a stand-in
        # for general market-moving headlines rather than one company's news.
        market_news = get_news("^GSPC", limit=5)

        if not indices:
            return {"error": "Could not fetch market index data."}

        return {"indices": indices, "market_news": market_news}
    except Exception as e:
        return {"error": f"Failed to fetch market overview: {str(e)}"}


AVAILABLE_TOOLS = {
    "get_stock_data": get_stock_data,
    "get_news": get_news,
    "get_institutional_holders": get_institutional_holders,
    "get_price_history": get_price_history,
    "get_sector_top_stocks": get_sector_top_stocks,
    "get_crypto_data": get_crypto_data,
    "get_forex_rate": get_forex_rate,
    "get_market_overview": get_market_overview,
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