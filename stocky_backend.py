import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import numpy as np
import feedparser
from google import genai
from dotenv import load_dotenv

# LOAD ENV
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")

# INIT AI
if not api_key:
    print("WARNING: No Gemini API Key found. Using fallback AI.")
    client = None
else:
    client = genai.Client(api_key=api_key)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

stock_map = {
    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "microsoft": "MSFT", "meta": "META"
}

def get_symbol(query):
    query = query.lower().strip()
    return stock_map.get(query, query.upper())

def format_market_cap(value):
    if value in (None, "N/A"):
        return "N/A"
    try:
        val = float(value)
        abs_val = abs(val)
        if abs_val >= 1_000_000_000_000:
            return f"${val / 1_000_000_000_000:.2f}T"
        if abs_val >= 1_000_000_000:
            return f"${val / 1_000_000_000:.2f}B"
        if abs_val >= 1_000_000:
            return f"${val / 1_000_000:.2f}M"
        return f"${val:,.0f}"
    except Exception:
        return "N/A"

def fetch_snapshot(symbol):
    ticker = yf.Ticker(symbol)
    # Using ticker.info is necessary for reliable fundamental data like PE and Market Cap
    info = ticker.info or {}
    fast_info = ticker.fast_info or {}
    history = ticker.history(period="1y")
    return ticker, fast_info, info, history

# ---------------- STOCK CHART ----------------
@app.route("/stock")
def stock():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        data = yf.Ticker(symbol).history(period="6mo")
        if data.empty:
            return jsonify({"error": "No data found"}), 404
        prices = data["Close"].ffill().round(2).tolist()
        return jsonify({
            "symbol": symbol,
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": prices
        })
    except Exception as e:
        print("STOCK ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- DETAILS ----------------
@app.route("/details")
def details():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        ticker, fast_info, info, hist_1y = fetch_snapshot(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            raise Exception("No data")

        latest_price = round(hist["Close"].iloc[-1], 2)
        prev_price = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else latest_price
        change = round(latest_price - prev_price, 2)

        # Aggressive Market Cap checking
        market_cap = info.get("marketCap") or fast_info.get("market_cap") or info.get("enterpriseValue")
        if not market_cap:
            shares = info.get("sharesOutstanding")
            if shares: market_cap = shares * latest_price

        # Reliable P/E
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        
        return jsonify({
            "symbol": symbol,
            "price": latest_price,
            "change": change,
            "market_cap": format_market_cap(market_cap),
            "pe_ratio": round(float(pe_ratio), 2) if pe_ratio else "N/A",
            "high_52": round(float(info.get("fiftyTwoWeekHigh", latest_price)), 2),
            "low_52": round(float(info.get("fiftyTwoWeekLow", latest_price)), 2)
        })
    except Exception as e:
        print("DETAILS ERROR:", e)
        return jsonify({"symbol": symbol, "price": 0, "change": 0, "market_cap": "N/A", "pe_ratio": "N/A", "high_52": 0, "low_52": 0})

# ---------------- PREDICTION & SENTIMENT ----------------
# [Keep your existing PREDICTION code here]

@app.route("/sentiment")
def sentiment():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        # Use headers to mimic a browser and avoid Google blocking
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://news.google.com/rss/search?q={symbol}"
        response = requests.get(url, headers=headers)
        feed = feedparser.parse(response.content)
        headlines = [entry.title for entry in feed.entries[:10]]
        if not headlines: return jsonify({"sentiment": "No news found."})
        
        if client:
            prompt = f"Analyze sentiment for {symbol}: {' | '.join(headlines)}"
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            return jsonify({"sentiment": response.text})
        # [Fallback logic here]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- CHAT ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    # ... [Same logic as your current chat, ensuring mentioned_symbol detection is preserved]
    # Ensure this uses the fetch_snapshot() function defined above!
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
