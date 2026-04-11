import os
from flask import Flask, request, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from google import genai
from dotenv import load_dotenv

# LOAD ENV
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")

# INIT AI
if not api_key:
    print("CRITICAL ERROR: No API Key found in .env file!")
    client = None
else:
    client = genai.Client(api_key=api_key)

app = Flask(__name__)

# CORS FIX
CORS(app, resources={r"/*": {"origins": "*"}})

stock_map = {
    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "microsoft": "MSFT", "meta": "META"
}

def get_symbol(query):
    query = query.lower().strip()
    return stock_map.get(query, query.upper())

# ---------------- STOCK CHART ----------------
@app.route("/stock")
def stock():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        data = yf.Ticker(symbol).history(period="6mo")

        if data.empty:
            return jsonify({"error": "No data found"}), 404

        return jsonify({
            "symbol": symbol,
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": data["Close"].fillna(method="ffill").round(2).tolist()
        })

    except Exception as e:
        print("STOCK ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- DETAILS (FIXED) ----------------
@app.route("/details")
def details():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        ticker = yf.Ticker(symbol)

        # 🔥 IMPORTANT FIX: fallback using history
        hist = ticker.history(period="2d")

        if hist.empty:
            return jsonify({"error": "No data"}), 404

        latest_price = round(hist["Close"].iloc[-1], 2)
        prev_price = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else latest_price
        change = round(latest_price - prev_price, 2)

        # Try info but don't depend on it
        info = ticker.info if ticker.info else {}

        market_cap = info.get("marketCap", 0)
        pe_ratio = info.get("trailingPE", "N/A")
        high_52 = info.get("fiftyTwoWeekHigh", latest_price)
        low_52 = info.get("fiftyTwoWeekLow", latest_price)

        # Format market cap
        if market_cap:
            if market_cap > 1_000_000_000_000:
                mc_str = f"${market_cap / 1_000_000_000_000:.2f}T"
            elif market_cap > 1_000_000_000:
                mc_str = f"${market_cap / 1_000_000_000:.2f}B"
            else:
                mc_str = f"${market_cap:,}"
        else:
            mc_str = "N/A"

        return jsonify({
            "symbol": symbol,
            "price": latest_price,
            "change": change,
            "market_cap": mc_str,
            "pe_ratio": pe_ratio,
            "high_52": high_52,
            "low_52": low_52
        })

    except Exception as e:
        print("DETAILS ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- PREDICTION ----------------
@app.route("/predict")
def predict():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        data = yf.Ticker(symbol).history(period="1y")

        if len(data) < 10:
            return jsonify({"error": "Not enough data"}), 400
        
        prices = data["Close"].values
        x = np.arange(len(prices))

        coeff = np.polyfit(x, prices, 1)
        future_indices = np.arange(len(prices), len(prices) + 5)

        prediction = (coeff[0] * future_indices + coeff[1]).round(2).tolist()

        return jsonify({"symbol": symbol, "prediction": prediction})

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- SENTIMENT ----------------
@app.route("/sentiment")
def sentiment():
    if client is None:
        return jsonify({"error": "AI not configured"}), 500

    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        feed = feedparser.parse(url)

        headlines = [entry.title for entry in feed.entries[:5]]

        if not headlines:
            return jsonify({"sentiment": "No news found."})

        prompt = f"Analyze sentiment for {symbol} stock: {' | '.join(headlines)}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return jsonify({"sentiment": response.text})

    except Exception as e:
        print("SENTIMENT ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- CHAT ----------------
@app.route("/chat")
def chat():
    if client is None:
        return jsonify({"reply": "AI not available"})

    user_message = request.args.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask something."})

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message
        )

        return jsonify({"reply": response.text})

    except Exception as e:
        print("CHAT ERROR:", e)
        return jsonify({"reply": "AI error occurred"})

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
