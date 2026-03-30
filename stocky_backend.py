import os
from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from google import genai
from dotenv import load_dotenv

# 1. LOAD THE SECRET KEY
load_dotenv() 
api_key = os.getenv("GEMINI_API_KEY")

# 2. INITIALIZE THE AI CLIENT
if not api_key:
    print("CRITICAL ERROR: No API Key found in .env file!")
    client = None
else:
    client = genai.Client(api_key=api_key)

app = Flask(__name__)

# 3. CONFIGURATION
stock_map = {
    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "microsoft": "MSFT", "meta": "META"
}

def get_symbol(query):
    query = query.lower().strip()
    return stock_map.get(query, query.upper())

# 4. ROUTES
@app.route("/")
def home():
    return render_template("index.html")

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
            "prices": data["Close"].round(2).tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/details")
def details():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Format large numbers for readability (e.g., billions/trillions)
        market_cap = info.get("marketCap", 0)
        if market_cap > 1_000_000_000_000:
            mc_str = f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap > 1_000_000_000:
            mc_str = f"${market_cap / 1_000_000_000:.2f}B"
        else:
            mc_str = f"${market_cap:,}"

        return jsonify({
            "symbol": symbol,
            "price": info.get("currentPrice", "N/A"),
            "change": round(info.get("currentPrice", 0) - info.get("regularMarketPreviousClose", 0), 2),
            "market_cap": mc_str,
            "pe_ratio": info.get("trailingPE", "N/A"),
            "high_52": info.get("fiftyTwoWeekHigh", "N/A"),
            "low_52": info.get("fiftyTwoWeekLow", "N/A")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare")
def compare():
    # Example query: "AAPL, TSLA, MSFT"
    query = request.args.get("symbols", "AAPL,TSLA")
    
    # Clean up the input (remove spaces and convert to uppercase)
    raw_symbols = [s.strip() for s in query.split(",")]
    
    # Limit to 3 stocks so the chart doesn't get too messy
    symbols = [get_symbol(s) for s in raw_symbols][:3] 
    
    combined_data = {}
    shared_dates = None
    
    try:
        for sym in symbols:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="6mo")
            
            if not hist.empty:
                # Use the dates from the first successful stock to align the X-axis
                if shared_dates is None:
                    shared_dates = hist.index.strftime('%Y-%m-%d').tolist()
                
                # Save the prices for this specific symbol
                combined_data[sym] = hist['Close'].round(2).tolist()
                
        return jsonify({
            "dates": shared_dates,
            "prices": combined_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict")
def predict():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    data = yf.Ticker(symbol).history(period="1y")
    if len(data) < 10:
        return jsonify({"error": "Not enough data"}), 400
    
    prices = data["Close"].values
    x = np.arange(len(prices))
    coeff = np.polyfit(x, prices, 1)
    future_indices = np.arange(len(prices), len(prices) + 5)
    prediction = (coeff[0] * future_indices + coeff[1]).round(2).tolist()
    return jsonify({"symbol": symbol, "prediction": prediction})

# --------------------------------
# SENTIMENT ANALYSIS
# --------------------------------
@app.route("/sentiment")
def sentiment():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    
    url = f"https://news.google.com/rss/search?q={symbol}+stock"
    feed = feedparser.parse(url)
    headlines = [entry.title for entry in feed.entries[:5]]

    if not headlines:
        return jsonify({"sentiment": "No news found."})

    prompt = f"Analyze these {symbol} headlines: {' | '.join(headlines)}"

    try:
        # UPDATED: Using the 2026 Stable Model
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return jsonify({"sentiment": response.text})
    except Exception as e:
        print("Sentiment Error:", e)
        return jsonify({"error": "AI failed"})

# --------------------------------
# CHATBOT
# --------------------------------
@app.route("/chat")
def chat():
    user_message = request.args.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask a question."})

    try:
        # UPDATED: Using the 2026 Stable Model
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=user_message
        )

        # UPDATED: Simplified response reading
        reply = response.text

    except Exception as e:
        print("Gemini Error:", e)
        reply = "Sorry, I'm having trouble connecting to my brain. Check terminal for details."

    return jsonify({"reply": reply})
@app.route("/trending")
def trending():
    symbols = ["AAPL", "TSLA", "NVDA", "AMZN", "MSFT"]
    results = []
    for sym in symbols:
        data = yf.Ticker(sym).history(period="2d")
        if len(data) >= 2:
            price = round(data["Close"].iloc[-1], 2)
            change = round(price - data["Close"].iloc[-2], 2)
            results.append({"symbol": sym, "price": price, "change": change})
    return jsonify(results)

if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
