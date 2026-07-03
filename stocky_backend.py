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
client = genai.Client(api_key=api_key) if api_key else None

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
    if value in (None, "N/A"): return "N/A"
    try:
        val = float(value)
        abs_val = abs(val)
        if abs_val >= 1e12: return f"${val / 1e12:.2f}T"
        if abs_val >= 1e9: return f"${val / 1e9:.2f}B"
        return f"${val / 1e6:.2f}M"
    except: return "N/A"

def fetch_snapshot(symbol):
    ticker = yf.Ticker(symbol)
    # Using .info is more reliable for fundamental data
    info = ticker.info or {}
    fast_info = ticker.fast_info or {}
    history = ticker.history(period="1y")
    return ticker, fast_info, info, history

# ---------------- CHAT (FIXED) ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_message = request.args.get("message", "")
    if not user_message and request.is_json:
        user_message = request.get_json().get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask something."})

    # TRY GEMINI
    if client:
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=user_message
            )
            if response and hasattr(response, "text"):
                return jsonify({"reply": response.text})
        except Exception as e:
            print("GEMINI CHAT FAILED:", e)

    # FALLBACK BOT
    msg = user_message.lower()
    mentioned_symbol = None
    for company, ticker in stock_map.items():
        if company in msg or ticker.lower() in msg:
            mentioned_symbol = ticker
            break

    # Improved logic to detect intent
    if any(term in msg for term in ["price", "market cap", "p/e"]):
        symbol = mentioned_symbol or "AAPL"
        try:
            _, fast_info, info, hist_1y = fetch_snapshot(symbol)
            latest = fast_info.get("last_price")
            if not latest and not hist_1y.empty:
                latest = float(hist_1y["Close"].iloc[-1])
            
            cap = format_market_cap(fast_info.get("market_cap") or info.get("marketCap"))
            pe = info.get("trailingPE") or info.get("forwardPE")
            
            reply = (f"Snapshot for {symbol}: Price ${latest:.2f}, "
                     f"Market Cap {cap}, P/E {pe:.2f if pe else 'N/A'}.")
            return jsonify({"reply": reply})
        except Exception as e:
            print("SNAPSHOT ERROR:", e)
            return jsonify({"reply": "I couldn't pull a live snapshot right now."})

    return jsonify({"reply": "I'm in offline mode. Try: 'AAPL price' or 'TSLA sentiment'."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
