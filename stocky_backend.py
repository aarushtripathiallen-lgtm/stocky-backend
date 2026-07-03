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

# INIT AI - Upgraded to latest model
if api_key:
    client = genai.Client(api_key=api_key)
    MODEL_NAME = "gemini-2.0-flash" 
else:
    print("WARNING: No Gemini API Key found.")
    client = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

stock_map = {
    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "microsoft": "MSFT", "meta": "META"
}

def get_symbol(query):
    query = query.lower().strip()
    return stock_map.get(query, query.upper())

def fetch_snapshot(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    fast_info = ticker.fast_info or {}
    hist = ticker.history(period="1d")
    return fast_info, info, hist

# ---------------- CHAT (FIXED) ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    # Supports both URL params and JSON body
    user_message = request.args.get("message") or (request.get_json().get("message") if request.is_json else "")
    
    if not user_message:
        return jsonify({"reply": "Please ask something."})

    # TRY GEMINI (LATEST MODEL)
    if client:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_message
            )
            return jsonify({"reply": response.text})
        except Exception as e:
            print("GEMINI API ERROR:", e)

    # FALLBACK LOGIC
    msg = user_message.lower()
    symbol = next((ticker for company, ticker in stock_map.items() if company in msg or ticker.lower() in msg), None)

    if symbol and any(term in msg for term in ["price", "market cap"]):
        try:
            fast_info, info, _ = fetch_snapshot(symbol)
            price = fast_info.get("last_price", "N/A")
            return jsonify({"reply": f"The current price of {symbol} is ${price}."})
        except:
            return jsonify({"reply": "I couldn't fetch data for that stock."})

    return jsonify({"reply": "I'm in offline mode. Try asking about a specific stock like 'What is the price of Apple?'"})

# ---------------- RUN (FIXED PORT BINDING) ----------------
if __name__ == "__main__":
    # Render sets the PORT environment variable.
    # If it is not set (like when running locally), default to 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
