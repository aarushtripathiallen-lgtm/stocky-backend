import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import feedparser
from google import genai
from google.genai import types
from dotenv import load_dotenv

# LOAD ENV
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# INIT AI
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

def format_market_cap(value):
    if value in (None, "N/A"): return "N/A"
    try:
        val = float(value)
        if val >= 1e12: return f"${val / 1e12:.2f}T"
        if val >= 1e9: return f"${val / 1e9:.2f}B"
        return f"${val / 1e6:.2f}M"
    except: return "N/A"

# ---------------- 1. STOCK CHART ROUTE ----------------
@app.route("/stock")
def stock():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        data = yf.Ticker(symbol).history(period="6mo")
        if data.empty: return jsonify({"error": "No data"}), 404
        return jsonify({
            "symbol": symbol,
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": data["Close"].ffill().round(2).tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- 2. DETAILS ROUTE (FIXED) ----------------
@app.route("/details")
def details():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Robust price fetching
        latest_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev_price = info.get("previousClose") or latest_price
        change = round(latest_price - prev_price, 2)

        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")

        return jsonify({
            "symbol": symbol,
            "price": round(latest_price, 2),
            "change": change,
            "market_cap": format_market_cap(market_cap),
            "pe_ratio": round(float(pe_ratio), 2) if pe_ratio else "N/A",
            "high_52": round(float(info.get("fiftyTwoWeekHigh", latest_price)), 2),
            "low_52": round(float(info.get("fiftyTwoWeekLow", latest_price)), 2)
        })
    except Exception as e:
        print("DETAILS ERROR:", e)
        return jsonify({"symbol": symbol, "price": 0, "change": 0, "market_cap": "N/A", "pe_ratio": "N/A", "high_52": 0, "low_52": 0})

# ---------------- 3. SENTIMENT ROUTE ----------------
@app.route("/sentiment")
def sentiment():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        response = requests.get(url, headers=headers)
        feed = feedparser.parse(response.content)
        headlines = [entry.title for entry in feed.entries[:5]]
        
        if not headlines: return jsonify({"sentiment": "No news found."})
        if client:
            prompt = f"Analyze sentiment for {symbol}: {' | '.join(headlines)}"
            res = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return jsonify({"sentiment": res.text})
        return jsonify({"sentiment": "AI Offline."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- 4. CHAT ROUTE (TUTOR UPGRADE) ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_message = request.args.get("message") or (request.get_json().get("message") if request.is_json else "")
    if not user_message: return jsonify({"reply": "Please ask something."})

    if client:
        try:
            # We add a System Instruction to make the AI act like a teacher
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction="You are Stocky, an expert financial tutor. Your goal is to teach the user about the stock market. Explain complex financial concepts in simple, easy-to-understand terms. Keep answers concise."
                )
            )
            return jsonify({"reply": response.text})
        except Exception as e:
            print("GEMINI API ERROR:", e)

    return jsonify({"reply": "I'm in offline mode right now."})

# ---------------- 5. PREDICT ROUTE ----------------
@app.route("/predict")
def predict():
    return jsonify({"forecast": [0, 0, 0, 0, 0]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
