import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import feedparser
from google import genai
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

def fetch_snapshot(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    fast_info = ticker.fast_info or {}
    history = ticker.history(period="1y")
    return ticker, fast_info, info, history

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

# ---------------- 2. DETAILS ROUTE ----------------
@app.route("/details")
def details():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        ticker, fast_info, info, hist_1y = fetch_snapshot(symbol)
        hist = ticker.history(period="5d")
        latest_price = round(hist["Close"].iloc[-1], 2) if not hist.empty else 0
        prev_price = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else latest_price

        market_cap = info.get("marketCap") or fast_info.get("market_cap")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")

        return jsonify({
            "symbol": symbol,
            "price": latest_price,
            "change": round(latest_price - prev_price, 2),
            "market_cap": format_market_cap(market_cap),
            "pe_ratio": round(float(pe_ratio), 2) if pe_ratio else "N/A",
            "high_52": round(float(info.get("fiftyTwoWeekHigh", latest_price)), 2),
            "low_52": round(float(info.get("fiftyTwoWeekLow", latest_price)), 2)
        })
    except:
        return jsonify({"symbol": symbol, "price": 0, "change": 0, "market_cap": "N/A", "pe_ratio": "N/A", "high_52": 0, "low_52": 0})

# ---------------- 3. SENTIMENT ROUTE ----------------
@app.route("/sentiment")
def sentiment():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://news.google.com/rss/search?q={symbol}"
        response = requests.get(url, headers=headers)
        feed = feedparser.parse(response.content)
        headlines = [entry.title for entry in feed.entries[:10]]
        
        if not headlines: return jsonify({"sentiment": "No news found."})
        if client:
            prompt = f"Analyze sentiment for {symbol}: {' | '.join(headlines)}"
            res = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return jsonify({"sentiment": res.text})
        return jsonify({"sentiment": "AI Offline."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- 4. PREDICT ROUTE ----------------
@app.route("/predict")
def predict():
    # Placeholder to prevent 404 errors. 
    # If you had machine learning logic here previously, you can paste it back inside this block.
    return jsonify({"forecast": [0, 0, 0, 0, 0]})

# ---------------- 5. CHAT ROUTE ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_message = request.args.get("message") or (request.get_json().get("message") if request.is_json else "")
    if not user_message: return jsonify({"reply": "Please ask something."})

    if client:
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=user_message)
            return jsonify({"reply": response.text})
        except Exception as e:
            print("GEMINI API ERROR:", e)

    msg = user_message.lower()
    symbol = next((t for c, t in stock_map.items() if c in msg or t.lower() in msg), None)
    if symbol and any(term in msg for term in ["price", "market cap"]):
        try:
            _, fast_info, info, _ = fetch_snapshot(symbol)
            price = fast_info.get("last_price", "N/A")
            return jsonify({"reply": f"The current price of {symbol} is ${price:.2f}."})
        except:
            pass
    return jsonify({"reply": "I'm in offline mode. Try asking 'What is the price of Apple?'"})

# ---------------- RUN (FIXED PORT BINDING) ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
