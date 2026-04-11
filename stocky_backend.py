import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import yfinance as yf
import numpy as np
import feedparser
from google import genai
from dotenv import load_dotenv

# 1. LOAD ENV
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. INIT AI
# Using Gemini 3.1 Flash for both speed and cost-efficiency
MODEL_ID = "gemini-3.1-flash"

if not api_key:
    print("WARNING: No Gemini API Key found. Using fallback AI.")
    client = None
else:
    client = genai.Client(api_key=api_key)

app = Flask(__name__)
# Set a secret key to use Flask Sessions for chat memory
app.secret_key = os.getenv("FLASK_SECRET_KEY", "stocky_secret_123")
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
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")

        if hist.empty:
            return jsonify({"error": "No data"}), 404

        latest_price = round(hist["Close"].iloc[-1], 2)
        prev_price = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else latest_price
        change = round(latest_price - prev_price, 2)

        info = ticker.info if ticker.info else {}

        market_cap = info.get("marketCap", 0)
        pe_ratio = info.get("trailingPE", "N/A")
        high_52 = info.get("fiftyTwoWeekHigh", latest_price)
        low_52 = info.get("fiftyTwoWeekLow", latest_price)

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
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        feed = feedparser.parse(url)
        headlines = [entry.title for entry in feed.entries[:5]]

        if not headlines:
            return jsonify({"sentiment": "No news found."})

        # USE UPDATED MODEL: Gemini 3.1 Flash
        if client:
            try:
                prompt = f"Analyze sentiment for {symbol} stock based on these headlines: {' | '.join(headlines)}. Provide a brief summary."
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt
                )
                return jsonify({"sentiment": response.text})
            except Exception as ai_error:
                print("GEMINI SENTIMENT FAILED:", ai_error)

        # FALLBACK LOGIC
        positive_words = ["gain", "rise", "up", "surge", "profit", "growth"]
        negative_words = ["fall", "drop", "loss", "down", "decline"]
        score = sum(1 for h in headlines for w in positive_words if w in h.lower())
        score -= sum(1 for h in headlines for w in negative_words if w in h.lower())

        sentiment_msg = "📈 Positive" if score > 0 else "📉 Negative" if score < 0 else "⚖️ Neutral"
        return jsonify({"sentiment": f"{sentiment_msg} sentiment based on recent news."})

    except Exception as e:
        print("SENTIMENT ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- CHAT (WITH MEMORY & 3.1 FLASH) ----------------
@app.route("/chat")
def chat():
    user_message = request.args.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask something."})

    if client:
        try:
            # Simple list-based history in session
            if "history" not in session:
                session["history"] = []

            # Add user message to history
            session["history"].append({"role": "user", "parts": [{"text": user_message}]})

            # Generate response using unified Gemini 3.1 Flash
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=session["history"]
            )

            # Add AI response to history
            session["history"].append({"role": "model", "parts": [{"text": response.text}]})
            
            # Keep history manageable (last 10 messages)
            if len(session["history"]) > 10:
                session["history"] = session["history"][-10:]

            return jsonify({"reply": response.text})

        except Exception as e:
            print("GEMINI CHAT FAILED:", e)

    # FALLBACK BOT
    msg = user_message.lower()
    if "price" in msg:
        reply = "Search any stock above to see its latest price 📈"
    elif "hello" in msg:
        reply = "Hey! I'm your Stocky assistant 🤖"
    else:
        reply = "I'm currently in offline mode, but still here to help!"

    return jsonify({"reply": reply})

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
