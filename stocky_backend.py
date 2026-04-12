import os
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
<<<<<<< HEAD
=======

# CORS FIX
>>>>>>> 96d794b (Fixed chat display)
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

<<<<<<< HEAD
        prices = data["Close"].ffill().round(2).tolist()

        return jsonify({
            "symbol": symbol,
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": prices
=======
        return jsonify({
            "symbol": symbol,
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": data["Close"].fillna(method="ffill").round(2).tolist()
>>>>>>> 96d794b (Fixed chat display)
        })

    except Exception as e:
        print("STOCK ERROR:", e)
        return jsonify({"error": str(e)}), 500

<<<<<<< HEAD
# ---------------- DETAILS ----------------
=======
# ---------------- DETAILS (FIXED) ----------------
>>>>>>> 96d794b (Fixed chat display)
@app.route("/details")
def details():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        ticker = yf.Ticker(symbol)

        hist = ticker.history(period="5d")

        if hist.empty:
            raise Exception("No data")

        latest_price = round(hist["Close"].iloc[-1], 2)
        prev_price = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else latest_price
        change = round(latest_price - prev_price, 2)

        return jsonify({
            "symbol": symbol,
            "price": latest_price,
            "change": change,
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "high_52": latest_price,
            "low_52": latest_price
        })

    except Exception as e:
        print("DETAILS ERROR:", e)

        # 🔥 FALLBACK DATA (prevents undefined UI)
        return jsonify({
            "symbol": symbol,
            "price": 0,
            "change": 0,
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "high_52": 0,
            "low_52": 0
        })

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

<<<<<<< HEAD
# ---------------- SENTIMENT (WITH FALLBACK) ----------------
@app.route("/sentiment")
def sentiment():
=======
# ---------------- SENTIMENT ----------------
@app.route("/sentiment")
def sentiment():
    if client is None:
        return jsonify({"error": "AI not configured"}), 500

>>>>>>> 96d794b (Fixed chat display)
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        feed = feedparser.parse(url)
<<<<<<< HEAD

        headlines = [entry.title for entry in feed.entries[:5]]

        if not headlines:
            return jsonify({"sentiment": "No news found."})

        # TRY GEMINI
        if client:
            try:
                prompt = f"Analyze sentiment for {symbol} stock: {' | '.join(headlines)}"

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )

                return jsonify({"sentiment": response.text})

            except Exception as ai_error:
                print("GEMINI FAILED:", ai_error)

        # FALLBACK LOGIC
        positive_words = ["gain", "rise", "up", "surge", "profit", "growth"]
        negative_words = ["fall", "drop", "loss", "down", "decline"]

        score = 0
        for h in headlines:
            for word in positive_words:
                if word in h.lower():
                    score += 1
            for word in negative_words:
                if word in h.lower():
                    score -= 1

        if score > 0:
            sentiment = "📈 Positive sentiment based on recent news."
        elif score < 0:
            sentiment = "📉 Negative sentiment based on recent news."
        else:
            sentiment = "⚖️ Neutral sentiment based on recent news."

        return jsonify({"sentiment": sentiment})
=======

        headlines = [entry.title for entry in feed.entries[:5]]

        if not headlines:
            return jsonify({"sentiment": "No news found."})

        prompt = f"Analyze sentiment for {symbol} stock: {' | '.join(headlines)}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return jsonify({"sentiment": response.text})
>>>>>>> 96d794b (Fixed chat display)

    except Exception as e:
        print("SENTIMENT ERROR:", e)
        return jsonify({"error": str(e)}), 500

<<<<<<< HEAD
# ---------------- CHAT (WITH FALLBACK) ----------------
@app.route("/chat")
def chat():
    user_message = request.args.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask something."})

    # TRY GEMINI
    if client:
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=user_message
            )
            return jsonify({"reply": response.text})

        except Exception as e:
            print("GEMINI CHAT FAILED:", e)

    # FALLBACK BOT
    msg = user_message.lower()

    if "price" in msg:
        reply = "Search any stock above to see its latest price 📈"
    elif "trend" in msg:
        reply = "Check the chart above — green means upward trend 🚀"
    elif "hello" in msg:
        reply = "Hey! I'm your Stocky assistant 🤖"
    else:
        reply = "I'm currently in offline mode, but still here to help!"

    return jsonify({"reply": reply})
=======
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
>>>>>>> 96d794b (Fixed chat display)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
