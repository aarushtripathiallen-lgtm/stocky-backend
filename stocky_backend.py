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
CORS(app, resources={r"/*": {"origins": "*"}})

# Pre-defined map for common tickers
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
    info = ticker.fast_info or {}
    quote_type = ticker.info or {}
    history = ticker.history(period="1y")
    return ticker, info, quote_type, history

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

        market_cap = (
            fast_info.get("market_cap")
            or info.get("marketCap")
            or info.get("enterpriseValue")
        )
        pe_ratio = (
            info.get("trailingPE")
            or info.get("forwardPE")
            or fast_info.get("trailing_pe")
        )
        high_52 = fast_info.get("year_high")
        low_52 = fast_info.get("year_low")

        if hist_1y is not None and not hist_1y.empty:
            if not high_52:
                high_52 = float(hist_1y["High"].max())
            if not low_52:
                low_52 = float(hist_1y["Low"].min())

        return jsonify({
            "symbol": symbol,
            "price": latest_price,
            "change": change,
            "market_cap": format_market_cap(market_cap),
            "pe_ratio": round(float(pe_ratio), 2) if pe_ratio else "N/A",
            "high_52": round(float(high_52), 2) if high_52 else latest_price,
            "low_52": round(float(low_52), 2) if low_52 else latest_price
        })

    except Exception as e:
        print("DETAILS ERROR:", e)
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

# ---------------- SENTIMENT (WITH FALLBACK) ----------------
@app.route("/sentiment")
def sentiment():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        feed = feedparser.parse(url)
        headlines = [entry.title for entry in feed.entries[:10]]

        if not headlines:
            return jsonify({"sentiment": "No news found."})

        # TRY GEMINI AI
        if client:
            try:
                prompt = (
                    f"You are a stock news analyst. Analyze sentiment for {symbol} "
                    f"using these headlines: {' | '.join(headlines)}. "
                    "Respond in 4 short sections with labels exactly as:\n"
                    "Overall sentiment: <Positive/Neutral/Negative>\n"
                    "Confidence: <0-100>%\n"
                    "Drivers: 3 bullet points\n"
                    "Risks to watch: 2 bullet points\n"
                    "Keep it concise but informative."
                )

                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt
                )

                return jsonify({"sentiment": response.text})

            except Exception as ai_error:
                print("GEMINI FAILED:", ai_error)

        # FALLBACK ANALYSIS
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
        
        if score > 1:
            sentiment_label = "Positive"
        elif score < -1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        confidence = min(95, 55 + abs(score) * 8)
        top_drivers = "\n".join([f"- {headline}" for headline in headlines[:3]])
        sentiment_summary = (
            f"Overall sentiment: {sentiment_label}\n"
            f"Confidence: {confidence}%\n"
            "Drivers:\n"
            f"{top_drivers}\n"
            "Risks to watch:\n"
            "- Headline momentum can reverse quickly.\n"
            "- Confirm with earnings guidance and valuation metrics."
        )

        return jsonify({"sentiment": sentiment_summary})

    except Exception as e:
        print("SENTIMENT ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- CHAT (WITH FALLBACK) ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_message = request.args.get("message", "")
    if not user_message and request.is_json:
        body = request.get_json(silent=True) or {}
        user_message = body.get("message", "")
    if not user_message and request.method == "POST":
        user_message = request.form.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask something."})

    # TRY GEMINI AI
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

    if "price" in msg or "market cap" in msg or "p/e" in msg or mentioned_symbol:
        symbol = mentioned_symbol or "AAPL"
        try:
            _, fast_info, info, hist_1y = fetch_snapshot(symbol)
            latest = fast_info.get("last_price")
            if not latest and hist_1y is not None and not hist_1y.empty:
                latest = float(hist_1y["Close"].iloc[-1])
            cap = format_market_cap(fast_info.get("market_cap") or info.get("marketCap"))
            pe = info.get("trailingPE") or info.get("forwardPE")
            pe_text = f"{float(pe):.2f}" if pe else "N/A"
            latest_text = f"${float(latest):.2f}" if latest else "N/A"
            reply = (
                f"Here’s a quick snapshot for {symbol}: Price {latest_text}, "
                f"Market Cap {cap}, P/E {pe_text}. "
                "Ask for sentiment or trend if you want a deeper view."
            )
        except Exception:
            reply = "I couldn't pull a live snapshot right now. Try again in a moment."
    elif "trend" in msg:
        reply = "Open the 6-month chart above and compare recent highs/lows for trend direction 🚀"
    elif "hello" in msg or "hi" in msg:
        reply = "Hey! I'm your Stocky assistant 🤖 Ask me about a ticker, valuation, or sentiment."
    else:
        reply = (
            "I'm in offline AI mode but still here to help. "
            "Try: 'AAPL price', 'TSLA market cap', or 'NVDA sentiment'."
        )

    return jsonify({"reply": reply})

# ---------------- RUN ----------------
if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
