import os
import logging
import requests
import feedparser
import yfinance as yf
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---------------- CONFIG / ENV ----------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stocky")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# CORS: with session cookies, "*" cannot be combined with credentials in real
# browsers. Set FRONTEND_ORIGIN to your actual frontend URL in production.
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

# How much conversation history to retain in the session.
MAX_HISTORY_TURNS = 8          # number of user+model turn PAIRS to keep
MAX_HISTORY_CHARS = 6000       # hard cap on total serialized history size
MAX_MESSAGE_CHARS = 2000       # reject absurdly long single messages

# ---------------- INIT AI CLIENT ----------------
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        client = None
else:
    logger.warning("GEMINI_API_KEY not set - chatbot will run in offline fallback mode.")

# ---------------- INIT FLASK ----------------
app = Flask(__name__)

# Sessions require a secret key to sign the cookie. A random fallback stops
# the app from crashing, but it also means all sessions reset on every
# restart/deploy. Set FLASK_SECRET_KEY in your environment for production.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)
if not os.getenv("FLASK_SECRET_KEY"):
    logger.warning("FLASK_SECRET_KEY not set - using an ephemeral key; sessions won't survive a restart.")

# Needed for cross-origin requests that carry the session cookie.
app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
)

CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}}, supports_credentials=True)

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
        if val >= 1e12:
            return f"${val / 1e12:.2f}T"
        if val >= 1e9:
            return f"${val / 1e9:.2f}B"
        return f"${val / 1e6:.2f}M"
    except Exception:
        return "N/A"


# ---------------- CHAT HISTORY (SESSION) HELPERS ----------------

def get_history():
    return session.get("chat_history", [])


def truncate_history(history):
    """
    Keep the conversation bounded so it doesn't blow past the session
    cookie's ~4KB limit or the model's context window.

    Strategy:
      1. Keep only the most recent MAX_HISTORY_TURNS user+model pairs.
      2. On top of that, enforce a hard character budget by dropping the
         oldest remaining turns first.
    """
    if len(history) > MAX_HISTORY_TURNS * 2:
        history = history[-MAX_HISTORY_TURNS * 2:]

    total_chars = sum(len(turn.get("text", "")) for turn in history)
    while total_chars > MAX_HISTORY_CHARS and len(history) > 2:
        removed = history.pop(0)
        total_chars -= len(removed.get("text", ""))

    return history


def save_history(history):
    session["chat_history"] = truncate_history(history)
    session.modified = True


def history_to_contents(history):
    """Convert stored {role, text} turns into genai Content objects."""
    contents = []
    for turn in history:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn.get("text", ""))]))
    return contents


SYSTEM_INSTRUCTION = (
    "You are Stocky, a friendly financial tutor and market-insight assistant. "
    "Explain financial concepts clearly and simply. Do not give personalized "
    "investment advice or tell users what to buy/sell - focus on education "
    "and general market context. Keep answers concise."
)


# ---------------- 1. STOCK CHART ROUTE ----------------
@app.route("/stock")
def stock():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        data = yf.Ticker(symbol).history(period="6mo")
        if data.empty:
            return jsonify({"error": "No data"}), 404
        return jsonify({
            "symbol": symbol,
            "dates": data.index.strftime("%Y-%m-%d").tolist(),
            "prices": data["Close"].ffill().round(2).tolist()
        })
    except Exception as e:
        logger.warning(f"/stock failed for {symbol}: {e}")
        return jsonify({"error": "Could not fetch chart data."}), 500


# ---------------- 2. DETAILS ROUTE ----------------
@app.route("/details")
def details():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

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
        logger.warning(f"/details failed for {symbol}: {e}")
        return jsonify({
            "symbol": symbol, "price": 0, "change": 0, "market_cap": "N/A",
            "pe_ratio": "N/A", "high_52": 0, "low_52": 0
        })


# ---------------- 3. SENTIMENT ROUTE ----------------
@app.route("/sentiment")
def sentiment():
    query = request.args.get("symbol", "AAPL")
    symbol = get_symbol(query)

    # Fetch headlines - network calls get their own try/except so a dead
    # news feed never takes down the whole route.
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        response = requests.get(url, headers=headers, timeout=6)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"News fetch failed for {symbol}: {e}")
        return jsonify({"sentiment": "News feed is unavailable right now."})

    try:
        feed = feedparser.parse(response.content)
        headlines = [entry.title for entry in feed.entries[:5] if getattr(entry, "title", None)]
    except Exception as e:
        logger.warning(f"Feed parsing failed for {symbol}: {e}")
        headlines = []

    if not headlines:
        return jsonify({"sentiment": "No recent news found."})

    if not client:
        return jsonify({"sentiment": "AI is offline - cannot analyze sentiment."})

    try:
        prompt = (
            f"Based only on these headlines, give the overall market sentiment "
            f"(bullish, bearish, or neutral) for {symbol} in 2-3 sentences: "
            f"{' | '.join(headlines)}"
        )
        res = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = (getattr(res, "text", None) or "").strip()
        return jsonify({"sentiment": text or "No sentiment could be generated."})
    except Exception as e:
        logger.exception(f"Gemini sentiment call failed for {symbol}")
        return jsonify({"sentiment": "Sentiment analysis is temporarily unavailable."})


# ---------------- 4. CHAT ROUTE ----------------
@app.route("/chat", methods=["GET", "POST"])
def chat():
    body = request.get_json(silent=True) or {}
    user_message = (request.args.get("message") or body.get("message") or "").strip()

    if not user_message:
        return jsonify({"reply": "Please ask something."}), 400

    if len(user_message) > MAX_MESSAGE_CHARS:
        return jsonify({"reply": "That message is too long - please shorten it and try again."}), 400

    history = get_history()

    if client:
        try:
            past_contents = history_to_contents(history)
            chat_session = client.chats.create(
                model=MODEL_NAME,
                history=past_contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=800,
                    temperature=0.6,
                ),
            )
            response = chat_session.send_message(user_message)
            reply_text = (getattr(response, "text", None) or "").strip()
            if not reply_text:
                reply_text = "I couldn't generate a response - could you rephrase that?"

            history.append({"role": "user", "text": user_message})
            history.append({"role": "model", "text": reply_text})
            save_history(history)

            return jsonify({"reply": reply_text})

        except Exception as e:
            # Log the real error server-side; never echo raw exception text
            # (stack traces, internal URLs, etc.) back to the client.
            logger.exception("Gemini API error in /chat")
            return jsonify({
                "reply": "Sorry, I'm having trouble reaching the AI service right now. Please try again shortly."
            }), 502

    logger.warning("Gemini client unavailable - serving offline fallback for /chat.")
    return offline_chat_fallback(user_message)


def offline_chat_fallback(user_message):
    """Best-effort reply when no GEMINI_API_KEY is configured."""
    msg = user_message.lower()
    symbol = next((t for c, t in stock_map.items() if c in msg or t.lower() in msg), None)

    if symbol and any(term in msg for term in ["price", "market cap"]):
        try:
            ticker = yf.Ticker(symbol)
            fast_info = ticker.fast_info
            # fast_info behaves like a dict in recent yfinance versions;
            # fall back to attribute access for older ones.
            price = None
            try:
                price = fast_info["lastPrice"]
            except (TypeError, KeyError):
                price = getattr(fast_info, "last_price", None)

            if price:
                return jsonify({"reply": f"The current price of {symbol} is ${float(price):.2f}."})
        except Exception as e:
            logger.warning(f"Offline price fetch failed for {symbol}: {e}")

    return jsonify({
        "reply": "I'm in offline mode. Add a GEMINI_API_KEY environment variable to enable the AI assistant."
    })


@app.route("/chat/reset", methods=["POST"])
def reset_chat():
    """Lets the frontend start a fresh conversation (e.g. a 'New chat' button)."""
    session.pop("chat_history", None)
    return jsonify({"status": "cleared"})


# ---------------- 5. PREDICT ROUTE ----------------
@app.route("/predict")
def predict():
    return jsonify({"forecast": [0, 0, 0, 0, 0]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
