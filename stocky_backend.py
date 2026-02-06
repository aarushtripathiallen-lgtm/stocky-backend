from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# =========================
# CHATBOT API
# =========================
@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '').upper()

    greetings = ['HI', 'HELLO', 'HEY']
    symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMZN']

    # Greeting
    if any(greet in msg for greet in greetings):
        reply = "Hello 👋 I'm Stocky. Ask me about stock prices or investing."

    # Live stock price in chat
    elif any(sym in msg for sym in symbols):
        symbol = next(sym for sym in symbols if sym in msg)
        data = yf.Ticker(symbol).history(period='1d')

        if not data.empty:
            price = round(data['Close'].iloc[-1], 2)
            reply = f"The latest price of {symbol} is ${price}"
        else:
            reply = "Sorry, I couldn't fetch the stock price right now."

    # Investment advice
    elif 'INVEST' in msg:
        reply = "Long-term investing with diversification helps reduce risk 📈"

    elif 'RISK' in msg:
        reply = "Market risk depends on volatility, company fundamentals, and global factors."

    elif 'STOCK' in msg:
        reply = "Stocks represent ownership in a company and can grow via price appreciation."

    else:
        reply = "Ask me about stock prices (AAPL, TSLA, MSFT) or investing."

    return jsonify({'reply': reply})


# =========================
# REAL-TIME STOCK DATA API
# =========================
@app.route('/stock')
def stock():
    symbol = request.args.get('symbol', 'AAPL')
    data = yf.Ticker(symbol).history(period='5d')

    return jsonify({
        'symbol': symbol,
        'dates': data.index.strftime('%Y-%m-%d').tolist(),
        'prices': data['Close'].tolist()
    })


# =========================
# ML PRICE PREDICTION API
# =========================
@app.route('/predict')
def predict():
    symbol = request.args.get('symbol', 'AAPL')
    data = yf.Ticker(symbol).history(period='1mo')

    data['Day'] = np.arange(len(data))
    X = data[['Day']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array(range(len(data), len(data) + 5)).reshape(-1, 1)
    predictions = model.predict(future_days).tolist()

    return jsonify({
        'symbol': symbol,
        'prediction': predictions
    })


# =========================
# RUN SERVER
# =========================
if __name__ == '__main__':
    app.run(debug=True)
