import json
from flask import Flask, jsonify, send_from_directory, request
import config

app = Flask(__name__)

@app.route('/data')
def get_data():
    pair = request.args.get('pair', default='BTC_USDT', type=str)
    filename = f"data/{pair}_visualization_data.json"
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Data not found for the specified pair"}), 404

@app.route('/pairs')
def list_pairs():
    try:
        pairs = [p.replace('/', '_') for p in getattr(config, 'TRADING_PAIRS', [])]
        return jsonify(pairs)
    except Exception:
        return jsonify([])

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
