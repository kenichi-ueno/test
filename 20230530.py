from flask import Flask, render_template, request
import requests
import pandas as pd
import datetime

app = Flask(__name__)

def get_crypto_data(symbol):
    start_date = datetime.datetime.now() - datetime.timedelta(days=365 * 10)
    end_date = datetime.datetime.now()

    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": 3650,  # 10 years * 365 days
        "aggregate": 1,
        "toTs": int(end_date.timestamp()),
        "e": "CCCAGG"
    }

    try:
        response = requests.get("https://min-api.cryptocompare.com/data/v2/histoday", params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"{symbol}のデータの取得に失敗しました。エラー: {e}")

    data = response.json()

    if data["Response"] == "Success":
        price = []

        for item in data["Data"]["Data"]:
            date = datetime.datetime.fromtimestamp(item["time"]).strftime("%Y-%m-%d")
            close_price = item["close"]
            price.append({"date": date, "close_price": close_price})

        df = pd.DataFrame(price)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    else:
        raise ValueError(f"{symbol}のデータの取得に失敗しました。")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    symbol = request.form["symbol"]
    try:
        crypto_data = get_crypto_data(symbol)
        if crypto_data is None:
            result_message = f"{symbol}のデータが見つかりませんでした。"
        else:
            result_message = f"{symbol}のデータを取得しました。"
            # データの表示や処理をここに記述
    except ValueError as e:
        result_message = str(e)

    return render_template("result.html", result_message=result_message)

if __name__ == "__main__":
    app.run()

