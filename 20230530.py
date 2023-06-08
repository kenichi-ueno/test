from flask import Flask, render_template, request
import pandas_datareader as web
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import requests
import datetime

app = Flask(__name__, template_folder='templates')

def get_monthly_stock_data(stock_code, start, end):
    data_source = 'stooq'
    try:
        df = web.DataReader(stock_code, data_source=data_source, start=start, end=end)
    except:
        raise ValueError(f"{stock_code}のデータの取得に失敗しました。")
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').last()
    if len(df) == 0:
        return None
    if 'Close' in df.columns:
        df = df[['Close']]
    else:
        raise KeyError("データに'Close'列が存在しません。")
    return df

def get_monthly_crypto_data(symbol, start, end):
    price = []
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": 2000,
        "aggregate": 1,
        "to": int(end.timestamp()),
        "e": "CCCAGG"  # 使用する取引所の指定（ここではCCCAGGを指定）
    }

    try:
        response = requests.get("https://min-api.cryptocompare.com/data/v2/histoday", params=params, timeout=10)
        response.raise_for_status()  # HTTPエラーチェック
    except requests.exceptions.RequestException as e:
        raise ValueError(f"{symbol}のデータの取得に失敗しました。エラー: {e}")

    data = response.json()

    if data["Response"] == "Success":
        for item in data["Data"]["Data"]:
            date = datetime.datetime.fromtimestamp(item["time"]).strftime("%Y-%m-%d")
            close_price = item["close"]
            price.append({"date": date, "close_price": close_price})

        df = pd.DataFrame(price)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.resample("M").last()

        if len(df) == 0:
            return None
        if 'close_price' in df.columns:
            df = df[['close_price']]
        else:
            raise KeyError("データに'close_price'列が存在しません。")
        return df
    else:
        raise ValueError(f"{symbol}のデータの取得に失敗しました。")

@app.route('/')
def home():
    return render_template('portfolio.html')

@app.route('/result', methods=['POST'])
def result():
    selected_symbols = request.form.getlist('symbol1')
    for i in range(2, 11):
        symbol = request.form.get(f'symbol{i}', '')
        if symbol:
            selected_symbols.append(symbol)
    if len(selected_symbols) == 0:
        return "選択された銘柄がありません。"
    if len(selected_symbols) > 10:
        return "選択できる銘柄数の上限は10です。"

    start_date = pd.Timestamp.now() - pd.DateOffset(years=10)
    end_date = pd.Timestamp.now()

    portfolio_data = pd.DataFrame()
    if len(selected_symbols) == 0:
        return "選択された銘柄がありません。"

    for symbol in selected_symbols:
        if symbol.startswith('CRYPTO:'):
            crypto_data = get_monthly_crypto_data(symbol, start=start_date, end=end_date)
            if crypto_data is None:
                return f"{symbol}のデータが取得できませんでした."
            portfolio_data[symbol] = crypto_data['close_price'].values
        else:
            stock_data = get_monthly_stock_data(symbol, start=start_date, end=end_date)
            if stock_data is None:
                return f"{symbol}のデータが取得できませんでした."
            portfolio_data[symbol] = stock_data['Close'].values

    num_symbols = len(selected_symbols)

    returns = portfolio_data.pct_change().mean() * np.sqrt(12)  # 年率換算修正
    returns = pd.Series(returns.values, index=returns.index)  # 追加

    # 共分散行列を計算
    cov_matrix = portfolio_data.pct_change().cov().to_numpy()

    # 目的関数: ポートフォリオの分散
    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # 投資比率の制約条件
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # 投資比率の総和は1
        {'type': 'ineq', 'fun': lambda weights: weights}
    ]

    # 目標リターンの範囲
    min_return = returns.min()
    max_return = returns.max()
    target_returns = np.arange(min_return, max_return + 0.01, 0.01)

    # 最小分散のポートフォリオを格納するリスト
    min_variance_portfolios = []

    # 各目標リターンごとに最小分散のポートフォリオを算出
    for target_return in target_returns:
        return_constraint = {'type': 'eq', 'fun': lambda weights: np.dot(weights, returns.values) - target_return}
        # 最小分散ポートフォリオの初期値
        initial_weights = np.ones(num_symbols) / num_symbols
        # 最適化の実行
        result = minimize(objective, initial_weights, method='SLSQP', constraints=constraints + [return_constraint])
        # 最適な投資比率を取得
        optimal_weights = result.x
        # 最小分散ポートフォリオの情報を格納
        min_variance_portfolios.append(optimal_weights)

    # タイトルの表示
    title = " ".join(selected_symbols)

    # 結果の表示
    result_table = []
    for i, target_return in enumerate(target_returns):
        result_row = []
        for j in range(num_symbols):
                    result_row.append(f"{np.dot(min_variance_portfolios[i], returns) * 100:.2f}")
        result_row.append(f"{np.dot(min_variance_portfolios[i], np.dot(cov_matrix, min_variance_portfolios[i])):.6f}")
        std_dev = np.sqrt(np.dot(min_variance_portfolios[i], np.dot(cov_matrix, min_variance_portfolios[i]))) * np.sqrt(12) * 100
        result_row.append(f"{std_dev:.2f}")
        result_table.append(result_row)

    returns = portfolio_data.pct_change().mean() * np.sqrt(12)  # 年率換算修正
    std_devs = portfolio_data.pct_change().std() * np.sqrt(12)  # 年率換算修正

    # 各銘柄のリターンと標準偏差の表示
    symbol_results = []
    for j in range(num_symbols):
        symbol = selected_symbols[j]
        return_percentage = returns[j] * 100
        std_dev_percentage = std_devs[j] * 100
        symbol_results.append([symbol, f"{return_percentage:.2f}%", f"{std_dev_percentage:.2f}%"])

    # 結果をテンプレートに渡す
    result_html = render_template('result.html', title=title, result_table=result_table, symbol_results=symbol_results, symbol_count=len(selected_symbols))
    return result_html

@app.route('/error')
def error():
    return render_template('error.html')

@app.errorhandler(Exception)
def handle_error(e):
    error_message = "An error occurred while processing your request. Please try again later."
    app.logger.exception(error_message)
    return render_template('error.html', error=error_message), 500

@app.errorhandler(404)
def handle_not_found_error(e):
    error_message = "ページが見つかりません"
    app.logger.error(error_message)
    return render_template('error.html', error=error_message), 404

if __name__ == '__main__':
    app.debug = True
    app.run()

