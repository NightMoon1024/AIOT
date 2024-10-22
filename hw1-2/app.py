import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import dash_table
from itertools import combinations

# 讀取 CSV 資料，指定日期格式
df = pd.read_csv('2330-training.csv', parse_dates=['Date'], dayfirst=False)

# 清理資料
def clean_column(column):
    return column.str.replace(',', '').astype(float)

df['y'] = clean_column(df['y'].astype(str))
df['x1'] = clean_column(df['x1'].astype(str))
df['x2'] = clean_column(df['x2'].astype(str))
df['x3'] = clean_column(df['x3'].astype(str))
df['x4'] = clean_column(df['x4'].astype(str))
df['x5'] = clean_column(df['x5'].astype(str))

df.set_index('Date', inplace=True)

# 計算所有特徵組合的最佳 MSE
def calculate_best_combinations():
    all_combinations = []
    # 測試所有特徵組合
    for n in range(1, 6):  # 1 到 5 個特徵
        for features in combinations(df.columns[1:], n):  # 假設 y 是第一列
            X = df[list(features)]
            y = df['y']

            # 分割資料集
            train_size = int(len(df) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # 建立多變量回歸模型
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 預測
            y_pred_test = model.predict(X_test)

            # 計算 MSE
            mse = mean_squared_error(y_test, y_pred_test)

            all_combinations.append({'features': ', '.join(features), 'mse': mse})
    
    # 根據 MSE 對特徵組合進行排序
    return sorted(all_combinations, key=lambda x: x['mse'])

# 初始化最佳組合
best_combinations = calculate_best_combinations()
best_features = best_combinations[0]['features'].split(', ')  # 取 MSE 最小的特徵

# 建立 Dash 應用
app = Dash(__name__)

# 定義應用佈局
app.layout = html.Div([
    html.H1("股價預測"),
    html.Div(f"最佳特徵組合: {best_features}"),
    dcc.Graph(id='prediction-graph'),
    dash_table.DataTable(
        id='feature-combinations-table',
        columns=[
            {'name': '特徵組合', 'id': 'features'},
            {'name': '均方誤差 (MSE)', 'id': 'mse'}
        ],
        data=best_combinations,  # 顯示最佳組合
        page_size=10
    )
])

# 定義回調以更新圖表
@app.callback(
    Output('prediction-graph', 'figure'),
    Input('feature-combinations-table', 'data')  # 不需要用戶選擇特徵
)
def update_graph(data):
    # 定義特徵與目標變量
    X = df[best_features]
    y = df['y']

    # 分割資料集
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 使用選擇的特徵進行最終預測
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 預測
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 計算上下限 (這裡假設用標準差來表示)
    residuals = y_test - y_pred_test
    std_dev = np.std(residuals)
    lower_bound = y_pred_test - 1.96 * std_dev  # 95% 置信區間下限
    upper_bound = y_pred_test + 1.96 * std_dev  # 95% 置信區間上限

    # 繪製圖表
    fig = go.Figure()

    # 繪製訓練集的實際值和預測值
    fig.add_trace(go.Scatter(x=df.index[:train_size], y=y_train,
                             mode='lines+markers',
                             name='Actual Train',
                             line=dict(color='blue', width=2)))

    fig.add_trace(go.Scatter(x=df.index[:train_size], y=y_pred_train,
                             mode='lines+markers',
                             name='Predicted Train',
                             line=dict(color='orange', dash='dash', width=2)))

    # 繪製測試集的實際值和預測值
    fig.add_trace(go.Scatter(x=df.index[train_size:], y=y_test,
                             mode='lines+markers',
                             name='Actual Test',
                             line=dict(color='green', width=2)))

    fig.add_trace(go.Scatter(x=df.index[train_size:], y=y_pred_test,
                             mode='lines+markers',
                             name='Predicted Test',
                             line=dict(color='red', dash='dash', width=2)))

    # 添加上下限填充
    fig.add_trace(go.Scatter(x=df.index[train_size:], y=lower_bound,
                             mode='lines',
                             name='Lower Bound',
                             line=dict(color='gray', dash='dash', width=1)))

    fig.add_trace(go.Scatter(x=df.index[train_size:], y=upper_bound,
                             mode='lines',
                             name='Upper Bound',
                             line=dict(color='gray', dash='dash', width=1)))

    # 填充上下限之間的區域
    fig.add_trace(go.Scatter(
        x=np.concatenate([df.index[train_size:], df.index[train_size:][::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prediction Interval',
        showlegend=True
    ))

    # 預測未來30天的股價
    last_days = df[best_features].tail(5).values.flatten()  # 取得最後5天的資料
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

    future_predictions = []
    future_lower_bounds = []
    future_upper_bounds = []

    for _ in range(30):
        # 將最佳特徵的最後5天的資料用作特徵
        X_future = last_days[-len(best_features):].reshape(1, -1)  # 確保維度正確
        future_pred = model.predict(X_future)
        future_predictions.append(future_pred[0])

        # 計算未來預測的上下限
        future_std_dev = std_dev  # 假設未來的標準差與過去相同
        future_lower_bounds.append(future_pred[0] - 1.96 * future_std_dev)
        future_upper_bounds.append(future_pred[0] + 1.96 * future_std_dev)

        # 更新 last_days 以包含新的預測
        last_days = np.append(last_days[1:], future_pred[0])  # 更新 last_days 以使用新預測

    # 將未來預測轉換為圖形
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions,
                             mode='lines+markers',
                             name='Future Predictions',
                             line=dict(color='purple', width=2)))

    # 繪製未來預測的上下限
    fig.add_trace(go.Scatter(x=future_dates, y=future_lower_bounds,
                             mode='lines',
                             name='Future Lower Bound',
                             line=dict(color='orange', dash='dash', width=1)))

    fig.add_trace(go.Scatter(x=future_dates, y=future_upper_bounds,
                             mode='lines',
                             name='Future Upper Bound',
                             line=dict(color='orange', dash='dash', width=1)))

    # 填充未來預測的上下限區域
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([future_upper_bounds, future_lower_bounds[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Future Prediction Interval',
        showlegend=True
    ))

    # 更新圖表佈局
    fig.update_layout(title='Stock Price Predictions with Prediction Intervals',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      hovermode='x unified')

    return fig  # 返回圖表

# 啟動 Dash 應用
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
