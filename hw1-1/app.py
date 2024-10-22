import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import linear_regression_data as lr_data  # 導入數據生成模塊

# 建立 Dash 應用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("線性回歸"),
    dcc.Graph(id='regression-graph'),
    
    # Sliders for parameters a, b, c
    html.Label('a:'),
    dcc.Slider(id='slider-a', min=-10, max=10, step=1, value=1, updatemode='drag'),
    
    html.Label('b:'),
    dcc.Slider(id='slider-b', min=0, max=100, step=5, value=50, updatemode='drag'),
    
    html.Label('c:'),
    dcc.Slider(id='slider-c', min=0, max=10, step=0.5, value=1, updatemode='drag'),
    
    html.Label('Variance:'),
    dcc.Slider(id='slider-variance', min=0, max=10, step=0.5, value=1, updatemode='drag')
])

# 回調函數用來更新圖形
@app.callback(
    Output('regression-graph', 'figure'),
    [Input('slider-a', 'value'),
     Input('slider-b', 'value'),
     Input('slider-c', 'value'),
     Input('slider-variance', 'value')]
)
def update_graph(a, b, c, variance):
    # 使用 linear_regression_data.py 中的函數生成數據
    x, y = lr_data.generate_data(a, b, c, variance)

    # 擬合回歸線並獲取方程式
    fitted_line, regression_eq = lr_data.fit_regression_line(x, y)

    # 更新圖形
    figure = {
        'data': [
            go.Scatter(x=x, y=y, mode='markers', name='Data points'),
            go.Scatter(x=x, y=fitted_line, mode='lines', name=regression_eq, line=dict(color='red'))
        ],
        'layout': go.Layout(title='線性回歸圖', xaxis={'title': 'x'}, yaxis={'title': 'y'})
    }
    return figure

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
