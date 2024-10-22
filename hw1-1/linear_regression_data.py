import numpy as np

# 生成數據點
def generate_data(a, b, c, variance, n=100):
    x = np.linspace(-10, 10, n)
    noise = np.random.normal(0, variance, n)
    y = a * x + b + c * noise
    return x, y

# 擬合回歸線並返回方程式
def fit_regression_line(x, y):
    coefficients = np.polyfit(x, y, 1)  # 進行線性回歸擬合
    regression_eq = f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}'  # 動態生成方程式字串
    fitted_line = coefficients[0] * x + coefficients[1]  # 擬合的回歸線
    return fitted_line, regression_eq
