import numpy as np

_row = lambda x: x

def series2X(series, size, func=_row):
    # 把时间序列转换为滑动窗口形式
    X = np.array([series[i:i+size] for i in range(len(series)-size+1)])
    return np.apply_along_axis(func, 1, X)

def series2Xy(series, size, func=_row):
    # 把时间序列转换为单步带标注形式数据
    X = np.array([series[:-1][i:i+size] for i in range(len(series)-size)])
    y = np.array(series[size:])
    return np.apply_along_axis(func, 1, X), y
