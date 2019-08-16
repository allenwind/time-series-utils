import numpy as np
from statsmodels.tsa.stattools import adfuller

def check_time_series(series):
    if not isinstance(series, np.ndarray):
        raise ValueError("time series invalidation")

def check_none_stationarity(series):
    # 检测时间序列是否具有非平稳性
    # wiki:
    # https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
    return adfuller(series)[1] > 0.05

def check_white_noise(series):
    # 检测时间序列或残差是否为白噪声
    # wiki:
    # https://en.wikipedia.org/wiki/White_noise
    # https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test
    return adfuller(series)[1] > 0.05

def check_random_walk(series):
    # 检测时间序列是否为随机游走
    # 差分后的白噪声检验
    return adfuller(np.diff(series))[1] > 0.05

def check_time_series_degree(series, threshold=0.05):
    # 计算时间序列的阶
    # wiki
    # https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test

    n = 0
    adf = adfuller(series)
    while adf[1] > threshold:
        n += 1
        series = np.diff(series)
        adf = adfuller(series)
    return n

def check_time_series_periodic(series):
    # 检测时间序列中周期性的强度
    # 根据自相关性对时序进行分片
    # 每段分别使用 DTW 度量它们的距离
    # 如果这个距离越小，周期性越明显
    # 如果输入序列并没有完整的周期，这种方法可能会失效
    pass
