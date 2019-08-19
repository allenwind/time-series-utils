import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller

def _time_series_all_autocorrelation(series):
    # 计算所有 lag 的自相关值, 计算方法可参考
    # wiki:
    # https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    n = series.size
    rs = []
    for i in range(n):
        r = 0
        for j in range(n-i):
            r += series[j+i] * series[j]
        rs.append(r)
    return np.array(rs)

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

def check_time_series_trending(series):
    # 检测时间序列是否存在趋势
    pass

def find_time_series_max_periodic(series, offset=1):
    # 如何时序存在周期, 那么自相关函数会呈现明显的规律
    # offset 表示忽略自相关函数中的前 n 个自相关系数
    # 通常来说，较长的序列且复杂的序列会在 lag 较小时
    # 表示较大的自相关性.

    # 寻找时序中的最大周期，即最大 lag
    # 确定最佳的滑动窗口大小
    # 该实现是经验方法, 经验方法
    # 也可以通过自相关的方式计算最佳值
    # 但是计算十分耗时
    # 此外, 还可以使用傅里叶方法
    # 详细见 fourier.py 模块

    # 计算最大自相关系数的 lag
    # wiki:
    # https://en.wikipedia.org/wiki/Autocorrelation

    # 方法：
    # 1. 检测时序的趋势性
    # 2. 差分去趋势
    # 3. 计算自相关函数
    # 4. 获取最大自相关的 lag
    if not offset:
        offset = max(1, len(series)//100)

    auto = _time_series_all_autocorrelation(series)[offset:]
    return int(np.argmax(auto)) + offset

def time_series_move_lag(series, lag=1, pad="first"):
    # 预测的滞后性，把预测结果往后移动 lag 个时间步，并使用 pad 进行填充
    # 理论上，滑动窗口的预测都有这个情况，可以理解为模型为了最优化，选择和
    # 最近一个时间步相近的取值。

    if pad == "first":
        v = series[0]
    elif pad == "last":
        v = series[0]
    elif pad == "mean":
        v = np.mean(series)
    elif pad == "zero":
        v = 0
    elif pad == "median":
        v = np.median(series)
    elif pad == "mode":
        v = stats.mode(series)

    values = [v] * lag
    values.extend(series.tolist())
    return np.array(values)
