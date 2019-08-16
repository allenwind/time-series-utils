import numpy as np
from statsmodels.tsa.stattools import adfuller

from .check import check_time_series_degree

_row = lambda x: x

class Pipeline:
    
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, series):
        return self

    def fit_transform(self, series):
        for e in self.estimators:
            series = e.fit_transform(series)
        return series

    def inverse_transform(self, series):
        for e in reversed(self.estimators):
            series = e.inverse_transform(series)
        return series

def series2X(series, size, func=_row):
    # 把时间序列转换为滑动窗口形式
    X = np.array([series[i:i+size] for i in range(len(series)-size+1)])
    return np.apply_along_axis(func, 1, X)

def series2Xy(series, size, func=_row):
    # 把时间序列转换为单步带标注形式数据
    X = np.array([series[:-1][i:i+size] for i in range(len(series)-size)])
    y = np.array(series[size:])
    return np.apply_along_axis(func, 1, X), y

def series2d2X(series2d, size, func=_row):
    pass

def series2d2Xy(series2d, size, func=_row):
	pass

class FuncTransfer:

    def __init__(self, func, ifunc):
        self.func = func
        self.ifunc = ifunc

    def fit_transform(self, series):
        return self.func(series)

    def inverse_transform(self, series):
        return self.ifunc(series)

class StationaryTransfer:
    
    # 平稳时间序列与非平稳时间序列的转换
    # 确定序列是否平稳可以通过 ACF
    # 非平稳化为平稳序列可以通过差分方法
    # 从平稳序列还原为源序列需要保留每次
    # 差分的序列的首值.
    
    # TODO 整合自动定阶方法

    def __init__(self, k=1):
        self.k = k
        self._prefix = []
    
    def fit_transform(self, series):
        # 迭代地执行高阶差分
        k = self.k
        while k:
            self._prefix.append(series[0])
            series = np.diff(series)
            k -= 1
        return series

    def inverse_transform(self, series):
        # 迭代地还原高阶差分
        k = self.k
        while k:
            k -= 1
            values = [self._prefix[k]]
            values = np.append(values, series)
            values = np.cumsum(values)
            series = values
        return values

class AutoStationaryTransfer:

    # 自动定价的平稳化转换
    
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self._prefix = []

    def fit_transform(self, series):
        self.k = 0
        adf = adfuller(series)
        while adf[1] > self.threshold:
            self.k += 1
            series = np.diff(series)
            adf = adfuller(series)
        return series

    def inverse_transform(self, series):
        k = self.k
        while k:
            k -= 1
            values = [self._prefix[k]]
            values = np.append(values, series)
            values = np.cumsum(values)
            series = values
        return values
