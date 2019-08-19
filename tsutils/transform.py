import numpy as np
from statsmodels.tsa.stattools import adfuller

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

def series2d2X(series2d, size):
    # 把多维时间序列转换为滑动窗口形式
    return np.array([series2d[i:i+size,:] for i in range(len(series2d)-size+1)])

def series2d2Xy(series2d, size):
    # 把多维时间序列转换为单步带标注形式数据
    X = np.array([series2d[:-1][i:i+size,:] for i in range(len(series2d)-size)])
    y = np.array(series2d[size:,:])
    return X, y

class TimeSeriesLabelizer:

    #  把时序标签化

    def fit(self, X):
        pass

    def fit_transform(self, X):
        pass

class FuncTransfer:

    def __init__(self, func, ifunc):
        self.func = func # 正变换函数
        self.ifunc = ifunc # 逆变换函数

    def fit_transform(self, series):
        return self.func(series)

    def inverse_transform(self, series):
        return self.ifunc(series)

square_transfer = FuncTransfer(np.square, np.sqrt)
sqrt_transfer = FuncTransfer(np.sqrt, np.square)
exp_transfer = FuncTransfer(np.exp, np.log)
log_transfer = FuncTransfer(np.log, np.exp)

class StationaryTransfer:
    
    # 平稳时间序列与非平稳时间序列的转换
    # 确定序列是否平稳可以通过 ACF
    # 非平稳化为平稳序列可以通过差分方法
    # 从平稳序列还原为源序列需要保留每次
    # 差分的序列的首值.

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

    # adfuler 自动定价的平稳化转换
    
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

class E2ETransfer:
    
    # 端到端的数据转换, 包括:
    # 数据变换, 逆变换, 滤波
    # 离群点处理

    def fit(self, series):
        pass

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass
