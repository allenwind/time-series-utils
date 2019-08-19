from collections import deque
import numpy as np

_row = lambda x: x

def time_series_shift(series, shift):
    # 弃用，直接使用 np.roll
    # shift > 0 向右 shift
    idx = shift % series.size
    return np.concatenate([series[-idx:], series[:-idx]])

class Rolling:

    def init(self, series):
        pass

    def fit(self, value):
        pass

    def transform(self):
        pass

    def fit_transform(self, value):
        pass

class TimeSeriesRolling(Rolling):
    
    # 时间序列预测的滑动窗口
    
    def __init__(self, size, series=None, func=_row):
        self.size = size
        self.func = func
        self.window = deque(maxlen=size)
        if series is not None:
            self.init(series)

    def init(self, series):
        # 初始化时间窗口
        self.window.extend(list(series))
    
    def fit(self, value):
        # 滑动窗口向前移动更新值
        self.window.append(value)

    def transform(self):
        # 计算当前窗口的特征值
        return self.func(np.array(self.window))

    def fit_transform(self, value):
        # 更新滑动窗口并计算特征值
        self.fit(value)
        return self.transform()

def simple_sliding_window(series, size, step=1):
    # 把滑动窗口直接转换成矩阵形式
    w = np.hstack([series[i:i+i-size or None:step] for i in range(size)])
    return w.reshape((-1, size))
