from collections import deque
import numpy as np

_row = lambda x: x

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
