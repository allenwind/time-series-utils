from collections import deque
import numpy as np

_row = lambda x: x

class Rolling:
    
    # 时间序列预测的滑动窗口
    
    def __init__(self, size, series=None, func=_row):
        self.size = size
        self.func = func
        self.window = deque(maxlen=size)
        if series is not None:
            self.init(series)

    def init(self, series):
        self.window.extend(list(series))
    
    def fit(self, value):
        self.window.append(value)

    def transform(self):
        return self.func(np.array(self.window))

    def fit_transform(self, value):
        self.fit(value)
        return self.transform()
