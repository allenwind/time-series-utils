class SimpleScaler:
    
    EPS = 0.1
    
    def __init__(self, type="std"):
        if type == "std":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(self.EPS, 1+self.EPS))

    def fit(self, series):
        self.scaler.fit(series.reshape((-1, 1)))
    
    def fit_transform(self, series):
        return self.scaler.fit_transform(series.reshape((-1, 1))).ravel()
    
    def inverse_transform(self, series):
        return self.scaler.inverse_transform(series.reshape((-1, 1))).ravel()

class PowerTransfer:

    # 把时间序列变换为正太分布
    
    def __init__(self):
        pass

    def fit(self):
        return self

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass

class SeriesMinMaxScaler:
    
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, series):
        pass

    def inverse_transform(self, series):
        pass
