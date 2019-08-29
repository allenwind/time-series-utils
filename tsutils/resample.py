import scipy.signal as signal

def time_series_resample(series, size):
    return signal.resample(series, size)

class TimeSeriesResamplier:

    def __init__(self, size):
        self.size = size

    def fit(self, X):
        return self

    def fit_resample(self, X):
        return time_series_resample(X, self.size)
