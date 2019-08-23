import scipy.signal as signal

def time_series_resample(series, size):
    return signal.resample(series, size)

class TimeSeriesResamplier:

    def fit(self, X):
        pass

    def fit_resample(self, X):
        pass
