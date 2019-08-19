import scipy.signal as signal

def time_series_resample(series, size):
    return signal.resample(series, size)
