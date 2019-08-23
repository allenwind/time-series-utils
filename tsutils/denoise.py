import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from sklearn.preprocessing import PowerTransformer

__all__ = ["FourierDenoiser", "time_series_fourier_denoise"]

def time_series_fourier_denoise(series, threshold=0.2):
    # 基于傅里叶变换的 denoising 方法
    # 把原始时序变换到频域上，过滤掉高频信号
    fs = fft.rfft(series)
    freqs = fft.rfftfreq(len(series), 0.1)
    # 过滤高频信号
    fs[freqs > threshold] = 0
    return fft.irfft(fs)

class FourierDenoiser:

    def __init__(self, threshold=0.2):
        self.threshold = threshold
    
    def fit(self, series):
        return series

    def fit_transform(self, series):
        return time_series_fourier_denoise(series, self.threshold)
