import numpy as np
import scipy.spatial as spatial

from ._dtw import dtw

__all__ = ["time_series_cid_distance", "time_series_dtw_distance", "mahalanobis_distance"]

def time_series_cid_ce(series):
    # CID distance
    # paper:
    # http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf

    d = np.diff(series)
    return np.sqrt(np.dot(d, d))

def time_series_cid_distance(series1, series2):
    # a distance measurement of two time series.
    # paper:
    # http://www.cs.ucr.edu/~eamonn/Complexity-Invariant%20Distance%20Measure.pdf

    ce1 = time_series_cid_ce(series1)
    ce2 = time_series_cid_ce(series2)
    ds = series1 - series2

    d = np.sqrt(np.dot(ds, ds)) * max(ce1, ce2) / min(ce1, ce2)
    return d

def time_series_dtw_distance(series1, series2):
    # 简单说, dtw 的基本原理是给定一个适应性窗口, 时序间两个时间点的距离是这个窗口
    # 内的点距离的最小值.
    # http://www.mathcs.emory.edu/~lxiong/cs730_s13/share/slides/searching_sigkdd2012_DTW.pdf
    # https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf
    return dtw(series1, series2, 10)

def mahalanobis_distance(obs, X, center="zero"):
    # mahalanobis distance of obs to X
    # wiki:
    # https://en.wikipedia.org/wiki/Mahalanobis_distance

    # 使用这个方法的参考 paper:
    # Online Anomaly Detection for Hard Disk Drives Based on Mahalanobis Distance
    # 事实上，它可以结合 KNN 使用，提升分类性能

    # 计算协方差矩阵
    cov = np.cov(X.T)

    # 计算数据集的中心
    if center == "zero":
        center = np.zeros(cov.shape[1])
    else:
        center = np.mean(X, axis=0)

    # 矩阵的逆不一定存在，这里使用矩阵的伪逆
    icov = np.linalg.pinv(cov)
    # 计算 obs 到 center 的 Mahalanobis distance
    d = spatial.distance.mahalanobis(obs, center, icov)
    return d
