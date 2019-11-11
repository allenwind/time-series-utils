import numpy as np

class Binnizer:

    # 连续信号的离散化

    def __init__(self, bin_range, bin_size):
        r_min, r_max = bin_range
        self.r_min = r_min
        self.r_max = r_max
        self.bin_size = bin_size
        self.bins = np.linspace(self.r_min, self.r_max, self.bin_size)
        self.onehot = np.eye(self.bin_size)

    def fit(self, series):
        bs = self.digitize(series)
        return self.onehot[bs]

    def transform(self, series):
        bs = self.digitize(series)
        return self.onehot[bs]

    def inverse_transform(self, X):
        # 还原模拟信号
        idx = np.where(np.not_equal(X, 0))[1]
        return self.sample_func(idx)

    def sample_func(self, idx):
        # 采样方法
        # 正太分布
        # 均匀分布
        series = self.bins[idx]
        return series

    def digitize(self, series):
        return np.digitize(series, self.bins) - 1

    def scores_f(self, s1, s2):
        return np.sqrt(np.sum(np.square(s1-s2)))

class BeamSearcher:

    # 寻找一条定长路径，使其观察概率最大化
    # conditional beam search
    # 如何在组合爆炸中寻找最优观察概率的方法

    def __init__(self, topk, bin_size, n_steps, binnizer):
        self.topk = topk
        self.bin_size = bin_size
        self.n_steps = n_steps
        self.binnizer = binnizer
        self.topk_path = [[]] * topk
        self.scores = np.zeros(topk)

    def search(self, model):
        pass

    def to_topk_series(self):
        pass

    def to_best_series(self):
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    b = Binnizer(bin_range=(-1.1, 1.1), bin_size=100)
    x = np.linspace(0, 2*np.pi, 10000)
    series = np.sin(x) + np.random.uniform(-0.1, 0.1, len(x))
    tseries = b.fit(series)
    plt.imshow(tseries)
    plt.show()
    series2 = b.inverse_transform(tseries)

    s = b.scores_f(series, series2)
    print(s)

    plt.plot(series)
    plt.plot(series2)
    plt.plot(series-series2, color="red")
    plt.show()