import numpy as np

# shape = (batch_size, window_size, n_features, bin_size)

def build_cycle_atrix(series):
    # 创建循环矩阵
    return np.array([np.roll(series, i) for i in range(len(series))])

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
        return self

    def transform(self, series):
        # 从模拟信号中采样
        bs = self.digitize(series)
        return self.onehot[bs]

    def inverse_transform(self, X):
        # 还原模拟信号
        # 从 one-hot 中还原模拟信号
        idx = np.where(np.not_equal(X, 0))[1]
        return self.inverse_sample_func(idx)

    def inverse_transform_from_digits(self, idx):
        # 从离散信号中还原模拟信号
        return self.inverse_sample_func(idx)

    def sample_func(self, series):
        # 采样函数
        pass

    def inverse_sample_func(self, idx):
        # 逆采样
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

    def __init__(self, model, topk, bin_size, n_steps, binnizer):
        self.model = model
        self.topk = topk
        self.bin_size = bin_size
        self.n_steps = n_steps
        self.binnizer = binnizer
        self.topk_path = [[]] * topk
        self.scores = np.expand_dims(np.ones(topk)/topk, 1) # (topk, 1)

    def search(self, model):
        # (1) 生成 topk 个最有可能的解
        # (2) 接下来每一步，从 topk*V 中 选择最后可能 topk 个解
        for i in range(self.n_steps):
            X = self.roller.transform()
            X = np.expand_dims(X, 0)
            # (1, n)
            y_proba = self.model.predict(X) 
            # how to zeros?
            y_proba = np.log(y_proba) 
            # 利用广播 (topk, n)
            # np.kron(a, b)
            S = self.scores + y_proba 
            scores = self._find_topk(S)
            self.scores = scores

    def _find_topk(self, S):
        ix, iy = np.unravel_index(np.argsort(S, axis=None), S.shape)
        scores = []
        for i, j in zip(ix[-self.topk:], iy[-self.topk:]):
            scores.append(S[i][j])
            self.topk_path[i].append(j)
        return scores

    def to_topk_series(self):
        pass

    def to_best_series(self):
        return self.binnizer()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    b = Binnizer(bin_range=(-1.1, 1.1), bin_size=50)
    x = np.linspace(0, 2*np.pi, 10000)
    series = np.sin(x) #+ np.random.uniform(-0.1, 0.1, len(x))
    tseries = b.transform(series)
    plt.imshow(tseries)
    plt.show()
    series2 = b.inverse_transform(tseries)

    s = b.scores_f(series, series2)
    print(s)

    plt.plot(series)
    plt.plot(series2)
    plt.plot(series-series2, color="red")
    plt.show()