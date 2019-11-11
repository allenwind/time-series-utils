import numpy as np

class Binnizer:

	def __init__(self, bin_range, bin_size):
		r_min, r_max = bin_range
		self.r_min = r_min
		self.r_max = r_max
		self.bin_size = bin_size
		self.bins = np.linspace(self.r_min, self.r_max, self.bin_size)
		self.onehot = np.eye(self.bin_size)

	def fit(self, series):
		bs = np.digitize(series, self.bins)
		return self.onehot(bs)

	def transform(self, series):
		bs = np.digitize(series, self.bins)
		return self.onehot(bs)

	def inverse_transform(self, X):
		# 还原模拟信号
		idx = np.where(np.not_equal(r, 0))[1]
		return self.sample_func(idx)

	def sample_func(self, idx):
		# 采样方法
		# 正太分布
		# 均匀分布
		series = self.bins[idx]
		return series

	def scores_f(self, s1, s2):
		return np.sqrt(np.square(s1-s2))

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