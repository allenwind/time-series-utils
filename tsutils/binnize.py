import numpy as np

class Binnizer:

	def __init__(self, r_min, r_max, bin_size):
		self.r_min = r_min
		self.r_max = r_max
		self.bin_size = bin_size
		self.bins = np.linspace(self.r_min, self.r_max, self.bin_size)

	def fit(self, X):
		pass

	def transform(self, X):
		pass

	def inverse_transform(self, X):
		# 采样方法
		# 根据采样定理还原模拟信号
		# 正太分布
		# 均匀分布
		pass

class BeamSearcher:

	# 寻找一条定长路径，使其观察概率最大化
	# conditional beam search
	# 如何在组合爆炸中寻找最优观察概率的方法