import numpy as np

# dtw 的 numpy 实现

def dtw(s1, s2, w):
    w = max(abs(s1.size-s2.size), w)
    s = np.eye(s1.size, s2.size)
    for i in range(1, s1.size):
        for j in range(max(1, w-i), min(s2.size, w+i)):
            d = abs(s1[i]-s2[j])
            s[i,j] = d + min(s[i-1,j-1], s[i, j-1], s[i-1,j])
    return s[-1,-1], np.argmin(s, axis=0)

def dtw_2d(seq1, seq2, w):
	# 可参考论文中处理 2d match 的思路
	# https://arxiv.org/pdf/1604.04378.pdf
	pass