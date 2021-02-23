# time series utils

时间序列预处理工具：

1. 包括基于傅里叶分解的去噪
2. 时序间的距离度量
3. 时序标注
4. 滑动窗口
5. transform
6. 交叉验证
7. 平稳化转换



预处理包括如下:

1. 缺失值处理
2. 白噪声检验或随机游走检验
3. 去噪
4. 平稳化
5. 数据正太化
6. 归一化
7. stationary  --> detrend



原始时序中包含大量噪声, 可以使用如下方法去噪:

1. 平滑法
2. 数字滤波器



此外, 还要通过白噪声检验或随机游走检验检验数据是否可预测.




## TODO

1. E2ETransfer
2. smooth function
3. 均衡采样与非均衡采样

深度学习去噪方法 papers:
1. noise2noise https://arxiv.org/pdf/1803.04189.pdf
2. PAE denoising https://arxiv.org/pdf/1509.05982.pdf
