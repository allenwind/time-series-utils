def time_series_train_test_split(x, y, train_size=0.7):
    pass

def train_val_split(series, train_rate=0.7, test_rate=0.7):
    # 训练与检验集的分离
    # TODO 交叉检验

    idx1 = int(train_rate * len(series))
    idx2 = int(test_rate * len(series))
    s1 = series[:idx1+1]
    s2 = series[idx1:]
    return s1, s2
