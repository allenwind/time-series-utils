def view_rolling_features(series, size):
    transfer = FeaturesTimeSeriesTransfer(series)
    X, y = transfer.transform_features(size)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    plt.subplot(211)
    plt.plot(series)
    plt.subplot(212)
    plt.imshow(X.T)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def visualize_autocorrelation(series, offset=0):
    auto = np.array(time_series_all_autocorrelation(series))
    plt.subplot(211)
    plt.plot(series)
    plt.subplot(212)
    plt.plot(auto[offset:], "+")
    plt.show()
