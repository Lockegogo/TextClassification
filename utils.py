import matplotlib.pyplot as plt


def plot_metric(dfhistory, metric):
    train_metrics = dfhistory['train_' + metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    # 将图片保存到 results 文件夹
    plt.savefig('results/' + metric + '.png')
