__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: plot
Create Time: 2018-12-02 14:45
'''

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def svm_plot(x, y):
    def get_data(x, y):
        # clf = svm.SVC(kernel='linear', C=10)

        n_samples, n_features = len(x), len(x[0])
        return x, y, n_samples, n_features

    def plot_embedding(data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(label[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig

    def main():
        data, label, n_samples, n_features = get_data(x, y)
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=3, init='pca', random_state=0)
        t0 = time()
        result = tsne.fit_transform(data)
        fig = plot_embedding(result, label,
                             't-SNE embedding of the messages (time %.2fs)'
                             % (time() - t0))
        plt.show()

    main()


if __name__ == '__main__':
    pass
