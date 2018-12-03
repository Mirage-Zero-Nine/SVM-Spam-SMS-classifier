__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: t-SNE-demo
Create Time: 2018-11-21 18:31
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import logging


def dataset():
    """
    This data set comes from sklearn sample data set.
    It contains 1797 pictures. Each picture's size is 8 * 8.
    Each digit is presented as an array.
    """
    logging.info('Loading dataset...')

    ds = datasets.load_digits()
    X, y = ds[0], ds[1]
    # print(X[0].reshape(8, 8))
    # logging.info('Dataset loaded.')

    # n_samples, n_features = X.shape

    '''
    Show the origin data that is randomly chosen from data set
    Put them into a numpy array for later t-SNE process.
    In this demo, 30 * 30 = 900 digits were chosen.
    '''
    n = 30  # 30 * 30
    img = np.zeros((10 * n, 10 * n))
    for i in range(n):
        ix = 10 * i + 1
        for j in range(n):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))

    logging.info('Plotting dataset...')

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=plt.cm.binary)  # Set color map
    # plt.imshow(img, cmap=plt.cm.gray

    # Plot image
    plt.xticks([])
    plt.yticks([])
    plt.title("Data set")
    plt.savefig('Data.png')
    plt.show()

    logging.info('Dataset logging completed.')
    return X, y


def t_SNE(X, y):
    """t-SNE"""
    t_sne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_t_sne = t_sne.fit_transform(X)
    title = "Original data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_t_sne.shape[-1])

    logging.info('Normalization for visualization...')

    x_min, x_max = X_t_sne.min(0), X_t_sne.max(0)
    X_norm = (X_t_sne - x_min) / (x_max - x_min)  # Normalization

    plt.figure(figsize=(8, 8))

    logging.info('Converting result...')

    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Dark2(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})

    logging.info('Plotting...')

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig('t-SNE result.png')
    plt.show()

    logging.info('Plot completed.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S'
                        )
    X, y = dataset()
    t_SNE(X, y)
