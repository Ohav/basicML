import torch
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import numpy as np

CIFAR_PATH = "cifar-10-batches-py/"
H = W = 32
C = 3
IMAGE_SIZE = H * W * C


def load_data(path):
    with open(path, 'rb') as f:
        # Dictionary
        data = pickle.load(f, encoding='bytes')
    return data


def get_10percent_cifar():
    batch = np.random.choice(range(1, 6), 1)[0]
    data = load_data(CIFAR_PATH + f'data_batch_{batch}')
    labels = np.random.choice(data[b'labels'], 5000)
    samples = np.random.choice(data[b'data'], 5000) / 255
    samples = np.moveaxis(samples.reshape(-1, C, H, W), 1, -1)

    test_data = load_data(CIFAR_PATH + "test_batch")
    test_labels = np.random.choice(test_data[b"labels"], 1000)
    test_samples = np.random.choice(test_data[b"data"], 1000) / 255
    test_samples = np.moveaxis(test_samples.reshape((-1, C, H, W)), 1, -1)
    return (samples, labels), (test_samples, test_labels)


def train_svm():
    (train_X, train_y), (test_X, test_y) = get_10percent_cifar()
    clf = svm.SVC()

    clf.fit(train_X.reshape(-1, IMAGE_SIZE), train_y)

    y_pred = clf.predict(test_X.reshape(-1, IMAGE_SIZE))
    print("SVM done, recall: {}".format(np.sum(y_pred == test_y) / len(test_y)))

    rbf = svm.SVC(kernel='rbf')
    rbf.fit(train_X.reshape(-1, IMAGE_SIZE), train_y)

    y_pred = rbf.predict(test_X.reshape(-1, IMAGE_SIZE))
    print("RBF done, recall: {}".format(np.sum(y_pred == test_y) / test_y))





if __name__ == "__main__":
    train_svm()
