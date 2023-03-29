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
    train_idx = np.random.choice(len(data[b'labels']), 5000, replace=False)
    labels = np.array(data[b'labels'])[train_idx]
    samples = np.array(data[b'data'])[train_idx] / 255
    samples = np.moveaxis(samples.reshape(-1, C, H, W), 1, -1)

    test_data = load_data(CIFAR_PATH + "test_batch")
    test_idx = np.random.choice(len(test_data[b'labels']), 1000, replace=False)
    test_labels = np.array(test_data[b"labels"])[test_idx]
    test_samples = np.array(test_data[b"data"])[test_idx] / 255
    test_samples = np.moveaxis(test_samples.reshape((-1, C, H, W)), 1, -1)
    return (samples, labels), (test_samples, test_labels)


def train_svm():
    (train_X, train_y), (test_X, test_y) = get_10percent_cifar()
    train_X = train_X.reshape(-1, IMAGE_SIZE)
    test_X = test_X.reshape(-1, IMAGE_SIZE)

    clf = svm.SVC(kernel="linear")
    clf.fit(train_X, train_y)
    y_train_pred = clf.predict(train_X)
    y_test_pred = clf.predict(test_X)
    print("SVM Accuracy: Train: {} - Test: {}".format(np.sum(y_train_pred == train_y) / len(train_y), np.sum(y_test_pred == test_y) / len(test_y)))

    rbf = svm.SVC(kernel='rbf')
    rbf.fit(train_X, train_y)
    y_train_pred = rbf.predict(train_X)
    y_test_pred = rbf.predict(test_X)
    print("RBF Accuracy: Train: {} - Test: {}".format(np.sum(y_train_pred == train_y) / len(train_y), np.sum(y_test_pred == test_y) / len(test_y)))



if __name__ == "__main__":
    train_svm()
