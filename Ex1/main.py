import torch
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import numpy as np
from consts import *
import torch.optim as optim
import torch.nn as nn
from networks import FCN
import tqdm


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
    train_X = train_X.reshape(-1, CIFAR_IMAGE_SIZE)
    test_X = test_X.reshape(-1, CIFAR_IMAGE_SIZE)

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

def calc_accuracy_and_loss(net, test_X, test_y, criterion):
    outputs = net(test_X)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total = test_y.size(0)
    correct = (predicted == test_y).sum().item()
    accuracy = correct/total
    loss = criterion(outputs, test_y)
    return accuracy, loss.detach().numpy().reshape(1)[0]


def train_nn(criterion=nn.CrossEntropyLoss(), lr=0.001, momentum=0.9, std=0.1, number_of_epochs=50, optimizer='sgd', init='normal'):
    (train_X, train_y), (test_X, test_y) = get_10percent_cifar()
    train_X = train_X.reshape(-1, CIFAR_IMAGE_SIZE)
    test_X = test_X.reshape(-1, CIFAR_IMAGE_SIZE)
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.long)
    net = FCN(256)
    if init == 'normal':
        net.init_weights(std)
    elif init == 'xavier':
        net.init_weights_xavier()
    losses = np.zeros(number_of_epochs)
    test_loss = []
    test_acc = []
    if optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        print("Invalid optimizer " + str(optimizer))
        return 0
    for epoch in tqdm.tqdm(range(number_of_epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        indices = list(range(len(train_X)))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]
        for i in range(0, len(train_X), 64):
            inputs = train_X[i:i + 64]
            labels = train_y[i:i + 64]
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses[epoch] = running_loss
        stats = calc_accuracy_and_loss(net, test_X, test_y, criterion)
        test_loss.append(stats[1])
        test_acc.append(stats[0])

    return test_loss, test_acc

def grid_search():
    results = {}
    min_loss = 10000000
    min_params = ()
    for lr in [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
        for momentum in [0.8, 0.85, 0.9, 0.95]:
            for std in [0.001, 0.01, 0.1, 0.5, 1]:
                acc, loss = train_nn(criterion=nn.CrossEntropyLoss(), lr=lr, momentum=momentum, std=std, number_of_epochs=10)
                results[(lr, momentum, std)] = (acc, loss)
                print(f"The trio {lr}, {momentum}, {std} got a loss/acc: {loss}/{acc}")
                if loss < min_loss:
                    min_params = (lr, momentum, std)
    print(results)
    print(f"Best: {min_loss} made by {min_params}")


def compare_sgd_adam():
    # Q2.2
    # Run SGD and ADAM and plot accuracies & loss.
    epoch_count = 20
    sgd_stats = train_nn(criterion=nn.CrossEntropyLoss(), lr=5e-3, momentum=0.95, std=0.001, number_of_epochs=epoch_count, optimizer='sgd')
    adam_stats = train_nn(criterion=nn.CrossEntropyLoss(), lr=5e-3, momentum=0.95, number_of_epochs=epoch_count, optimizer='adam')
    plt.figure()
    plt.subplot(121)
    plt.title("SGD vs Adam test set loss, over epochs")
    plt.plot(range(epoch_count), sgd_stats[0], 'b', label="SGD loss")
    plt.plot(range(epoch_count), adam_stats[0], 'r', label='Adam loss')
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Test loss")
    plt.subplot(122)
    plt.title("SGD vs Adam test set accuracy, over epochs")
    plt.plot(range(epoch_count), sgd_stats[1], 'b', label="SGD accuracy")
    plt.plot(range(epoch_count), adam_stats[1], 'r', label='Adam accuracy')
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Test accuracy")
    plt.show()

def xavier_init():
    # Q2.3
    # Run SGD with std init and Xavier init and plot accuracies & loss.
    epoch_count = 20
    sgd_stats = train_nn(criterion=nn.CrossEntropyLoss(), lr=5e-3, momentum=0.95, std=0.001,
                         number_of_epochs=epoch_count, optimizer='sgd')
    xavier_stats = train_nn(criterion=nn.CrossEntropyLoss(), lr=5e-3, momentum=0.95,
                          number_of_epochs=epoch_count, optimizer='sgd', init='xavier')
    plt.figure()
    plt.subplot(121)
    plt.title("Normal vs Xavier init SGD test set loss, over epochs")
    plt.plot(range(epoch_count), sgd_stats[0], 'b', label="Normal Distribution loss")
    plt.plot(range(epoch_count), xavier_stats[0], 'r', label='Xavier loss')
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Test loss")
    plt.subplot(122)
    plt.title("Normal vs Xavier init SGD test set accuracy, over epochs")
    plt.plot(range(epoch_count), sgd_stats[1], 'b', label="Normal Distribution accuracy")
    plt.plot(range(epoch_count), xavier_stats[1], 'r', label='Xavier accuracy')
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Test accuracy")
    plt.show()


if __name__ == "__main__":
    #train_svm()
    # print(grid_search())
    # compare_sgd_adam()
    xavier_init()