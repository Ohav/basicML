import torch
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
import numpy as np
from consts import *
import torch.optim as optim
import torch.nn as nn
from networks import FCN, CNN
import tqdm
from itertools import cycle

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
    samples = samples.reshape(-1, C, H, W)

    test_data = load_data(CIFAR_PATH + "test_batch")
    test_idx = np.random.choice(len(test_data[b'labels']), 1000, replace=False)
    test_labels = np.array(test_data[b"labels"])[test_idx]
    test_samples = np.array(test_data[b"data"])[test_idx] / 255
    test_samples = test_samples.reshape((-1, C, H, W))
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
    net.eval()
    outputs = net(test_X)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total = test_y.size(0)
    correct = (predicted == test_y).sum().item()
    accuracy = correct/total
    loss = criterion(outputs, test_y)
    return accuracy, loss.detach().numpy().reshape(1)[0]


def train_nn(net, criterion=nn.CrossEntropyLoss(), lr=0.001, momentum=0.9, std=0.1, number_of_epochs=50, weight_decay=0,
             optimizer='sgd', init='normal'):
    (train_X, train_y), (test_X, test_y) = get_10percent_cifar()
    # train_X = train_X.reshape(-1, CIFAR_IMAGE_SIZE)
    # test_X = test_X.reshape(-1, CIFAR_IMAGE_SIZE)
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.long)
    if init == 'normal':
        net.init_weights(std)
    elif init == 'xavier':
        net.init_weights_xavier()

    test_loss = []
    test_acc = []
    train_loss = []
    train_acc = []

    if optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        print("Invalid optimizer " + str(optimizer))
        return 0

    for epoch in tqdm.tqdm(range(number_of_epochs)):  # loop over the dataset multiple times
        indices = list(range(len(train_X)))
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_y = train_y[indices]
        net.train()
        for i in range(0, len(train_X), 64):
            inputs = train_X[i:i + 64]
            labels = train_y[i:i + 64]
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        test_stats = calc_accuracy_and_loss(net, test_X, test_y, criterion)
        test_loss.append(test_stats[1])
        test_acc.append(test_stats[0])
        train_stats = calc_accuracy_and_loss(net, train_X, train_y, criterion)
        train_loss.append(train_stats[1])
        train_acc.append(train_stats[0])

    return (test_loss, test_acc), (train_loss, train_acc)

def grid_search(net):
    results = {}
    min_loss = 10000000
    min_params = ()
    for lr in [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
        for momentum in [0.8, 0.85, 0.9, 0.95]:
            for std in [0.001, 0.01, 0.1, 0.5, 1]:
                test_stats, train_stats = train_nn(net=net, criterion=nn.CrossEntropyLoss(), lr=lr, momentum=momentum, std=std, number_of_epochs=10)
                loss, acc = test_stats
                loss = loss[-1]
                acc = acc[-1]
                results[(lr, momentum, std)] = (acc, loss)
                print(f"The trio {lr}, {momentum}, {std} got a loss/acc: {loss}/{acc}")
                if loss < min_loss:
                    min_params = (lr, momentum, std)
                    min_loss = loss
    print(results)
    print(f"Best: {min_loss} made by {min_params}")


def draw_plot(stats, labels, options, title, xlabel, ylabel):
    assert(len(stats) == len(labels))
    if options is None:
        options = plt.get_cmap('hsv', len(stats))
    plt.title(title)
    for i in range(len(stats)):
        plt.plot(range(len(stats[i])), stats[i], options[i], label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_two_compared_configuration_stats(stat1, stat2, name1, name2, epoch_count):
    plt.figure()
    plt.subplot(121)
    test_stats1, train_stats1 = stat1
    test_stats2, train_stats2 = stat2

    plt.title(f"{name1} vs {name2} test & train loss, over epochs")
    plt.plot(range(epoch_count), test_stats1[0], 'b', label=f"{name1} test loss")
    plt.plot(range(epoch_count), test_stats2[0], 'r', label=f'{name2} test loss')
    plt.plot(range(epoch_count), train_stats1[0], 'b--', label=f"{name1} train loss")
    plt.plot(range(epoch_count), train_stats2[0], 'r--', label=f'{name2} train loss')

    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Test loss")
    plt.subplot(122)
    plt.title(f"{name1} vs {name2} test set accuracy, over epochs")
    plt.plot(range(epoch_count), test_stats1[1], 'b', label=f"{name1} test accuracy")
    plt.plot(range(epoch_count), test_stats2[1], 'r', label=f'{name2} test accuracy')
    plt.plot(range(epoch_count), train_stats1[1], 'b--', label=f"{name1} train accuracy")
    plt.plot(range(epoch_count), train_stats2[1], 'r--', label=f'{name2} train accuracy')
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Test accuracy")
    plt.show()


def compare_sgd_adam(network, lr=5e-3, momentum=0.95, std=0.001):
    # Q2.2
    # Run SGD and ADAM and plot accuracies & loss.
    epoch_count = 10
    network_sgd = network()
    sgd_stats = train_nn(net=network_sgd, criterion=nn.CrossEntropyLoss(), lr=lr, momentum=momentum,
                         std=std, number_of_epochs=epoch_count, optimizer='sgd')
    network_adam = network()
    adam_stats = train_nn(net=network_adam, criterion=nn.CrossEntropyLoss(), lr=1e-5, momentum=momentum,
                          std=std, number_of_epochs=epoch_count, optimizer='adam')
    plot_two_compared_configuration_stats(sgd_stats, adam_stats, 'SGD', 'Adam', epoch_count)


def xavier_init(network, lr=5e-3, momentum=0.95, std=0.01):
    # Q2.3
    # Run SGD with std init and Xavier init and plot accuracies & loss.
    epoch_count = 50
    net = network()
    normal_stats = train_nn(net=net, criterion=nn.CrossEntropyLoss(), lr=lr, momentum=momentum,
                            std=std, number_of_epochs=epoch_count, optimizer='sgd', init='normal')
    net_xavier = network()
    xavier_stats = train_nn(net=net_xavier, criterion=nn.CrossEntropyLoss(), lr=lr, momentum=momentum,
                            number_of_epochs=epoch_count, optimizer='sgd', init='xavier')
    plot_two_compared_configuration_stats(normal_stats, xavier_stats, 'Normal', 'Xaviar', epoch_count)


def regularization_train(network, epoch_count=25):
    losses = []
    accs = []
    labels = []
    options = []


    weights = [0, 1, 10]
    dropouts = [0, 0.25, 0.5]
    # options = plt.get_cmap('hsv', len(weights) * len(dropouts))
    colors = cycle('brgmc')
    for weight in weights:
        for dropout in dropouts:
            cur_net = network(dropout=dropout)
            test_stats, train_stats = train_nn(net=cur_net, criterion=nn.CrossEntropyLoss(), lr=5e-3, momentum=0.95,
                     number_of_epochs=epoch_count, optimizer='sgd', init='normal', weight_decay=weight)
            losses.append(test_stats[0])
            accs.append(test_stats[1])
            losses.append(train_stats[0])
            accs.append(train_stats[1])
            color = colors.__next__()
            options.append(color)
            options.append(color + '--')
            labels.append("Test Weight decay={}, dropout p={}".format(weight, dropout))
            labels.append("Train Weight decay={}, dropout p={}".format(weight, dropout))
    plt.figure()
    plt.subplot(121)
    draw_plot(losses, labels, options,
              "Regularization test loss over Epochs\nwith different configurations",
              "Epoch Count", "Loss")
    plt.subplot(122)
    draw_plot(accs, labels, options,
              "Regularization test accuracy over Epochs\nwith different configurations",
              "Epoch Count", "Accuracy")

    plt.show()

def width_train(network, epoch_count=25):
    losses = []
    accs = []
    labels = []
    options = []

    colors = cycle('brgmc')
    # TODO: Add net init
    for i in [6, 10, 12]:
        width = 2 ** i
        cur_net = network(hidden_width=width)
        test_stats, train_stats = train_nn(net=cur_net, criterion=nn.CrossEntropyLoss(), lr=5e-3, momentum=0.95,
                            number_of_epochs=epoch_count, optimizer='sgd', init='normal',
                            weight_decay=0)
        losses.append(test_stats[0])
        accs.append(test_stats[1])
        losses.append(train_stats[0])
        accs.append(train_stats[1])
        color = colors.__next__()
        options.append(color)
        options.append(color + '--')
        labels.append("Test Width {}".format(width))
        labels.append("Train Width {}".format(width))

    plt.figure()
    plt.subplot(121)
    draw_plot(losses, labels, options,
              "Network test loss over Epochs\nWith different hidden layer width",
              "Epoch Count", "Loss")
    plt.subplot(122)
    draw_plot(accs, labels, options,
              "Network test Accuracy over Epochs\nWith different hidden layer width",
              "Epoch Count", "Accuracy")

    plt.show()


def depth_train(network, epoch_count=25):
    losses = []
    accs = []
    labels = []
    options = []
    colors = cycle('brgmc')
    for depth in [3, 4, 10]:
        cur_net = network(depth=depth)
        test_stats, train_stats = train_nn(net=cur_net, criterion=nn.CrossEntropyLoss(), lr=5e-3, momentum=0.95,
                             number_of_epochs=epoch_count, optimizer='sgd', init='normal',
                             weight_decay=0)
        losses.append(test_stats[0])
        losses.append(train_stats[0])
        accs.append(test_stats[1])
        accs.append(train_stats[1])
        color = colors.__next__()
        options.append(color)
        options.append(color + '--')
        labels.append("Test Depth {}".format(depth))
        labels.append("Train Depth {}".format(depth))

    plt.figure()
    plt.subplot(121)
    draw_plot(losses, labels, options,
              "Network Loss over Epochs\nWith different network depth",
              "Epoch Count", "Loss")
    plt.subplot(122)
    draw_plot(accs, labels, options,
              "Network Accuracy over Epochs\nWith different network depth",
              "Epoch Count", "Accuracy")

    plt.show()


if __name__ == "__main__":
    #train_svm()
    # print(grid_search())
    # compare_sgd_adam()
    # xavier_init()
    # regularization_train()
    # width_train()
    # xavier_init(network=CNN, lr=5e-3, momentum=0.8, std=0.1)
    # depth_train()
    # network = CNN()
    # compare_sgd_adam(network, 5e-3, 0.8, 0.1)
    depth_train(FCN, 25)
