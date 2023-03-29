import torch
import sklearn
import pickle

CIFAR_PATH = ""


def load_data(path):
    with open(path, 'rb') as f:
        # Dictionary
        data = pickle.load(f, encoding='bytes')
    return data





if __name__ == "__main__":
    print("Hi")
    pass