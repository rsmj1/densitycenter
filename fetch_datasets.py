import numpy as np
import os, csv, glob
import requests
from mnist import MNIST #This will load mnist from the folder it is stored in
from sklearn.datasets import fetch_openml
import pickle


def load_mnist(type):
    '''
    type is whether to get training or eval set. "train" for training part.
    '''
    base_directory = 'datasets'
    mnist_data_path = os.path.join(base_directory, 'mnist')

    with open(os.path.join(mnist_data_path, 'data.pkl'), 'rb') as file:
        data = pickle.load(file)

    with open(os.path.join(mnist_data_path, 'labels.pkl'), 'rb') as file:
        labels = pickle.load(file)

    train_x = data[:60000]
    train_y = labels[:60000]
    eval_x = data[60000:]
    eval_y = labels[60000:]
    if type == "train":
        return train_x, train_y
    else:
        return eval_x, eval_y

def load_f_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np
    #Taken from the fashion mnist github. https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#loading-data-with-other-machine-learning-libraries
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def fetch_datasets():
    #Needs to check whether folder structure is present first
    base_directory = 'datasets'
    os.makedirs(base_directory, exist_ok=True)
    
    '''
    Datasets to be fetched and their URLs:
    1. KDDCUP04BIO : https://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt
    2. MNIST
    '''


    #1 KDDCUP04BIO
    KDDCUP04BIO = os.path.join(base_directory, 'KDDCUP04BIO')
    if not os.path.exists(KDDCUP04BIO):
        os.makedirs(KDDCUP04BIO, exist_ok=True)
        testurl = 'https://cs.joensuu.fi/sipu/datasets/pathbased.txt'
        print("KDDCUP04Bio.txt file does not exist, now fetching...")
        with open(os.path.join(KDDCUP04BIO, 'KDDCUP04Bio.txt'), 'wb') as file:
            content = requests.get(testurl, stream=True).content
            file.write(content)
    else:
        print("skipping #1")

    #2 MNIST
    mnist_data_path = os.path.join(base_directory, 'mnist')
    if not os.path.exists(mnist_data_path):
        os.makedirs(mnist_data_path, exist_ok=True)

        mnist = fetch_openml('mnist_784', version=1, cache=False)
        x, y = mnist['data'], mnist['target']
        
        with open(os.path.join(mnist_data_path, 'data.pkl'), 'wb') as file:
            pickle.dump(x, file)
        with open(os.path.join(mnist_data_path, 'labels.pkl'), 'wb') as file:
            pickle.dump(y, file)
        print("Downloaded 1")
    else:
        print("skipping #2")

    #3 Fashion MNIST
    f_mnist_data_path = os.path.join(base_directory, 'fashion_mnist')
    if not os.path.exists(f_mnist_data_path):
        os.makedirs(f_mnist_data_path, exist_ok=True)

        with open(os.path.join(f_mnist_data_path, 'train-images-idx3-ubyte.gz'), 'wb') as file:
            content = requests.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", stream=True).content
            file.write(content)
        with open(os.path.join(f_mnist_data_path, 'train-labels-idx1-ubyte.gz'), 'wb') as file:
            content = requests.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", stream=True).content
            file.write(content)
        with open(os.path.join(f_mnist_data_path, 't10k-images-idx3-ubyte.gz'), 'wb') as file:
            content = requests.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", stream=True).content
            file.write(content)
        with open(os.path.join(f_mnist_data_path, 't10k-labels-idx1-ubyte.gz'), 'wb') as file:
            content = requests.get("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", stream=True).content
            file.write(content) 

        print("Downloaded 3")
    else:
        print("skipping #3")




    return


fetch_datasets()