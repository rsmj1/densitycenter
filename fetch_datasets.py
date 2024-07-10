import numpy as np
import os, csv, glob
import requests
from mnist import MNIST #This will load mnist from the folder it is stored in
from sklearn.datasets import fetch_openml
import pickle
import tarfile
import ucimlrepo

def load_mnist(type="train"):
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
    import gzip
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


def load_cifar_batch(file):
    """Load a single batch of CIFAR-10."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar_data(data_dir, num, type="train"):
    '''
    Load CIFAR dataset
    They are pickled files as described on the website that stores the datasets - thus we load with pickle as seen here.
    '''
    batches = []
    if num == 10:
        data_dir = os.path.join(data_dir, 'CIFAR'+str(num), 'cifar-'+str(num)+'-batches-py')
        if type == "train":
            for i in range(1, 6): 
                batch_file = os.path.join(data_dir, f'data_batch_{i}')
                batch = load_cifar_batch(batch_file)
                batches.append(batch)
        else:
            batches.append(load_cifar_batch(os.path.join(data_dir, "test_batch")))
        
        data = np.concatenate([batch[b'data'] for batch in batches])
        labels = np.concatenate([batch[b'labels'] for batch in batches])
    else:
        data_dir = os.path.join(data_dir, 'CIFAR'+str(num), 'cifar-'+str(num)+'-python')
        if type == "train":
            batches.append(load_cifar_batch(os.path.join(data_dir, "train")))
        else:
            batches.append(load_cifar_batch(os.path.join(data_dir, "test")))
        data = np.concatenate([batch[b'data'] for batch in batches])
        labels = np.concatenate([batch[b'fine_labels'] for batch in batches])
    

    return data, labels

def load_covertype(path):

    with open(os.path.join(path, 'data.pkl'), 'rb') as file:
        data = pickle.load(file)

    with open(os.path.join(path, 'labels.pkl'), 'rb') as file:
        labels = pickle.load(file)

    data = data.to_numpy()
    labels = labels.to_numpy()
    return data, labels

def fetch_datasets(dataset_folder_name='datasets'):
    #Needs to check whether folder structure is present first
    base_directory = dataset_folder_name

    os.makedirs(base_directory, exist_ok=True)
    
    '''
    The big dataset thing already made:
    https://clustpy.readthedocs.io/en/latest/clustpy.data.html



    Categorical
    [Adult](https://archive.ics.uci.edu/dataset/2/adult)
    Biological
    [KDDCUP04BIO](https://cs.joensuu.fi/sipu/datasets/)
    [Elegans](https://data.caltech.edu/records/j4ycn-dpv05)*
    [Molecular typography](https://data.caltech.edu/records/qaqhb-r9m40)*
    Geographic
    [Taxi dataset](https://figshare.com/articles/dataset/Porto_taxi_trajectories/12302165) (just the start locations)
    Classics
    [Covertype](https://archive.ics.uci.edu/dataset/31/covertype)
    Point Cloud
    nuScenes scene; tutorial [here](https://www.nuscenes.org/nuscenes?tutorial=lidarseg_panoptic) shows how to get a small version of it.




    Datasets to be fetched and their URLs:
    1. KDDCUP04BIO : https://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt
    2. MNIST : https://www.openml.org/search?type=data&status=active&id=554
    3. FASHION MNIST : https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#loading-data-with-other-machine-learning-libraries
    4. CIFAR 10 : https://www.cs.toronto.edu/~kriz/cifar.html
    5. CIFAR 100 : https://www.cs.toronto.edu/~kriz/cifar.html
    6. Covertype : https://archive.ics.uci.edu/dataset/31/covertype

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


    #https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    #4 CIFAR 10
    CIFAR10 = os.path.join(base_directory, 'CIFAR10')
    if not os.path.exists(CIFAR10):
        os.makedirs(CIFAR10, exist_ok=True)
        fileurl = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        with open(os.path.join(CIFAR10, 'cifar-10-python.tar.gz'), 'wb') as file:
            content = requests.get(fileurl, stream=True).content
            file.write(content)

            tar = tarfile.open(os.path.join(CIFAR10, 'cifar-10-python.tar.gz'), 'r:gz')
            tar.extractall(path=CIFAR10)
            tar.close()
        print("Downloaded 4")
    else:
        print("skipping #4")

    #5 CIFAR 100
    CIFAR100 = os.path.join(base_directory, 'CIFAR100')
    if not os.path.exists(CIFAR100):
        os.makedirs(CIFAR100, exist_ok=True)
        fileurl = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        with open(os.path.join(CIFAR100, 'cifar-100-python.tar.gz'), 'wb') as file:
            content = requests.get(fileurl, stream=True).content
            file.write(content)

            tar = tarfile.open(os.path.join(CIFAR100, 'cifar-100-python.tar.gz'), 'r:gz')
            tar.extractall(path=CIFAR100)
            tar.close()
        print("Downloaded 5")
        
    else:
        print("skipping #5")


    #6 Covertype
    Covertype = os.path.join(base_directory, 'covertype')
    if not os.path.exists(Covertype):
        os.makedirs(Covertype, exist_ok=True)
        
        covertype = ucimlrepo.fetch_ucirepo(id=31) 
        x = covertype.data.features
        y = covertype.data.targets
        with open(os.path.join(Covertype, 'data.pkl'), 'wb') as file:
            pickle.dump(x, file)
        with open(os.path.join(Covertype, 'labels.pkl'), 'wb') as file:
            pickle.dump(y, file)

        print("Downloaded 6")
    else:
        print("skipping #6")

    return



fetch_datasets()