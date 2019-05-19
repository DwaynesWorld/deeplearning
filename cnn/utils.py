import matplotlib.pyplot as plt
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar_datasets(data_dir):
    """
    Handles reading and unpickling of CIFAR dataset batches.
    Returns dictionary of data.
    """

    dirs = [
        'batches.meta', 'data_batch_1', 'data_batch_2',
        'data_batch_3', 'data_batch_4', 'data_batch_5',
        'test_batch'
    ]

    all_data = [0, 1, 2, 3, 4, 5, 6]

    for i, direc in zip(all_data, dirs):
        all_data[i] = unpickle(data_dir+direc)

    return {
        'batch_meta': all_data[0],
        'data_batch1': all_data[1],
        'data_batch2': all_data[2],
        'data_batch3': all_data[3],
        'data_batch4': all_data[4],
        'data_batch5': all_data[5],
        'test_batch': all_data[6],
    }


def preview_transformed_data(data, preview=False):
    """Preview batched CIFAR data"""
    # Each batch contains 10000 rows of flattened 32*32 rgb images (3072 data points each row).
    # First 1024 data points contains the red image
    # Next 1024 data points contains the green image
    # Last 1024 data points contains the blue image
    # We will unflatten the images and move the rgb inline for every image
    X = np.array(data)
    X = X.reshape(10000, 3, 32, 32)
    X = X.transpose(0, 2, 3, 1)

    if (preview):
        print(X.shape)
        print(X)

        # Show single image
        plt.imshow(X[1])
        plt.show()


def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class Cifar:

    def __init__(self, cifar_data):
        self.i = 0

        # Grabs a list of all the data batches for training
        self.all_train_batches = [
            cifar_data['data_batch1'],
            cifar_data['data_batch2'],
            cifar_data['data_batch3'],
            cifar_data['data_batch4'],
            cifar_data['data_batch5']
        ]

        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [cifar_data['test_batch']]

        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):

        print("Setting Up Training Images and Labels")

        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and Labels")

        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1)/255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i+batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y
