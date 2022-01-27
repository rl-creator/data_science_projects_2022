
'''Module containing function to import dataset.'''

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def get_data ():
    '''Import included cifar10 dataset in the tensorflow kears library.'''
    return cifar10.load_data ()

def get_train_test_data (data):
    '''Unpack data into X and y training and test sets.'''
    (X_train, y_train), (X_test, y_test) = data
    return X_train, y_train, X_test, y_test

#if __name__ == "__main__":