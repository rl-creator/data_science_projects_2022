
'''Module containing functions to import dataset and prepare independent and 
response variables.'''

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from storage import store_data

def get_data ():
    '''Import included cifar10 dataset in the tensorflow kears library.'''
    return cifar10.load_data ()

def get_train_test_data (data):
    '''Unpack data into X and y training and test sets.'''
    (X_train, y_train), (X_test, y_test) = data
    return store_data ('X_train', 'y_train', 'X_test', 'y_test') (X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    pass