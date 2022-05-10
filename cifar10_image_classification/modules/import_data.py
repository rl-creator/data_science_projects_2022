
'''Module containing functions to import dataset and prepare independent and 
response variables.'''

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from storage import store_data

def get_data (n):
    '''Import included cifar10 dataset in the tensorflow kears library.
    
    n: Argument serves no purpose. It is needed because the 'create_pipeline'
    function requires an 'input' argument. However, if 'get_data' does not have 
    an argument, there will be an error where 'get_data' will have issue with an
    argument being fed to it from the pipeline. So, to resolve the conflict, 
    'get_data' has an argument that does not have a purpose for the function.
    '''
    return cifar10.load_data ()

def get_train_test_data (data):
    '''Unpack data into X and y training and test sets.'''
    (X_train, y_train), (X_test, y_test) = data
    return store_data ('X_train', 'y_train', 'X_test', 'y_test') (X_train, y_train, X_test, y_test)

def normalise_data (train_test_data):
    '''Normalise data to a 0 to 1 scale.

    The RBG colour scale for photos measures red, blue, and green on a 0 to 255 
    point scale. Dividing by 255 coverts all RBG measurements to a 0 to 1 scale.
    '''
    X_train_norm = train_test_data.X_train /255
    X_test_norm = train_test_data.X_test /255
    y_train = train_test_data.y_train
    y_test = train_test_data.y_test

    data_normalised = (X_train_norm, y_train, X_test_norm, y_test)

    train_test_data_norm = store_data ('X_train_norm', 'y_train', 'X_test_norm', 'y_test') (*data_normalised)
    
    return store_data ('data_raw', 'data_normalised') (train_test_data, train_test_data_norm)



if __name__ == "__main__":
    pass