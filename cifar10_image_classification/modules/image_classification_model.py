
'''Module containing functions for creating, train, and evaluating a 
neural network model.'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.layers import Conv2D, Dense, Input

from create_pipeline import create_pipeline
from storage import store_data

def create_model (input_layer, hidden_layers):
    '''Function to define the input and output layers of the model and
    create the model to be tested.
    '''
    def pass_through_data (data):
        '''Input the data to be passed through the pipeline.
        
        The data is not used in this function. This function only creates the 
        model being developed. The created model will be passed to the next 
        function for training. The data will pass through the function and join 
        the model as output.
        '''

        ########
        #   Data
        data_raw = data.data_raw
        data_normalised = data.data_normalised
        ########

        input = input_layer

        neural_net = create_pipeline (hidden_layers)

        outputs = neural_net (input)
        
        model = keras.Model (inputs = input, outputs = outputs, name = "cifar10_first_model")

        return store_data ("data_raw", "data_normalised", "model") (data_raw, data_normalised, model)


    return pass_through_data


###   Compile model

def compile_model (optimizer = "Adam", loss = "catagorical_crossentropy", metrics = ["accuracy"]):
    '''Function to set optimizer, loss function, and metrics for the model and 
    compile the model.
    '''
    def input_model_n_pass_data (data_model):
        '''Input the model to be compiled.
        Input data to be passed through the pipeline.
        
        The data is not being used in this function. 
        This function will only compile the model. Once the model is compiled,
        the compiled model and the data will be passed to the next function for
        training.
        '''

        model = data_model.model

        model.compile (optimizer = optimizer, 
                       loss = loss, 
                       metrics = metrics)
        
        ########
        #   Data
        data_raw = data_model.data_raw
        data_normalised = data_model.data_normalised
        ########

        return store_data ("data_raw", "data_normalised", "model") (data_raw, data_normalised, model)
        
    return input_model_n_pass_data


def compile_model2 (optimizer = "Adam", loss = "catagorical_crossentropy", metrics = ["accuracy"]):
    '''Function to set optimizer, loss function, and metrics for the model and 
    compile the model.
    '''
    def input_model_n_pass_data (data_model):
        '''Input the model to be compiled.
        Input data to be passed through the pipeline.
        
        The data is not being used in this function. 
        This function will only compile the model. Once the model is compiled,
        the compiled model and the data will be passed to the next function for
        training.


        ###############################
        This way of calling the model and compiling works.
        It is not necessary to assign data_model.model to a new variable.
        Possible explanation: model is an instance of keras.Model which is a 
        mutable object. So, by calling the object and running the compile 
        method, the object is being altered in the tuple 'data_model,' and after,
        when returned, the tuple 'data_model,' with the changed 'model' is 
        returned.
        ###############################
        '''

        data_model.model.compile (optimizer = optimizer, 
                                  loss = loss, 
                                  metrics = metrics)
        
        print ("Model compiled for compile_model2")

        return data_model
        
    return input_model_n_pass_data


###   Train model

def train_model (batch_size = None, validation_split = 0.0, **model_param):
    '''Function to set the parameters of the model fitting.'''
    def input_model_n_data (data_model):
        '''Function to train the model'''

        #   Training data
        X_train, y_train = data_model.data_normalised.X_train_norm, data_model.data_normalised.y_train

        data_model.model.fit (X_train, y_train, 
                              batch_size = batch_size,
                              validation_split = validation_split)
        
        print ("Completed model training.")

        return data_model

    return input_model_n_data

if __name__ == "__main__":
    pass