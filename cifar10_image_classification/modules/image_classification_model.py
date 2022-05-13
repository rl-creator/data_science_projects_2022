
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



if __name__ == "__main__":
    pass