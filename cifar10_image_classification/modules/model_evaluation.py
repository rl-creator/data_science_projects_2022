
'''Module containing functions for evaluating a 
neural network model.'''

import numpy as np
import tensorflow as tf
from tensorflow import keras


###   Evaluate model

def evaluate_model (**parameters_for_model_evaluation):
    '''Function to set the parameters for the model evaluation.'''
    def input_model_n_data (data_model):
        '''Function to evaluate the model'''

        X_test, y_test = data_model.data_normalised.X_test_norm, data_model.data_normalised.y_test

        results = data_model.model.evaluate (x = X_test, y = y_test, **parameters_for_model_evaluation)

        print (f"The accuracy of the model is {results[1]}.")
        print (f"The sparce categorical accuracy of the model is {results[2]}.")
        #   results[0] is the loss of the model.

        return data_model

    return input_model_n_data


def return_confusion_matrix (parameters_for_prediction : dict = {}, parameters_for_confusion_matrix : dict = {}):
    '''Function to set the parameters for model predictions and the parameters 
    for creating the confusion matrix.'''
    def input_model_n_data (data_model):
        '''Function to produce the confusion matrix of the model.'''

        y_labels = data_model.data_normalised.y_test

        y_pred = data_model.model.predict (data_model.data_normalised.X_test_norm, **parameters_for_prediction)

        #confusion_matrix = tf.math.confusion_matrix (y_labels, y_pred, **parameters_for_confusion_matrix)

        return y_pred
    
    return input_model_n_data


if __name__ == "__main__":
    pass