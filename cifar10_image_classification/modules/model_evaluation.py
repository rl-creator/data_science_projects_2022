
'''Module containing functions for evaluating a 
neural network model.'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt




###   Evaluate model

def evaluate_model (**parameters_for_model_evaluation):
    '''Function to set the parameters for the model evaluation.'''
    def input_model_n_data (data_model):
        '''Function to evaluate the model'''

        X_test, y_test = data_model.data_normalised.X_test_norm, data_model.data_normalised.y_test

        results = data_model.model.evaluate (x = X_test, y = y_test, **parameters_for_model_evaluation)

        for ind in range (len (results)):
            print (f"The {data_model.model.metrics_names[ind]} of the model is {results[ind]}.")

        return data_model

    return input_model_n_data


def return_confusion_matrix (parameters_for_prediction : dict = {}, parameters_for_confusion_matrix : dict = {}):
    '''Function to set the parameters for model predictions and the parameters 
    for creating the confusion matrix.'''
    def input_model_n_data (data_model):
        '''Function to produce the confusion matrix of the model.'''

        y_labels = data_model.data_normalised.y_test

        y_pred = np.argmax (data_model.model.predict (data_model.data_normalised.X_test_norm, **parameters_for_prediction), axis = 1)

        confusion_matrix = tf.math.confusion_matrix (y_labels, y_pred, **parameters_for_confusion_matrix)

        plot_confusion_matrix (confusion_matrix)

        return data_model
    
    return input_model_n_data


def plot_confusion_matrix (matrix):
    '''Function to return graphical confusion matrix'''
    plt.figure (figsize = (12, 10))
    sns.heatmap (matrix, annot = True, fmt = "g")
    plt.xlabel ("Prediction")
    plt.ylabel ("Label")

    plt.show ()



if __name__ == "__main__":
    pass