
'''Model containing functions to return details about the neural network.'''

from tensorflow import keras
import matplotlib.pyplot as plt


def model_summary (summary = False, model_plot = False):
    '''Function to display the structure of a model.'''
    def return_model_summary (data_model):
        '''Input model to display.'''

        if summary is True:
            data_model.model.summary ()

        if model_plot is True:
            display (keras.utils.plot_model (data_model.model, show_shapes = True))

        return data_model

    return return_model_summary



def plot_metrics (data_model):
    '''Function for plotting the change in metrics during the model training.'''

    accuracy = data_model.history.history['sparse_categorical_accuracy']
    val_accuracy = data_model.history.history['val_sparse_categorical_accuracy']

    loss = data_model.history.history['loss']
    val_loss = data_model.history.history['val_loss']

    epochs = range (1, len (accuracy) + 1)

    plt.plot (epochs, accuracy, label = "Training accuracy")
    plt.plot (epochs, val_accuracy, label = "Validation accuracy")
    plt.title ("Training and validation accuracy")
    plt.xlabel ("epochs")
    plt.legend ()

    plt.figure ()

    plt.plot (epochs, loss, label = "Training loss")
    plt.plot (epochs, val_loss, label = "Validation loss")
    plt.title ("Training and validation loss")
    plt.xlabel ("epochs")
    plt.legend ()

    plt.show ()

    return data_model