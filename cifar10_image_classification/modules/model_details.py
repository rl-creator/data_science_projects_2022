
'''Model containing functions to return details about the neural network.'''

from tensorflow import keras


def return_model_summary (data_model):
    ''''''

    data_model.model.summary ()

    keras.utils.plot_model (data_model.model, show_shapes = True)

    return (data_model)