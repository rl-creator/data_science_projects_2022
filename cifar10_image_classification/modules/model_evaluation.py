
'''Module containing functions for evaluating a 
neural network model.'''


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


if __name__ == "__main__":
    pass