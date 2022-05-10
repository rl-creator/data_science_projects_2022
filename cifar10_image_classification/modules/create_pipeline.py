
'''Module containing function for creating functional pipeline.'''

def create_pipeline (list_of_functions : list):
    '''Input a list of functions to be converted to a functional pipeline.'''
    def pipeline (input = None):
        '''Input for the pipeline function.
        
        Input is whatever initiates the pipeline. For example: the input layer 
        of a neural network. 

        The list_of_functions would be the layers of the neural network. Using 
        the Keras functional API, a pipeline of the layers can be created. The 
        input for that pipeline would be the initial input layer for the neural
        network.
        
        input: Default value is None so pipeline can be initialised without the 
        need to assign a value to 'input'.
        '''
        res = input

        for function in list_of_functions:
            res = function (res)

        return res
    return pipeline