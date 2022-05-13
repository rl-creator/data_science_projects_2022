
'''Module containing functions for creating functional pipeline.'''


def create_pipeline (list_of_functions):
    '''Input a list of functions to be converted to a functional pipeline.
    
    Removed "*" from *list_of_functions, so, list_of_functions must be a proper
    list object (i.e. "[]"), rather than a sequence of positional arguments.
    The "*" needed to be removed because calling the pipeline function from 
    within another function causes issue. 
    If the create_pipeline function is called from within a function that takes 
    a list as an input argument, the list cannot be passed to the 
    create_pipeline function with *list_of_function, as there is an issue trying
    to pass a list to an input that want a sequence of positional arguments.
    The other issue is, if create_pipeline takes a sequenece of positional 
    arguments (i.e. *list_of_functions) and it is called within a function 
    that also takes a sequence of positional arguments, the sequence of 
    arguments cannot to passed to create_pipeline.
    So, because of these two issues, *list_of_functions needed to be removed 
    and create_pipeline needs to take a list object as input.
    
    If a list is input as 
    list_of_functions, *list_of_functions cannot read the list. If another 
    '''
    def apply_pipeline (input):
        '''Run the created pipeline by inputing the argument needed to start
        the pipeline.
        
        Input is whatever initiates the pipeline. For example: the input layer 
        of a neural network. 

        The list_of_functions would be the layers of the neural network. Using 
        the Keras functional API, a pipeline of the layers can be created. The 
        input for that pipeline would be the initial input layer for the neural
        network.
        
        input: The input that initiate the pipelines.
        Example: file path, web address, dataset object, 
        '''
        res = input

        for function in list_of_functions:
            res = function (res)

        return res
    return apply_pipeline

if __name__ == "__main__":
    pass