
'''Functions to produce data structures to help with the storage of outputs
from functions.'''

from collections import namedtuple

def store_data (*function_output):
    '''Create a namedtuple to store output from a function.
    Namedtuples are used for labelling output of a function and more easily 
    organise the input in the following function.
    '''
    return namedtuple ('storage', function_output)

if __name__ == "__main__":
    pass