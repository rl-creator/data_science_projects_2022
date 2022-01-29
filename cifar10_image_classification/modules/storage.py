
'''Function to create data structures for better organisation of output 
from functions.
This makes it easier to understand what is output from a function, as the 
outputs can be identified from given labels, and as a result, it is easier to 
identify what is being input into a subsequent function.
'''

from collections import namedtuple

def store_data (*function_output):
    '''Creates a namedtuple to organise output from a function.
    Namedtuples are used because the function's output can be labeled making 
    it easier to identify and understand what was output.
    '''
    return namedtuple ('storage', function_output)

if __name__ == "__main__":
    pass