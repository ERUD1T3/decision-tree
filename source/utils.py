############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: utils.py
#   Description: utility functions for the decision tree
#############################################################


from math import log2 as lg
import random
from itertools import chain, combinations


def all_subsets(ss):
    '''returns all subsets of a set'''
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

def log2(x):
    '''log2 of x with support of 0'''
    if x == 0:
        return 0
    else:
        return lg(x)


def corrupt_data(data, classes, percent):
    '''corrupt the class labels of training examples from 0% to 20% (2% in-
    crement) by changing from the correct class to another class; output the
    accuracy on the uncorrupted test set with and without rule post-pruning.'''

    # get the number of training examples
    num_examples = len(data)
    # get the number of classes to corrupt
    num_examples_to_corrupt = int(percent * num_examples)
    # get the elements to corrupt
    corrupt_elements = random.sample(range(num_examples), num_examples_to_corrupt)

    # corrupt the data
    for e in corrupt_elements:
        # get the class label
        correct_label = data[e][-1]
        
        random_class = random.choice(classes)

        # while the random class is the same as the correct class
        while random_class == correct_label:
            random_class = random.choice(classes)
    
        # change the class label
        data[e][-1] = random_class
        
    return data