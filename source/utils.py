############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: utils.py
#   Description: utility functions for the decision tree
#############################################################


from math import log2 as lg
import random

def log2(x):
    '''log2 of x with support of 0'''
    if x == 0:
        return 0
    else:
        return lg(x)

# TODO: fix this function
def corrupt_data(data, percent):
    '''corrupt the class labels of training examples from 0% to 20% (2% in-
    crement) by changing from the correct class to another class; output the
    accuracy on the uncorrupted test set with and without rule post-pruning.'''

    # corrupt the data
    for i in range(len(data)):
        # get the class label
        label = data[i][-1]
        # get the index of the class label
        index = data[i].index(label)
        # get the number of classes
        num_classes = len(data[i]) - 1
        # get the number of classes to corrupt
        num_classes_corrupt = int(num_classes * percent)
        # get the classes to corrupt
        classes_corrupt = []
        for j in range(num_classes_corrupt):
            # get a random class
            class_corrupt = random.randint(0, num_classes - 1)
            # check if the class is already in the list
            while class_corrupt in classes_corrupt:
                # get a random class
                class_corrupt = random.randint(0, num_classes - 1)
            # add the class to the list
            classes_corrupt.append(class_corrupt)
        # corrupt the class labels
        for j in classes_corrupt:
            # get a random class
            class_corrupt = random.randint(0, num_classes - 1)
            # check if the class is already in the list
            while class_corrupt in classes_corrupt:
                # get a random class
                class_corrupt = random.randint(0, num_classes - 1)
            # add the class to the list
            classes_corrupt.append(class_corrupt)
        # corrupt the class labels
        for j in classes_corrupt:
            # corrupt the class label
            data[i][j] = random.choice(data[i])
    return data