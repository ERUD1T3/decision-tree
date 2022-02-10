############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: learner.py
#   Description: : input training examples/instances, 
#   output a tree (or rule set)
#############################################################

from distutils.log import debug
from math import log2
from dtree import DNode, DTree

class learner:

    def __init__(self, attr_path, training_path, testing_path, debug=False):
        '''
        Initialize the learner object
        '''
        self.attr_path = attr_path
        self.training_path = training_path
        self.testing_path =  testing_path
        self.debug = debug
        # attributes and their order
        self.attributes, self.order = self.read_attributes(self.attr_path)
        self.training = self.read_data(self.training_path)
        self.testing = self.read_data(self.testing_path)
        self.n_examples = len(self.training)
        self.n_pos = 0 # number of positive examples
        self.n_neg = 0 # number of negative examples



    def read_attributes(self, attr_path:str):
        '''
        Read in the attributes
        '''

        attributes = {}
        order = []

         # read in the attributes
        with open(attr_path, 'r') as f:
            for line in f:
                if len(line) > 1:
                    words = line.strip().split()
                    
                    # storing the attributes
                    attributes[words[0]] = words[1:]
                    order.append(words[0])

                
        if self.debug:
            print('Attributes: ', attributes)
            print('Order: ', order)
            print('Final Attribute: ', order[-1])

        return attributes, order


    def read_data(self, data_path: str):
        '''
        Read in the training data and testing data
        '''

        data = []

         # read in the attributes
        with open(data_path, 'r') as f:
            for line in f:
                    words = line.strip().split()
                    data.append(words)
               
        if self.debug:
            print('Read data: ', data)

        return data

    def entropy(self, data):
        '''
        Calculate the entropy of a data set
        '''
        # reading values of the target attribute
        target_classes = self.attributes[self.order[-1]] 
        entropy = 0.0
        n = len(data)
        c_dist = {c: 0.0 for c in target_classes}
        
        # getting the count for the target classes
        for d in data:
            c_dist[d[-1]] += 1

        if self.debug:
            print('distribution: ', c_dist)

        # calculating the entropy
        for c in target_classes:

            if self.debug:
                print('Class: ', c, ' prob: ', c_dist[c] / n)

            p = c_dist[c] / n

            if p != 0:
                entropy += -p * log2(p)
            # else entropy += 0 but we don't need to do that
        
        return entropy

    def gain(self, data, attribute: str):
        '''
        Calculate the information gain
        '''

        # calulating prior entropy
        prior = self.entropy(data)

        # getting the values of the attribute
        values = self.attributes[attribute]

        attr_index = self.order.index(attribute)

        # calculating the entropys over the subset
        posterior = 0.0
        for v in values:
            # getting the subset of the data
            subset = [d for d in data if d[attr_index] == v]
            # calculating the entropy
            p = len(subset) / len(data)
            e = self.entropy(subset)
            # calculating the gain
            posterior += p * e
        
        return prior - posterior

    def get_best_attribute(self, data, attrs_to_test: list()):
        '''
        Get the best attribute to split the data
        '''
        # getting the attributes
        attr_iter = iter(attrs_to_test)

        # getting the first attribute
        attr = next(attr_iter)

        max_gain, max_attr = self.gain(data, attr), attr
        if self.debug:
            print('Gain for attribute: ', max_attr, ' is: ', max_gain)

        # getting the best attribute for root node
        for attr in attr_iter:
            # getting the gain for each attribute
            gain = self.gain(data, attr)
            if self.debug:
                print('Gain for attribute: ', attr, ' is: ', gain)

            # updating the best attribute
            if gain > max_gain:
                max_gain, max_attr = gain, attr

        if debug:
            print('Best attribute: ', max_attr, ' with gain: ', max_gain)

        return max_attr


    # TODO: implement the decision tree learning algorithm
    def learn(self, training=None):
        '''learn the decision tree'''
        training = self.training if training is None else training
        # initialize the tree
        tree = DTree(self.attributes, self.order, self.debug)

        # setting the root node
        tree.root = DNode(self.get_best_attribute(training, self.order[:-1]))
            
            