############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: learner.py
#   Description: : input training examples/instances, 
#   output a tree (or rule set)
#############################################################

from math import log2
from dtree import DNode, DTree

class Learner:

    def __init__(self, attr_path, training_path, testing_path, debug=False):
        '''
        Initialize the learner object
        '''
        self.attr_path = attr_path
        self.training_path = training_path
        self.testing_path =  testing_path
        self.debug = debug
        # attributes and their order
        self.attr_values, self.order = self.read_attributes(self.attr_path)
        self.training = self.read_data(self.training_path)
        self.testing = self.read_data(self.testing_path)
        self.n_examples = len(self.training)



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

        if len(order) == 0:
            raise Exception('No attributes found')


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

        if len(data) == 0:
            raise Exception('No data found')

        return data

    def entropy(self, data):
        '''
        Calculate the entropy of a data set
        '''
        # reading values of the target attribute
        target_classes = self.attr_values[self.order[-1]] 
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
        values = self.attr_values[attribute]

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

        if self.debug:
            print('Best attribute: ', max_attr, ' with gain: ', max_gain)

        return max_attr

    # TODO: test this function
    def are_same(self, data):
        '''
        Check if all the examples are the same final class
        '''
        # getting data iterator
        data_iter = iter(data)

        # getting the first example
        first = next(data_iter)

        # checking if all the examples are the same
        for d in data_iter:
            if d[-1] != first[-1]:
                return False

        return True

    # TODO: test this function
    def get_best_class(self, data):
        '''
        return the best class in the data
        Currently the best class is the most common class in the data
        '''

        # getting the classes
        classes = self.attr_values[self.order[-1]]

        # getting the counts
        counts = {c: 0 for c in classes}
        for d in data:
            counts[d[-1]] += 1

        # getting the best class
        best_class = max(counts, key=counts.get)

        if self.debug:
            print('Best class: ', best_class)

        return best_class
    
    def get_class_distribution(self, data):
        '''
        return the class distribution of the data
        '''

        # getting the classes
        classes = self.attr_values[self.order[-1]]

        # getting the counts
        counts = [0 for _ in classes]
        for d in data:
            counts[classes.index(d[-1])] += 1

        if self.debug:
            print('Classes: ', classes)
            print('Class distribution: ', counts)

        return counts


    # TODO: fix this function
    def ID3_build(self, data, target_attr, attrs_to_test: list()):
        '''
        Build a decision tree using ID3
        '''
        # check if all the examples are the same
        if self.are_same(data):
            return DNode(data[0][-1])

        # check if there are no attributes to test
        if len(attrs_to_test) == 0:
            return DNode(self.get_best_class(data))

        # get the best attribute to split the data
        best_attr = self.get_best_attribute(data, attrs_to_test)

        # get the values of the attribute
        values = self.attr_values[best_attr]

        # remove the attribute from the list of attributes to test
        attrs_to_test.remove(best_attr)

        # create the root node
        root = DNode(best_attr)

        # create the children nodes
        for v in values:
            # getting the subset of the data
            subset = [d for d in data if d[self.order.index(best_attr)] == v]

            # if the subset is empty
            if len(subset) == 0:
                # set the child to the most common class
                leaf = DNode(self.get_best_class(data))
                leaf.parent = root
                leaf.is_terminal = True
                leaf.class_distribution = self.get_class_distribution(data)
                root.add_child(v, leaf)

            else:
                # remove the attribute from the list of attributes
                
                child = self.ID3_build(subset, target_attr, attrs_to_test)
                child.parent = root
                # create the child node
                root.add_child(v, child)

        return root


    # TODO: test this function
    def learn(self, training=None):
        '''learn the decision tree'''
        training = self.training if training is None else training

        # creating the tree
        tree = DTree(self.attr_values, self.order, self.debug)

        # learning the tree using ID3
        tree.root = self.ID3_build(training, self.order[-1], self.order[:-1])

        # printing the tree
        if self.debug:
            tree.print_tree()
        
        return tree

        
            