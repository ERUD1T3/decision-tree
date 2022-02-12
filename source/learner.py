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

        # tracking attributes generated from continuous data
        self.tmp_attr_values = {}
        self.tmp_order = []



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

        # for no data available, return 0.0
        if n == 0:
            return 0.0

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

    # TODO: test
    def gain(self, data, attribute: str):
        '''
        Calculate the information gain
        '''

        # calulating prior entropy
        prior = self.entropy(data)
        posterior = 0.0

        # case for continuous attribute
        if '<' in attribute or '>' in attribute:
            # getting the subset of the data
            pos_subset, neg_subset = [], []
            for d in data:
                if self.eval_continuous(d, attribute):
                    pos_subset.append(d)
                else:
                    neg_subset.append(d)

            # getting the probabilities for the subsets
            p_pos = len(pos_subset) / len(data)
            p_neg = len(neg_subset) / len(data)

            # calculating the entropy for the subsets
            e_pos = self.entropy(pos_subset)
            e_neg = self.entropy(neg_subset)

            # calculating the posterior entropy
            posterior = p_pos * e_pos + p_neg * e_neg

        # case of discrete data
        else:
            # getting the values of the attribute
            values = self.attr_values[attribute]
            attr_index = self.order.index(attribute)

            # calculating the entropys over the subset
            for v in values:
                # getting the subset of the data
                subset = [d for d in data if d[attr_index] == v]
                # calculating the entropy
                p = len(subset) / len(data)
                e = self.entropy(subset)
                # calculating the gain
                posterior += p * e
        
        return prior - posterior

    def get_best_discretized_attribute(self, data, d_attr_to_test):
        '''
        Get best discretize attribute
        '''
        # getting the attributes
        attr_iter = iter(d_attr_to_test)

        # getting the first attribute
        attr = next(attr_iter)

        # getting the best attribute
        best_gain, best_attr = self.gain(data, attr), attr

        # getting the best attribute
        for attr in attr_iter:
            # getting the gain
            gain = self.gain(data, attr)
            if gain > best_gain:
                best_attr, best_gain = attr, gain

        return best_gain, best_attr

    # TODO: test
    def get_best_attribute(self, data, attrs_to_test: list()):
        '''
        Get the best attribute to split the data
        '''
        # getting the attributes
        attr_iter = iter(attrs_to_test)
        # getting the first attribute
        attr = next(attr_iter)
        gain = 0.0
        max_gain, max_attr = 0.0, None

        # attribute is continuous
        if self.is_continuous(attr):
            # get discretized attributes from attr
            discretized, _ = self.continuous_to_discrete(attr, data)
            # get the best discretized attribute
            max_gain, max_attr = self.get_best_discretized_attribute(data, discretized)
        # attribute is discrete
        else:
            max_gain, max_attr = self.gain(data, attr), attr

            if self.debug:
                print('Gain for attribute: ', max_attr, ' is: ', max_gain)

        # getting the best attribute for root node
        for a in attr_iter:
            # attribute is continuous
            if self.is_continuous(a):
                # get discretized attributes from attr
                discretized, _ = self.continuous_to_discrete(a, data)
                # get the best discretized attribute
                gain, attr = self.get_best_discretized_attribute(data, discretized)
            # attribute is discrete
            else:
                # getting the gain for each attribute
                gain = self.gain(data, a)
                attr = a

            if self.debug:
                print('Gain for attribute: ', attr, ' is: ', gain)

            # updating the best attribute
            if gain > max_gain:
                max_gain, max_attr = gain, attr

        if self.debug:
            print('Best attribute: ', max_attr, ' with gain: ', max_gain)

        return max_attr

    
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

    # TODO: Test this function
    def continuous_to_discrete(self, attr, data):
        '''
        Convert the continuous attribute to discrete
        '''
        candidates_thresholds = []

        # getting the attribute target values
        # attr_target_pairs = {d[self.order.index(attr)]:d[-1] for d in data}
        attr_values = [d[self.order.index(attr)] for d in data]
        target_values = [d[-1] for d in data]

        # sort both list by the attribute values
        attr_values, target_values = zip(*sorted(zip(attr_values, target_values)))

        if self.debug:
            print('Sorted attribute values: ', attr_values)
            print('Sorted target values: ', target_values)

        low, high = 0, 1
        # getting candidate thresholds
        while high < len(attr_values):
            # find the change in the target values
            if target_values[low] != target_values[high]:
                # calculate the threshold
                threshold = (attr_values[low] + attr_values[high]) / 2
                candidates_thresholds.append(threshold)

            low_index += 1
            high_index += 1

        if self.debug:
            print('Candidate thresholds: ', candidates_thresholds)
        
        # create attributes based on the candidate thresholds
        for threshold in candidates_thresholds:
            # make a new attributes
            new_attr_gt = f'{attr} > {threshold}'
            new_attr_lt = f'{attr} < {threshold}'
            # should there be equal? No, very unlikely

            if self.debug:
                print('New attribute greater than: ', new_attr_gt)
                print('New attribute less than: ', new_attr_lt)

            # add the new attributes to the attribute list
            self.tmp_attr_values[new_attr_gt] = ['T', 'F']
            self.tmp_attr_values[new_attr_lt] = ['T', 'F']

            # add the new attributes to the attribute order
            self.tmp_order.append(new_attr_gt)
            self.tmp_order.append(new_attr_lt)

        # remove the old attribute (maybe not necessary)
        # self.attr_values.pop(attr)
        # self.order.remove(attr)

        if self.debug:
            print('New attribute order: ', self.tmp_order)
            print('New attribute values: ', self.tmp_attr_values)

        return self.tmp_order, self.tmp_attr_values


    def eval_continuous(self, datum, attr):
        '''
        Takes a the discretized version of a
        continuous attribute and the datum and 
        returns true if the inequality is satisfied,
        false otherwise
        '''
        # get less than attribute
        decomp = attr.split('<')
        if len(decomp) == 2:
            attr_lt = decomp[0]
            threshold = float(decomp[1])
            return datum[self.order.index(attr_lt)] < threshold

        # get greater than attribute
        decomp = attr.split('>')
        if len(decomp) == 2:
            attr_gt = decomp[0]
            threshold = float(decomp[1])
            return datum[self.order.index(attr_gt)] > threshold


    def is_continuous(self, attr):
        '''
        check if an attribute is continuous
        '''
        return self.attr_values[attr][0] == 'continuous'
    



    def ID3_build(self, data, target_attr, attrs_to_test: list()):
        '''
        Build a decision tree using ID3
        '''

        # check if there is no data
        # if len(data) == 0:
        #     return None

        # check if all the examples are the same
        if self.are_same(data):
            root = DNode(data[0][-1])
            root.is_terminal = True
            root.class_distribution = self.get_class_distribution(data)
            return root

        # check if there are no attributes to test
        if len(attrs_to_test) == 0:
            root = DNode(self.get_best_class(data))
            root.is_terminal = True
            root.class_distribution = self.get_class_distribution(data)
            return root

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

            if self.debug:
                print('Subset: ', subset)

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

        
            