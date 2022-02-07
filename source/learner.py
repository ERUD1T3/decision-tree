############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: learner.py
#   Description: : input training examples/instances, 
#   output a tree (or rule set)
#############################################################

class learner:

    def __init__(self, attr_path, training_path, testing_path, debug=False):
        '''
        Initialize the learner object
        '''
        self.attr_path = attr_path
        self.training_path = training_path
        self.testing_path =  testing_path
        self.debug = debug

        self.attributes, self.final_attr_name = self.read_attributes(self.attr_path)


    # TODO: implement the attribute reader
    def read_attributes(self, attr_path):
        '''
        Read in the attributes
        '''

        attributes = {}
        final_attr_name = None
        is_final = False

         # read in the attributes
        with open(self.attr_path, 'r') as f:
            for line in f:
                if len(line) > 1:
                    words = line.strip().split()
                    
                    # storing the attributes
                    attributes[words[0]] = words[1:]

                    if is_final:
                        final_attr_name = words[0]
                else:
                    is_final = True
                



        if self.debug:
            print('Attributes: ', attributes)
            print('Final Attribute Name: ', final_attr_name)

        return attributes, final_attr_name

    def read_training_data(self, training_path):
        '''
        Read in the training data
        '''
        pass

    def read_testing_data(self, testing_path):
        '''
        read in the testing data
        '''
        pass

    def learn(self):
        '''learn the decision tree'''
        pass