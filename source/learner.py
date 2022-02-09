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
        self.attributes, self.order = self.read_attributes(self.attr_path)
        self.training = self.read_data(self.training_path)
        self.testing = self.read_data(self.testing_path)



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


    # TODO: implement the decision tree learning algorithm
    def learn(self):
        '''learn the decision tree'''
        pass