############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: classifier.py
#   Description: : input a tree (or
#   rule set) and labeled instances, output
#   the classifications/predictions and how
#   accurate the tree is with respect to the
#   correct labels (% of correct classifications).
#############################################################
from dtree import DNode, DTree
class classifier:

    def __init__(self, tree, debug=False):
        '''
        Initialize the classifier object
        '''
        self.tree = tree
        self.debug = debug

    def classify(self, tes):
        '''
        Classify the testing data
        '''
        pass