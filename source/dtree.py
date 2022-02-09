############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: dtree.py
#   Description: : main tree class and contains
#   all the functions to build a decision tree
#   with Tree printer (pre-order traversal,
#   deeper nodes are indented more, leaves
#   have class distribution) and Rule set printer.
#############################################################
class Node:
    def __init__(self, attribute, parent=None):
        self.attribute = attribute
        self.parent = parent # parent node
        self.children = {} # keys are attribute values
        # self.class_distribution = {}




class DTree:

    def __init__(self, attributes, order, debug=False):
        '''
        Initialize the tree object
        '''

        self.attributes = attributes
        self.order = order
        self.debug = debug

        self.root = None


    def print_tree(self, node=None, indent=0):
        '''
        Display the tree
        '''
        pass

    def print_rules(self, node=None, indent=0):
        '''
        Display the rules
        '''
        pass


    def build(self, training):
        '''
        Build the decision tree
        '''
        pass


    def test(self, testing):
        '''
        Test the decision tree
        '''
        pass

   