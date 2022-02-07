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

class DTree:

    def __init__(self, attributes, final_attr_name, debug=False):
        '''
        Initialize the tree object
        '''
        self.root = None
        self.attributes = attributes
        self.final_attr_name = final_attr_name
        self.debug = debug


    def build(self, training_path):
        '''
        Build the decision tree
        '''
        pass

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