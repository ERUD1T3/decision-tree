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
class DNode:
    def __init__(self, attribute, parent=None):
        '''
        Inintialize the node
        '''
        self.attribute = attribute
        self.parent = parent # parent node
        self.children = {} # keys are attribute values
        self.class_distribution = [] # keys are class values
        self.is_terminal = False

    def add_child(self, value, child):
        '''
        Add a child node
        '''
        self.children[value] = child


class DTree:
    def __init__(self, attributes, order, debug=False):
        '''
        Initialize the tree object
        '''

        self.attributes = attributes
        self.order = order
        self.debug = debug

        self.root = None


    def preorder(self, node, indent):
        '''
        Print a node
        '''
        if node is None:
            return

        attr = node.attribute

        # visiting a leaf node
        if node.is_terminal:
            c_dist = node.class_distribution
            c_dist_str =  [str(c) for c in c_dist]
            print(f': {attr} (' + ','.join(c_dist_str) + ')', end=' ')
        else:
            # visiting a non-leaf node children
            for value, child in node.children.items():
                print('\n', end='')
                print('| ' * indent, f'{attr} = {value}', end=' ')
                self.preorder(child, indent + 1)

    

    def print_tree(self):
        '''
        Display the tree via pre-order traversal
        '''
        # recusively print the tree starting at the root
        self.preorder(self.root, 0)


    def print_rules(self):
        '''
        Display the rules
        '''
        pass

   