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
        # self.value = None

    def add_child(self, value, child):
        '''
        Add a child node
        '''
        # child.value = value
        self.children[value] = child
        

    def get_class_dist_str(self) -> str:
        '''
        Return the string representation of
        the class distribution 
        '''
        c_dist = self.class_distribution
        c_dist_str =  [str(c) for c in c_dist]

        return '(' + ','.join(c_dist_str) + ')'


    # TODO: test this function
    def get_rule(self):
        '''
        Get the rule for the node
        '''
        rule = ''
        if self.parent:
            rule += self.parent.get_rule()
        if self.attribute:
            if self.is_terminal:
                rule += f' => {self.attribute} {self.get_class_dist_str()}'
            elif self.parent is None:
                rule += f'{self.attribute} = {self.value}'
            else:
                rule += f' ^ {self.attribute} = {self.value}'
        return rule

class DTree:
    def __init__(self, debug=False):
        '''
        Initialize the tree object
        '''

        self.debug = debug
        self.root = None
        self.rules = []

    def preorder(self, node, indent):
        '''
        Print a node
        '''
        if node is None:
            return

        attr = node.attribute

        # visiting a leaf node
        if node.is_terminal:
            # c_dist = node.class_distribution
            # c_dist_str =  [str(c) for c in c_dist]
            print(f': {attr} {node.get_class_dist_str()}', end=' ')
        else:
            # visiting a non-leaf node children
            for value, child in node.children.items():
                print('\n', end='')
                print(('|  ' * indent) + f'{attr} = {value}', end=' ')
                self.preorder(child, indent + 1)
            

    def print_tree(self):
        '''
        Display the tree via pre-order traversal
        '''
        # recusively print the tree starting at the root
        self.preorder(self.root, 0)
        print('\n')
        
    def print_rules(self):
        '''
        Display the rules
        '''
        for rule in self.rules:
            print(rule)

   