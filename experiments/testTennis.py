############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: testTennis.py
#   Description: print the tree, tree accuracy on
#   the training and test sets, the rules, rule
#   accuracy on the training and test sets (no
#   pruning, the dataset is too small)
#############################################################


# imports
from source.learner import Learner

def main():
    '''main of the program'''

    training_path = '../data/tennis/tennis-attr.txt'
    testing_path = '../data/tennis/tennis-test.txt'
    attributes_path = '../data/tennis/tennis-attr.txt'
    debugging = False


    print('\nLearning the decision tree...\n')
    # run the program
    dtl = Learner(attributes_path, training_path, testing_path, debugging)
    tree = dtl.learn()

    print('\nPrinting the decision tree...\n')
    tree.print_tree()

    print('\nTesting the tree on training data\n')
    # testing tree on training data
    dtl.test(tree, dtl.training)

    print('\nTesting the tree on testing data\n')
    # testing tree on test data
    dtl.test(tree, dtl.testing)

    print('\nPrinting the decision tree rules\n')
    # print the rules
    dtl.tree_to_rules(tree)    

    print('\nTesting the rules on Training data\n')
    # testing tree on test data
    dtl.test(tree, dtl.training)

    print('\nTesting the rules on Testing data\n')
    # testing tree on test data
    dtl.test(tree, dtl.testing)


    
if __name__ == '__main__':
    main()
