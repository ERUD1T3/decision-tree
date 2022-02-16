############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: testTennis.py
#   Description: corrupt the class labels of
#   training examples from 0% to 20% (2% in-
#   crement) by changing from the correct class
#   to another class; output the accuracy on the
#   uncorrupted test set with and without rule
#   post-pruning.
#############################################################


# imports
from source.learner import Learner
from source.utils import corrupt_data

def main():
    '''main of the program'''

    training_path = '../data/iris/iris-attr.txt'
    testing_path = '../data/iris/iris-test.txt'
    attributes_path = '../data/iris/iris-attr.txt'
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
