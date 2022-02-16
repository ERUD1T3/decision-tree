############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: testIris.py
#   Description: print the tree, tree accuracy on the
#   training and test sets, the rules after post-
#   pruning, rule accuracy on the training and
#   test sets
#############################################################


# imports
from source.learner import Learner

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

    print('\n Pruning the tree...\n')
    # prune the tree
    dtl.rule_post_pruning(tree, dtl.testing)

    print('\nTesting the rules on training data\n')
    # testing tree on test data
    dtl.test(tree, dtl.training)

    print('\nTesting the rules on testing data\n')
    # testing tree on test data
    dtl.test(tree, dtl.testing)


    
if __name__ == '__main__':
    main()
