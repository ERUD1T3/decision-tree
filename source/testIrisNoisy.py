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
from learner import Learner
from utils import corrupt_data
# from sys import setrecursionlimit


# change recursion limit to 1 million to prevent stack overflow
# setrecursionlimit(10**9)

def main():
    '''main of the program'''

    training_path = 'C:\\Users\\the_3\\Desktop\\College\\Spring2022\\ML\\decision-tree\\data\\iris\\iris-train.txt'
    testing_path = 'C:\\Users\\the_3\\Desktop\\College\\Spring2022\\ML\\decision-tree\\data\\iris\\iris-test.txt'
    attributes_path = 'C:\\Users\\the_3\\Desktop\\College\\Spring2022\\ML\\decision-tree\\data\\iris\\iris-attr.txt'
    debugging = False
    validation = True # higher accuracy when test data is used for vaildation

    # create learner
    dtl = Learner(
        attributes_path, 
        training_path, 
        testing_path, 
        debugging,
        validation
    )

    # start of the experiment 
    for p in range(0, 22, 2):


        print('////////////////////////////////////////////////////////////')
        # corrupt the data'
        print('\nCorrupting the data by changing from the correct class to another class...')
        print('\nPercentage of corruption: {}%'.format(p))
        dtl.training = corrupt_data(dtl.training, dtl.get_classes(), p / 100.)
        # print('\nCorrupted data:')
        # print(dtl.training)

        print('\nLearning the decision tree...\n')
        # run the program
        tree = dtl.learn()

        # print('\nTesting the tree on training data\n')
        # # testing tree on training data
        # training_acc = dtl.test(tree, dtl.training)
        # print('\nTraining accuracy: ', training_acc)

        print('\nTesting the tree on uncorrupted testing data\n')
        # testing tree on test data
        testing_acc = dtl.test(tree, dtl.testing)
        print('\nTesting accuracy: ', testing_acc)

        print('\nPrinting the decision tree rules\n')
        # print the rules
        dtl.tree_to_rules(tree)    
        tree.print_rules()

        print('\n Pruning the tree...\n')
        # prune the tree
        tree = dtl.rule_post_pruning(tree, dtl.validation)
        tree.print_rules()

        # print('\nTesting the rules on training data\n')
        # # testing tree on test data
        # training_acc = dtl.test(tree, dtl.training)
        # print('\nTraining accuracy: ', training_acc)

        print('\nTesting the rules on uncorrupted testing data\n')
        # testing tree on test data
        testing_acc = dtl.test(tree, dtl.testing)
        print('\nTesting accuracy: ', testing_acc)

        # break

    
if __name__ == '__main__':
    main()
