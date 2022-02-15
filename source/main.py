############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse
from learner import Learner


def parse_args():
    '''parse the arguments for the titcactoe game'''

    parser = argparse.ArgumentParser(
        description='Decision Tree Learning program to classify both \
            discrete and continuous data'
    )

    parser.add_argument(
        '-a', '--attributes',
        type=str,
        required=True,
        help='path to the attributes files (required)'
    )

    parser.add_argument(
        '-d', '--training',
        type=str, 
        required=True,
        help='path to the training data files (required)'
    )
    
    parser.add_argument(
        '-t', '--testing',
        type=str , 
        required=True,
        help='path to the test data files (required)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug mode, prints statements activated (optional)'
    )
    
    # parse arguments
    args = parser.parse_args()
    return args


def main():
    '''main of the program'''
    args = parse_args() # parse arguments
    print(' args entered',args)

    training_path = args.training
    testing_path = args.testing
    attributes_path = args.attributes
    debugging = args.debug


    print('\nLearning the decision tree...\n')
    # run the program
    dtl = Learner(attributes_path, training_path, testing_path, debugging)
    tree = dtl.learn()

    print('\nTesting the tree on training data\n')
    # testing tree on training data
    dtl.test(tree, dtl.training)

    print('\nTesting the tree on testing data\n')
    # testing tree on test data
    dtl.test(tree)

    print('\nPrinting the decision tree rules\n')
    # print the rules
    dtl.tree_to_rules(tree)    

    print('\nTesting the rules on testing data\n')
    # testing tree on test data
    dtl.test(tree)

    print('\n Pruning the tree...\n')
    # prune the tree
    dtl.rule_post_pruning(tree, dtl.testing)

    print('\nTesting the tree post pruning\n')
    # testing tree on test data
    dtl.test(tree)


    
if __name__ == '__main__':
    main()
