############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse
from learner import Learner
from utils import consolidate_rules


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
    
    parser.add_argument(
        '--prune',
        action='store_true',
        default=False,
        help='prune the tree using rule post pruning (optional)'
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
    prune = args.prune


    print('\nLearning the decision tree...\n')
    # run the program
    dtl = Learner(attributes_path, training_path, testing_path, debugging)
    tree = dtl.learn()

    print('\nTesting the tree on training data\n')
    # testing tree on training data
    dtl.test(tree, dtl.training)

    print('\nTesting the tree on testing data\n')
    # testing tree on test data
    dtl.test(tree, dtl.testing)


    if prune:
        print('\nPrinting the decision tree rules before pruning\n')
        # print the rules
        dtl.tree_to_rules(tree)  
        tree.print_rules()  

    
        print('\n Pruning the tree...\n')
        # prune the tree
        dtl.rule_post_pruning(tree, dtl.testing)
        print('\nPrinting the decision tree rules after pruning\n')
        # consolidate rules
        tree.rules = consolidate_rules(tree.rules)
        tree.print_rules()

        print('\nTesting the tree post pruning\n')
        # testing tree on test data
        dtl.test(tree, dtl.testing)


    
if __name__ == '__main__':
    main()
