############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: utils.py
#   Description: utility functions for the decision tree
#############################################################


from math import log2 as lg
import random

def log2(x):
    '''log2 of x with support of 0'''
    if x == 0:
        return 0
    else:
        return lg(x)


def corrupt_data(data, classes, percent):
    '''corrupt the class labels of training examples from 0% to 20% (2% in-
    crement) by changing from the correct class to another class; output the
    accuracy on the uncorrupted test set with and without rule post-pruning.'''

    # get the number of training examples
    num_examples = len(data)
    # get the number of classes to corrupt
    num_examples_to_corrupt = int(percent * num_examples)
    # get the elements to corrupt
    corrupt_elements = random.sample(range(num_examples), num_examples_to_corrupt)

    # corrupt the data
    for e in corrupt_elements:
        # get the class label
        correct_label = data[e][-1]
        
        random_class = random.choice(classes)

        # while the random class is the same as the correct class
        while random_class == correct_label:
            random_class = random.choice(classes)
    
        # change the class label
        data[e][-1] = random_class
        
    return data

def add_dist(dist1, dist2):
    '''
    add 2 string distribution and return the string result
    '''
    x, y, z = dist1.strip('(').strip(')').split(',')
    x2, y2, z2 = dist2.strip('(').strip(')').split(',')
    x = int(x) + int(x2)
    y = int(y) + int(y2)
    z = int(z) + int(z2)
    return '({},{},{})'.format(x, y, z)

def consolidate_rules(rules: list):
    '''consolidate rules with the same attribute and value'''
   # get the number of rules
    num_rules = len(rules)

    for r in range(num_rules):
        # get the rule
        rule = rules[r]
       
        if rule == '':
            continue

        # get the antecedent and consequent
        ante, cons = rule.split(' => ')
        target_class, dist = cons.split(' ')

        # for each other rule
        for r2 in range(r + 1, num_rules):
            # get the other rule
            rule2 = rules[r2]
           
            if rule2 == '':
                continue
            
            # get the antecedent and consequent
            ante2, cons2 = rule2.split(' => ')
            target_class2, dist2 = cons2.split(' ')

            # if the antecedents are the same
            if ante == ante2:
                # if the target class is the same
                if target_class == target_class2:
                    # add the distributions
                    dist = add_dist(dist, dist2)
                    # add the rule to the list of rules to remove
                    rules[r2] = ''
                    # set the new consequent
                    rules[r] = ante + ' => ' + target_class + ' ' + dist


    # remove the rules
    rules = list(filter(lambda val: val != '', rules))

    return rules


