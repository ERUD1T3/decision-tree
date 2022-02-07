############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/7/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse


def parse_args():
    '''parse the arguments for the titcactoe game'''
    parser = argparse.ArgumentParser(description='Tic Tac Toe with AI')
    # parser.add_argument('exp_path', type=str,
    #                     help='path to the experience data (required)')
    # parser.add_argument('--no-teacher', action='store_true', default=False,
    #                     help='disable the teacher option, the model will learn through self-play')
    # parser.add_argument('--ng', type=int , help='number of games to play against itself', default=False)
    # parse arguments
    args = parser.parse_args()
    return args


def main():
    '''main of the program'''
    args = parse_args() # parse arguments
    




if __name__ == '__main__':
    main()
