import argparse
import utils
import game

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Run regressor, classification, or game algorithm on multi or single data'
    )
    parser.add_argument('-data_path',default='../datasets-part1/',help='data path of file containing tic tac toe data')
    parser.add_argument('-label',default='single',help="Type of data. single or multi.")
    parser.add_argument('-model_type', default='classification',help='Options: regression/classification')
    parser.add_argument('-game', '--game',action='store_true',help='If True play tic-tac-toe game.')
    parser.add_argument('-regression_model',default='linear',help='Options: k-nearest, linear, perceptron')
    parser.add_argument('-classification_model',default='SVM',help='Options: k-nearest, SVM, perceptron')
    return parser.parse_args(args)

def main(args):
    if args.game:
        final = 'tictac_final.txt'
        
        data = utils.load_final(args.data_path + final)
        print("Starting game")
        game.run()
    else:
        multi_label = 'tictac_multi.txt'
        single_label = 'tictac_single.txt'

        if args.label == 'single':
            data = utils.load_single(args.data_path+single_label)
        elif args.label == 'multi':
            data = utils.load_multi(args.data_path+multi_label)
        else:
            print("Invalid label type")
        
        if args.model_type == 'regression':
            model = args.regression_model
            #write regression code here

        elif args.model_type == 'classification':
            model = args.classification_model
            #write classification function calls here
        else:
            print("Invald model type. Please enter regression or classification.")


if __name__ == '__main__':
    main(parse_args())