import argparse
import utils
import regression
import game

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Run regressor, classification, or game algorithm on multi or single data'
    )
    parser.add_argument('-data_path',default='../datasets-part1/',help='data path of file containing tic tac toe data')
    parser.add_argument('-label',default='single',help="Type of data. single or multi.")
    parser.add_argument('-model_type', default='classification',help='Options: regression/classification')
    parser.add_argument('-game', '--game',action='store_true',help='If True play tic-tac-toe game.')
    parser.add_argument('-regression_model',default='linear',help='Options: k-nearest, linear, MLP')
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
            print("ERROR: Invalid label type. Please enter single or multi")
        
        if args.model_type == 'regression':
            model = args.regression_model
            if model == 'k-nearest':
                regression.knearest(data)
            elif model == 'linear':
                regression.linaer_regression(data)
            elif model == 'MLP':
                regression.multilayer_perceptron(data)
            else:
                print("EERROR: Please enter a valid regression model type. (k-nearest, linear, MLP)")

        elif args.model_type == 'classification':
            model = args.classification_model
            #write classification function calls here
        else:
            print("ERROR: Invald model type. Please enter regression or classification.")


if __name__ == '__main__':
    main(parse_args())