import argparse
import utils
from regression import knearest, linear_regression, multilayer_perceptron
from classifier import linear_svm_classifier, knn_classifier, mlp_classifier
import game


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Run regressor, classification, or game algorithm on multi or single data'
    )
    parser.add_argument('-data_path', default='../datasets-part1/', help='data path of file containing tic tac toe data')
    # parser.add_argument('-label',default='single',help="Type of data. single or multi.")
    parser.add_argument('-model_type', default='classification', help='Options: regression/classification')
    parser.add_argument('-game', '--game', action='store_true', help='If True play tic-tac-toe game.')
    # parser.add_argument('-regression_model',default='linear',help='Options: k-nearest, linear, MLP')
    # parser.add_argument('-classification_model',default='SVM',help='Options: k-nearest, SVM, MLP')
    parser.add_argument('-e', '--encode', action='store_true', help='encode the multi label into single label.')
    parser.add_argument('-o', '--oneTenth', action='store_true', help='only use 1/10 of data train the model.')

    return parser.parse_args(args)


def main(args):
    if args.game:
        final = 'tictac_final.txt'
        multi_label = 'tictac_multi.txt'

        #function to load multi data and run regressor
        final_data = utils.load_final(args.data_path + final)
        multi_data = utils.load_final(args.data_path + multi_label)
        print("Starting game")
        game.play(multi_data)
    else:
        dataset = ['tictac_single.txt', 'tictac_multi.txt', 'tictac_final.txt']
        regressors = ['k-nearest', 'linear', 'MLP']
        classifiers = ['k-nearest', 'SVM', 'MLP']

        if args.model_type == 'classification':
            for filename in dataset:
                print(filename+'\n')
                if 'single' in filename:
                    data = utils.load_single(args.data_path+filename, ONE_TENTH_DATA=args.oneTenth)
                elif 'multi' in filename:
                    data = utils.load_multi(args.data_path+filename, ONE_TENTH_DATA=args.oneTenth)
                else:
                    data = utils.load_final(args.data_path + filename, ONE_TENTH_DATA=args.oneTenth)
                for classification in classifiers:
                    if classification == 'SVM':
                        linear_svm_classifier(data, data_type=filename, ONE_TENTH_DATA=args.oneTenth)
                    elif classification == 'k-nearest':
                        knn_classifier(data, encode=args.encode, data_type=filename, ONE_TENTH_DATA=args.oneTenth)
                    elif classification == 'MLP':
                        mlp_classifier(data, encode=args.encode, data_type=filename, ONE_TENTH_DATA=args.oneTenth)
            
        elif args.model_type == 'regression':
            data = utils.load_multi(args.data_path+'tictac_multi.txt')
            for regressor in regressors:
                if regressor == 'k-nearest':
                    knearest(data)
                elif regressor == 'linear':
                    linear_regression(data)
                elif regressor == 'MLP':
                    multilayer_perceptron(data)

        else:
            print("ERROR: Invalid model type. Please enter regression or classification.")


if __name__ == '__main__':
    main(parse_args())
