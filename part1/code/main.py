import argparse
import utils
import regression
import classifier
import game

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Run regressor, classification, or game algorithm on multi or single data'
    )
    parser.add_argument('-data_path',default='../datasets-part1/',help='data path of file containing tic tac toe data')
    # parser.add_argument('-label',default='single',help="Type of data. single or multi.")
    parser.add_argument('-model_type', default='classification',help='Options: regression/classification')
    parser.add_argument('-game', '--game',action='store_true',help='If True play tic-tac-toe game.')
    # parser.add_argument('-regression_model',default='linear',help='Options: k-nearest, linear, MLP')
    # parser.add_argument('-classification_model',default='SVM',help='Options: k-nearest, SVM, MLP')
    return parser.parse_args(args)

def main(args):
    if args.game:
        final = 'tictac_final.txt'
        multi_label = 'tictac_multi.txt'

        #function to load multi data and run regressor
        final_data = utils.load_final(args.data_path + final)
        multi_data = utils.load_final(args.data_path + multi_label)
        print("Starting game")
        game.run()
    else:
        # multi_label = 'tictac_multi.txt'
        # single_label = 'tictac_single.txt'

        # single_data = utils.load_single(args.data_path+single_label)
        # multi_data = utils.load_multi(args.data_path+multi_label)

        dataset = ['tictac_single.txt', 'tictac_multi.txt']
        regressors = ['k-nearest', 'linear', 'MLP']
        classifiers = ['k-nearest', 'SVM', 'MLP']

        if args.model_type == 'classification':
            for filename in dataset:
                print(filename+'\n')
                if 'single' in filename:
                    data = utils.load_single(args.data_path+filename)
                else:
                    data = utils.load_multi(args.data_path+filename)
                for classification in classifiers:
                    if classification == 'SVM':
                        classifier.linear_svm_classifier(data,filename)
                    elif classification == 'k-nearest':
                        classifier.knn_classifier(data)
                    elif classification == 'MLP':
                        classifier.mlp_classifier(data)
            
        elif args.model_type == 'regression':
            data = utils.load_multi(args.data_path+'tictac_multi.txt')
            for regressor in regressors:
                if regressor == 'k-nearest':
                    regression.knearest(data)
                elif regressor == 'linear':
                    regression.linear_regression(data)
                elif regressor == 'MLP':
                    regression.multilayer_perceptron(data)

        else:
            print("ERROR: Invald model type. Please enter regression or classification.")

        # if args.label == 'single':
        #     data = utils.load_single(args.data_path+single_label)
        # elif args.label == 'multi':
        #     data = utils.load_multi(args.data_path+multi_label)
        # else:
        #     print("ERROR: Invalid label type. Please enter single or multi")
        
        # if args.model_type == 'regression':
        #     model = args.regression_model
        #     if model == 'k-nearest':
        #         regression.knearest(data,args.label)
        #     elif model == 'linear':
        #         regression.linear_regression(data,args.label)
        #     elif model == 'MLP':
        #         regression.multilayer_perceptron(data,args.label)
        #     else:
        #         print("EERROR: Please enter a valid regression model type. (k-nearest, linear, MLP)")

        # elif args.model_type == 'classification':
        #     model = args.classification_model
        #     if model == 'SVM':
        #         classifier.linear_svm_classifier(data,args.label)
        #     elif model == 'k-nearest':
        #         classifier.knn_classifier(data,args.label)
        #     elif model == 'MLP':
        #         classifier.mlp_classifier(data,args.label)
        #     else:
        #         print("EERROR: Please enter a valid classification model type. (SVM, k-nearest, MLP)")

        # else:
        #     print("ERROR: Invald model type. Please enter regression or classification.")


if __name__ == '__main__':
    main(parse_args())