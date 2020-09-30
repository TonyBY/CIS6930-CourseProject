import numpy as np

from sklearn.model_selection import StratifiedKFold
from classifier import linear_svm_classifier, knn_classifier, mlp_classifier
from utils import multi_label_y_encoder
import regression



def evaluate(model_names, model_list, data):
    X, y = data[0], data[1]
    print("data[0] shape: ", X.shape)
    print("data[1] shape: ", y.shape)

    if data[1][0].size == 1:
        reshaped_y = y.reshape(y.size)
    else:
        reshaped_y = multi_label_y_encoder(y).copy()
        print("encoded_data[1] shape: ", reshaped_y.shape)

    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)

    for model_name, model in zip(model_names, model_list):
        print("-------------------------- %s ------------------------------------" % model_name)
        scores = []
        cnt = 1
        for train_index, test_index in skf.split(X, reshaped_y):
            print("------------- Fold: %s -------------" % cnt)
            cnt += 1
            print("len(train_index)", len(train_index), "len(test_index)", len(test_index),)
            # print("TRAIN:", train_index, "TEST:", test_index)

            training_data = (X[train_index], y[train_index])
            testing_data = (X[test_index], y[test_index])

            estimator = model(training_data)
            if "linear" in model_name.lower() and y[test_index][0].size == 9:
                score = estimator.score(testing_data[0], multi_label_y_encoder(testing_data[1]))
            else:
                score = estimator.score(testing_data[0], testing_data[1])
            print("score: ", score)
            scores.append(score)
        scores = np.array(scores)
        print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    from utils import load_multi, load_single
    import sys

    single_label_data_path = '../datasets-part1/tictac_single.txt'
    multi_label_data_path = '../datasets-part1/tictac_multi.txt'

    argv = sys.argv

    if len(argv) == 2 and argv[1] == "multi":
            data = load_multi(multi_label_data_path)
    else:
        data = load_single(single_label_data_path)

    classifier_names = ["Linear SVM Classifier", "K Nearest Neighbors Classifier", "MLP Classifier"]
    classifier_model_list = [linear_svm_classifier, knn_classifier, mlp_classifier]

    Regressor_names = ["Linear Regression", "K Nearest Neighbors Regressor", "MLP Regressor"]

    evaluate(classifier_names, classifier_model_list, data)


