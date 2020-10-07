import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, plot_confusion_matrix, hamming_loss
from sklearn.model_selection import StratifiedKFold, KFold
import math
# from classifier import svm_classifier, knn_classifier, mlp_classifier

from sklearn.metrics import confusion_matrix

from utils import multi_label_y_encoder, multi_label_y_decoder
from linearRegression import linear_regression_fit, linear_regression_predict


def get_exact_match_accuracy(predicted,labels):
    return accuracy_score(labels, predicted, normalize=True, sample_weight=None)


#motivated by https://stackoverflow.com/questions/32239577/getting-the-accuracy-for-multi-label-prediction-in-scikit-learn
def get_hamming_score(predicted,labels):
    score = 1 - hamming_loss(labels, predicted)
    return score


def get_intersect_accuracy(predicted,labels):
    intersect = len(labels.intersection(predicted))
    return float(intersect/len(labels))


def get_single_accuracy(predicted, labels):
    return get_exact_match_accuracy(predicted,labels)


def get_multi_accuracy(predicted, labels):
    exact = get_exact_match_accuracy(predicted,labels)
    hamming = get_hamming_score(predicted,labels)
    return (exact,hamming)


def make_confusion_matrix(model, X, labels, title, ONE_TENTH_DATA=False):
    fig, ax = plt.subplots()

    if ONE_TENTH_DATA:
        title = title + '_oneTenth'
        file_path = '../results/%s.pdf' % title
    else:
        file_path = '../results/%s.pdf' % title
    ax.set_title(title)

    if max(labels) > 9:
        labels_sorted_idx = labels.argsort()
        sorted_labels = labels[labels_sorted_idx]
        sorted_X = X[labels_sorted_idx]

        subsample = np.arange(0, labels.size, 2).tolist()
        plot_confusion_matrix(model, sorted_X[subsample], sorted_labels[subsample], labels=sorted_labels[subsample],
                              normalize='true', ax=ax, include_values=False)

        cnt = 0
        for label in ax.xaxis.get_ticklabels():
            interval = math.floor(len(subsample) // 9)
            if cnt % interval == 0:
                cnt += 1
                continue
            label.set_visible(False)
            cnt += 1

        cnt = 0
        for label in ax.yaxis.get_ticklabels():
            interval = math.floor(len(subsample) // 9)
            if cnt % interval == 0:
                cnt += 1
                continue
            label.set_visible(False)
            cnt += 1
    elif max(labels) < 2:
        plot_confusion_matrix(model, X, labels, labels=[-1, +1], normalize='true', ax=ax, include_values=False)
    else:
        plot_confusion_matrix(model, X, labels, labels=range(0, 9), normalize='true', ax=ax, include_values=False)

    plt.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=0)


def evaluate_models(class_type, model, X, y, data_type=None, encode=False, ONE_TENTH_DATA=False):
    accuracy_list = []
    hamming_list = []
    cnt = 1
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(X,y):
        print("------------- Fold: %s -------------" % cnt)
        cnt += 1

        training_data = (X[train_index], y[train_index])
        testing_data = (X[test_index], y[test_index])
       
        if model == 'linearRegression':
            weight_list = linear_regression_fit(training_data[0], training_data[1])
            predict = np.transpose(linear_regression_predict(testing_data[0], weight_list))
            # print(predict)
            # print(aaa)
        else:
            model.fit(training_data[0],training_data[1])
            predict = model.predict(testing_data[0])
            
        if class_type == 'regressor':
            predict = np.rint(predict) 
        
        if encode:
            predict = multi_label_y_decoder(predict)
            labels = multi_label_y_decoder(testing_data[1])
        else:
            if 'LinearSVC' in str(type(model)) and 'multi' in data_type:
                predict = multi_label_y_decoder(predict)
                labels = multi_label_y_decoder(testing_data[1])
            else:
                labels = testing_data[1]
        
        accuracy, hamming = get_multi_accuracy(predict,labels)
        accuracy_list.append(accuracy)
        hamming_list.append(hamming)
        print("Exact Accuracy: %0.2f" % (accuracy))
        print("Hamming score (multi-class accuracy): %0.2f" % (hamming))
    accuracy_list = np.array(accuracy_list)
    hamming_list = np.array(hamming_list)
    if (data_type is not None and 'single' in data_type) or (data_type is not None and 'final' in data_type):
        print("\nMean Exact Accuracy: %0.2f (+/- %0.2f)" % (accuracy_list.mean(), accuracy_list.std() * 2))
    else:
        print("\nMean Exact Accuracy: %0.2f (+/- %0.2f)" % (accuracy_list.mean(), accuracy_list.std() * 2))
        print("Mean Hamming Score: %0.2f (+/- %0.2f)\n\n" % (hamming_list.mean(), hamming_list.std() * 2))

    if class_type == 'classifier':
        if encode or 'LinearSVC' in str(type(model)) or 'single' in data_type or 'final' in data_type:
            print('\n Building Confusiton Matrix...')
            title = str(type(model)).strip('>').strip("'").split('.')[-1] + '_' + class_type + '_' + \
                    data_type.split('.')[0]
            make_confusion_matrix(model, testing_data[0], testing_data[1], title, ONE_TENTH_DATA)
