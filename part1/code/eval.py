import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, hamming_loss, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import KFold
import math

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


def make_confusion_matrix(model, X, labels):  # labels is actually y_true
    if max(labels) > 9:
        labels_sorted_idx = labels.argsort()
        sorted_labels = labels[labels_sorted_idx]
        sorted_X = X[labels_sorted_idx]

        subsample = np.arange(0, labels.size, 2).tolist()

        X = sorted_X[subsample]
        y_pred = model.predict(X)
        y_true = sorted_labels[subsample]

        labels = sorted_labels[subsample]
    else:
        y_pred = model.predict(X)
        y_true = labels

    return confusion_matrix(y_true, y_pred, normalize='true'), labels


def get_average_of_matrieces(matrix_list):
    number_of_matrix = len(matrix_list)
    matrix_shape = matrix_list[0].shape

    temp_matrix = np.zeros(matrix_shape)

    for matrix in matrix_list:
        try:
            temp_matrix = np.add(temp_matrix, matrix)
        except ValueError:
            number_of_matrix -= 1
            continue

    return temp_matrix / number_of_matrix


def plot_confusion_matrix_single(cm, labels, title, ONE_TENTH_DATA=False):
    fig, ax = plt.subplots()

    if ONE_TENTH_DATA:
        title = title + '_oneTenth'
        file_path = '../results/%s.pdf' % title
    else:
        file_path = '../results/%s.pdf' % title
    ax.set_title(title)

    if max(labels) > 9:
        display_labels = labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(include_values=False, cmap='viridis', ax=ax, xticks_rotation='horizontal', values_format=None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    elif max(labels) < 2:
        display_labels = [-1, +1]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(include_values=False, cmap='viridis', ax=ax, xticks_rotation='horizontal', values_format=None)
    else:
        display_labels = range(0, 9)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(include_values=False, cmap='viridis', ax=ax, xticks_rotation='horizontal', values_format=None)

    plt.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=0)


def plot_confusion_matrix_multi(model, X, labels, title, ONE_TENTH_DATA=False):
    fig, ax = plt.subplots()

    if ONE_TENTH_DATA:
        title = title + '_oneTenth'
        file_path = '../results/%s.pdf' % title
    else:
        file_path = '../results/%s.pdf' % title
    ax.set_title(title)

    # print(max(labels))
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

    confusion_matrix_list = []
    cm_labels_to_use = [0]
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

        if class_type == 'classifier' and ('single' in data_type or 'final' in data_type):
            print('\n Building Confusiton Matrix...')
            cm, cm_labels = make_confusion_matrix(model, testing_data[0], testing_data[1])
            if max(cm_labels) > max(cm_labels_to_use):
                cm_labels_to_use = cm_labels
            confusion_matrix_list.append(cm)

    accuracy_list = np.array(accuracy_list)
    hamming_list = np.array(hamming_list)
    if (data_type is not None and 'single' in data_type) or (data_type is not None and 'final' in data_type):
        print("\nMean Exact Accuracy: %0.2f (+/- %0.2f)" % (accuracy_list.mean(), accuracy_list.std() * 2))
    else:
        print("\nMean Exact Accuracy: %0.2f (+/- %0.2f)" % (accuracy_list.mean(), accuracy_list.std() * 2))
        print("Mean Hamming Score: %0.2f (+/- %0.2f)\n\n" % (hamming_list.mean(), hamming_list.std() * 2))

    if class_type == 'classifier' and 'multi' in data_type:
        print('\n Ploting Confusiton Matrix...')
        title = str(type(model)).strip('>').strip("'").split('.')[-1] + '_' + class_type + '_' + \
                data_type.split('.')[0]
        plot_confusion_matrix_multi(model, testing_data[0], testing_data[1], title, ONE_TENTH_DATA)

    elif len(confusion_matrix_list) != 0:
        print("Calculation the averaged confusion matrix...")
        averaged_confusion_matrix = get_average_of_matrieces(confusion_matrix_list)

        print('\n Ploting Averaged Confusiton Matrix...')
        title = str(type(model)).strip('>').strip("'").split('.')[-1] + '_' + class_type + '_' + \
                data_type.split('.')[0]

        plot_confusion_matrix_single(averaged_confusion_matrix, cm_labels_to_use, title, ONE_TENTH_DATA)
