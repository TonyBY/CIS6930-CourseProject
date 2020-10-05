import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
# from classifier import svm_classifier, knn_classifier, mlp_classifier

from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils import multi_label_y_encoder, multi_label_y_decoder
from linearRegression import linear_regression_fit, linear_regression_predict


def get_exact_match_accuracy(predicted,labels):
    return accuracy_score(labels, predicted, normalize=True, sample_weight=None)


#motivated by https://stackoverflow.com/questions/32239577/getting-the-accuracy-for-multi-label-prediction-in-scikit-learn
def get_hamming_score(predicted,labels):
    acc_list = []
    for i in range(labels.shape[0]):
        labels_set = set( np.where(labels[i])[0] )
        predicted_set = set( np.where(predicted[i])[0] )
        
        temp_accuracy = None
        if len(labels_set) == 0 and len(predicted_set) == 0:
            temp_accuracy = 1
        else:
            intersect = len(labels_set.intersection(predicted_set))
            union = len(labels_set.union(predicted_set))
            temp_accuracy = float(intersect/union)
        acc_list.append(temp_accuracy)
    return np.mean(acc_list)


def get_intersect_accuracy(predicted,labels):
    intersect = len(labels.intersection(predicted))
    return float(intersect/len(labels))


def get_single_accuracy(predicted, labels):
    return get_exact_match_accuracy(predicted,labels)


def get_multi_accuracy(predicted, labels):
    exact = get_exact_match_accuracy(predicted,labels)
    hamming = get_hamming_score(predicted,labels)
    return (exact,hamming)


def make_confusion_matrix(model, X, labels, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    plot_confusion_matrix(model, X, labels, normalize='true', ax=ax)
    file_path = '../results/%s.pdf' % title
    plt.savefig(file_path, dpi=600, bbox_inches='tight', pad_inches=0)
    # plt.show()


def evaluate_models(class_type, model, X, y, data_type=None, encode=False):
    accuracy_list = []
    hamming_list = []
    cnt = 1
    kf = KFold(n_splits=10, shuffle=True)
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
    print("\nMean Exact Accuracy: %0.2f (+/- %0.2f)" % (accuracy_list.mean(), accuracy_list.std() * 2))
    print("\nMean Hamming Score: %0.2f (+/- %0.2f)\n\n" % (hamming_list.mean(), hamming_list.std() * 2))

    print(str(type(model)))
    print(class_type)
    if class_type == 'classifier':
        print(encode)
        if encode or 'LinearSVC' in str(type(model)) or 'single' in data_type:
            print('\n Building Confusiton Matrix...')
            title = str(type(model)).strip('>').strip("'").split('.')[-1] + '_' + class_type
            make_confusion_matrix(model, testing_data[0], testing_data[1], title)



