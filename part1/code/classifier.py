from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from utils import multi_label_y_encoder
from eval import evaluate_models


def linear_svm_classifier(data, data_type):
    print("Linear_SVM Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        y = y.reshape(y.size)
    else:
        y = multi_label_y_encoder(y)

    lsvm = LinearSVC(random_state=0, tol=1e-5)
    evaluate_models('classifier', lsvm, X, y, data_type=data_type)


def svm_classifier(data, encode=False, data_type=None):
    X, y = data[0], data[1]
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    evaluate_models('classifier', clf, X, y, encode=encode, data_type=data_type)


def knn_classifier(data, encode=False, data_type=None):
    print("K-Newrest Neighbors Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        y = y.reshape(y.size)
    elif encode:
        y = multi_label_y_encoder(y)

    knn = KNeighborsClassifier(n_neighbors=1)
    evaluate_models('classifier', knn, X, y, encode=encode, data_type=data_type)


def mlp_classifier(data, encode=False, data_type=None):
    print("Multilayer Perceptron Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        y = y.reshape(y.size)
    elif encode:
        y = multi_label_y_encoder(y)

    mlp = MLPClassifier(random_state=1, max_iter=500)
    evaluate_models('classifier', mlp, X, y, encode=encode, data_type=data_type)
