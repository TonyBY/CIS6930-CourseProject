from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from utils import multi_label_y_encoder
from eval import evaluate_models


def linear_svm_classifier(data, data_type, ONE_TENTH_DATA=False):
    print("Linear_SVM Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        y = y.reshape(y.size)
    else:
        y = multi_label_y_encoder(y)

    lsvm = LinearSVC(random_state=0, tol=1e-5)
    evaluate_models('classifier', lsvm, X, y, data_type=data_type, ONE_TENTH_DATA=ONE_TENTH_DATA)

def knn_classifier(data, encode=False, data_type=None, ONE_TENTH_DATA=False):
    print("K-Newrest Neighbors Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        y = y.reshape(y.size)
    elif encode:
        y = multi_label_y_encoder(y)

    knn = KNeighborsClassifier(n_neighbors=1)
    evaluate_models('classifier', knn, X, y, encode=encode, data_type=data_type, ONE_TENTH_DATA=ONE_TENTH_DATA)


def mlp_classifier(data, encode=False, data_type=None, ONE_TENTH_DATA=False):
    print("Multilayer Perceptron Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        y = y.reshape(y.size)
    elif encode:
        y = multi_label_y_encoder(y)

    mlp = MLPClassifier(random_state=1, max_iter=500)
    evaluate_models('classifier', mlp, X, y, encode=encode, data_type=data_type, ONE_TENTH_DATA=ONE_TENTH_DATA)
