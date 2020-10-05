from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from utils import multi_label_y_encoder
import eval

def linear_svm_classifier(data, data_type):
    print("Linear_SVM Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        # print("Using single-label dataset: ")
        y = y.reshape(y.size)
    else:
        # print("Using multi-label dataset: ")
        y = multi_label_y_encoder(y)

    lsvm = LinearSVC(random_state=0, tol=1e-5)
    eval.evaluate_models('classifier',lsvm,X,y,data_type)
    # lsvm.fit(X, y)

    # print("Score: ", lsvm.score(X, y))
    # return lsvm

def svm_classifier(data):
    X, y = data[0], data[1]
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    eval.evaluate_models('classifier',clf,X,y)

def knn_classifier(data):
    print("K-Newrest Neighbors Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        # print("Using single-label dataset: ")
        y = y.reshape(y.size)
    # else:
    #     print("Using multi-label dataset: ")

    knn = KNeighborsClassifier(n_neighbors=1)
    eval.evaluate_models('classifier',knn,X,y)
    # knn.fit(X, y)
    # #
    # # print("Score: ", knn.score(X, y))
    # return knn


def mlp_classifier(data):
    print("Multilayer Perceptron Classifier")

    X, y = data[0], data[1]

    if data[1][0].size == 1:
        # print("Using single-label dataset: ")
        y = y.reshape(y.size)
    # else:
        # print("Using multi-label dataset: ")

    mlp = MLPClassifier(random_state=1, max_iter=500)#.fit(X, y)
    eval.evaluate_models('classifier',mlp,X,y)
    mlp.fit(X, y)
    #
    # print("Score: ", mlp.score(X, y))
    return mlp