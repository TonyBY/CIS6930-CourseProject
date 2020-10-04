import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import svm

import utils

def knearest(data):
    print("K-nearest regression")
    lin_clf = svm.LinearSVC()
    lin_clf.fit(data[0], data[1])
    print(in_clf.predict([[1, -1, 0, 0, 0, 0, 0, 1, 0]]))


def multilayer_perceptron(data):
    print('MLP')
    mlp = MLPRegressor(random_state=1, max_iter=500).fit(data[0], data[1])
    

def normal_equation(inputs,output):
    first = np.dot(inputs.transpose(),inputs)
    first_inv = np.linalg.inv(first)
    second = np.dot(inputs.transpose(),output)
    weights = np.dot(first_inv,second)
    return weights

def split_outputs(inputs,outputs):
    output_list = []
    for output in outputs.transpose():
        output_list.append((inputs,output))
    return output_list

def lin_regression_calc(inputs, weight):
    return np.dot(inputs, weight)

def linaer_regression(data):
    practice_input = [1, 1, -1, 1, 0, 1, -1, -1, 0]    
    weights_list = []
    outputs_list = []
    
    data_list = split_outputs(data[0], data[1])
    for data in data_list:
        weight = normal_equation(data[0],data[1])
        weights_list.append(weight)

    for weight in weights_list:
        Y = lin_regression_calc(practice_input, weight)
        outputs_list.append(Y)
        print(outputs_list)
        print(outputs_list.index(max(outputs_list)))

