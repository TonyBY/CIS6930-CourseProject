import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm

import utils
import eval

def knearest(data):
    print("K-nearest regression")
    X, y = data[0], data[1]
    knn = KNeighborsRegressor(n_neighbors=3)
    eval.evaluate_models('regressor',knn,X,y)

def multilayer_perceptron(data):
    print('MLP')
    X, y = data[0], data[1]
    mlp = MLPRegressor(random_state=1, solver='lbfgs', max_iter=396).fit(data[0], data[1])
    eval.evaluate_models('regressor',mlp,X,y)

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

def linear_regression(data):
    print('Linear Regression')
    X, y = data[0], data[1]   
    eval.evaluate_models('regressor','linearRegression',X,y)

def linear_regression_fit(X,y):
    weights_list = []    
    data_list = split_outputs(X, y)
    for data in data_list:
        weight = normal_equation(data[0],data[1])
        weights_list.append(weight)
    return weights_list

def linear_regression_predict(data,weight_list):
    outputs_list = []
    for weight in weight_list:
        Y = lin_regression_calc(data, weight)
        outputs_list.append(Y)
    return outputs_list
        

