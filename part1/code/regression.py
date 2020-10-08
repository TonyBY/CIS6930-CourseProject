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
    mlp = MLPRegressor(random_state=1, solver='adam').fit(data[0], data[1])
    eval.evaluate_models('regressor',mlp,X,y)

def linear_regression(data):
    print('Linear Regression')
    X, y = data[0], data[1]   
    eval.evaluate_models('regressor','linearRegression',X,y)

        

