import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm

import linearRegression
import utils

def knearest(data):
    print("K-nearest regression is coming...")
    X, y = data[0], data[1]
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(data[0], data[1])
    return knn 

def multilayer_perceptron(data):
    print('multilayer perceptron is coming...')
    X, y = data[0], data[1]
    mlp = MLPRegressor(random_state=1).fit(data[0], data[1])
    return mlp 
 
def choose_model_player(modelName, data):
    if modelName.lower() == 'knn':
        model = knearest(data)
    elif modelName.lower() == 'mlp':
        model = multilayer_perceptron(data)
    elif modelName.lower() == 'linear':
        print("linear regression is coming...")
        model = linearRegression.linear_regression_fit(data[0],data[1])
    return model

def predict_next_step(model, model_name, test):
    if model_name == 'linear':
        test = np.array(test)
        return np.transpose(linearRegression.linear_regression_predict(test,model))
    else:
        return model.predict(test)
'''
model_name = 'linear'
data = utils.load_multi('../datasets-part1/tictac_multi.txt')
player = choose_model_player(model_name, data)

if model_name == 'linear':
    predicts = predict_next_step(player, 'linear', [[1,0,0,0,-1,-1,0,1,1]])
print(predicts)
'''





            
