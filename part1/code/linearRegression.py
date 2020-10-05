import numpy as np

def normal_equation(inputs,output):
    inputs_with_bias = np.ones((inputs.shape[0],inputs.shape[1]+1))
    inputs_with_bias[:,:-1] = inputs
    first = np.dot(inputs_with_bias.transpose(),inputs_with_bias)
    first_inv = np.linalg.inv(first)
    second = np.dot(inputs_with_bias.transpose(),output)
    weights = np.dot(first_inv,second)
    return weights

def split_outputs(inputs,outputs):
    output_list = []
    for output in outputs.transpose():
        output_list.append((inputs,output))
    return output_list

def lin_regression_calc(inputs, weight):
    inputs_with_bias = np.ones((inputs.shape[0],inputs.shape[1]+1))
    inputs_with_bias[:,:-1] = inputs
    return np.dot(inputs_with_bias, weight)

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