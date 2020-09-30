import numpy as np
from sklearn.model_selection import StratifiedKFold

def load_single(path):
    data = np.loadtxt(path)
    inputs = data[:,:-1]
    outputs = data[:,-1:]
    return(inputs,outputs)

def load_multi(path):
    data = np.loadtxt(path)
    inputs = data[:,:9]
    outputs = data[:,9:]
    return(inputs,outputs)

def load_final(path):
    data = np.loadtxt(path)
    inputs = data[:,:9]
    outcome = data[:,9:]
    return(inputs,outcome)


def multi_label_y_encoder(multi_y):
    encoded_label_list = []
    for y_array in multi_y:
        y_list = list(y_array)
        y_int = int("".join(str(int(x)) for x in y_list), 2)
        encoded_label_list.append(y_int)

    encoded_label_array = np.array(encoded_label_list).reshape(len(encoded_label_list))
    return encoded_label_array


def multi_label_y_decoder(encoded_multi_y):
    decoded_multi_y = np.empty((0, 9), float)
    for y_int in encoded_multi_y:
        lower_bits = str(bin(int(y_int))).split('b')[1]
        higher_bits = '0'*(9-len(lower_bits))
        y_string = higher_bits + lower_bits
        y_list = [float(bit) for bit in y_string]
        y_array = np.array(y_list).reshape(1, 9)
        decoded_multi_y = np.append(decoded_multi_y, y_array, axis=0)
    return decoded_multi_y
