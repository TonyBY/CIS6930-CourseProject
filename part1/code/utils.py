from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def load_single(path, ONE_TENTH_DATA=False):
    data = np.loadtxt(path)

    inputs = data[:, :-1]
    outputs = data[:, -1:]

    if ONE_TENTH_DATA:
        X = inputs
        y = outputs
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break
        return (X_test, y_test)
    else:
        return (inputs, outputs)


def load_multi(path, ONE_TENTH_DATA=False):
    data = np.loadtxt(path)

    inputs = data[:, :9]
    outputs = data[:, 9:]

    if ONE_TENTH_DATA:
        X = inputs
        y = outputs
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break
        return (X_test, y_test)
    else:
        return (inputs, outputs)


def load_final(path, ONE_TENTH_DATA=False):
    data = np.loadtxt(path)

    inputs = data[:, :-1]
    outputs = data[:, -1:]

    if ONE_TENTH_DATA:
        X = inputs
        y = outputs
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break
        print("X.shape: ", X_test.shape)
        print("y.shape: ", y_test.shape)
        return (X_test, y_test)
    else:
        return (inputs, outputs)



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
