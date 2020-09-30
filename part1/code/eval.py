import numpy as np
from sklearn.model_selection import StratifiedKFold

import regression
import utils



def kfold(inputs,outputs):
    kfold_split = StratifiedKFold(n_splits=10)
    kfold = kfold.split(inputs, outputs)
    return kfold

def evaluate(model_type, model, inputs, outputs):
    scores = []
    kfold = kfold(inputs,outputs)
    for fold, (train, test) in enumerate(kfold):
        if model_type == 'regression' and model == 'MLP'
        regression.MLP(inputs.iloc[train, :], outputs.iloc[train])
        score = MLP.score(inputs.iloc[test, :], outputs.iloc[test])
        scores.append(score)
        print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (fold+1, np.bincount(outputs.iloc[train]), score))
 
    print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

