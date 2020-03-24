from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np


def evaluation_on_subtask(X,Y, folds = 5):

    clf = svm.SVC()
    scores = cross_val_score(clf, X, Y, cv=folds)
    
    return np.mean(scores)