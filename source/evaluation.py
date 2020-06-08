from sklearn.model_selection import cross_val_score, cross_validate
from tqdm.notebook import tqdm
import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import r2_score
from source.utils.load_utils import just_load

    
def r2_score_raw():
    
    def scorer_0 (estimator, X, y):
        y_pred = estimator.predict(X) 
        scores = r2_score(y, y_pred, multioutput = "raw_values")
        return scores[0]
    
    def scorer_1 (estimator, X, y):
        y_pred = estimator.predict(X) 
        scores = r2_score(y, y_pred, multioutput = "raw_values")
        return scores[1]
    
    def scorer_2 (estimator, X, y):
        y_pred = estimator.predict(X) 
        scores = r2_score(y, y_pred)
        return scores
        
    scorer_dict = {
        "arousal": scorer_0,
        "valence": scorer_1,
        "overall": scorer_2
    }
    
    return scorer_dict
        

def evaluate_on_task (estimator, dataset, scorer_fn, k_fold=5):
    
    # scorer(estimator, X, y) --> single value
    
    X, Y = [], []
    
    for sample in tqdm(dataset, leave = False):
        X.append(sample["song_features"])
        
        if sample.get("encoded_class") is not None:
            Y.append(sample['encoded_class'])
            classification = True
        else:
            Y.append(sample['target'])
            classification = False
        
    X, Y = np.array(X), np.array(Y).squeeze()
    
    if len(Y.shape) > 1:
        classification = False
    
    if classification:
        cv = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2)
    else:
        cv = ShuffleSplit(n_splits=k_fold, test_size=0.2)
            
    #scores = cross_val_score(estimator, X, Y, cv = cv, scoring=scorer_fn, error_score="raise")
    scores = cross_validate(estimator, X, Y,
                            cv = cv,
                            scoring=scorer_fn,
                            n_jobs = -1,
                            error_score="raise")
    
    scores = {k:v for k,v in scores.items() if "test" in k}
    
    return scores