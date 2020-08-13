from tqdm.notebook import tqdm
import numpy as np

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, cross_validate

from source.utils.load_utils import just_load, split_and_load
from source.models_task_specific.mb_classification import MusicBertClassifier
from source.models_task_specific.mb_regression import MusicBertRegression

    
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
        

### SVM EVALUATION ###
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


### FINE TUNING FUNCTION ###
def fine_tune_model(dataset_class, evals = 0.04, epochs=500, load_model_path=None, split_size=0.75, workers = 4, name=""):
    
    dataset = dataset_class()
    if split_size is None:
        split_size = 0.75
    
    train_dataloader, val_dataloader = split_and_load(dataset, workers=workers, batch_size=4, split_size=split_size)
    
    if "classification" in dataset.tags:
        num_classes = dataset.get_num_classes()
        model = MusicBertClassifier(num_classes, name=name, num_encoder_layers=4, multi_label="multi_label" in dataset.tags).cuda()
        
    else:
        sample = dataset[0]
        target_size = sample['target'].shape[-1]
        model = MusicBertRegression(target_size, name=name, num_encoder_layers=4).cuda()

    if load_model_path is not None:
        model.load_model(load_model_path)

    model.train_model(train_dataloader, val_dataloader, epochs = epochs, eval_per_epoch=evals)

    loss = model.loss_curve.cpu().numpy()
    val_loss = model.validation_curve.cpu().numpy()
    
    del model
    
    return loss, val_loss