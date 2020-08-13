from source.datasets.fast_datasets import DeezerFastDataset

from source.models_task_specific.mb_classification import MusicBertClassifier
from source.models_task_specific.mb_regression import MusicBertRegression

## MODEL Setup
model_class = MusicBertRegression
workers = 1
epochs = 10
evals_per_epoch = 1
split = 0.95

## Dataset Setup
dataset_class = DeezerFastDataset

dataset_kwargs = {}
# dataset_kwargs = {
#     'use_cache': True,
# }

config = [
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_deezer_no_pretrain', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': None,
        'split_size': split,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_deezer_infonce', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': 'models/mb_infoNCE.pth',
        'split_size': split,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_deezer_jsd', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': 'models/mb_JSD.pth',
        'split_size': split,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_deezer_dv', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': 'models/mb_DV.pth',
        'split_size': split,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
]