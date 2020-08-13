from source.datasets.fast_datasets import EmoMusicFastDataset

from source.models_task_specific.mb_classification import MusicBertClassifier
from source.models_task_specific.mb_regression import MusicBertRegression

## MODEL Setup
model_class = MusicBertRegression
workers = 2
epochs = 200
evals_per_epoch = 0.1

## Dataset Setup
dataset_class = EmoMusicFastDataset
dataset_kwargs = {
    'use_cache': True,
}

config = [
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_emomusic_no_pretrain', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': None,
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_emomusic_infonce', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': 'models/mb_infoNCE.pth',
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_emomusic_jsd', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': 'models/mb_JSD.pth',
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
    {
        'model_class': model_class,
        'model_args': [2], #num_features
        'model_kwargs': {
            'name': 'mb_emomusic_dv', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': evals_per_epoch,
        'epochs': epochs,
        'model_path': 'models/mb_DV.pth',
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4,
        'dataset_kwargs': dataset_kwargs,
    },
]