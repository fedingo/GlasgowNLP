from .run import schedule_runs
from .datasets.fast_datasets import *

from .models_task_specific.mb_classification import MusicBertClassifier
from .models_task_specific.mb_regression import MusicBertRegression

config = [
    {
        'model_class': MusicBertClassifier,
        'model_args': [10], #num_classes
        'model_kwargs': {
            'num_encoder_layers': 4
        },
        'dataset_class': GTZANFastDataset,
        'device': 'cuda',
        'evals': 1,
        'epochs': 1,
        'model_path': None,
        'split_size': 0.75,
        'workers': 2,
        'batch_size': 4
    },
    {
        'model_class': MusicBertClassifier,
        'model_args': [10], #num_classes
        'model_kwargs': {
            'num_encoder_layers': 4
        },
        'dataset_class': GTZANFastDataset,
        'device': 'cuda',
        'evals': 1,
        'epochs': 1,
        'model_path': 'models/music_bert_cpc_frozen.pth',
        'split_size': 0.75,
        'workers': 2,
        'batch_size': 4
    },
    {
        'model_class': MusicBertClassifier,
        'model_args': [10], #num_classes
        'model_kwargs': {
            'num_encoder_layers': 4
        },
        'dataset_class': GTZANFastDataset,
        'device': 'cuda',
        'evals': 1,
        'epochs': 1,
        'model_path': 'models/music_bert_mlm_ce.pth',
        'split_size': 0.75,
        'workers': 2,
        'batch_size': 4
    },
]


schedule_runs(config)