from source.datasets.fast_datasets import GTZANFastDataset

from source.models_task_specific.mb_classification import MusicBertClassifier
from source.models_task_specific.mb_regression import MusicBertRegression

model_class = MusicBertClassifier
dataset_class = GTZANFastDataset
workers = 2

config = [
    {
        'model_class': model_class,
        'model_args': [10], #num_classes
        'model_kwargs': {
            'name': 'mb_gtzan_no_pretrain', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': 0.1,
        'epochs': 500,
        'model_path': None,
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4
    },
    {
        'model_class': model_class,
        'model_args': [10], #num_classes
        'model_kwargs': {
            'name': 'mb_gtzan_infonce', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': 0.1,
        'epochs': 500,
        'model_path': 'models/mb_infoNCE.pth',
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4
    },
    {
        'model_class': model_class,
        'model_args': [10], #num_classes
        'model_kwargs': {
            'name': 'mb_gtzan_jsd', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': 0.1,
        'epochs': 500,
        'model_path': 'models/mb_JSD.pth',
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4
    },
    {
        'model_class': model_class,
        'model_args': [10], #num_classes
        'model_kwargs': {
            'name': 'mb_gtzan_dv', 
            'num_encoder_layers': 4
        },
        'dataset_class': dataset_class,
        'device': 'cuda',
        'evals': 0.1,
        'epochs': 500,
        'model_path': 'models/mb_DV.pth',
        'split_size': 0.75,
        'workers': workers,
        'batch_size': 4
    },
]