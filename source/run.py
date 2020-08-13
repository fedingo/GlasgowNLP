from source.utils.load_utils import split_and_load
from concurrent.futures import ProcessPoolExecutor

from tqdm.auto import tqdm

### run experiment function ###
def run_experiment(model_class,
                   model_args,
                   model_kwargs,
                   dataset_class,
                   device,
                   evals = 0.04,
                   epochs=500,
                   model_path=None,
                   split_size=0.75,
                   workers = 4,
                   batch_size = 4,
                   dataset_kwargs = {}):

    
    # Hack to make TQDM work in notebook
    print(' ', end='', flush=True)
    
    dataset = dataset_class(**dataset_kwargs)
    train_dataloader, val_dataloader = split_and_load(dataset,
                                                      workers=workers,
                                                      batch_size=batch_size,
                                                      split_size=split_size)
    

    model = model_class(*model_args, **model_kwargs).to(device)

    if model_path is not None:
        model.load_model(model_path)

    model.train_model(train_dataloader, val_dataloader, epochs = epochs, eval_per_epoch=evals)

    loss = model.loss_curve.cpu().numpy()
    val_loss = model.validation_curve.cpu().numpy()
    
    del model
    
    return loss, val_loss


def schedule_runs(config_list, run_fn=run_experiment, max_workers=None):
    
    losses, vals = [], []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        pool = []
        for idx, kwargs in enumerate(config_list):
            process = executor.submit(run_fn, **kwargs)
            
            pool.append(process)
        
        for process in pool:
            try:
                loss, val = process.result()
                
                losses.append(loss)
                vals.append(val)
            except Exception as e:
                print(e)
                process.join() # Cleans the Zombie process

                
    return losses, vals
