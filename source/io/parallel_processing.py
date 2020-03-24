from joblib import Parallel, delayed
import tqdm.notebook as tqdm

def parallel_extract(array, process_fn):

    Parallel(n_jobs=-1, backend='multiprocessing')\
             (delayed(process_fn)(item)\
              for item in tqdm.tqdm(array))

    print("Done")