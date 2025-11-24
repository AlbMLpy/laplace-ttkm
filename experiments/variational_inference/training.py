import os
import sys
import argparse
from pathlib import Path
from itertools import chain
from joblib import Parallel, delayed
sys.path.append(str(Path.cwd().parents[1]))

from configs.variational_inference import ( 
    FMAP,
    FSHIFT,
    MODELS,
    DATASETS,
)

MODEL_HELP = "Choose model: 'all', 'vi', 'cv';"
PARALLEL_HELP = "Turn on/off parallel mode;"
N_JOBS_HELP = "How many processes to use for computations;"

def argparse_uci():
    parser = argparse.ArgumentParser(description='VI vs CV experiment')
    parser.add_argument('model', type=str, help=MODEL_HELP)
    parser.add_argument('-p', '--parallel', action='store_true', help=PARALLEL_HELP)
    parser.add_argument('-n', '--n_jobs', type=int, default=12, help=N_JOBS_HELP)
    return parser.parse_args()

def get_options(model, datasets):
    if model in MODELS:
        datasets = [v for v in datasets]
        return [(model, data) for data in datasets]
    else:
        raise ValueError(f"Bad model name: {model}")

def run(model_name, data_name, fmap):
    run_str = f"python train_part.py {model_name} {data_name} {fmap} -tqdm"
    os.system(run_str)

if __name__ == "__main__":
    args = argparse_uci()
    if args.model == 'all':
        options = [get_options(model, DATASETS) for model in MODELS]
        options = list(chain.from_iterable(options))
    else:
        options = get_options(args.model, DATASETS)
    
    if args.parallel:
        Parallel(n_jobs=args.n_jobs)(
            delayed(run)(model_name, data_name, FMAP) 
            for model_name, data_name in options
        )
    else:
        for model_name, data_name in options:
            run(model_name, data_name, FMAP)
