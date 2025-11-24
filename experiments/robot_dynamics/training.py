import os
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path.cwd().parents[1]))

MODELS = ['vi', 'cv', 'gp']
MODEL_HELP = "Choose model: 'all', 'vi', 'cv', 'gp';"

def argparse_uci():
    parser = argparse.ArgumentParser(description='Robot Dynamics Experiment')
    parser.add_argument('model', type=str, help=MODEL_HELP)
    return parser.parse_args()

def run(model_name):
    run_str = f"python train_part.py {model_name}"
    os.system(run_str)

if __name__ == "__main__":
    args = argparse_uci()
    options = MODELS if args.model == 'all' else [args.model,]
    for model_name in options:
        print(f"Training {model_name.upper()}")
        run(model_name)
