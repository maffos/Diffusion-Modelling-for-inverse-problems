import pandas as pd
import numpy as np
import os


def traverse_subfolders(source_dir, exclude = None):
    best_params = {}
    best_score = np.infty
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            results_dir = os.path.join(subfolder_path, "results")
            if not 'exclude' in subfolder_path:
                if os.path.isdir(results_dir):
                    results_csv = os.path.join(results_dir, "results.csv")
                    if os.path.isfile(results_csv):
                        df = pd.read_csv(results_csv)
                        score = df['KL2'].mean()
                        if score <= best_score:
                            best_score = score
                            params = {}
                            path = subfolder_path.split('/')
                            params['Loss'] = path[4]
                            params['num_layers'] = path[5]
                            params['metric'] = path[6]
                            params['lam'] = path[7]
                            params['lr'] = path[-1]
                            if 'lam2' in path[-2]:
                                params['lam2'] = path[-2]
                            best_params = params
    return best_params, best_score

def compare_2_methods(path_method_1, path_method_2):
    best_params = {}
    best_score = np.infty
    results_csv_1 = os.path.join(path_method_1, "results.csv")
    results_csv_2 = os.path.join(path_method_2, "results.csv")

    if os.path.isfile(results_csv_1):
        df = pd.read_csv(results_csv_1)
        score1 = df['KL2'].mean()
    if os.path.isfile(results_csv_2):
        df = pd.read_csv(results_csv_2)
        score2 = df['KL2'].mean()

    if score1 <= score2:
        best_score = score1
        best_params = path_method_1
    else:
        best_score = score2
        best_params = path_method_2
    print('Method1: ', score1)
    print('Method2: ', score2)
    return best_params, best_score
# Usage
source_directory = 'examples/scatterometry/25-08-23/FPE/PINNLoss3/'
score, params = traverse_subfolders(source_directory)
#path1 = 'examples/scatterometry/dsm-loss'
#path2 = 'examples/scatterometry/scatterometry_runs_new/CFM/ErmonLoss/4layer/L2/lam:0.1/lr:0.0001/results'
#params,score = compare_2_methods(path1,path2)
print(params)
print(score)
