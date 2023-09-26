import pandas as pd
import numpy as np
import os

def get_params_from_path(path):
    # params['Loss'] = path[4]
    # params['num_layers'] = path[5]
    params = {}
    params['metric'] = path[6]
    params['ic-metric'] = path[7]
    params['lam'] = path[8]
    # params['lr'] = path[-1]
    if 'lam2' in path[-1]:
        params['lam2'] = path[-1]
    elif 'lam2' in path[-2]:
        params['lam2'] = path[-2]

    return params
def traverse_subfolders(source_dir, exclude = None):
    best_params_kl = {}
    best_params_nll = {}
    best_params_mse = {}
    best_kl = np.infty
    best_nll = np.infty
    best_mse = np.infty
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            results_dir = os.path.join(subfolder_path, "results")
            if not exclude in subfolder_path:
                if os.path.isdir(results_dir):
                    results_csv = os.path.join(results_dir, "results.csv")
                    if os.path.isfile(results_csv):
                        df = pd.read_csv(results_csv)
                        try:
                            kl = df['KL2'].mean()
                        except:
                            kl = df['KL'].mean()
                        nll_diff = np.mean(np.abs(df['NLL_mcmc']-df['NLL_diffusion']))
                        try:
                            mse = df['MSE'].mean()
                        except:
                            mse = np.infty
                        path = subfolder_path.split('/')
                        if (kl < best_kl):
                            best_params_kl = get_params_from_path(path)
                            best_kl = kl
                        if nll_diff < best_nll:
                            best_params_nll = get_params_from_path(path)
                            best_nll = nll_diff
                        if mse < best_mse:
                            best_params_mse = get_params_from_path(path)
                            best_mse = mse
    return best_params_kl,best_params_nll,best_params_mse, best_kl,best_nll,best_mse

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
source_directory = 'examples/scatterometry/results/FPE/PINNLoss4/3layer'
params_kl,params_nll,params_mse,kl,nll,mse = traverse_subfolders(source_directory)
#path1 = 'examples/scatterometry/dsm-loss'
#path2 = 'examples/scatterometry/scatterometry_runs_new/CFM/ErmonLoss/4layer/L2/lam:0.1/lr:0.0001/results'
#params,score = compare_2_methods(path1,path2)
print('Best KL: ', kl)
print(params_kl)
print('-------------------')
print('Best NLL: ', nll)
print(params_nll)
print('-------------------')
print('Best MSE: ', mse)
print(params_mse)
print('-------------------')