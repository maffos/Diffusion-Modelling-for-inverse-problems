"""
This script is designed for evaluating and selecting the best model parameters from multiple experiments stored in a directory structure.

1. Extracting Parameters:
   - `get_params_from_path_dsm_pde_loss` and `get_params_from_path` functions extract relevant parameters from the directory path.

2. Traversing Subfolders:
   - `traverse_subfolders` function iteratively goes through each subdirectory within a specified source directory. It is designed to find the best model parameters based on various metrics like KL divergence, NLL (Negative Log-Likelihood), and MSE (Mean Squared Error). This function skips specified directories and considers different calculation methods for NLL based on whether the model is linear.

3. Main Execution:
   - The script uses command line arguments to receive the source directory and any exclusion criteria for directories.
   - It then applies the `traverse_subfolders` function to find the best parameters across all subdirectories, based on the specified metrics.
   - The results, including the best metrics and their corresponding parameters, are outputted to the console.
"""

import pandas as pd
import numpy as np
import os
import argparse

def get_params_from_path_dsm_pde_loss(path):
    params = {}
    params['metric'] = path[-3]
    params['lam'] = path[-2]
    return params

def get_params_from_path(path):
    params = {}
    params['metric'] = path[-4]
    params['ic-metric'] = path[-3]
    params['lam'] = path[-2]
    params['lam2'] = path[-1]



    return params

def traverse_subfolders(source_dir, exclude = [], linear = False, result_key = 'results'):
    best_params_kl = {}
    best_params_nll = {}
    best_params_mse = {}
    best_params_kl_reverse = {}
    best_kl = np.infty
    best_nll = np.infty
    best_mse = np.infty
    best_kl_reverse = np.infty
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            if all(x not in subfolder_path for x in exclude) and result_key in subfolder_path:
                results_csv = os.path.join(subfolder_path, "results.csv")
                if os.path.isfile(results_csv):
                    df = pd.read_csv(results_csv)
                    try:
                        kl = df['KL2'].mean()
                    except:
                        kl = df['KL'].mean()
                    try:
                        kl_reverse = df['KL_reverse'].mean()
                    except:
                        kl_reverse = np.nan
                    if linear:
                        nll_diff = np.mean(np.abs(df['NLL_true']-df['NLL_diffusion']))
                    else:
                        nll_diff = np.mean(np.abs(df['NLL_mcmc']-df['NLL_diffusion']))
                    try:
                        mse = df['MSE'].mean()
                    except:
                        mse = np.infty
                    path = subfolder_path.split('/')
                    if (kl < best_kl):
                        if 'DSM_PDELoss' in path:
                            best_params_kl = get_params_from_path_dsm_pde_loss(path)
                        else:
                            best_params_kl = get_params_from_path(path)
                        best_kl = kl
                    if (kl_reverse < best_kl_reverse):
                        if 'DSM_PDELoss' in path:
                            best_params_kl_reverse = get_params_from_path_dsm_pde_loss(path)
                        else:
                            best_params_kl_reverse = get_params_from_path(path)
                        best_kl_reverse = kl_reverse
                    if nll_diff < best_nll:
                        if 'DSM_PDELoss' in path:
                            best_params_nll = get_params_from_path_dsm_pde_loss(path)
                        else:
                            best_params_nll = get_params_from_path(path)
                        best_nll = nll_diff
                    if mse < best_mse:
                        if 'DSM_PDELoss' in path:
                            best_params_mse = get_params_from_path_dsm_pde_loss(path)
                        else:
                            best_params_mse = get_params_from_path(path)
                        best_mse = mse

    return best_params_kl,best_params_kl_reverse, best_params_nll,best_params_mse, best_kl,best_kl_reverse,best_nll,best_mse

if  __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Load model parameters.")
    parser.add_argument('--src_dir', required=True, type=str,
                                            help='Source directory in which the best model in all subdirectories is searched for.')
    parser.add_argument('--exclude', required=False, default = [], type = list,
                                            help='Parameters that should be excluded.')

    args = parser.parse_args()

    if 'linearModel' in args.src_dir:
        linear = True

    params_kl,params_kl_reverse, params_nll,params_mse,kl,kl_reverse,nll,mse = traverse_subfolders(args.src_dir, args.exclude, result_key= 'results', linear = linear)

    print('---------------------------------')
    print('Best KL: ', kl)
    print(params_kl)
    print('---------------------------------')
    print('Best KL reverse: ', kl_reverse)
    print(params_kl_reverse)
    print('-------------------')
    print('Best NLL: ', nll)
    print(params_nll)
    print('-------------------')
    print('Best MSE: ', mse)
    print(params_mse)
    print('-------------------')