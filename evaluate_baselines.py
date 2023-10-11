import pandas as pd
import numpy as np
import os

def evaluate(result_dir, linear = False):

    if linear:
        gt_name = 'true'
    else:
        gt_name = 'mcmc'

    filename = os.path.join(result_dir, "results.csv")
    if os.path.isfile(filename):
        df = pd.read_csv(filename)

        kl_snf = df['KL_SNF'].mean()
        kl_inn = df['KL_INN'].mean()
        kl_dsm = df['KL_diffusion'].mean()
        nll_dsm = np.mean(np.abs(df['NLL_%s'%gt_name]-df['NLL_diffusion']))
        nll_inn = np.mean(np.abs(df['NLL_%s'%gt_name]-df['NLL_inn']))
        nll_snf = np.mean(np.abs(df['NLL_%s'%gt_name]-df['NLL_snf']))
        mse = df['MSE'].mean()

    print('KL INN: ', kl_inn)
    print('KL SNF: ', kl_snf)
    print('KL DSM: ', kl_dsm)

    print('NLL DSM: ', nll_dsm)
    print('NLL INN: ', nll_inn)
    print('NLL SNF: ', nll_snf)

    print('MSE: ', mse)

if __name__ == '__main__':
    #dirname = 'examples/scatterometry/results/baselines/results'
    dirname = 'examples/linearModel/results/baselines/results'
    evaluate(dirname,linear=True)
    #evaluate_linear(dirname)