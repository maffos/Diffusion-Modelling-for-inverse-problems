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
        kl_snf_reverse = df['KL_SNF_reverse'].mean()
        kl_inn = df['KL_INN'].mean()
        kl_inn_reverse = df['KL_INN_reverse'].mean()
        kl_dsm = df['KL_diffusion'].mean()
        kl_dsm_reverse = df['KL_diffusion_reverse'].mean()
        nll_dsm = np.mean(np.abs(df['NLL_%s'%gt_name]-df['NLL_diffusion']))
        nll_inn = np.mean(np.abs(df['NLL_%s'%gt_name]-df['NLL_inn']))
        nll_snf = np.mean(np.abs(df['NLL_%s'%gt_name]-df['NLL_snf']))
        try:
            mse = df['MSE'].mean()
        except:
            mse = None

    print('KL INN: ', kl_inn)
    print('KL SNF: ', kl_snf)
    print('KL DSM: ', kl_dsm)

    print('Reveres KL INN: ', kl_inn_reverse)
    print('Reverse KL SNF: ', kl_snf_reverse)
    print('Reverse KL DSM: ', kl_dsm_reverse)

    print('NLL DSM: ', nll_dsm)
    print('NLL INN: ', nll_inn)
    print('NLL SNF: ', nll_snf)

    print('MSE: ', mse)

if __name__ == '__main__':
    dirname = 'examples/scatterometry/results/baselines2/results'
    #dirname = 'examples/linearModel/results/baselines/results'
    evaluate(dirname,linear=False)
    #evaluate_linear(dirname)