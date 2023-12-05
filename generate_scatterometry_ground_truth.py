"""
This script is designed for generating ground truth (GT) samples in the context of scatterometry, a metrological technique for material characterization. It involves the following key components:

Function `generate_gt_samples`: This function is responsible for generating GT samples and plotting their distribution.

Main Execution Flow:
   - Configuration Loading: The script loads configuration parameters from a YAML file, setting up the environment and simulation parameters for scatterometry.
   - Model Loading: It loads a surrogate forward model for scatterometry, essential for simulating the scatterometry process.
   - Test Dataset Generation: A test dataset is generated using the forward model, which includes parameters and corresponding measurements for scatterometry.
   - Parallel Sample Generation: The script iterates over the test dataset, repeatedly generating GT samples in parallel. This process is tracked using a progress bar for better visibility of the execution progress.

The generated GT samples are used for evaluating the models trained in the other scripts.
"""

from models.diffusion import *
from models.SNF import anneal_to_energy
from utils_scatterometry import *
import utils
from tqdm import tqdm
from datasets import generate_dataset_scatterometry
import yaml
from joblib import Parallel, delayed
import os
import torch

def generate_gt_samples(i,j):
    x_true = anneal_to_energy(torch.rand(config['n_samples_x'], forward_model_params['xdim'], device=device) * 2 - 1, mcmc_energy, config['METR_STEPS'],
                              noise_std=config['NOISE_STD_MCMC'])[0].detach().cpu().numpy()
    out_dir = os.path.join(gt_dir, str(i))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = os.path.join(out_dir,'%d.npy'%j)
    with open(filename, 'wb') as f:
        np.save(f, x_true)

    # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
    if i in config['plot_y'] and j == config['n_repeats']-1:
        utils.plot_density(x_true, nbins = 75, limits=(-1.2,1.2), xticks=[-1, 0, 1], size=(12,12), labelsize=30,
                           fname=os.path.join(out_dir, 'posterior-mcmc-%d.svg' % i))

if __name__ == '__main__':

    config_dir = 'config/'
    surrogate_dir = 'trained_models/scatterometry'
    gt_dir = 'data/gt_samples_scatterometry'
    n_repeats = 10 #number of repeats for which metrics are calculated in the evaluation

    # load config params
    config = yaml.safe_load(open(os.path.join(config_dir, "config_scatterometry.yml")))

    # load the forward model
    forward_model, forward_model_params = load_forward_model(surrogate_dir)

    # generate test set
    x_test, y_test = generate_dataset_scatterometry(forward_model, forward_model_params['a'], forward_model_params['b'],
                                                    size=config['n_samples_y'])

    prog_bar = tqdm(total=config['n_samples_y'])
    for i, y in enumerate(y_test):
        inflated_ys = y[None, :].repeat(config['n_samples_x'], 1)
        mcmc_energy = lambda x: get_log_posterior(x, forward_model, forward_model_params['a'], forward_model_params['b'], inflated_ys, forward_model_params['lambd_bd'])
        Parallel(n_jobs=4)(delayed(generate_gt_samples)(i,j) for j in range(n_repeats))
        prog_bar.update()



