import matplotlib.pyplot as plt
from models.diffusion import CDE,CDiffE,PosteriorDiffusionEstimator
from losses import *
import seaborn as sns
import numpy as np
import pandas as pd
import os
import shutil
import itertools
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#code was taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_model_from_args(config, forward_model_params, score_posterior, forward_model):

    if config['model'] == 'CDE':
        model = CDE(forward_model_params['xdim'],forward_model_params['ydim'],config['hidden_layers'])
    elif config['model'] == 'CDiffE':
        model = CDiffE(forward_model_params['xdim'],forward_model_params['ydim'],config['hidden_layers'])
    elif config['model'] == 'Posterior':
        model = PosteriorDiffusionEstimator(forward_model_params['xdim'],forward_model_params['ydim'],config['hidden_layers'])
    else:
        raise ValueError('No valid value for "model" passed. Has to be one of "CDE", "CDiffE" or "Posterior".')


    if config['loss_fn'] == 'PINNLoss':
        loss_fn = PINNLoss(score_posterior, lam = config['lam'], lam2 = config['lam2'], pde_loss = config['pde_loss'],
                       ic_metric = config['ic_metric'], pde_metric=config['pde_metric'])
    elif config['loss_fn'] == 'PINNLoss2':
        loss_fn = PINNLoss2(score_posterior, lam=config['lam'], pde_loss=config['pde_loss'], pde_metric=config['pde_metric'])
    elif config['loss_fn'] == 'DSM_PDE':
        loss_fn = DSM_PDELoss(lam = config['lam'], pde_loss = config['pde_loss'], pde_metric=config['pde_metric'])
    elif config['loss_fn'] == 'DSM':
        loss_fn = DSMLoss()
    elif config['model'] == 'Posterior':
        loss_fn = model.loss_fn(forward_model,forward_model_params['a'],forward_model_params['b'], lam=config['lam'])
    else:
        raise ValueError('No valid loss_fn was specified. Options are: "PINNLoss","PINNLoss2","DSM" or "DSM_PDE".'
                         'When the model is PosteriorDiffusionEstimator, the PosteriorLoss is used as default.')
    return model,loss_fn

def set_directories(train_dir, out_dir,resume_training = False):

    if os.path.exists(out_dir) and not resume_training:
        shutil.rmtree(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_dir = os.path.join(train_dir, 'logs')

    if os.path.exists(log_dir) and not resume_training:
        shutil.rmtree(log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir

"""
def diffusion_parser(parser):
    # Add arguments
    parser.add_argument('--train_dir', required=False, default = 'test', type=str,
                        help='Directory where checkpoints and logs are saved during training.')
    parser.add_argument('--out_dir', required=False, default = 'test', type=str, help='Directory to save output results.')
    parser.add_argument('--model', required=False, type=str, default='CDE',
                        help='Type of model to be used. Currently supported are "CDE","CDiffE" and "Posterior"')
    parser.add_argument('--loss_fn', required=False, type=str, default='PINNLoss',
                        help='Loss function to use for training. Valid options are "PINNLoss", "PINNLoss2", "DSM" and "DSM_PDE".')
    parser.add_argument('--pde_loss', required=False, default='FPE', type=str,
                        help='Loss enforcing the underlying PDE of the score. Can be either "FPE" or "cScoreFPE.')
    parser.add_argument('--lam', required=False, default=0.001, type=float,
                        help='Regularization parameter lambda controlling the PDE term.')
    parser.add_argument('--lam2', required=False, default=0.01, type=float,
                        help='Second regularization parameter lambda controlling the initial condition term.')
    parser.add_argument('--pde_metric', required=False, default='L1',
                        help='Regularization metric to use for the pde term. Either "L1" or "L2".')
    parser.add_argument('--ic_metric', required=False, default='L2',
                        help='Regularization metric to use for the initial condition. Either "L1" or "L2".')

    # Parse the arguments
    args = parser.parse_args()

    return args

"""

def check_wd(required_dir_name):

    # Get the current working directory
    current_path = os.getcwd()

    # Check if 'main' is the last part of the current path
    if not current_path.endswith(required_dir_name):
        raise ValueError(
            f"The script must be executed from the 'main' directory of the project, current path is '{current_path}'.")

def plot_density(samples, nbins, size, labelsize = 12, show = False, cmap = 'viridis', limits=None, fname = None,xticks = None, show_mean = False):
    """
    Plot the density of the samples in a grid.
    Parameters:
    - samples: A numpy array of shape (n_samples, n_dimensions).
    - limit: A tuple defining the lower and upper limit for the histogram bin range.
    """
    n_samples, n_dims = samples.shape
    fig, axes = plt.subplots(n_dims, n_dims, figsize=size)
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                # 1D histogram on the diagonal
                if limits:
                    bins = np.linspace(limits[0], limits[1], nbins)
                else:
                    bins = np.linspace(np.min(samples[:, i]), np.max(samples[:, i]), nbins)

                # Calculate histogram data
                hist, edges = np.histogram(samples[:, i], bins=bins)

                # Plot the histogram
                axes[i, j].step(edges[:-1], hist, where='mid', color='steelblue', linewidth=2)
                axes[i, j].set_xlim(bins[0], bins[-1])
                axes[i, j].set_ylabel('')
                axes[i, j].set_xlabel('dim%d' % i, size=labelsize)

                # Draw a dashed line at the mode
                if show_mean:
                    # Find the mode (bin with maximum count)
                    mode_index = np.argmax(hist)
                    mode_value = (edges[mode_index] + edges[mode_index + 1]) / 2
                    # Calculate bin centers
                    bin_centers = (edges[:-1] + edges[1:]) / 2
                    # Calculate weighted mean (considering each bin's count)
                    weighted_mean = np.sum(hist * bin_centers) / np.sum(hist)
                    axes[i, j].axvline(x=mode_value, color='lightsteelblue', linestyle='--', linewidth = 2)

                # Remove y-axis labels
                axes[i, j].set_yticklabels([])

                # Set x-ticks
                if xticks is None:
                    x_min = .5*(edges[0]+edges[1])
                    x_max = .5*(edges[-2]+edges[-1])
                    if x_max < 0:
                        xticks = [x_min, x_max]
                    elif not show_mean:
                        xticks = [x_min, 0, x_max]
                    axes[i, j].set_xticks(xticks)
                if show_mean:
                    xticks = [xticks[0], weighted_mean, xticks[-1]]
                    xticklabels = [xticks[0], np.round(weighted_mean,1), xticks[-1]]
                else:
                    xticklabels = xticks

                axes[i,j].set_xticks(xticks)
                axes[i,j].set_xticklabels(xticklabels,size =labelsize)
                axes[i,j].set_yticks([])

                sns.despine(left=True,top=True,right=True)
            elif i < j:
                # 2D histogram off-diagonal
                if limits:
                    hist_range = [limits, limits]
                else:
                    hist_range = [(np.min(samples[:, j]), np.max(samples[:, j])),
                                  (np.min(samples[:, i]), np.max(samples[:, i]))]

                H, xedges, yedges = np.histogram2d(samples[:, j], samples[:, i], bins=nbins, range=hist_range)
                axes[i, j].imshow(H.T, origin='lower', aspect='auto', interpolation='nearest',
                                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                  cmap=cmap)
                axes[i, j].set_xlim(hist_range[0])
                axes[i, j].set_ylim(hist_range[1])
                sns.despine(right=True, top=True, bottom=True, left = True)

                axes[i, j].set_xticklabels([])
                axes[i, j].set_xticks([])
                axes[i,j].set_yticklabels([])
                axes[i,j].set_yticks([])
            else:
                # For the lower triangular plots, we make them blank
                axes[i, j].axis('off')
    #plt.tight_layout()
    if fname:
        plt.savefig(fname)
    if show:
        plt.show()
    else:
        plt.close()

def plot_csv(file_path, fname, labelsize, max_step = 1000, show_plot = True):
    """
    Reads a CSV file and plots 'step' on the x-axis and 'value' on the y-axis.

    Parameters:
    - file_path (str): The path to the CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if 'step' and 'value' columns exist
    if 'Step' not in df.columns or 'Value' not in df.columns:
        raise ValueError("Columns 'step' and 'value' must be in the CSV.")

    # Filter the DataFrame based on the step_limit
    df_limited = df[df['Step'] <= max_step]

    # Plotting
    plt.plot(df_limited['Step'], df_limited['Value'])

    # Plotting

    # Labeling axes
    plt.xlabel('Step', size = labelsize)
    plt.ylabel('Value', size = labelsize)

    # Title and grid
    plt.grid(True)

    # Display the plot
    plt.savefig(fname)

    if show_plot:
        plt.show()

    plt.close()

