import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import torch.utils.data
import os
import itertools
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#code was taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def rademacher_like(s):

    v = torch.distributions.bernoulli.Bernoulli(torch.ones_like(s)*.5).sample()
    v[torch.where(v==0)]=-1
    return v

#function copied from https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True, retain_graph=True)[0][..., i:i+1]
    return div

def batch_gradient(y,x):
    grad = torch.zeros_like(y)
    for i in range(y.shape[1]):
        dy_dx = torch.autograd.grad(y[:,i].sum(),x, retain_graph=True, create_graph=True)[0]
        dy_dx = dy_dx.view(-1)
        grad[:,i] += dy_dx
    return grad

def div_estimator(s,x,num_samples=1, rademacher = True):

    div = torch.zeros(s.shape[0],1)
    for _ in range(num_samples):
        if rademacher:
            v = rademacher_like(s)
        else:
            v = torch.randn_like(s)
        vjp = torch.autograd.grad(s,x,grad_outputs = v, create_graph=True, retain_graph=True)[0]
        div += (vjp[:,None,:]@v[:,:,None]).view(-1,1)

    div /= num_samples
    return div

def plot_density(samples, nbins, size, labelsize = 12, show = False, cmap = 'viridis', limits=None, fname = None,xticks = None, show_mode = False):
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
                if show_mode:
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
                    elif not show_mode:
                        xticks = [x_min, 0, x_max]
                    axes[i, j].set_xticks(xticks)
                if show_mode:
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


#code taken from https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super(GaussianFourierProjection, self).__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(torch.flatten(x_proj,start_dim= 1)), torch.cos(torch.flatten(x_proj, start_dim=1))], dim=-1)
