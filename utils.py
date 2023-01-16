import numpy as np
import torch
import torch.utils.data
import os
import itertools
import matplotlib.pyplot as plt
from torch import nn

__all__ = ['load_dataset', 'generate_dataset']

def get_dataset_names():
    return ['gaussian_ring']
    
def load_gaussian_ring_dataset(labels, tot_dataset_size):

    verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

    label_maps = {
              'all':  [0, 1, 2, 3, 4, 5, 6, 7],
              'some': [0, 0, 0, 0, 1, 1, 2, 3],
              'none': [0, 0, 0, 0, 0, 0, 0, 0],
             }


    # print('Generating artifical data for setup "%s"' % (labels))

    np.random.seed(0)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.2)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.

    shuffling = np.random.permutation(N)
    pos = torch.tensor(pos[shuffling], dtype=torch.float)
    labels = torch.tensor(labels[shuffling], dtype=torch.float)

    return pos, labels
    
def generate_dataset(name, **kwargs):
    if name not in get_dataset_names():
        raise ValueError(f"'{name}' is not one of the example datasets.")
        
    elif name == 'gaussian_ring':
        return load_gaussian_ring_dataset(**kwargs)
    
    else:
    
        raise ValueError('This should not be displayed, update the list of known Datasets.')
          
def load_dataset(filename):
    
    try:
        data = np.load(filename, allow_pickle=True)["data"].item()
    except:
        raise ValueError('You need to specify an existing dataset as filename.')
        
    x_labels = data['parameters']
    xs = data['x_train']

    #drop the age column
    xs = xs[:,1:]
    x_labels = x_labels[1:]
        
    # normalize x
    xs = (xs - xs.min(axis=0)) / (xs.max(axis=0) - xs.min(axis=0))
    xs = torch.from_numpy(xs).float()
    xs.requires_grad = True

    #y is already normalized. But throw away first entry as it is always 0
    ys = torch.from_numpy(data['y_train'][:,1:]).float()

    return xs,ys,x_labels

def get_x_minmax(filename):
    data = np.load(filename, allow_pickle=True)["data"].item()
    x = data['x_train'][:,1:]
    return x.min(axis=0),x.max(axis=0)
def reverse_preprocessing(x, filename):

    x_min, x_max = get_x_minmax(filename)
    return x*(x_max-x_min)+x_min
def get_dataloader(x, y, batch_size):
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]
    def data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    return data_loader
    
def get_dataloader_noise(x_train, y_train, sigma):
    perm = torch.randperm(len(x_train))
    x = x_train[perm]
    y = y_train[perm]
    y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size], y_train[i:i+batch_size]

    return epoch_data_loader

#code was taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def make_image(pred_samples,x_true, num_epochs, inds=None, show_plot = False, savefig = True):

    cmap = plt.cm.tab20
    range_param = 1.2
    if inds is None:
        no_params = pred_samples.shape[1]
        inds=range(no_params)
    else:
        no_params=len(inds)
    fig, axes = plt.subplots(figsize=[12,12], nrows=no_params, ncols=no_params, gridspec_kw={'wspace':0., 'hspace':0.});
    fig.suptitle('Epochs=%d'%num_epochs)
    for j, ij in enumerate(inds):
        for k, ik in enumerate(inds):
            axes[j,k].get_xaxis().set_ticks([])
            axes[j,k].get_yaxis().set_ticks([])
            # if k == 0: axes[j,k].set_ylabel(j)
            # if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), alpha=0.3, range=(-0,range_param))
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), histtype="step", range=(-0,range_param))
                axes[j,k].axvline(x_true[:,j])

            else:
                val, x, y = np.histogram2d(pred_samples[:,ij], pred_samples[:,ik], bins=25, range = [[-0, range_param], [-0, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

    if savefig:
        plt.savefig(os.path.join(output_dir, 'posterior_epochs=%d.png'%num_epochs))
    if show_plot:
        plt.show()

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