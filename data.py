from torch.utils.data import Dataset
from torch import nn
import torch
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class PPGDataset(Dataset):

    def __init__(self, src_dir, location):
        self.src_dir = src_dir
        self.location = location
        self.filename = os.path.join(src_dir, location, '_PPG.npz')

        try:
            data = np.load(self.filename, allow_pickle=True)["data"].item()
        except:
            raise ValueError('You need to specify an existing dataset as filename.')

        x_labels = data['parameters']
        xs = data['x_train']

        #drop the age column
        xs = xs[:,1:]
        self.x_labels = x_labels[1:]

        # normalize x
        self.xs = (xs - xs.min(axis=0)) / (xs.max(axis=0) - xs.min(axis=0))

        #y is already normalized. But throw away first entry as it is always 0
        self.ys = data['y_train'][:,1:]

class ScatterometryDataset(Dataset):

    def __init__(self, a,b, lambd_bd, checkpoint_file, n_samples, random_state):

        rand_gen = torch.random.manual_seed(random_state)
        # load forward model
        self.surrogate = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                      nn.Linear(256, 256), nn.ReLU(),
                                      nn.Linear(256, 256), nn.ReLU(),
                                      nn.Linear(256, 23)).to(device)
        self.surrogate = self.surrogate.load_state_dict(torch.load('models_scatterometry/forward_model_new.pt'))
        for param in self.surrogate.parameters():
            param.requires_grad = False

        self.x = torch.tensor(self.inverse_cdf_prior(np.random.uniform(size=(8 * n_samples, 3), generator = rand_gen), lambd_bd),
                         dtype=torch.float)
        self.y = self.surrogate(self.x)

    # returns (negative) log_posterior evaluation for the scatterometry model
    # likelihood is determined by the error model
    # uniform prior is approximated via boundary loss for a.e. differentiability
    def get_log_posterior(self, samples, forward_model, a, b, ys, lambd_bd):
        relu = torch.nn.ReLU()
        forward_samps = forward_model(samples)
        prefactor = ((a * forward_samps) ** 2 + b ** 2)
        p = .5 * torch.sum(torch.log(prefactor), dim=1)
        p2 = 0.5 * torch.sum((ys - forward_samps) ** 2 / prefactor, dim=1)
        p3 = lambd_bd * torch.sum(relu(samples - 1) + relu(-1 - samples), dim=1)
        return p + p2 + p3

    # returns samples from the boundary loss approximation prior
    # lambd_bd controlling the strength of boundary loss
    def inverse_cdf_prior(self, x, lambd_bd):
        x *= (2 * lambd_bd + 2) / lambd_bd
        y = np.zeros_like(x)
        left = x < 1 / lambd_bd
        y[left] = np.log(x[left] * lambd_bd) - 1
        middle = np.logical_and(x >= 1 / lambd_bd, x < 2 + 1 / lambd_bd)
        y[middle] = x[middle] - 1 / lambd_bd - 1
        right = x >= 2 + 1 / lambd_bd
        y[right] = -np.log(((2 + 2 / lambd_bd) - x[right]) * lambd_bd) + 1
        return y
