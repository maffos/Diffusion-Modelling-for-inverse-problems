from torch.utils.data import Dataset
from torch import nn
import torch
import os
import numpy as np
from utils_scatterometry import inverse_cdf_prior

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_dataset_scatterometry(forward_model,a,b,size=100):
    random_state = 13
    rand_gen = torch.manual_seed(random_state)
    xdim=3

    xs = torch.rand(size, xdim, generator= rand_gen).to(device) * 2 - 1
    ys = forward_model(xs)
    ys = ys + b * torch.randn(ys.shape, generator = rand_gen).to(device) + ys * a * torch.randn(ys.shape, generator = torch.manual_seed(random_state+1)).to(device)

    return xs, ys

def get_gt_samples_scatterometry(src_dir, y, i):
    filename = os.path.join(src_dir,str(y),'%d.npy'%i)
    with open(filename, 'rb') as f:
        x_true = np.load(f)

    return x_true

def get_dataloader_scatterometry(batch_size, forward_model,a, b,lambd_bd):
    x = torch.tensor(inverse_cdf_prior(np.random.uniform(size=(8*batch_size,3)),lambd_bd),dtype=torch.float,device=device)
    y = forward_model(x)
    y += torch.randn_like(y) * b + torch.randn_like(y)*a*y
    def epoch_data_loader():
        for i in range(0, 8*batch_size, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader


def generate_dataset_forward(xdim, f, n_samples, random_state=7):
    random_gen = torch.random.manual_seed(random_state)
    x = torch.randn(n_samples, xdim, generator=random_gen).to(device)
    y = f(x)
    return x, y


def get_dataloader_forward(x_train, y_train, sigma, batch_size):
    perm = torch.randperm(len(x_train))
    x = x_train[perm]
    y = y_train[perm]
    y += sigma * torch.randn_like(y)

    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)

    return epoch_data_loader