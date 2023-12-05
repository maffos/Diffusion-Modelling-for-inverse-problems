import numpy as np
import torch
import os
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_forward_model(src_dir):
    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)
    forward_model.load_state_dict(
        torch.load(os.path.join(src_dir, 'surrogate.pt'), map_location=torch.device(device)))
    for param in forward_model.parameters():
        param.requires_grad = False

    params = {}
    params['a'] = 0.2
    params['b'] = 0.01
    params['lambd_bd'] = 1000
    params['xdim'] = 3
    params['ydim'] = 23

    return forward_model, params

# returns (negative) log_posterior evaluation for the scatterometry model
# likelihood is determined by the error model
# uniform prior is approximated via boundary loss for a.e. differentiability
def get_log_posterior(samples, forward_model, a, b, ys,lambd_bd):
    relu=torch.nn.ReLU()
    forward_samps=forward_model(samples)
    prefactor = ((a*forward_samps)**2+b**2)
    p = .5*torch.sum(torch.log(prefactor), dim = 1)
    p2 = 0.5*torch.sum((ys-forward_samps)**2/prefactor, dim = 1)
    p3 = lambd_bd*torch.sum(relu(samples-1)+relu(-1-samples), dim = 1)
    log_prob = p+p2+p3
    return log_prob


# returns samples from the boundary loss approximation prior
# lambd_bd controlling the strength of boundary loss
def inverse_cdf_prior(x,lambd_bd):
    x*=(2*lambd_bd+2)/lambd_bd
    y=np.zeros_like(x)
    left=x<1/lambd_bd
    y[left]=np.log(x[left]*lambd_bd)-1
    middle=np.logical_and(x>=1/lambd_bd,x < 2+1/lambd_bd)
    y[middle]=x[middle]-1/lambd_bd-1
    right=x>=2+1/lambd_bd
    y[right]=-np.log(((2+2/lambd_bd)-x[right])*lambd_bd)+1
    return y
