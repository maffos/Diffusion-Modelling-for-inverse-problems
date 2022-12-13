import torch
import numpy as np

def gaussian_likelihood(y_obs, y_true, sigma):
    """Returns gaussian log-likelihood"""
    return (-0.5* (y_obs-y_true)**2/sigma**2).sum(axis=1)

def MLLoss(z, log_det_J, latent_prior):
    log_p_z = latent_prior.log_prob(z)
    return (-log_p_z+log_det_J).mean()

def ForwardBackwardKLLoss(x,z,jac_inv, log_det_J, y_true, y_noise, latent_prior, likelihood_fn, conv_lambda, sigma):
    relu = torch.nn.ReLU()
    forward_kl = MLLoss(z,-jac_inv, latent_prior)
    backward_kl = (-likelihood_fn(y_noise, y_true, sigma) - log_det_J).mean()
    loss_relu = 100 * torch.sum(relu(x - 1) + relu(-x))
    return forward_kl * (1 - conv_lambda) + (backward_kl + loss_relu) * conv_lambda