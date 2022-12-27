import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#todo: rename as metrics.py and write all Loss as classes
def norm_pdf(y_obs, y_true, sigma):
    """Returns gaussian log-likelihood"""
    return (-0.5* (y_obs-y_true)**2/sigma**2).sum(axis=1)

def mleLoss(z, log_det_J, latent_prior):
    log_p_z = latent_prior.log_prob(z)
    return torch.mean((-log_p_z+log_det_J))

def forwardBackwardKLLoss(x,z,jac_inv, log_det_J, y_true, y_noise, latent_prior, likelihood_pdf, conv_lambda, sigma):
    relu = torch.nn.ReLU()
    forward_kl = mleLoss(z,-jac_inv, latent_prior)
    backward_kl = torch.mean((-likelihood_pdf(y_noise, y_true, sigma) - log_det_J))
    loss_relu = 100 * torch.sum(relu(x - 1) + relu(-x))
    return forward_kl * (1 - conv_lambda) + (backward_kl + loss_relu) * conv_lambda

def mean_relative_error(y_true, y_predict, epsilon = 1e-8):
    #only add the epsilon in the denominator to avoid an infinite loss (the gradients can still explode however).
    return torch.mean((y_true-y_predict)**2/(torch.abs(y_true)+epsilon))

#code is taken from https://github.com/pdebench/PDEBench and modified according to Takamoto et al. (2022), to deal with ODEs
class FftMseLoss(object):
    """
    loss function in Fourier space
    June 2022, F.Alesiani
    """
    def __init__(self):
        super(FftMseLoss, self).__init__()
    def __call__(self, y_true, y_pred, flow=None,fhigh=None):
        yf_true = torch.fft.rfft(y_true,dim=1)
        yf_pred = torch.fft.rfft(y_pred,dim=1)
        if flow is None: flow = 0
        if fhigh is None: fhigh = np.max(yf_true.shape[1])

        #cut off unwanted frequencies
        yf_true = yf_true[:,flow:fhigh]
        yf_pred = yf_pred[:,flow:fhigh]

        #loss = torch.mean(torch.sqrt(torch.abs(yf_true-yf_pred)**2)).to(device)
        loss = torch.mean((yf_true-yf_pred).abs()**2)
        return loss