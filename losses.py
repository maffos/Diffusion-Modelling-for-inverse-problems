import torch
from torch import nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#function copied from https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True, retain_graph=True)[0][..., i:i+1]
    return div

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
class FftMseLoss(nn.Module):
    """
    loss function in Fourier space
    June 2022, F.Alesiani
    """
    def __init__(self):
        super(FftMseLoss, self).__init__()
    def forward(self, y_true, y_pred, flow=None,fhigh=None):
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

class MultipleLoss(nn.Module):

    def __init__(self, loss1, loss2):
        super(MultipleLoss, self).__init__()
        self.loss1_fn = loss1
        self.loss2_fn = loss2

    """
    So far only the MSE is considered as first loss. Probably we will not have to consider other types of loss functions.
    Therefore params can only be passed to the second loss function. So far it is not planned to integrate passing
    params to the first loss,too. But it might become necessary in the future.
    """
    def forward(self, y_true, y_predict, lmbd, **params):

        loss = (1-lmbd)*self.loss1_fn(y_true,y_predict) + lmbd * self.loss2_fn(y_true, y_predict, **params)
        return loss


class DSMLoss(nn.Module):
    
    def __init__(self,xdim):

        super(DSMLoss,self).__init__()
        self.xdim = xdim
        self.name = 'DSMLoss'
                 
    def forward(self,s,std,target):

        return ((s * std + target) ** 2).view(self.xdim, -1).sum(1, keepdim=False) / 2

class ScoreFPELoss(nn.Module):

    def __init__(self):

        super(ScoreFPELoss,self).__init__()

    def forward(self, s, x_t, t, beta):
        batch_size = x_t.shape[0]

        s_t = torch.autograd.grad(s,x_t,grad_outputs=torch.ones_like(s), create_graph = True, retain_graph=True)[0]
        divx_s =divergence(s,x_t)
        loss = torch.autograd.grad(divx_s.sum() + torch.sum(s ** 2), x_t, retain_graph=True)[0]
        loss = torch.sum((s_t - .5 * beta * (s + loss)) ** 2, dim=1).view(batch_size, 1)

        return loss

class ErmonLoss(nn.Module):

    def __init__(self, xdim, lam = 1.):

        super(ErmonLoss,self).__init__()
        self.lam = lam
        self.dsm_loss = DSMLoss(xdim)
        self.pde_loss = ScoreFPELoss()
        self.name = 'ErmonLoss'

    def forward(self,model,x,y,t):

        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        s = model.a(x_t, t,y)/g
        beta = model.base_sde.beta(t)

        loss = self.dsm_loss(s,std,target)+self.lam*self.pde_loss(s,x_t,t,beta)
        return loss.mean()

#calculates PINN loss without x-collocations term, e.g. no dsm loss included
class PINNLoss(nn.Module):

    def __init__(self, initial_condition, boundary_condition, lam1 = 1., lam2 = 1., lam3=1.):

        super(PINNLoss,self).__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        self.pde_loss = ScoreFPELoss()
        self.name = 'PinnLoss'

    def forward(self, model,x,y,t):

        batch_size, xdim = x.shape
        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        x_t = x_t.to(x.device)
        t0 = torch.zeros_like(t).to(x.device)
        T = torch.ones_like(t).to(x.device)
        g_0 = model.base_sde.g(t0, x_t)
        x_T, target_T, std_T, g_T = model.base_sde.sample(T, x, return_noise=True)
        x_T = x_T.to(x.device)
        beta = model.base_sde.beta(t)

        s_0 = model.a(x, t0, y) / g_0
        s = model.a(x_t, t, y) / g
        s_T = model.a(x_T, T, y)/g_T

        MSE_u = self.lam2 * self.initial_condition(s_0,x,y) + self.lam3 * self.boundary_condition(s_T, x_T)
        loss = torch.mean(MSE_u + self.lam1 * self.pde_loss(s,x_t,t,beta))
        
        return loss
    
#the difference between this and ermon_loss is that ermon_loss does not include the initial and boundary condition.
#doesn't work yet
class PINNLoss2(PINNLoss):

    def __init__(self,initial_condition,boundary_condition, xdim, lam1 = 1., lam2 = 1., lam3 = 1.):

        super(PINNLoss2, self).__init__(initial_condition,boundary_condition, lam1, lam2, lam3)
        self.dsm_loss = DSMLoss(xdim)
        self.pde_loss = ScoreFPELoss()
        self.name = 'PINNLoss2'

    def forward(self, model,x,y,t):
        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        t0 = torch.zeros_like(t)
        T = torch.ones_like(t)
        g_0 = model.base_sde.g(t0, x_t)
        x_T, target_T, std_T, g_T = model.base_sde.sample(T, x, return_noise=True)
        beta = model.base_sde.beta(t)
    
        s_0 = model.a(x, t0, y)/g_0
        s_T = model.a(x_T, T, y)/g_T
        s = model.a(x_t, t,y)/g

        MSE_u = self.lam2*self.initial_condition(s_0,x,y) + self.lam3*self.boundary_condition(s_T,x_T) + self.dsm_loss(s,std,target)
        loss = torch.mean(MSE_u+self.lam1*self.pde_loss(s,x_t,t,beta))

        return loss

#calculates PINN loss without boundary condition term
class PINNLoss3(ErmonLoss):

    def __init__(self, xdim, initial_condition, lam1 = 1., lam2 = 1.):

        super(PINNLoss3, self).__init__(xdim, lam1)
        self.lam2 = lam2
        self.initial_condition = initial_condition
        self.name = 'PINNLoss3'

    def forward(self,model,x,y,t):

        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        t0 = torch.zeros_like(t)
        g_0 = model.base_sde.g(t0, x_t)
        beta = model.base_sde.beta(t)

        s_0 = model.a(x, t0, y)/g_0
        s = model.a(x_t, t,y)/g

        MSE_u = self.lam2*self.initial_condiion(s_0,x,y) + self.dsm_loss(s,std,target)
        loss = torch.mean(MSE_u+self.lam*self.pde_loss(s,x_t,t,beta))

        return loss