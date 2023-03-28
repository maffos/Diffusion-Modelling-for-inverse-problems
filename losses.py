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

class DFMLoss(nn.Module):

    def __init__(self):
        super(DFMLoss, self).__init__()
        self.name = 'DFMLoss'

    def forward(self, model,sde,x,t,y):

        x_t, target, std, g = sde.sample(1-t, x, return_noise=True)
        v = model(x_t,t,y)
        alpha = sde.mean_weight(1-t)
        beta = g**2
        #u=-.5*beta*alpha*(alpha*x_t-x)/(2*std**2) #with x_t
        u = -.5*beta*alpha*(alpha*target-std*x)/std

        loss = torch.sum((u-v)**2, dim=1)

        if torch.any(torch.isnan(loss)):
            print('Std:',std)
            print('beta:',beta)
            print('alpha:',alpha)
            print('u:',u)
            print('v:',v)
            raise ValueError('loss is nan')
        return loss
       
class DSMLoss(nn.Module):
    
    def __init__(self):

        super(DSMLoss,self).__init__()
        self.name = 'DSMLoss'
                 
    def forward(self,s, std,target):

        batch_size = s.shape[0]
        return ((s * std + target) ** 2).view(batch_size, -1).sum(1, keepdim=False) / 2

class CFMLoss(nn.Module):

    def __init__(self):
        super(CFMLoss, self).__init__()
        self.name = 'CFMLoss'

    def forward(self, model,s,t,beta,target,std):
        v = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(s), create_graph=True, retain_graph=True)[0]
        # u=.5*beta*mu*(target*std/(1-mu**2)-1) #erste version
        # u= .5*target*beta*model.base_sde.mean_weight(t)**2 #sigma^5
        u = .5 * target * beta * model.base_sde.mean_weight(t) ** 2
        loss = torch.sum((u - std ** 4 * v) ** 2, dim=1)

        return loss
class ScoreFlowMatchingLoss(nn.Module):

    def __init__(self, lam = 1.):

        super(ScoreFlowMatchingLoss, self).__init__()
        self.name = 'ScoreFlowMatching'
        self.dsm_loss = DSMLoss()
        self.lam = lam

    def forward(self,model,x,t,y):

        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        t0 = torch.zeros_like(t)
        T = torch.ones_like(t)
        g_0 = model.base_sde.g(t0, x_t)
        x_T, target_T, std_T, g_T = model.base_sde.sample(T, x, return_noise=True)
        beta = model.base_sde.beta(t)

        s = model.a(x_t, t, y) / g
        v = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(s), create_graph=True, retain_graph=True)[0]
        #mu = model.base_sde.mean_weight(t)*x
        #u=.5*beta*mu*(target*std/(1-mu**2)-1) #erste version
        #u= .5*target*beta*model.base_sde.mean_weight(t)**2 #sigma^5
        u = .5*target*beta*model.base_sde.mean_weight(t)**2/std
        dsm_loss = self.dsm_loss(s,std,target)
        cfm_loss = self.lam*torch.sum((u-std**3*v)**2,dim=1)
        loss = dsm_loss+cfm_loss
        return loss.mean(), {'DSM-Loss':dsm_loss.mean(),'CFM-loss':cfm_loss.mean()}
class ScoreFPELoss(nn.Module):

    def __init__(self):

        super(ScoreFPELoss,self).__init__()
        self.name = 'FPELoss'

    def forward(self, s, x_t, t, beta, metric = 'L1'):
        batch_size = x_t.shape[0]

        s_t = torch.autograd.grad(s,t,grad_outputs=torch.ones_like(s), create_graph = True, retain_graph=True)[0]
        #logger.add_scalar('Train/ddt_s', s_t.mean(), i)
        divx_s = divergence(s,x_t)
        #logger.add_scalar('Train/Divergence-Term', divx_s.mean(), i)
        #sq_norm = s ** 2
        #logger.add_scalar('Train/square_norm', sq_norm.mean(),i)
        #scalar_product = (s[:,None,:]@x_t[:,:,None]).view(-1,1)
        #logger.add_scalar('Train/scalar_product', scalar_product.mean(),i)
        loss = torch.autograd.grad(divx_s.sum() + torch.sum(s ** 2, dim=1).sum() + (s[:, None, :] @ x_t[:, :, None]).sum(),
                                   x_t, retain_graph=True)[0]
        #loss = torch.autograd.grad(divx_s + s ** 2 + (s[:, None, :] @ x_t[:, :, None]).view(-1,1),
        #                           x_t, grad_outputs=torch.ones_like(s), retain_graph=True)[0]
        #logger.add_scalar('Train/Autodiff', loss.mean(), i)
        if metric == 'L1':
            loss = torch.mean(torch.abs(s_t - .5 * beta * loss), dim=1).view(batch_size, 1)
        elif metric == 'L2':
            loss = torch.sqrt(torch.sum((s_t-0.5*beta*loss)**2, dim = 1)).view(batch_size,1)
        else:
            loss = torch.mean((s_t - .5 * beta * loss) ** 2, dim=1).view(batch_size, 1)
        return loss

#Loss resulting from the HJB PDE. Only difference to the Score FPE is the minus sign of divergence term
class ScoreHJBLoss(nn.Module):

    def __init__(self):

        super(ScoreHJBLoss,self).__init__()
        self.name = 'HJBLoss'

    def forward(self, s, x_t, t, beta, metric = 'L1'):
        batch_size = x_t.shape[0]

        s_t = torch.autograd.grad(s,t,grad_outputs=torch.ones_like(s), create_graph = True, retain_graph=True)[0]
        divx_s = divergence(s,x_t)
        loss = torch.autograd.grad(-divx_s + s ** 2 + (s[:,None,:]@x_t[:,:,None]).view(-1,1),
                                   x_t,grad_outputs=torch.ones_like(s),retain_graph=True)[0]
        if metric == 'L1':
            loss = torch.mean(torch.abs(s_t - .5 * beta * loss), dim=1).view(batch_size, 1)
        elif metric == 'L2':
            loss = torch.sqrt(torch.sum((s_t-0.5*beta*loss)**2, dim = 1)).view(batch_size,1)
        else:
            loss = torch.mean((s_t - .5 * beta * loss) ** 2, dim=1).view(batch_size, 1)
        return loss

#Loss as used by Lai, Chieh-Hsin, et al. "Regularizing score-based models with score fokker-planck equations." 
class ErmonLoss(nn.Module):

    def __init__(self, lam=1., pde_loss='HJB'):

        super(ErmonLoss,self).__init__()
        self.lam = lam
        self.dsm_loss = DSMLoss()
        if pde_loss == 'HJB':
            self.pde_loss = ScoreHJBLoss()
        elif pde_loss == 'FPE':
            self.pde_loss = ScoreFPELoss()
        else:
            self.pde_loss = CFMLoss()
        self.name = 'ErmonLoss'

    def forward(self,model,x,t,y):

        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        s = model.a(x_t, t,y)/g
        beta = model.base_sde.beta(t)

        MSE_u = self.dsm_loss(s,std,target)
        if self.pde_loss.name == 'CFMLoss':
            MSE_pde = self.lam * self.pde_loss(model,s,t,beta,target,std)
        else:
            MSE_pde = self.lam*self.pde_loss(s,x_t,t,beta)
        loss = MSE_u+MSE_pde
        return loss.mean(), {'DSM-Loss': MSE_u.mean(), 'PDE-Loss': MSE_pde.mean()}


class PINNLoss(nn.Module):

    def __init__(self, initial_condition, boundary_condition, lam = 1., lam2 = 1., lam3=1., pde_loss = 'HJB'):

        super(PINNLoss,self).__init__()
        self.lam = lam
        self.lam2 = lam2
        self.lam3 = lam3
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        if pde_loss == 'HJB':
            self.pde_loss = ScoreHJBLoss()
        elif pde_loss == 'FPE':
            self.pde_loss = ScoreFPELoss()
        else:
            self.pde_loss = CFMLoss()
        self.name = 'PinnLoss'

    def forward(self, model,x,t,y):

        batch_size = x.shape[0]
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

        initial_condition_loss = self.lam2*torch.mean((s_0-self.initial_condition(x,y))**2, dim=1).view(batch_size,1)
        boundary_condition_loss = self.lam3*torch.mean((s_T-self.boundary_condition(x_T))**2, dim=1).view(batch_size,1)
        MSE_u = initial_condition_loss+boundary_condition_loss
        if self.pde_loss.name == 'CFMLoss':
            MSE_pde = self.lam * self.pde_loss(model,s,t,beta,target,std)
        else:
            MSE_pde = self.lam*self.pde_loss(s,x_t,t,beta)
        loss = torch.mean(MSE_u + MSE_pde)
        
        return loss, {'PDE-Loss':MSE_pde.mean(), 'Initial Condition':initial_condition_loss.mean(), 'Boundary Condition':boundary_condition_loss.mean()}
    
#PINNLoss+DSMLoss
class PINNLoss2(PINNLoss):

    def __init__(self,initial_condition,boundary_condition, lam = 1., lam2 = 1., lam3 = 1., pde_loss = 'HJB'):

        super(PINNLoss2, self).__init__(initial_condition,boundary_condition, lam, lam2, lam3)
        self.dsm_loss = DSMLoss()
        if pde_loss == 'HJB':
            self.pde_loss = ScoreHJBLoss()
        elif pde_loss == 'FPE':
            self.pde_loss = ScoreFPELoss()
        else:
            self.pde_loss = CFMLoss()
        self.name = 'PINNLoss2'

    def forward(self, model,x,t,y):

        batch_size= x.shape[0]
        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        t0 = torch.zeros_like(t)
        T = torch.ones_like(t)
        g_0 = model.base_sde.g(t0, x_t)
        x_T, target_T, std_T, g_T = model.base_sde.sample(T, x, return_noise=True)
        beta = model.base_sde.beta(t)
    
        s_0 = model.a(x, t0, y)/g_0
        s_T = model.a(x_T, T, y)/g_T
        s = model.a(x_t, t,y)/g

        initial_condition_loss = self.lam2 * torch.mean((s_0 - self.initial_condition(x, y)) ** 2, dim=1).view(
            batch_size, 1)
        boundary_condition_loss = self.lam3 * torch.mean((s_T - self.boundary_condition(x_T)) ** 2, dim=1).view(
            batch_size, 1)
        dsm_loss = self.dsm_loss(s, std, target)
        MSE_u = initial_condition_loss + boundary_condition_loss+dsm_loss
        if self.pde_loss.name == 'CFMLoss':
            MSE_pde = self.lam * self.pde_loss(model,s,t,beta,target,std)
        else:
            MSE_pde = self.lam*self.pde_loss(s,x_t,t,beta)
        loss = torch.mean(MSE_u + MSE_pde)

        return loss, {'PDE-Loss': MSE_pde.mean(), 'Initial Condition': initial_condition_loss.mean(),
                      'Boundary Condition': boundary_condition_loss.mean(), 'DSM-Loss': dsm_loss.mean()}


#calculates PINN loss without boundary condition term
class PINNLoss3(nn.Module):

    def __init__(self,initial_condition, lam = 1., lam2 = 1., pde_loss = 'HJB'):

        super(PINNLoss3, self).__init__()
        self.lam = lam
        self.lam2 = lam2
        self.initial_condition = initial_condition
        if pde_loss == 'HJB':
            self.pde_loss = ScoreHJBLoss()
        elif pde_loss == 'FPE':
            self.pde_loss = ScoreFPELoss()
        else:
            self.pde_loss = CFMLoss()
        self.name = 'PINNLoss3'

    def forward(self,model,x,t,y):

        batch_size = x.shape[0]
        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        t0 = torch.zeros_like(t)
        g_0 = model.base_sde.g(t0, x_t)
        beta = model.base_sde.beta(t)

        s_0 = model.a(x, t0, y)/g_0
        s = model.a(x_t, t,y)/g

        initial_condition_loss = self.lam2 * torch.mean((s_0 - self.initial_condition(x, y)) ** 2, dim=1).view(
            batch_size, 1)
        #dsm_loss = self.dsm_loss(s, std, target)
        if self.pde_loss.name == 'CFMLoss':
            MSE_pde = self.lam * self.pde_loss(model,s,t,beta,target,std)
        else:
            MSE_pde = self.lam*self.pde_loss(s,x_t,t,beta)
        loss = torch.mean(initial_condition_loss + MSE_pde)

        return loss, {'PDE-Loss': MSE_pde.mean(), 'Initial Condition': initial_condition_loss.mean()}
