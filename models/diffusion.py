from include.sdeflow_light.lib import sdes
from nets import TemporalMLP_small,MLP

import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_diffusion_model(xdim, ydim, embed_dim, hidden_layers):

    net_params = {'input_dim': xdim + ydim,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers,
                  'embed_dim': embed_dim}
    forward_process = sdes.VariancePreservingSDE()
    score_net = TemporalMLP_small(**net_params).to(device)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=True)
    return reverse_process

def train_diffusion_epoch(optimizer, loss_fn, model, epoch_data_loader, t_min):
    mean_loss = 0
    logger_info = {}

    for k, (x, y) in enumerate(epoch_data_loader()):

        t = sample_t(model,x)
        if loss_fn.name == 'DSMLoss':
            x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
            s = model.a(x_t, t, y) / g
            loss = loss_fn(s,std,target).mean()
        else:
            loss = loss_fn(model,x,t,y)
        if isinstance(loss, tuple):
            loss_info = loss[1]
            loss = loss[0]
            for key,value in loss_info.items():
                try:
                    logger_info[key] = logger_info[key] * k / (k + 1) + value.item() / (k + 1)
                except:
                    logger_info[key] = 0
                    logger_info[key] = logger_info[key] * k / (k + 1) + value.item() / (k + 1)

        if torch.min(t) < t_min:
            t_min = torch.min(t)
        if torch.isnan(loss):
            for key, value in loss_info.items():
                print(key + ':' + str(value))
            raise ValueError(
                'Loss is nan, min sampled t was %f. Minimal t during training was %f' % (torch.min(t), t_min))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss, logger_info, t_min

def create_diffusion_model2(xdim, ydim,hidden_layers):

    net_params = {'input_dim': xdim + ydim+1,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers,
                  'activation': nn.Tanh()}
    forward_process = sdes.VariancePreservingSDE()
    score_net = MLP(**net_params).to(device)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=True)
    return reverse_process

def get_grid(sde, cond1, xdim,ydim, num_samples = 2000, num_steps=200, transform=None,
             mean=0, std=1):
    cond = torch.zeros(num_samples, ydim).to(cond1.device)
    cond += cond1
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, xdim).to(cond1.device)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1) * sde.T
    ones = torch.ones(num_samples, 1)
    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu((ones * ts[i]).to(cond1.device), y0, cond)
            sigma = sde.sigma((ones * ts[i]).to(cond1.device), y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0).to(cond1.device)

    y0 = y0.data.cpu().numpy()
    return y0

def sample_t(model,x, eps = 1e-4):

    if model.debias:
        t_ = model.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])+eps
        t_[torch.where(t_>model.T)] -= eps
        t_.requires_grad = True
    else:
        #we cannot just uniformly sample when using the PINN-loss because the gradient explodes for t of order 1e-7
        t_ = eps+torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)], requires_grad=True).to(x) * model.T
        t_[torch.where(t_>model.T)] = model.T-eps
    return t_

"""
class SDE(nn.Module):

    def __init__(self, net, forward_sde, xdim, ydim, sde_type, noise_type='diagonal', T=1, t0=0):
        super(SDE, self).__init__()
        self.net = net
        self.forward_sde = forward_sde
        self.T = T
        self.t0 = t0
        self.xdim = xdim
        self.ydim = ydim
        self.sde_type = sde_type
        self.noise_type = noise_type

    #x and y are passed as one tensor to be compatible with sdeint
    def f(self, t, inputs, lmbd = 0.):
        #unpack x and y from the inputs to pass them to the net
        x_t = inputs[:,:xdim]
        y = inputs[:,-ydim:]
        if t.ndim <= 1:
            t = torch.full((x_t.shape[0], 1), t)
        f_x = (1. - 0.5 * lmbd) * self.forward_sde.g(self.T - t, x_t) * self.net(x_t, self.T - t, y) - \
            self.forward_sde.f(self.T - t, x_t)
        return torch.cat([f_x,y],dim=1)
    def g(self, t, x_t, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.forward_sde.g(self.T - t, x_t)

def sample(model, y, x_T=None, dt=0.005, t0=0.,t1=1., n_samples = 500):

    :param model: (Object) Object of class that implements functions f and g as drift and diffusion coefficient.
    :param y: (Tensor) of shape (1,ydim)
    :param x_T: (Tensor) of shape (n_samples, xdim)
    :param dt: (float) Timestep to integrate
    :param t0: (scalar) Start time. Default 0.
    :param t1: (scalar) End Time. Default 1.
    :param n_samples: (int) Number samples to draw.
    :return:    (Tensor:(n_samples,xdim)) Samples from the posterior.
    model.eval()
    with torch.no_grad():
        x_T = torch.randn((n_samples, model.xdim)) if x_T is None else x_T
        t = torch.linspace(t0,t1,int((t1-t0)/dt))
        #concatenate x and y to use with sdeint
        y = y.repeat(n_samples,1)
        xy = torch.concat([x_T,y],dim=1)
        x_pred = torchsde.sdeint(model, xy, t, dt=dt)[-1,:,:]

    return x_pred[:,:xdim]
"""