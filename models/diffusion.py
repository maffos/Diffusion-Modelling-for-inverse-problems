from include.sdeflow_light.lib import sdes
from nets import MLP, MLP2, PosteriorDrift

import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_diffusion_model_CDiffE(input_dim, hidden_layers):

    net_params = {'input_dim': input_dim+1,
                  'output_dim': input_dim,
                  'hidden_layers': hidden_layers,
                  'activation': nn.Tanh()}
    forward_process = sdes.VariancePreservingSDE()
    score_net = MLP2(**net_params).to(device)
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

        optimizer.zero_grad()
        loss.backward(retain_graph = True)
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss / (k + 1)
    return mean_loss, logger_info, t_min

def create_diffusion_model_CDE(xdim, ydim,hidden_layers):

    net_params = {'input_dim': xdim + ydim+1,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers,
                  'activation': nn.Tanh()}
    forward_process = sdes.VariancePreservingSDE()
    score_net = MLP(**net_params).to(device)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=True)
    return reverse_process

def create_diffusion_model_posterior(xdim, ydim,hidden_layers):

    prior_net_params = {'input_dim': xdim +1,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers,
                  'activation': nn.Tanh()}
    likelihood_net_params = {'input_dim': xdim +ydim+1,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers,
                  'activation': nn.Tanh()}
    forward_process = sdes.VariancePreservingSDE()
    prior_net = MLP2(**prior_net_params).to(device)
    likelihood_net = MLP(**likelihood_net_params).to(device)
    posterior_net = PosteriorDrift(prior_net,likelihood_net,forward_process)
    reverse_process = sdes.PluginReverseSDE(forward_process, posterior_net, T=1, debias=True)
    return reverse_process

def get_grid_CDiffE(sde, cond1, xdim,ydim, num_samples = 2000, num_steps=200,
             mean=0, std=1):

    cond = torch.zeros(num_samples, ydim).to(cond1.device)
    cond += cond1
    delta = sde.T / num_steps
    x0 = torch.randn(num_samples, xdim).to(cond1.device)
    x0 = x0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1) * sde.T
    ones = torch.ones(num_samples, 1)
    z_0 = torch.concat([x0,cond], dim=1)
    x_t = x0
    with torch.no_grad():
        for i in range(num_steps):
            y_t = sde.base_sde.sample(sde.T-ts[i],z_0)[:,xdim:]
            z_t = torch.concat([x_t,y_t], dim = 1)
            mu = sde.mu((ones * ts[i]).to(cond1.device), z_t)
            sigma = sde.sigma((ones * ts[i]).to(cond1.device), z_t)
            z_t = z_t + delta * mu + delta ** 0.5 * sigma * torch.randn_like(z_t).to(cond1.device)
            x_t = z_t[:,:xdim]

    x_t = x_t.data.cpu().numpy()
    return x_t

def get_grid_CDE(sde, cond1, xdim, ydim, num_samples = 2000, num_steps = 200,
    mean = 0, std = 1):

    cond = torch.zeros(num_samples, ydim).to(cond1.device)
    cond += cond1
    delta = sde.T / num_steps
    x0 = torch.randn(num_samples, xdim).to(cond1.device)
    x0 = x0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1) * sde.T
    ones = torch.ones(num_samples, 1)
    x_t = x0
    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu((ones * ts[i]).to(cond1.device), x_t)
            sigma = sde.sigma((ones * ts[i]).to(cond1.device), x_t)
            x_t = x_t + delta * mu + delta ** 0.5 * sigma * torch.randn_like(x_t).to(cond1.device)

        x_t = x_t.data.cpu().numpy()


    return x_t

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
