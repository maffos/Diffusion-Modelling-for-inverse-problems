import torch
from libraries.sdeflow_light.lib import sdes
from nets import TemporalMLP_small,MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_diffusion_model(xdim, ydim, embed_dim, hidden_layers):

    net_params = {'input_dim': xdim + ydim,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers,
                  'embed_dim': embed_dim}
    forward_process = sdes.VariancePreservingSDE()
    score_net = TemporalMLP_small(**net_params).to(device)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=False)
    return reverse_process

def create_diffusion_model2(xdim, ydim,hidden_layers):

    net_params = {'input_dim': xdim + ydim+1,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers}
    forward_process = sdes.VariancePreservingSDE()
    score_net = MLP(**net_params).to(device)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=False)
    return reverse_process

def train_diffusion_epoch(optimizer, model, epoch_data_loader):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):

        loss = model.dsm(x,y).mean()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss

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