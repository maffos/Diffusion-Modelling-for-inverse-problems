
import matplotlib.pyplot as plt
from sbi.analysis import pairplot, conditional_pairplot

from models.SNF import *
from losses import *
from tqdm import tqdm
import os
import scipy
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# mcmc parameters for "discovering" the ground truth
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000
lambd_bd = 1000

def create_diffusion_model2(xdim, ydim,hidden_layers):

    net_params = {'input_dim': xdim + ydim+1,
                  'output_dim': xdim,
                  'hidden_layers': hidden_layers,
                  'activation': nn.Tanh()}
    forward_process = sdes.VariancePreservingSDE()
    score_net = MLP(**net_params).to(device)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=False)
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
        t_ = model.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        t_.requires_grad = True
    else:
        #we cannot just uniformly sample when using the PINN-loss because the gradient explodes for t of order 1e-7
        t_ = eps+torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)], requires_grad=True).to(x) * model.T
        t_[torch.where(t_>model.T)] = model.T-eps
    return t_
    
#is added to KL Divergence to improve numerical stability
reg = 1e-10

def get_epoch_data_loader(batch_size, forward_model,a, b,lambd_bd):
    x = torch.tensor(inverse_cdf_prior(np.random.uniform(size=(8*batch_size,3)),lambd_bd),dtype=torch.float,device=device)
    y = forward_model(x)
    y += torch.randn_like(y) * b + torch.randn_like(y)*a*y
    def epoch_data_loader():
        for i in range(0, 8*batch_size, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader


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
    return p+p2+p3


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

def train(forward_model, num_epochs_SNF, num_epochs_diffusion, batch_size, lambd_bd, save_dir):
    # define networks
    log_posterior=lambda samples, ys:get_log_posterior(samples,forward_model,a,b,ys,lambd_bd)
    snf = create_snf(4,64,log_posterior,metr_steps_per_block=10,dimension=3,dimension_condition=23,noise_std=0.4)
    diffusion_model = create_diffusion_model2(xdim=3,ydim=23,hidden_layers=[512,512])
    optimizer = Adam(snf.parameters(), lr = 1e-4)

    loss_fn = PINNLoss(xdim)
    prog_bar = tqdm(total=num_epochs_SNF)
    for i in range(num_epochs_SNF):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_SNF_epoch(optimizer, snf, data_loader,forward_model, a, b,None)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_diffusion = Adam(diffusion_model.parameters(), lr = 1e-4)
    prog_bar = tqdm(total=num_epochs_diffusion)
    for i in range(num_epochs_diffusion):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_diffusion_epoch(optimizer_diffusion, loss_fn, diffusion_model, data_loader)
        prog_bar.set_description('determ diffusion loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    chkpnt_file_snf = os.path.join(save_dir, 'snf.pt')
    chkpnt_file_diff = os.path.join(save_dir, 'diffusion.pt')
    torch.save(snf.state_dict(),chkpnt_file_snf)
    torch.save(diffusion_model.a.state_dict(),chkpnt_file_diff)

    return snf, diffusion_model

def evaluate(snf,diffusion_model, out_dir, n_samples_y = 200, n_samples_x=5000,n_repeats=10, n_plots = 10):

    snf.eval()
    diffusion_model.eval()

    xs = torch.rand(n_samples_y, xdim, device=device) * 2 - 1
    ys = forward_model(xs)
    ys = ys + b * torch.randn_like(ys) + ys * a * torch.randn_like(ys)
    with torch.no_grad():

        nll_diffusion = []
        nll_mcmc = []
        nll_snf = []
        kl1_sum = 0.
        kl2_sum = 0.
        kl1_vals = []
        kl2_vals = []
        nbins = 75
        plot_y = np.random.choice(np.arange(n_samples_y), size=n_plots,
                                  replace=False)  # randomly select some y's to plot the posterior (otherwise we would get ~2000 plots)
        prog_bar = tqdm(total=n_samples_y)
        for i, y in enumerate(ys):
            # testing
            hist_mcmc_sum = np.zeros((nbins, nbins, nbins))
            hist_snf_sum = np.zeros((nbins, nbins, nbins))
            hist_diffusion_sum = np.zeros((nbins, nbins, nbins))
            nll_sum_mcmc = 0
            nll_sum_snf = 0
            nll_sum_diffusion = 0
            inflated_ys = y[None, :].repeat(n_samples_x, 1)

            mcmc_energy = lambda x: get_log_posterior(x, forward_model, a, b, inflated_ys, lambd_bd)
            
            for _ in range(n_repeats):
                x_pred_diffusion = get_grid(diffusion_model,y,xdim,ydim, num_samples=n_samples_x)
                x_pred_snf = snf.forward(torch.randn(n_samples_x, xdim, device=device), inflated_ys)[0].detach().cpu().numpy()
                x_true = anneal_to_energy(torch.rand(n_samples_x, xdim, device=device) * 2 - 1, mcmc_energy, METR_STEPS,
                                 noise_std=NOISE_STD_MCMC)[0].detach().cpu().numpy()
        
                # generate histograms
                hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                              range=((-1, 1), (-1, 1), (-1, 1)))
                hist_snf, _ = np.histogramdd(x_pred_snf, bins=(nbins, nbins, nbins), range=((-1, 1), (-1, 1), (-1, 1)))
                hist_diffusion, _ = np.histogramdd(x_pred_diffusion, bins=(nbins, nbins, nbins), range=((-1, 1), (-1, 1), (-1, 1)))
        
                hist_mcmc_sum += hist_mcmc
                hist_snf_sum += hist_snf
                hist_diffusion_sum += hist_diffusion

                #calculate negaitve log likelihood of the samples
                nll_sum_snf += mcmc_energy(torch.from_numpy(x_pred_snf).to(device)).sum() / n_samples_x
                nll_sum_mcmc += mcmc_energy(torch.from_numpy(x_true).to(device)).sum() / n_samples_x
                nll_sum_diffusion += mcmc_energy(torch.from_numpy(x_pred_diffusion).to(device)).sum() / n_samples_x
        
            if i in plot_y:
                # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
                fig, ax = pairplot([x_true])
                fig.suptitle('N=%d samples from the posterior with mcmc' % (n_samples_x))
                fname = os.path.join(out_dir, 'posterior-mcmc-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred_snf])
                fig.suptitle('N=%d samples from the posterior with SNF' % (n_samples_x))
                fname = os.path.join(out_dir, 'posterior-snf-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred_diffusion])
                fig.suptitle('N=%d samples from the posterior with diffusion' % (n_samples_x))
                fname = os.path.join(out_dir, 'posterior-sde-%d.png' % i)
                plt.savefig(fname)
                plt.close()

            hist_mcmc = hist_mcmc_sum / hist_mcmc_sum.sum()
            hist_snf = hist_snf_sum / hist_snf_sum.sum()
            hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
            hist_mcmc += reg
            hist_snf += reg
            hist_diffusion += reg
            hist_mcmc /= hist_mcmc.sum()
            hist_snf /= hist_snf.sum()
            hist_diffusion /= hist_diffusion.sum()
        
            kl1 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_snf))
            kl2 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_diffusion))
            kl1_sum += kl1
            kl2_sum += kl2
            kl1_vals.append(kl1)
            kl2_vals.append(kl2)
            nll_mcmc.append(nll_sum_mcmc.item() / n_repeats)
            nll_snf.append(nll_sum_snf.item() / n_repeats)
            nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
            prog_bar.set_description('KL_SNF: {:.3f}, KL_diffusion: {:.3f}'.format(np.mean(kl1_vals),np.mean(kl2_vals)))
            prog_bar.update()

        prog_bar.close()
        kl1_vals = np.array(kl1_vals)
        kl2_vals = np.array(kl2_vals)
        kl1_var = np.sum((kl1_vals - kl1_sum / n_samples_y) ** 2) / n_samples_y
        kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
        nll_snf = np.array(nll_snf)
        nll_mcmc = np.array(nll_mcmc)
        nll_diffusion = np.array(nll_diffusion)
        df = pd.DataFrame(
            {'KL1': kl1_vals, 'KL2': kl2_vals, 'NLL_mcmc': nll_mcmc, 'NLL_snf': nll_snf, 'NLL_diffusion': nll_diffusion})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('KL1:', kl1_sum / n_samples_y, '+-', kl1_var)
        print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)

if __name__ == '__main__':

    # load forward model
    xdim=3
    ydim=23

    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)

    forward_model.load_state_dict(torch.load('models/surrogate/scatterometry_surrogate.pt', map_location=torch.device(device)))
    for param in forward_model.parameters():
        param.requires_grad = False

    a = 0.2
    b = 0.01
    n_epochs_snf = 200
    n_epochs_diffusion = 20000

    plot_dir='/scatterometry'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    save_dir = 'models/scatterometry/no_embed'
    snf,diffusion_model = train(forward_model, n_epochs_snf, n_epochs_diffusion, batch_size=1000, lambd_bd=1000, save_dir = save_dir)
    evaluate(snf,diffusion_model, plot_dir, n_samples_y=100, n_samples_x = 64000, n_plots = 10)
