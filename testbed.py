import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from examples.scatterometry.utils_scatterometry import *
from models.SNF import anneal_to_energy, energy_grad
from models.diffusion import *
import pandas as pd
from torch.optim import Adam
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import os
import pickle
import time
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
from sklearn.model_selection import train_test_split
import utils
import losses


def create_INN(num_layers, sub_net_size,dimension,dimension_condition):
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))
    nodes = [InputNode(dimension, name='input')]
    cond = ConditionNode(dimension_condition, name='condition')
    for k in range(num_layers):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.4},
                          conditions = cond,
                          name=F'coupling_{k}'))
    nodes.append(OutputNode(nodes[-1], name='output'))

    model = ReversibleGraphNet(nodes + [cond], verbose=False).to(device)
    return model

    # trains an epoch of the INN
# given optimizer, the model and the data_loader
# returns mean loss

def train_inn_epoch(optimizer, model, epoch_data_loader, backward_training = True, **loss_params):
    model.train()
    mean_loss = 0
    relu = torch.nn.ReLU()
    for k, (x, y_noise, y_true) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        z, jac_inv = model(x, c = y_noise, rev = True)
        #log_p_z = latent.log_prob(z)
        #l5 = 0.5 * torch.sum(z**2, dim=1) - jac_inv
        #mle_loss = (torch.sum(l5) / cur_batch_size)
        #mle_loss = (-log_p_z+jac_inv).mean()
        if backward_training:
            z_samples = torch.randn(cur_batch_size, DIMENSION, device=device)
            x, log_det_J = model(z_samples, c=y_noise)
            loss = utils.ForwardBackwardKLLoss(x,z,jac_inv, log_det_J, y_true, y_noise, **loss_params)
        else:
            loss = losses.mleLoss(z,-jac_inv, **loss_params)
        #loss_kl = -torch.mean(jac) + torch.sum((y - MLP(x)) ** 2) / (cur_batch_size * 2 * sigma ** 2)
        #loss_kl = -torch.mean(jac) + torch.sum((y_noise - y_true) ** 2) / (cur_batch_size * 2 * sigma ** 2) #anmerkung: vorzeichen unklar. eventuell plus und minus vertauschen
        #loss_relu = 100 * torch.sum(relu(x - 1) + relu(-x))
        #likelihood = MultivariateNormal(y_true, torch.eye(COND_DIM)*sigma)
        #loss_kl = (-log_likelihood(y_noise, y_true) - jac).mean()
        #loss = mle_loss*(1-conv_lambda) + (loss_kl+loss_relu)*conv_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)

    return mean_loss
    
def train_MLP_epoch(optimizer, model, epoch_data_loader):
    model.train()
    mean_loss = 0
    mse = nn.MSELoss()
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        y_pred = model(x)
        loss = mse(y,y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss

def make_image(pred_samples,x_true, num_epochs, inds=None):

    cmap = plt.cm.tab20
    range_param = 1.2
    if inds is None:
        no_params = pred_samples.shape[1]
        inds=range(no_params)
    else:
        no_params=len(inds)
    fig, axes = plt.subplots(figsize=[12,12], nrows=no_params, ncols=no_params, gridspec_kw={'wspace':0., 'hspace':0.});
    fig.suptitle('Epochs=%d'%num_epochs)
    for j, ij in enumerate(inds):
        for k, ik in enumerate(inds):
            axes[j,k].get_xaxis().set_ticks([])
            axes[j,k].get_yaxis().set_ticks([])
            # if k == 0: axes[j,k].set_ylabel(j)
            # if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), alpha=0.3, range=(-0,range_param))
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), histtype="step", range=(-0,range_param))
                axes[j,k].axvline(x_true[:,j])

            else:
                val, x, y = np.histogram2d(pred_samples[:,ij], pred_samples[:,ik], bins=25, range = [[-0, range_param], [-0, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

    plt.savefig(os.path.join(output_dir, 'posterior_epochs=%d.png'%num_epochs))

def get_epoch_dataloader(x_train, y_train):
    perm = torch.randperm(len(x_train))
    x = x_train[perm]
    y = y_train[perm]
    #y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    return epoch_data_loader

def get_epoch_dataloader_noise(x_train, y_train):
    perm = torch.randperm(len(x_train))
    x = x_train[perm]
    y = y_train[perm]
    y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size], y_train[i:i+batch_size]

    return epoch_data_loader

def evaluate(model,ys,forward_model, a,b,lambd_bd, out_dir, gt_path, n_samples_x=5000,n_repeats=10, epsilon=1e-10):
    n_samples_y = ys.shape[0]
    model.eval()
    nll_diffusion = []
    nll_mcmc = []
    kl2_sum = 0.
    kl2_vals = []
    mse_score_vals = []
    nbins = 75
    # randomly select some y's to plot the posterior (otherwise we would get ~2000 plots)
    plot_y = [0,5,6,20,23,42,50,77,81,93]
    prog_bar = tqdm(total=n_samples_y)
    for i, y in enumerate(ys):
        # testing
        hist_mcmc_sum = np.zeros((nbins, nbins, nbins))
        hist_diffusion_sum = np.zeros((nbins, nbins, nbins))
        nll_sum_mcmc = 0
        nll_sum_diffusion = 0
        mse_score_sum = 0
        inflated_ys = y[None, :].repeat(n_samples_x, 1)
        mcmc_energy = lambda x: get_log_posterior(x, forward_model, a, b, inflated_ys, lambd_bd)

        for j in range(n_repeats):
            x_pred = get_grid(model, y, xdim, ydim, num_samples=n_samples_x)
            x_true = get_gt_samples(gt_path, i,j)
            x_true_tensor = torch.from_numpy(x_true).to(device)
            # calculate MSE of score on test set
            t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1).to(device)
            g_0 = model.base_sde.g(t0, x_true_tensor)
            score_predict = model.a(x_true_tensor, t0.to(device), inflated_ys.to(device)) / g_0
            score_true = score_posterior(x_true_tensor,inflated_ys)
            #score_true = -energy_grad(x_true_tensor,mcmc_energy)
            mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))
            # generate histograms
            hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                          range=((-1, 1), (-1, 1), (-1, 1)))
            hist_diffusion, _ = np.histogramdd(x_pred, bins=(nbins, nbins, nbins),
                                               range=((-1, 1), (-1, 1), (-1, 1)))

            hist_mcmc_sum += hist_mcmc
            hist_diffusion_sum += hist_diffusion

            # calculate negaitve log likelihood of the samples
            nll_sum_mcmc += mcmc_energy(x_true_tensor).sum() / n_samples_x
            nll_sum_diffusion += mcmc_energy(torch.from_numpy(x_pred).to(device)).sum() / n_samples_x

        # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
        if i in plot_y:
            fig, ax = pairplot([x_true])
            fig.suptitle('MCMC')
            fname = os.path.join(out_dir, 'posterior-mcmc-nolimits%d.png' % i)
            plt.savefig(fname)
            plt.close()

            fig, ax = pairplot([x_pred], limits = [[-1,1],[-1,1],[-1,1]])
            fig.suptitle('PINN-Loss')
            fname = os.path.join(out_dir, 'posterior-diffusion-limits%d.png' % i)
            plt.savefig(fname)
            plt.close()
            fig, ax = pairplot([x_pred])
            fig.suptitle('PINN-Loss' % (n_samples_x))
            fname = os.path.join(out_dir, 'posterior-diffusion-nolimits%d.png' % i)
            plt.savefig(fname)
            plt.close()

        hist_mcmc = hist_mcmc_sum / hist_mcmc_sum.sum()
        hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
        hist_mcmc += epsilon
        hist_diffusion += epsilon
        hist_mcmc /= hist_mcmc.sum()
        hist_diffusion /= hist_diffusion.sum()

        kl2 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_diffusion))
        kl2_sum += kl2
        kl2_vals.append(kl2)
        nll_mcmc.append(nll_sum_mcmc.item() / n_repeats)
        nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
        mse_score_vals.append(mse_score_sum.item()/n_repeats)
        prog_bar.set_description(
            'KL_diffusion: {:.3f}'.format(np.mean(kl2_vals)))
        prog_bar.update()

    prog_bar.close()
    kl2_vals = np.array(kl2_vals)
    kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
    nll_mcmc = np.array(nll_mcmc)
    nll_diffusion = np.array(nll_diffusion)
    df = pd.DataFrame(
        {'KL2': kl2_vals, 'NLL_mcmc': nll_mcmc,'NLL_diffusion': nll_diffusion,'MSE':np.array(mse_score_vals)})
    df.to_csv(os.path.join(out_dir, 'results.csv'))
    print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
def load_data(filename):

    data = np.load(filename, allow_pickle=True)["data"].item()
    x_labels = data['parameters'][1:]
    #data = data[age]
    xs = data['x_train'][:,1:]

    # normalize x
    xs = (xs - xs.min(axis=0)) / (xs.max(axis=0) - xs.min(axis=0))
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(data['y_train'][:,1:]).float()

    return xs,ys,x_labels

# trains and evaluates both the INN and SNF and returns the Wasserstein distance on the mixture example
# parameters are the mixture params (parameters of the mixture model in the prior), b (likelihood parameter)
# a set of testing_ys and the forward model (forward_map)

if __name__ == '__main__':
    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)
    src_dir = 'examples/scatterometry'
    gt_dir = os.path.join(src_dir, 'gt_samples')
    forward_model.load_state_dict(
        torch.load(os.path.join(src_dir, 'surrogate.pt'), map_location=torch.device(device)))
    for param in forward_model.parameters():
        param.requires_grad = False

    a = 0.2
    b = 0.01
    lambd_bd = 1000
    xdim = 3
    ydim = 23

    n_samples_y = 100
    n_samples_x = 30000
    n_epochs = 20000
    x_test, y_test = get_dataset(forward_model, a, b, size=n_samples_y)

    score_posterior = lambda x, y: -energy_grad(x, lambda x: get_log_posterior(x, forward_model, a, b, y, lambd_bd))[0]
    score_prior = lambda x: -x

    subfolder_path = os.path.join(src_dir, 'results/CFM/PINNLoss4/3layer/L2/L1/lam:0.1/lam2:0.01')
    chkpnt_path = os.path.join(subfolder_path, 'diffusion.pt')
    if os.path.isfile(chkpnt_path):
        out_dir = os.path.join(src_dir, 'test')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model = create_diffusion_model2(xdim, ydim, hidden_layers=[512, 512, 512])
        checkpoint = torch.load(chkpnt_path, map_location=torch.device(device))
        model.a.load_state_dict(checkpoint)
        evaluate(model, y_test, forward_model, a, b, lambd_bd, out_dir, gt_dir, n_samples_x=n_samples_x)

