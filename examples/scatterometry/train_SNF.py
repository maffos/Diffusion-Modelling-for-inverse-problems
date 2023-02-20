import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from models.SNF import *
from utils import *
import os
from tqdm import tqdm
import scipy
import pandas as pd
import torch
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# mcmc parameters for "discovering" the ground truth
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000

def loss_fn(optimizer, snf, epoch_data_loader,forward_model,a,b):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        loss = 0
        out = snf.backward(x, y)
        invs = out[0]
        jac_inv = out[1]

        l5 = 0.5* torch.sum(invs**2, dim=1) - jac_inv
        loss += torch.sum(l5) / cur_batch_size
        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mean_loss

def train(model, optimizer, forward_model, a,b,num_epochs, batch_size, lambd_bd,save_dir):
    # define networks
    optimizer = Adam(model.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = loss_fn(optimizer, model, data_loader,forward_model, a, b,None)
        prog_bar.set_description('model loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    chkpnt_file = os.path.join(save_dir, 'snf.pt')
    torch.save(model.state_dict(),chkpnt_file)

    return model

def evaluate(model,ys,forward_model, a,b,lambd_bd, out_dir, n_samples_x=5000,n_repeats=10, n_plots = 10, epsilon=1e-10):
    n_samples_y = ys.shape[0]
    model.eval()
    with torch.no_grad():
        nll_snf = []
        nll_mcmc = []
        nll_snf = []
        kl1_sum = 0.
        kl2_sum = 0.
        kl1_vals = []
        kl2_vals = []
        nbins = 75
        # randomly select some y's to plot the posterior (otherwise we would get ~2000 plots)
        plot_y = np.random.choice(np.arange(n_samples_y), size=n_plots,replace=False)  
        prog_bar = tqdm(total=n_samples_y)
        for i, y in enumerate(ys):
            # testing
            hist_mcmc_sum = np.zeros((nbins, nbins, nbins))
            hist_snf_sum = np.zeros((nbins, nbins, nbins))
            nll_sum_mcmc = 0
            nll_sum_snf = 0
            inflated_ys = y[None, :].repeat(n_samples_x, 1)

            mcmc_energy = lambda x: get_log_posterior(x, forward_model, a, b, inflated_ys, lambd_bd)

            for _ in range(n_repeats):
                x_pred = model(torch.randn(n_samples_x, xdim, device=device), inflated_ys)[0].detach().cpu().numpy()
                x_true = anneal_to_energy(torch.rand(n_samples_x, xdim, device=device) * 2 - 1, mcmc_energy, METR_STEPS,
                                          noise_std=NOISE_STD_MCMC)[0].detach().cpu().numpy()

                # generate histograms
                hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                              range=((-1, 1), (-1, 1), (-1, 1)))
                hist_snf, _ = np.histogramdd(x_pred, bins=(nbins, nbins, nbins),
                                                   range=((-1, 1), (-1, 1), (-1, 1)))

                hist_mcmc_sum += hist_mcmc
                hist_snf_sum += hist_snf

                # calculate negaitve log likelihood of the samples
                nll_sum_mcmc += mcmc_energy(torch.from_numpy(x_true).to(device)).sum() / n_samples_x
                nll_sum_snf += mcmc_energy(torch.from_numpy(x_pred).to(device)).sum() / n_samples_x

            if i in plot_y:
                # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
                fig, ax = pairplot([x_true])
                fig.suptitle('N=%d samples from the posterior with mcmc' % (n_samples_x))
                fname = os.path.join(out_dir, 'posterior-mcmc-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred])
                fig.suptitle('N=%d samples from the posterior with snf' % (n_samples_x))
                fname = os.path.join(out_dir, 'posterior-snf-%d.png' % i)
                plt.savefig(fname)
                plt.close()

            hist_mcmc = hist_mcmc_sum / hist_mcmc_sum.sum()
            hist_snf = hist_snf_sum / hist_snf_sum.sum()
            hist_mcmc += epsilon
            hist_snf += epsilon
            hist_mcmc /= hist_mcmc.sum()
            hist_snf /= hist_snf.sum()

            kl2 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_snf))
            kl2_sum += kl2
            kl2_vals.append(kl2)
            nll_mcmc.append(nll_sum_mcmc.item() / n_repeats)
            nll_snf.append(nll_sum_snf.item() / n_repeats)
            prog_bar.set_description(
                'KL_snf: {:.3f}'.format(np.mean(kl2_vals)))
            prog_bar.update()

        prog_bar.close()
        kl2_vals = np.array(kl2_vals)
        kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
        nll_mcmc = np.array(nll_mcmc)
        nll_snf = np.array(nll_snf)
        df = pd.DataFrame(
            {'KL2': kl2_vals, 'NLL_mcmc': nll_mcmc, 'NLL_snf': nll_snf,
             'NLL_snf': nll_snf})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
        
if __name__ == '__main__':

    # load forward model
    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)

    forward_model.load_state_dict(
        torch.load('examples/scatterometry/surrogate.pt', map_location=torch.device(device)))
    for param in forward_model.parameters():
        param.requires_grad = False

    out_dir = 'examples/scatterometry/snf'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_dir = 'examples/scatterometry'

    a = 0.2
    b = 0.01
    lambd_bd = 1000
    xdim = 3
    ydim = 23
    n_samples_y = 100
    x_test = torch.rand(n_samples_y, xdim, device=device) * 2 - 1
    y_test = forward_model(x_test)
    y_test = y_test + b * torch.randn_like(y_test) + y_test * a * torch.randn_like(y_test)
    n_epochs = 200
    log_posterior=lambda samples, ys:get_log_posterior(samples,forward_model,a,b,ys,lambd_bd=1000)
    model = create_snf(4,64,log_posterior,metr_steps_per_block=10,dimension=3,dimension_condition=23,noise_std=0.4)
    optimizer = Adam(model.parameters())
    model = train(model, optimizer, forward_model, a,b,n_epochs, batch_size = 1000, lambd_bd=1000,save_dir = save_dir)
    evaluate(model, y_test,forward_model, a,b,lambd_bd, out_dir, n_samples_x=64000)