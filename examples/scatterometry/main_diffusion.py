import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from models.diffusion import *
from models.SNF import anneal_to_energy, energy_grad
from utils_scatterometry import *
from losses import *
import scipy
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# mcmc parameters for "discovering" the ground truth
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000
RANDOM_STATE = 13

def train_epoch(optimizer, loss_fn, model, epoch_data_loader, t_min):
    mean_loss = 0
    logger_info = {}

    for k, (x, y) in enumerate(epoch_data_loader()):

        t = sample_t(model,x)
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
        #loss = model.dsm(x,y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss, logger_info, t_min

def train(model, optimizer, loss_fn, forward_model, a,b,lambd_bd, num_epochs, batch_size, save_dir, log_dir):

    logger = SummaryWriter(log_dir)
    #track the smallest t during training for debugging
    t_min = torch.inf
    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(batch_size, forward_model, a, b, lambd_bd)
        loss,logger_info, t_min = train_epoch(optimizer, loss_fn, model, data_loader, t_min)
        prog_bar.set_description('diffusion loss:{:.3f}'.format(loss))
        logger.add_scalar('Train/Loss', loss, i)
        for key,value in logger_info.items():
            logger.add_scalar('Train/'+key, value, i)
        prog_bar.update()
    prog_bar.close()

    print('Minimal t during training was %f'%t_min)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    chkpnt_file = os.path.join(save_dir, 'diffusion.pt')
    torch.save(model.a.state_dict(), chkpnt_file)

    return model

def evaluate(model,ys,forward_model, a,b,lambd_bd, out_dir, n_samples_x=5000,n_repeats=10, n_plots = 10, epsilon=1e-10):
    n_samples_y = ys.shape[0]
    model.eval()
    with torch.no_grad():
        nll_diffusion = []
        nll_mcmc = []
        kl2_sum = 0.
        kl2_vals = []
        nbins = 75
        #hardcoded ys to plot the posterior for reproducibility (otherwise we would get ~2000 plots)
        plot_y = [0,5,6,20,23,42,50,77,81,93]
        prog_bar = tqdm(total=n_samples_y)
        for i, y in enumerate(ys):
            # testing
            hist_mcmc_sum = np.zeros((nbins, nbins, nbins))
            hist_diffusion_sum = np.zeros((nbins, nbins, nbins))
            nll_sum_mcmc = 0
            nll_sum_diffusion = 0
            inflated_ys = y[None, :].repeat(n_samples_x, 1)

            mcmc_energy = lambda x: get_log_posterior(x, forward_model, a, b, inflated_ys, lambd_bd)

            for _ in range(n_repeats):
                x_pred = get_grid(model, y, xdim, ydim, num_samples=n_samples_x)
                x_true = anneal_to_energy(torch.rand(n_samples_x, xdim, device=device) * 2 - 1, mcmc_energy, METR_STEPS,
                                          noise_std=NOISE_STD_MCMC)[0].detach().cpu().numpy()

                # generate histograms
                hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                              range=((-1, 1), (-1, 1), (-1, 1)))
                hist_diffusion, _ = np.histogramdd(x_pred, bins=(nbins, nbins, nbins),
                                                   range=((-1, 1), (-1, 1), (-1, 1)))

                hist_mcmc_sum += hist_mcmc
                hist_diffusion_sum += hist_diffusion

                # calculate negaitve log likelihood of the samples
                nll_sum_mcmc += mcmc_energy(torch.from_numpy(x_true).to(device)).sum() / n_samples_x
                nll_sum_diffusion += mcmc_energy(torch.from_numpy(x_pred).to(device)).sum() / n_samples_x

            # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
            if i in plot_y:
                fig, ax = pairplot([x_true])
                fig.suptitle('N=%d samples from the posterior with mcmc' % (n_samples_x))
                fname = os.path.join(out_dir, 'posterior-mcmc-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred])
                fig.suptitle('N=%d samples from the posterior with diffusion' % (n_samples_x))
                fname = os.path.join(out_dir, 'posterior-sde-%d.png' % i)
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
            prog_bar.set_description(
                'KL_diffusion: {:.3f}'.format(np.mean(kl2_vals)))
            prog_bar.update()

        prog_bar.close()
        kl2_vals = np.array(kl2_vals)
        kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
        nll_mcmc = np.array(nll_mcmc)
        nll_diffusion = np.array(nll_diffusion)
        df = pd.DataFrame(
            {'KL2': kl2_vals, 'NLL_mcmc': nll_mcmc,'NLL_diffusion': nll_diffusion})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
        
if __name__ == '__main__':

    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)
    src_dir = '.'
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
    n_epochs = 500
    x_test,y_test = get_dataset(forward_model,a,b,size=n_samples_y)

    score_posterior = lambda x,y: -energy_grad(x, lambda x:  get_log_posterior(x,forward_model,a,b,y,lambd_bd))[0]
    score_prior = lambda x: -x

    hidden_layers = [512,512]
    model = create_diffusion_model2(xdim,ydim,hidden_layers)
    optimizer = Adam(model.a.parameters(), lr=1e-4)
    loss_fn = PINNLoss4(initial_condition = score_posterior, lam=.1, pde_loss='FPE')
    #loss_fn = ErmonLoss(lam=.1)

    train_dir = os.path.join(src_dir,loss_fn.name, 'L1', 'lam=.1')
    log_dir = os.path.join(train_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    out_dir = os.path.join(train_dir, 'results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = train(model, optimizer, loss_fn, forward_model, a,b,lambd_bd, n_epochs, batch_size=1000,save_dir=train_dir, log_dir = log_dir)
    evaluate(model, y_test, forward_model, a,b,lambd_bd, out_dir, n_samples_x=n_samples_x, n_plots=10)