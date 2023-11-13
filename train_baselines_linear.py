from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from examples.linearModel.main_diffusion import generate_dataset, f,get_posterior, score_posterior,get_evidence,get_likelihood,get_dataloader_dsm
from models.diffusion import *
from models.SNF import *
from models.INN import *
from losses import *
from utils import get_dataloader_noise,plot_density
import scipy
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch

# define parameters of the forward and inverse problem
epsilon = 1e-6
xdim = 2
ydim = 2
# f is a shear by factor 0.5 in x-direction and tranlsation by (0.3, 0.5).
A = torch.Tensor([[1, 0.5], [0, 1]])
b = torch.Tensor([0.3, 0.5])
scale = .3  # measurement noise
Sigma = scale * torch.eye(ydim)
Lam = torch.eye(xdim)
Sigma_inv = 1 / scale * torch.eye(ydim)
Sigma_y_inv = torch.linalg.inv(Sigma + A @ Lam @ A.T + epsilon * torch.eye(ydim))
mu = torch.zeros(xdim)
cov = Lam - A.T @ Sigma_y_inv @ A #covariance of the posterior
cov_inv = torch.linalg.inv(cov+epsilon*torch.eye(xdim))

def log_posterior(xs,ys):

    y_res = ys - (A @ mu + b)
    mean = y_res@(A.T @ Sigma_y_inv)
    x_res = xs-mean
    log_probs = .5*x_res@cov_inv
    log_probs = log_probs[:,None,:]@x_res[:,:,None]

    return log_probs.view(-1,1)
def train(xs,ys, num_epochs_INN,num_epochs_SNF,num_epochs_dsm, save_dir, log_dir, batch_size=1000):
    snf = create_snf(4, 64, log_posterior, metr_steps_per_block=10, dimension=xdim, dimension_condition=ydim, noise_std=0.4)
    diffusion_model = create_diffusion_model2(xdim=xdim, ydim=ydim, hidden_layers=[512, 512, 512])
    INN = create_INN(4, 64, dimension=xdim, dimension_condition=ydim)
    optimizer = Adam(snf.parameters(), lr=1e-4)

    logger = SummaryWriter(log_dir)

    loss_fn_diffusion = DSMLoss()
    prog_bar = tqdm(total=num_epochs_SNF)
    for i in range(num_epochs_SNF):
        data_loader = get_dataloader_noise(xs, ys,scale,batch_size)
        loss = train_SNF_epoch(optimizer, snf, data_loader)
        logger.add_scalar('Train/SNF-Loss', loss, i)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_diffusion = Adam(diffusion_model.parameters(), lr=1e-4)
    prog_bar = tqdm(total=num_epochs_dsm)
    t_min = torch.inf
    for i in range(num_epochs_dsm):
        data_loader = get_dataloader_noise(xs, ys,scale,batch_size)
        loss, logger_info, t = train_diffusion_epoch(optimizer_diffusion, loss_fn_diffusion, diffusion_model,
                                                     data_loader, t_min)
        logger.add_scalar('Train/diffusion-Loss', loss, i)
        prog_bar.set_description('diffusion loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_inn = Adam(INN.parameters(), lr=1e-3)
    prog_bar = tqdm(total=num_epochs_INN)
    for i in range(num_epochs_INN):
        data_loader = get_dataloader_noise(xs, ys,scale,batch_size)
        loss = train_inn_epoch(optimizer_inn, INN, data_loader)
        logger.add_scalar('Train/INN-Loss', loss, i)
        prog_bar.set_description('INN loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    chkpnt_file_snf = os.path.join(save_dir, 'snf.pt')
    chkpnt_file_diff = os.path.join(save_dir, 'diffusion.pt')
    chkpnt_file_inn = os.path.join(save_dir, 'INN.pt')
    torch.save(snf.state_dict(), chkpnt_file_snf)
    torch.save(diffusion_model.a.state_dict(), chkpnt_file_diff)
    torch.save(INN.state_dict(), chkpnt_file_inn)

    return snf, diffusion_model, INN


def evaluate(ys, snf, diffusion_model, INN, out_dir, n_samples_x=5000, n_repeats=10):
    snf.eval()
    INN.eval()
    diffusion_model.eval()

    nll_diffusion = []
    nll_true = []
    nll_snf = []
    nll_inn = []
    kl1_sum = 0.
    kl2_sum = 0.
    kl3_sum = 0.
    kl1_vals = []
    kl2_vals = []
    kl3_vals = []
    mse_score_vals = []
    nbins = 75
    xlim = (-4,4)

    # hardcoded ys to plot the posterior for reproducibility (otherwise we would get ~2000 plots)
    plot_y = [3, 5, 22, 39, 51, 53, 60, 71, 81, 97]
    n_samples_y = len(ys)
    prog_bar = tqdm(total=len(ys))
    for i, y in enumerate(ys):
        # testing
        hist_true_sum = np.zeros((nbins, nbins))
        hist_snf_sum = np.zeros((nbins, nbins))
        hist_diffusion_sum = np.zeros((nbins, nbins))
        hist_inn_sum = np.zeros((nbins, nbins))
        nll_sum_true = 0
        nll_sum_snf = 0
        nll_sum_diffusion = 0
        nll_sum_inn = 0.
        mse_score_sum = 0
        inflated_ys = y[None, :].repeat(n_samples_x, 1)
        posterior = get_posterior(y)

        for j in range(n_repeats):
            x_pred_diffusion = get_grid(diffusion_model, y, xdim, ydim, num_samples=n_samples_x)
            x_pred_snf = snf.forward(torch.randn(n_samples_x, xdim, device=device), inflated_ys)[
                0]
            x_pred_inn = INN(torch.randn(n_samples_x, xdim, device=device), c=inflated_ys)[0]
            x_true = posterior.sample((n_samples_x,))

            # calculate MSE of score on test set
            t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1).to(device)
            g_0 = diffusion_model.base_sde.g(t0, x_true)
            score_predict = diffusion_model.a(x_true, t0.to(device), inflated_ys.to(device)) / g_0
            score_true = score_posterior(x_true,inflated_ys)
            mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

            # generate histograms
            hist_true, _ = np.histogramdd(x_true.numpy(), bins=(nbins, nbins),
                                          range=(xlim, xlim))
            hist_snf, _ = np.histogramdd(x_pred_snf.detach().numpy(), bins=(nbins, nbins), range=(xlim,xlim))
            hist_diffusion, _ = np.histogramdd(x_pred_diffusion, bins=(nbins, nbins),
                                               range=(xlim, xlim))
            hist_inn, _ = np.histogramdd(x_pred_inn.detach().numpy(), bins=(nbins, nbins),
                                         range=(xlim, xlim))
            hist_true_sum += hist_true
            hist_snf_sum += hist_snf
            hist_diffusion_sum += hist_diffusion
            hist_inn_sum += hist_inn

            # calculate negaitve log likelihood of the samples
            nll_sum_snf -= torch.mean(posterior.log_prob(x_pred_snf)) 
            nll_sum_true -= torch.mean(posterior.log_prob(x_true))
            nll_sum_diffusion -= torch.mean(posterior.log_prob(torch.from_numpy(x_pred_diffusion)))
            nll_sum_inn -= torch.mean(posterior.log_prob(x_pred_inn))
        if i in plot_y:
            # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
            fig, ax = pairplot([x_pred_snf.detach().numpy()], limits=[xlim, xlim])
            fname = os.path.join(out_dir, 'posterior-snf-%d.svg' % i)
            plt.savefig(fname)
            plt.close()

            fig, ax = pairplot([x_pred_snf.detach().numpy()])
            fname = os.path.join(out_dir, 'posterior-snf-nolim-%d.svg' % i)
            plt.savefig(fname)
            plt.close()

            plot_density(x_pred_snf.detach().numpy(), nbins = nbins, limits = xlim, fname = os.path.join(out_dir,'posterior-snf-limits-own%d.svg'%i))
            plot_density(x_pred_snf.detach().numpy(), nbins=nbins, fname=os.path.join(out_dir, 'posterior-snf-nolimits-own%d.svg' % i))

            fig, ax = pairplot([x_pred_diffusion], limits=[xlim, xlim])
            fname = os.path.join(out_dir, 'posterior-dsm-%d.svg' % i)
            plt.savefig(fname)
            plt.close()

            fig, ax = pairplot([x_pred_diffusion])
            fname = os.path.join(out_dir, 'posterior-dsm-nolim-%d.svg' % i)
            plt.savefig(fname)
            plt.close()

            plot_density(x_pred_diffusion, nbins=nbins, limits=xlim,
                         fname=os.path.join(out_dir, 'posterior-dsm-limits-own%d.svg' % i))
            plot_density(x_pred_diffusion, nbins=nbins, fname=os.path.join(out_dir, 'posterior-dsm-nolimits-own%d.svg' % i))

            fig, ax = pairplot([x_pred_inn.detach().numpy()], limits=[xlim, xlim])
            fname = os.path.join(out_dir, 'posterior-inn-%d.png' % i)
            plt.savefig(fname)
            plt.close()

            fig, ax = pairplot([x_pred_inn.detach().numpy()])
            fname = os.path.join(out_dir, 'posterior-inn-nolim-%d.svg' % i)
            plt.savefig(fname)
            plt.close()

            plot_density(x_pred_inn.detach().numpy(), nbins=nbins, limits=xlim,
                         fname=os.path.join(out_dir, 'posterior-inn-limits-own%d.svg' % i))
            plot_density(x_pred_inn.detach().numpy(), nbins=nbins, fname=os.path.join(out_dir, 'posterior-inn-nolimits-own%d.svg' % i))

        hist_true = hist_true_sum / hist_true_sum.sum()
        hist_snf = hist_snf_sum / hist_snf_sum.sum()
        hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
        hist_inn = hist_inn_sum / hist_inn_sum.sum()
        hist_true += reg
        hist_snf += reg
        hist_diffusion += reg
        hist_inn += reg
        hist_true /= hist_true.sum()
        hist_snf /= hist_snf.sum()
        hist_diffusion /= hist_diffusion.sum()
        hist_inn /= hist_inn.sum()

        kl1 = np.sum(scipy.special.rel_entr(hist_true, hist_snf))
        kl2 = np.sum(scipy.special.rel_entr(hist_true, hist_diffusion))
        kl3 = np.sum(scipy.special.rel_entr(hist_true, hist_inn))
        kl1_sum += kl1
        kl2_sum += kl2
        kl3_sum += kl3
        kl1_vals.append(kl1)
        kl2_vals.append(kl2)
        kl3_vals.append(kl3)
        nll_true.append(nll_sum_true.item() / n_repeats)
        nll_snf.append(nll_sum_snf.item() / n_repeats)
        nll_inn.append(nll_sum_inn.item() / n_repeats)
        nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
        mse_score_vals.append(mse_score_sum.item() / n_repeats)
        prog_bar.set_description('KL_SNF: {:.3f}, KL_diffusion: {:.3f}'.format(np.mean(kl1_vals), np.mean(kl2_vals)))
        prog_bar.update()

    prog_bar.close()
    kl1_vals = np.array(kl1_vals)
    kl2_vals = np.array(kl2_vals)
    kl3_vals = np.array(kl3_vals)
    kl1_var = np.sum((kl1_vals - kl1_sum / n_samples_y) ** 2) / n_samples_y
    kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
    kl3_var = np.sum((kl3_vals - kl3_sum / n_samples_y) ** 2) / n_samples_y
    nll_snf = np.array(nll_snf)
    nll_true = np.array(nll_true)
    nll_diffusion = np.array(nll_diffusion)
    nll_inn = np.array(nll_inn)
    df = pd.DataFrame(
        {'KL_SNF': kl1_vals, 'KL_diffusion': kl2_vals, 'KL_INN': kl3_vals, 'NLL_true': nll_true, 'NLL_snf': nll_snf,
         'NLL_diffusion': nll_diffusion, 'NLL_inn': nll_inn, 'MSE': np.array(mse_score_vals)})
    df.to_csv(os.path.join(out_dir, 'results.csv'))
    print('KL1:', kl1_sum / n_samples_y, '+-', kl1_var)
    print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
    print('KL3:', kl3_sum / n_samples_y, '+-', kl3_var)

if __name__ =='__main__':

    # create data
    xs, ys = generate_dataset(n_samples=100000)
    n_epochs_dsm = 5000
    n_epochs_INN = 500
    n_epochs_SNF = 100
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=.9, random_state=7)
    reg = 1e-8
    train_dir = 'examples/linearModel/results/baselines'
    log_dir = os.path.join(train_dir, 'logs')
    out_dir = os.path.join(train_dir, 'results')

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)
    else:
        os.makedirs(log_dir)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    else:
        os.makedirs(out_dir)

    snf, diffusion_model, INN = train(x_train,y_train, n_epochs_INN, n_epochs_SNF, n_epochs_dsm, batch_size=1000, save_dir=train_dir, log_dir=log_dir)
    evaluate(y_test[:100],snf, diffusion_model, INN, out_dir, n_samples_x=30000)