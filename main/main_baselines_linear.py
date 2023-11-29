import matplotlib.pyplot as plt
from linear_problem import LinearForwardProblem
from models.diffusion import *
from models.SNF import *
from models.INN import *
from losses import *
from datasets import generate_dataset_linear, get_dataloader_linear
import utils
import argparse
import scipy
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch

def train(snf,diffusion_model,INN,forward_model,xs,ys, num_epochs_INN,num_epochs_SNF,num_epochs_dsm, save_dir, log_dir, batch_size=1000):
    logger = SummaryWriter(log_dir)
    loss_fn_diffusion = DSMLoss()
    optimizer_snf = Adam(snf.parameters(), lr=1e-4)
    prog_bar = tqdm(total=num_epochs_SNF)
    for i in range(num_epochs_SNF):
        data_loader = get_dataloader_linear(xs, ys,forward_model.scale,batch_size)
        loss = train_SNF_epoch(optimizer_snf, snf, data_loader)
        logger.add_scalar('Train/SNF-Loss', loss, i)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_diffusion = Adam(diffusion_model.parameters(), lr=1e-4)
    prog_bar = tqdm(total=num_epochs_dsm)
    t_min = torch.inf
    for i in range(num_epochs_dsm):
        data_loader = get_dataloader_linear(xs, ys,forward_model.scale,batch_size)
        loss, logger_info = diffusion_model.train_epoch(optimizer_diffusion, loss_fn_diffusion, diffusion_model,
                                                     data_loader, t_min)
        logger.add_scalar('Train/diffusion-Loss', loss, i)
        prog_bar.set_description('diffusion loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_inn = Adam(INN.parameters(), lr=1e-3)
    prog_bar = tqdm(total=num_epochs_INN)
    for i in range(num_epochs_INN):
        data_loader = get_dataloader_linear(xs, ys,forward_model.scale,batch_size)
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


def evaluate(ys, snf, diffusion_model, INN, forward_model, out_dir, n_samples_x=5000, n_repeats=10, epsilon = 1e-10, xlim = (-3.5,3.5),nbins = 75, figsize = (12,12), labelsize = 30):
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
        posterior = forward_model.get_posterior(y)

        for j in range(n_repeats):
            x_pred_diffusion = diffusion_model.get_grid(y, num_samples=n_samples_x)
            x_pred_snf = snf(torch.randn(n_samples_x, forward_model.xdim, device=device), inflated_ys)[
                0]
            x_pred_inn = INN(torch.randn(n_samples_x, forward_model.xdim, device=device), c=inflated_ys)[0]
            x_true = posterior.sample((n_samples_x,))

            # calculate MSE of score on test set
            t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1).to(device)
            g_0 = diffusion_model.base_sde.g(t0, x_true)
            score_predict = diffusion_model.a(x_true, t0.to(device), inflated_ys.to(device)) / g_0
            score_true = forward_model.score_posterior(x_true,inflated_ys)
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

            # calculate negative log likelihood of the samples
            nll_sum_snf -= torch.mean(posterior.log_prob(x_pred_snf)) 
            nll_sum_true -= torch.mean(posterior.log_prob(x_true))
            nll_sum_diffusion -= torch.mean(posterior.log_prob(torch.from_numpy(x_pred_diffusion)))
            nll_sum_inn -= torch.mean(posterior.log_prob(x_pred_inn))
            
        if i in plot_y:
            
            utils.plot_density(x_true, nbins, limits=xlim, xticks=xlim, size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-true-%d.svg' % i), show_mean=True)

            utils.plot_density(x_pred_diffusion, nbins, limits=xlim, xticks=xlim, size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-diffusion-%d.svg' % i), show_mean=True)

            utils.plot_density(x_pred_snf, nbins, limits=xlim, xticks=xlim, size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-snf-%d.svg' % i), show_mean=True)

            utils.plot_density(x_pred_inn, nbins, limits=xlim, xticks=xlim, size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-inn-%d.svg' % i), show_mean=True)

        hist_true = hist_true_sum / hist_true_sum.sum()
        hist_snf = hist_snf_sum / hist_snf_sum.sum()
        hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
        hist_inn = hist_inn_sum / hist_inn_sum.sum()
        hist_true += epsilon
        hist_snf += epsilon
        hist_diffusion += epsilon
        hist_inn += epsilon
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

    #parse arguments
    parser = argparse.ArgumentParser(description="Load model parameters.")
    parser.add_argument('--train_dir', required=True, type=str,
                        help='Directory where checkpoints and logs are saved during training.')
    parser.add_argument('--out_dir', required=True, type=str, help='Directory to save output results.')
    parser.add_argument('--model', required=False, default='CDE', type=str, help='Which diffusion model to use. Can be either "CDE" or "CDiffE".')
    parser.add_argument('--hidden_layers', required=False, default=[512,512,512], help='Number of hidden layers.')
    parser.add_argument('--dataset_size', required=False, default=100000, type=int, help='Size of the Dataset.')
    parser.add_argument('--n_epochs_dsm', required=False, default = 5000, type = int, help='Number epochs to train the dsm baseline')
    parser.add_argument('--n_epochs_INN', required=False, default = 500, type = int, help='Number epochs to train the Normalizing Flow baseline')
    parser.add_argument('--n_epochs_SNF', required=False, default = 100, type = int, help='Number epochs to train the Stochastic Normalizing Flow baseline')

    args = parser.parse_args()
    # load linear forward problem
    f = LinearForwardProblem()
    
    # create data
    xs, ys = generate_dataset_linear(f.xdim, f, args['dataset_size'])
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=.9, random_state=7)

    log_dir = utils.set_directories(args)

    snf = create_snf(4, 64, f.log_posterior, metr_steps_per_block=10, dimension=f.xdim, dimension_condition=f.ydim,
                     noise_std=0.4)
    if args['model'] == 'CDE':
        diffusion_model = CDE(xdim=f.xdim, ydim=f.ydim, hidden_layers=args['hidden_layers'])
    elif args['model'] == 'CDiffE':
        diffusion_model = CDiffE(xdim=f.xdim, ydim=f.ydim, hidden_layers=args['hidden_layers'])
    INN = create_INN(4, 64, dimension=f.xdim, dimension_condition=ydim)

    snf, diffusion_model, INN = train(x_train,y_train, n_epochs_INN, n_epochs_SNF, n_epochs_dsm, batch_size=1000, save_dir=train_dir, log_dir=log_dir)
    evaluate(y_test[:100],snf, diffusion_model, INN, out_dir, n_samples_x=30000)