import matplotlib.pyplot as plt
import utils
from models.diffusion import *
from models.SNF import energy_grad
from utils_scatterometry import *
from datasets import (generate_dataset_scatterometry,get_gt_samples_scatterometry,get_dataloader_scatterometry)
from losses import *
import argparse
import scipy
import yaml
import shutil
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, optimizer, loss_fn, forward_model, a,b,lambd_bd, num_epochs, batch_size, save_dir, log_dir):

    logger = SummaryWriter(log_dir)
    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_dataloader_scatterometry(batch_size, forward_model, a, b, lambd_bd)
        loss,logger_info = model.train_epoch(optimizer, loss_fn, data_loader)
        prog_bar.set_description('diffusion loss:{:.3f}'.format(loss))
        logger.add_scalar('Train/Loss', loss, i)
        for key,value in logger_info.items():
            logger.add_scalar('Train/'+key, value, i)
        prog_bar.update()
    prog_bar.close()

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    chkpnt_file = os.path.join(save_dir, 'diffusion.pt')
    torch.save(model.sde.a.state_dict(), chkpnt_file)

    return model

def evaluate(model,ys,forward_model, a,b,lambd_bd, out_dir, gt_dir, n_samples_x=5000,n_repeats=10, epsilon=1e-10,xlim = (-1.2,1.2),nbins = 75, figsize = (12,12), labelsize = 30):
    n_samples_y = ys.shape[0]
    model.sde.eval()
    nll_diffusion = []
    nll_mcmc = []
    kl2_sum = 0.
    kl2_vals = []
    kl2_reverse_vals = []
    mse_score_vals = []
    # plotted y's are hardcoded for reproducability
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
            x_pred = model(y, num_samples=n_samples_x)
            x_true = get_gt_samples_scatterometry(gt_dir,i,j)
            x_true_tensor = torch.from_numpy(x_true).to(device)

            # calculate MSE of score on test set
            t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1).to(device)
            g_0 = model.sde.base_sde.g(t0, x_true_tensor)
            #score_predict = model.sde.a(x_true_tensor, t0.to(device), inflated_ys.to(device)) / g_0
            score_predict = model.sde.a(x_true_tensor, inflated_ys.to(device), t0.to(device))/g_0
            #score_predict = score_predict / g_0
            score_true = score_posterior(x_true_tensor,inflated_ys)
            mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

            # generate histograms
            hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                          range=(xlim, xlim, xlim))
            hist_diffusion, _ = np.histogramdd(x_pred, bins=(nbins, nbins, nbins),
                                               range=(xlim, xlim, xlim))

            hist_mcmc_sum += hist_mcmc
            hist_diffusion_sum += hist_diffusion

            # calculate negative log likelihood of the samples
            nll_sum_mcmc += mcmc_energy(x_true_tensor).sum() / n_samples_x
            nll_sum_diffusion += mcmc_energy(torch.from_numpy(x_pred).to(device)).sum() / n_samples_x

        # only plot samples of the last repeat otherwise it gets too much and plot only for some selected y
        if i in plot_y:

            utils.plot_density(x_true, nbins, limits=xlim, xticks=[-1, 0, 1], size=figsize, labelsize=labelsize,
                         fname=os.path.join(out_dir, 'posterior-mcmc-%d.svg'%i))

            utils.plot_density(x_pred, nbins, limits=xlim, xticks=[-1, 0, 1], size=figsize, labelsize=labelsize,
                     fname=os.path.join(out_dir, 'posterior-diffusion-%d.svg' % i))


        hist_mcmc = hist_mcmc_sum / hist_mcmc_sum.sum()
        hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
        hist_mcmc += epsilon
        hist_diffusion += epsilon
        hist_mcmc /= hist_mcmc.sum()
        hist_diffusion /= hist_diffusion.sum()

        kl2 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_diffusion))
        kl2_reverse = np.sum(scipy.special.rel_entr(hist_diffusion,hist_mcmc))
        kl2_sum += kl2
        kl2_vals.append(kl2)
        kl2_reverse_vals.append(kl2_reverse)
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
        {'KL2': kl2_vals, 'KL_reverse':kl2_reverse_vals, 'NLL_mcmc': nll_mcmc,'NLL_diffusion': nll_diffusion,'MSE':np.array(mse_score_vals)})
    df.to_csv(os.path.join(out_dir, 'results.csv'))
    print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
        
if __name__ == '__main__':

    # Define the required directory name
    #required_dir_name = 'main'
    #utils.check_wd(required_dir_name)

    # Create the parser
    parser = argparse.ArgumentParser(description="Load model parameters.")
    args = utils.diffusion_parser(parser)

    config_dir = 'config/'
    surrogate_dir = 'trained_models/scatterometry'
    gt_dir = 'data/gt_samples_scatterometry'

    # load config params
    config = yaml.safe_load(open(os.path.join(config_dir, "config_scatterometry.yml")))

    # load the forward model
    forward_model, forward_model_params = load_forward_model(surrogate_dir)

    #generate test set
    x_test,y_test = generate_dataset_scatterometry(forward_model,forward_model_params['a'],forward_model_params['b'],size=config['n_samples_y'])
#    x_test, y_test = generate_dataset_scatterometry(forward_model, forward_model_params.a, forward_model_params.b,
                                                    #size=args.n_samples_y)

    # define the score of the posterior
    score_posterior = lambda x, y: -energy_grad(x,
                                                lambda x: get_log_posterior(x, forward_model, forward_model_params['a'],
                                                                            forward_model_params['b'], y,
                                                                            forward_model_params['lambd_bd']))[0]

    model,loss_fn = utils.get_model_from_args(args, forward_model_params,score_posterior,forward_model, config)
    optimizer = Adam(model.sde.a.parameters(), lr=1e-4)
    log_dir = utils.set_directories(args)

    print('---------------------')
    model = train(model, optimizer, loss_fn, forward_model, forward_model_params['a'],forward_model_params['b'],forward_model_params['lambd_bd'], config['n_epochs'], batch_size=config['batch_size'],save_dir=args.train_dir, log_dir = log_dir)
    print('----------------------')
    evaluate(model, y_test, forward_model, forward_model_params['a'],forward_model_params['b'],forward_model_params['lambd_bd'], args.out_dir, gt_dir, n_samples_x=config['n_samples_x'], n_repeats = 2)