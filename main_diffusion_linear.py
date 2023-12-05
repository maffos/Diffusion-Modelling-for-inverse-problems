import os
import yaml
import utils
from datasets import get_dataloader_linear,generate_dataset_linear
from linear_problem import LinearForwardProblem
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import scipy
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model,optim, loss_fn, forward_model_params,save_dir, log_dir, num_epochs, batch_size, xs, ys, resume_training = False):

    model.sde.train()
    logger = SummaryWriter(log_dir)
    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):

        epoch_data_loader = get_dataloader_linear(xs, ys,forward_model_params['scale'],batch_size)
        loss,logger_info = model.train_epoch(optim,loss_fn,epoch_data_loader)

        logger.add_scalar('Train/Loss', loss, i)
        for key, value in logger_info.items():
            logger.add_scalar('Train/' + key, value, i)

        if resume_training:
            logger.add_scalar('Train/Loss', loss, i+5000)
            for key, value in logger_info.items():
                logger.add_scalar('Train/' + key, value, i+5000)

        else:
            logger.add_scalar('Train/Loss', loss, i)
            for key, value in logger_info.items():
                logger.add_scalar('Train/' + key, value, i)

        prog_bar.set_description('loss: {:.4f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    current_model_path = os.path.join(save_dir, 'current_model.pt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.sde.a.state_dict(), current_model_path)
    return model

def evaluate(model,ys, forward_model, out_dir, plot_ys, n_samples_x=5000,n_repeats=10, epsilon=1e-10, xlim = (-3.5,3.5),nbins = 75, figsize = (12,12), labelsize = 30):

    n_samples_y = ys.shape[0]
    model.sde.eval()
    with (torch.no_grad()):
        nll_diffusion = []
        nll_true = []
        kl2_sum = 0.
        mse_score_vals = []
        kl2_vals = []

        prog_bar = tqdm(total=n_samples_y)
        for i, y in enumerate(ys):
            hist_true_sum = np.zeros((nbins, nbins))
            hist_diffusion_sum = np.zeros((nbins, nbins))
            nll_sum_true = 0
            nll_sum_diffusion = 0
            mse_score_sum = 0
            posterior = forward_model.get_posterior(y)

            for _ in range(n_repeats):
                x_pred = model(y, num_samples=n_samples_x)
                x_true = posterior.sample((n_samples_x,))

                # calculate MSE of score on test set
                t_0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1)
                g_0 = model.sde.base_sde.g(t_0, x_true)
                inflated_ys = torch.ones_like(x_true) * y
                score_predict = model.sde.a(x_true, inflated_ys, t_0) / g_0
                score_true = forward_model.score_posterior(x_true, inflated_ys)
                mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

                # generate histograms
                hist_true, _ = np.histogramdd(x_true, bins=(nbins, nbins),
                                              range=(xlim, xlim))
                hist_diffusion, _ = np.histogramdd(x_pred, bins=(nbins, nbins),
                                                   range=(xlim, xlim))

                hist_true_sum += hist_true
                hist_diffusion_sum += hist_diffusion

                # calculate negative log likelihood of the samples
                nll_sum_true -= torch.mean(posterior.log_prob(x_true))
                nll_sum_diffusion -= torch.mean(posterior.log_prob(torch.from_numpy(x_pred)))

            # only plot samples of the last repeat otherwise it gets too much and plot only for some fixed y
            if i in plot_ys:

                utils.plot_density(x_true, nbins, limits=xlim, xticks=xlim, size=figsize,
                                   labelsize=labelsize,
                                   fname=os.path.join(out_dir, 'posterior-true-%d.svg' % i), show_mean = True)

                utils.plot_density(x_pred, nbins, limits=xlim, xticks=xlim, size=figsize,
                                   labelsize=labelsize,
                                   fname=os.path.join(out_dir, 'posterior-diffusion-%d.svg' % i), show_mean = True)

            hist_true = hist_true_sum / hist_true_sum.sum()
            hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
            hist_true += epsilon
            hist_diffusion += epsilon
            # re-normalize after adding epsilon
            hist_true /= hist_true.sum()
            hist_diffusion /= hist_diffusion.sum()

            kl2 = np.sum(scipy.special.rel_entr(hist_true, hist_diffusion))
            kl2_sum += kl2
            kl2_vals.append(kl2)
            nll_true.append(nll_sum_true.item() / n_repeats)
            nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
            mse_score_vals.append(mse_score_sum.item() / n_repeats)
            prog_bar.set_description(
                'KL_diffusion: {:.3f}'.format(np.mean(kl2_vals)))
            prog_bar.update()

        prog_bar.close()
        kl2_vals = np.array(kl2_vals)
        kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
        nll_true = np.array(nll_true)
        nll_diffusion = np.array(nll_diffusion)
        nlpd = np.abs(nll_true-nll_diffusion)
        df = pd.DataFrame(
            {'KL2': kl2_vals, 'NLL_true': nll_true, 'NLL_diffusion': nll_diffusion, 'MSE': np.array(mse_score_vals)})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
        return kl2_vals.mean(),nlpd.mean(),mse_score_vals.mean()

if __name__ == '__main__':

    # Create the parser and parse arguments
    parser = argparse.ArgumentParser(description="Load model parameters.")
    args = utils.diffusion_parser(parser)

    # load config params
    config_dir = 'config/'
    config = yaml.safe_load(open(os.path.join(config_dir, "config_linear.yml")))

    #load linear forward problem
    f = LinearForwardProblem()

    #create data
    xs,ys = generate_dataset_linear(f.xdim, f, config['dataset_size'])
    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=config['train_size'], random_state = config['random_state'])

    model,loss_fn = utils.get_model_from_args(vars(args), vars(f),f.score_posterior,f, config)

    if config['resume_training']:
        model.sde.a.load_state_dict(
            torch.load(os.path.join(args.train_dir, 'current_model.pt'), map_location=torch.device(device)))

    log_dir = utils.set_directories(args.train_dir, args.out_dir, config['resume_training'])

    optimizer = Adam(model.sde.a.parameters(), lr=config['lr'])

    model = train(model, optimizer, loss_fn, vars(f), args.train_dir, log_dir, config['n_epochs'], config['batch_size'], x_train,y_train, resume_training = config['resume_training'])
    _ = evaluate(model, y_test[:config['n_samples_y']], f, args.out_dir, config['plot_ys'], n_samples_x = config['n_samples_x'], n_repeats = 2)