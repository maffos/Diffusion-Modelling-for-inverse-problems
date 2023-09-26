import shutil

import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from sbi.analysis import pairplot, conditional_pairplot
import os

import utils
from include.sdeflow_light.lib import sdes
import sys
from sklearn.model_selection import train_test_split
#sys.path.append("/home/matthias/Uni/SoSe22/Master/Inverse-Modelling-of-Hemodynamics/")
from tqdm import tqdm
import nets
from models.diffusion import *
import pandas as pd
import numpy as np
import scipy
def generate_dataset(n_samples, random_state = 7):

    random_gen = torch.random.manual_seed(random_state)
    x = torch.randn(n_samples,xdim, generator = random_gen)
    y = f(x)
    #x = torch.from_numpy(x)
    #y = torch.from_numpy(y)
    return x.float(),y.float()

def check_posterior(x,y,posterior, prior, likelihood, evidence):


    log_p1 = posterior.log_prob(x)
    log_p2 = prior.log_prob(x)+likelihood.log_prob(y)-evidence.log_prob(y)

    print(log_p2, log_p1)
    #assert torch.allclose(log_p1, log_p2, atol = 1e-5), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)

def get_grid(sde, cond1,dim, n=4, num_samples = 2000, num_steps=200, transform=None,
             mean=0, std=1, clip=True):

    cond = torch.zeros(num_samples,dim)
    cond += cond1
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, dim)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1) * sde.T
    ones = torch.ones(num_samples, 1)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, cond)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    y0 = y0.data.cpu().numpy()
    return y0

#toy function as forward problem
def f(x):
    return (A@x.T).T+b

def get_likelihood(x):

    mean = A@x+b
    return MultivariateNormal(mean,Sigma)

def get_evidence():
    mean = A@mu+b
    cov = Sigma+A@Lam@A.T

    return MultivariateNormal(mean,cov)

def get_posterior(y):
    y_res = y-(A@mu+b)
    mean = Lam@A.T@Sigma_y_inv@y_res
    cov = Lam-Lam@A.T@Sigma_y_inv@A@Lam

    return MultivariateNormal(mean,cov)

#analytical score of the posterior
def score_posterior(x,y):
    y_res = y-(x@A.T+b)
    score_prior = -x
    score_likelihood = (y_res@Sigma_inv.T)@A
    return score_prior+score_likelihood

def evaluate(model,ys, out_dir, n_samples_x=5000,n_repeats=10, epsilon=1e-10):
    n_samples_y = ys.shape[0]
    model.eval()
    with (torch.no_grad()):
        nll_diffusion = []
        nll_true = []
        kl2_sum = 0.
        mse_score_vals = []
        kl2_vals = []
        nbins = 200
        # hardcoded ys to plot the posterior for reproducibility (otherwise we would get ~2000 plots)
        plot_ys = [3,5,22,39,51,53,60,71,81,97]

        prog_bar = tqdm(total=n_samples_y)
        for i, y in enumerate(ys):
            # testing
            hist_true_sum = np.zeros((nbins, nbins))
            hist_diffusion_sum = np.zeros((nbins, nbins))
            nll_sum_true = 0
            nll_sum_diffusion = 0
            mse_score_sum = 0
            posterior = get_posterior(y)

            for _ in range(n_repeats):
                x_pred = get_grid(model, y, xdim, ydim, num_samples=n_samples_x)
                x_true = posterior.sample((n_samples_x,))

                # calculate MSE of score on test set
                t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1)
                g_0 = model.base_sde.g(t0, x_true)
                inflated_ys = torch.ones_like(x_true)*y
                score_predict = model.a(x_true, t0, inflated_ys) / g_0
                score_true = score_posterior(x_true, inflated_ys)
                mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

                # generate histograms
                hist_true, _ = np.histogramdd(x_true, bins=(nbins, nbins),
                                              range=((-4, 4), (-4, 4)))
                hist_diffusion, _ = np.histogramdd(x_pred, bins=(nbins, nbins),
                                                   range=((-4, 4), (-4, 4)))

                hist_true_sum += hist_true
                hist_diffusion_sum += hist_diffusion

                # calculate negaitve log likelihood of the samples
                nll_sum_true -= torch.mean(posterior.log_prob(x_true))
                nll_sum_diffusion -= torch.mean(posterior.log_prob(torch.from_numpy(x_pred)))

            # only plot samples of the last repeat otherwise it gets too much and plot only for some fixed y
            if i in plot_ys:
                fig, ax = conditional_pairplot(posterior, condition=y, limits=[[-4, 4], [-4, 4]])
                fig.suptitle('Posterior at y=(%.2f,%.2f)' % (y[0], y[1]))
                fname = os.path.join(out_dir, 'posterior-true%d.svg' % i)
                plt.savefig(fname)
                plt.close()

                #utils.plot_density(x_pred, nbins = 80, title='PINN-Loss',
                                  # limits = (-4,4), fname = os.path.join(out_dir,'posterior-diffusion-limits%d.svg'%i))
                #utils.plot_density(x_pred, nbins=80, title='PINN-Loss', fname=os.path.join(out_dir,'posterior-diffusion-nolimits%d.svg'%i))
                pairplot([x_pred], limits = [[-4,4],[-4,4]])
                fig.suptitle('PINN-Loss')
                fname = os.path.join(out_dir, 'posterior-pinn-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                pairplot([x_pred])
                fig.suptitle('PINN-Loss')
                fname = os.path.join(out_dir, 'posterior-pinn-nolimits-%d.png' % i)
                plt.savefig(fname)
                plt.close()
            hist_true = hist_true_sum / hist_true_sum.sum()
            hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
            hist_true += epsilon
            hist_diffusion += epsilon
            #re-normalize after adding epsilon
            hist_true /= hist_true.sum()
            hist_diffusion /= hist_diffusion.sum()

            kl2 = np.sum(scipy.special.rel_entr(hist_true, hist_diffusion))
            kl2_sum += kl2
            kl2_vals.append(kl2)
            nll_true.append(nll_sum_true.item() / n_repeats)
            nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
            mse_score_vals.append(mse_score_sum.item()/n_repeats)
            prog_bar.set_description(
                'KL_diffusion: {:.3f}'.format(np.mean(kl2_vals)))
            prog_bar.update()

        prog_bar.close()
        kl2_vals = np.array(kl2_vals)
        kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
        nll_true = np.array(nll_true)
        nll_diffusion = np.array(nll_diffusion)
        df = pd.DataFrame(
            {'KL2': kl2_vals, 'NLL_true': nll_true, 'NLL_diffusion': nll_diffusion, 'MSE': np.array(mse_score_vals)})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
        
if __name__ == '__main__':
    
    # define parameters of the inverse problem
    epsilon = 1e-6
    xdim = 2
    ydim = 2
    # f is a shear by factor 0.5 in x-direction and tranlsation by (0.3, 0.5).
    A = torch.Tensor([[1, 0.5], [0, 1]])
    b = torch.Tensor([0.3, 0.5])
    scale = .3
    Sigma = scale * torch.eye(ydim)
    Lam = torch.eye(xdim)
    Sigma_inv = 1 / scale * torch.eye(ydim)
    Sigma_y_inv = torch.linalg.inv(Sigma + A @ Lam @ A.T + epsilon * torch.eye(ydim))
    mu = torch.zeros(xdim)

    # create data
    xs, ys = generate_dataset(n_samples=100000)
    x_train,x_test,y_train,y_test = train_test_split(xs, ys, train_size=.9, random_state=7)
    src_dir = 'examples/linearModel/results'
    for root, dirs, files in os.walk(src_dir):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            chkpnt_path = os.path.join(subfolder_path, 'current_model.pt')
            if os.path.isfile(chkpnt_path):
                out_dir = os.path.join(subfolder_path, 'results2')
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                model = create_diffusion_model2(xdim,ydim,hidden_layers=[512,512,512])
                checkpoint = torch.load(chkpnt_path, map_location=torch.device(device))
                model.a.load_state_dict(checkpoint)
                evaluate(model,y_test[:100], out_dir, n_samples_x=30000,n_repeats=10)