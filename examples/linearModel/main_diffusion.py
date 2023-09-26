import matplotlib.pyplot as plt
from sbi.analysis import pairplot, conditional_pairplot
import os
import shutil
import utils
from models.diffusion import *
from losses import *
from sklearn.model_selection import train_test_split
#import torchsde
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import scipy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def generate_dataset(n_samples, random_state = 7):

    random_gen = torch.random.manual_seed(random_state)
    x = torch.randn(n_samples,xdim, generator = random_gen)
    y = f(x)
    return x.to(device),y.to(device)

def get_dataloader_dsm(scale,batch_size, nx,ny,nt):

    eps = 1e-6
    xs = []
    ys = []
    x_probe = torch.randn(ny,xdim)
    y_probe = f(x_probe)
    for y in y_probe:
        posterior = get_posterior(y)
        xs.append(posterior.sample((nx,)).repeat(nt,1))
        ys.append(y.repeat(nx*nt,1))

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys,dim=0)
    ys += torch.randn_like(ys) * scale
    ts = eps + torch.rand([nx*nt*ny,1])
    ts[torch.where(ts > 1)] = model.T - eps

    perm = torch.randperm(len(xs))
    xs = xs[perm]
    ys = ys[perm]
    ts = ts[perm]
    def epoch_data_loader():
        for i in range(0, nx*ny*nt, batch_size):
            yield xs[i:i+batch_size].to(device), ys[i:i+batch_size].to(device), ts[i:i+batch_size].to(device)

    return epoch_data_loader
def check_posterior(x,y,posterior, prior, likelihood, evidence):


    log_p1 = posterior.log_prob(x)
    log_p2 = prior.log_prob(x)+likelihood.log_prob(y)-evidence.log_prob(y)

    print(log_p2, log_p1)
    #assert torch.allclose(log_p1, log_p2, atol = 1e-5), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)

def check_diffusion(model, n_samples, num_plots):
    for i in range(num_plots):
        x = torch.randn(2)
        y = f(x)
        posterior = get_posterior(y)
        x_0 = posterior.sample((n_samples,))
        T = torch.ones((x_0.shape[0],1))
        x_T, target_T, std_T, g_T = model.base_sde.sample(T, x_0, return_noise=True)
        x_prior = torch.randn((n_samples,2))
        kl_div = nn.functional.kl_div(x_T, x_prior)
        fig, ax = pairplot([x_0], condition=y, limits=[[-3, 3], [-3, 3]])
        fig.suptitle('Samples from the Posterior at y=(%.2f,%.2f)' % (y[0], y[1]))
        fname = 'posterior-true%d.png' % i
        plt.savefig(fname)
        plt.show()
        plt.close()
        heatmap, xedges, yedges = np.histogram2d(x_T[:,0].data.numpy(), x_T[:,1].data.numpy(), bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title('Samples from the prior by running the SDE')
        fname = 'prior-diffusion%d.png'%i
        plt.savefig(fname)
        plt.show()
        plt.close()
        heatmap, xedges, yedges = np.histogram2d(x_prior[:,0].data.numpy(), x_prior[:,1].data.numpy(), bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title('Samples from the true prior')
        fname = 'prior-true%d.png'%i
        plt.savefig(fname)
        plt.show()
        plt.close()
        print('KL Divergence = %.4f'%kl_div)

#affine function as forward problem
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
    score_likelihood = y_res@Sigma_inv@A.T
    return score_prior+score_likelihood

def log_plot(dist):
    x1 = np.linspace(-3,3,100)
    x2 = np.linspace(-3,3,100)
    logp = np.arange(100*100).reshape(100,100)
    for i,x in enumerate(x1):
        for j,y in enumerate(x2):
            logp[i,j] = dist.log_prob(torch.Tensor([x,y]))

    fig, axes = plt.subplots(2,2)
    h = axes[0,1].contourf(x1, x2, logp)
    plt.axis('scaled')
    plt.colorbar()
    plt.title('Prior Distribution')
    plt.show()

def train(model,xs,ys, optim, loss_fn, save_dir, log_dir, num_epochs, batch_size=1000, debug = False, resume_training = False):

    model.train()
    logger = SummaryWriter(log_dir)
    if debug:
        #track the minimum sampled t for debugging
        t_min = torch.inf
        min_epoch = 0
        ts = []
    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):

        train_loader = utils.get_dataloader_noise(xs, ys,scale,batch_size)
        #train_loader = get_dataloader_dsm(scale,batch_size,200,100,5)
        mean_loss = 0
        logger_info = {}

        for x,y in train_loader():

            x = torch.ones_like(x, requires_grad=True).to(x)*x
            #t = torch.ones_like(t,requires_grad=True).to(t)*t
            t = sample_t(model,x)
            if debug:
                ts+=t.flatten().tolist()
                if torch.min(t) < t_min:
                    t_min = torch.min(t)
                    min_epoch = i
            loss,loss_info = loss_fn(model,x,t,y)
            mean_loss += loss.data.item()

            for key,value in loss_info.items():
                try:
                    logger_info[key] +=value.item()
                except:
                    logger_info[key] = value.item()
        
            if debug:
                if torch.isnan(loss):
                    for key,value in loss_info.items():
                        print(key + ':' + str(value))
                    raise ValueError('Loss is nan, min sampled t was {}. Minimal t during training was {} in epoch {}. Current Epoch: {}'.format(torch.min(t),t_min, min_epoch, i))

            optim.zero_grad()
            loss.backward()
            optim.step()


        mean_loss /= (xs.shape[0]//batch_size)

        for key in logger_info.keys():
            logger_info[key] /= (xs.shape[0]//batch_size)

        if resume_training:
            logger.add_scalar('Train/Loss', mean_loss, i+5000)
            for key, value in logger_info.items():
                logger.add_scalar('Train/' + key, value, i+5000)

        else:
            logger.add_scalar('Train/Loss', mean_loss, i)
            for key, value in logger_info.items():
                logger.add_scalar('Train/' + key, value, i)

        prog_bar.set_description('loss: {:.4f}'.format(mean_loss))
        prog_bar.update()
    prog_bar.close()

    current_model_path = os.path.join(save_dir, 'current_model.pt')
    if debug:
        print('Minimal t during training was %f'%t_min)
        plt.hist(ts, bins=100)
        plt.title('Histogram of sampled ts (N=%d)'%num_epochs*100000)
        plt.xlabel('t')
        plt.savefig(os.path.join(save_dir, 'ts.png'))
        plt.show()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.a.state_dict(), current_model_path)
    return model

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
                inflated_ys = torch.ones_like(x_true) * y
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
                fname = os.path.join(out_dir, 'posterior-true%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred], limits = [[-4,4],[-4,4]])
                fig.suptitle('PINN-Loss')
                fname = os.path.join(out_dir, 'posterior-pinn-%d.png' % i)
                plt.savefig(fname)
                plt.close()

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
        df = pd.DataFrame(
            {'KL2': kl2_vals, 'NLL_true': nll_true, 'NLL_diffusion': nll_diffusion, 'MSE': np.array(mse_score_vals)})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)

if __name__ == '__main__':

    #create data
    xs,ys = generate_dataset(n_samples=100000)
    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=.9, random_state = 7)

    #define parameters
    src_dir = 'ScoreFPE/3layer/L1-pde/L2-IC'
    hidden_layers = [512,512,512]
    resume_training = False
    pde_loss = 'FPE'
    lam = .1
    lam2 = 1.
    lr = 1e-4
    metric = 'L1'

    #define models
    model = create_diffusion_model2(xdim,ydim, hidden_layers=hidden_layers)
    #loss_fn =PINNLoss2(initial_condition=score_posterior, boundary_condition=lambda x: -x, pde_loss=pde_loss, lam=lam)
    loss_fn = PINNLoss4(initial_condition=score_posterior, lam=lam,lam2=lam2, pde_loss = pde_loss, metric = metric)
    #loss_fn = ScoreFlowMatchingLoss(lam=.1)
    #loss_fn = PINNLoss3(initial_condition = score_posterior, lam = .1, lam2 = 1)
    #loss_fn = ErmonLoss(lam=0.1, pde_loss = 'FPE')
    optimizer = Adam(model.a.parameters(), lr = lr)

    train_dir = os.path.join(src_dir,loss_fn.name, 'lam:{}'.format(lam), 'lam2:{}'.format(lam2))
    if resume_training:
        model.a.load_state_dict(torch.load(os.path.join(train_dir,'current_model.pt'),map_location=torch.device(device)))
        out_dir = os.path.join(train_dir, 'results_resume')
    else:
        out_dir = os.path.join(train_dir, 'results')

    if os.path.exists(out_dir) and not resume_training:
        shutil.rmtree(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_dir = os.path.join(train_dir, 'logs')

    if os.path.exists(log_dir) and not resume_training:
        shutil.rmtree(log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    model = train(model,x_train,y_train, optimizer, loss_fn, train_dir, log_dir, num_epochs=5000, resume_training = resume_training)
    #we need to wrap the reverse SDE into an own class to use the integration method from torchsde
    #reverse_process = SDE(reverse_process.a, reverse_process.base_sde, xdim, ydim, sde_type='stratonovich')
    kl_div, nll, nll_diffusion, mse = evaluate(model, y_test[:100], out_dir, n_samples_x = 30000, n_repeats = 5)

    print('KL: %.3f, NLL: %.3f, NLL-diffusion: %.3f, MSE: %.7f'%(kl_div,nll,nll_diffusion,mse))
    #model = create_diffusion_model2(xdim,ydim, hidden_layers=[512,512])
    #check_diffusion(model, n_samples=20000,num_plots=3)