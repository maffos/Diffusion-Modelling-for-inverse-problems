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
import scipy
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.distributions import MultivariateNormal
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define parameters of the forward and inverse problem
epsilon = 1e-6
xdim = 2
ydim = 2
# f is a shear by factor 0.5 in x-direction and tranlsation by (0.3, 0.5).
A = torch.Tensor([[1, 0.5], [0, 1]]).to(device)
b = torch.Tensor([0.3, 0.5]).to(device)
scale = .3  # measurement noise
Sigma = scale * torch.eye(ydim).to(device)
Lam = torch.eye(xdim).to(device)
Sigma_inv = 1 / scale * torch.eye(ydim).to(device)
Sigma_y_inv = torch.linalg.inv(Sigma + A @ Lam @ A.T + epsilon * torch.eye(ydim).to(device))
mu = torch.zeros(xdim).to(device)

class KillSignal():

    def __init__(self):
        self.message = 'Loss was either nan or too high'
        
def eval_mode(A,b,Sigma,Lam,Sigma_inv,Sigma_y_inv,mu):
    #A,b,Sigma,Lam,Sigma_inv,Sigma_y_inv,mu
    pass
    
def generate_dataset(n_samples, random_state = 7):

    random_gen = torch.random.manual_seed(random_state)
    x = torch.randn(n_samples,xdim, generator = random_gen).to(device)
    y = f(x)
    return x,y

def check_posterior(x,y,posterior, prior, likelihood, evidence):


    log_p1 = posterior.log_prob(x)
    log_p2 = prior.log_prob(x)+likelihood.log_prob(y)-evidence.log_prob(y)

    print(log_p2, log_p1)
    #assert torch.allclose(log_p1, log_p2, atol = 1e-5), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)

#affine function as forward problem
def f(x):
    return (A@x.T).T+b

def get_likelihood(x):

    mean = A.to(x)@x+b.to(x)
    return MultivariateNormal(mean,Sigma)

def get_evidence():
    mean = A@mu+b
    cov = Sigma+A@Lam@A.T

    return MultivariateNormal(mean,cov)

def get_posterior(y, device = device):
    y_res = y-(A@mu+b)
    mean = Lam@A.T@Sigma_y_inv@y_res
    cov = Lam-Lam@A.T@Sigma_y_inv@A@Lam

    return MultivariateNormal(mean.to(device),cov.to(device))

#analytical score of the posterior
def score_posterior(x,y):
    y_res = y-(x@A.T+b)
    score_prior = -x
    score_likelihood = (y_res@Sigma_inv.T)@A
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

def train(model,xs,ys, optim, loss_fn, save_dir, log_dir, num_epochs, batch_size=1000, debug = False ,threshold = 1e6):

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
        mean_loss = 0
        logger_info = {}

        for x,y in train_loader():

            x = torch.ones_like(x, requires_grad=True)*x
            t = sample_t(model,x).to(device)
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
            if torch.isnan(loss) or loss >= threshold:
                return KillSignal()
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

        logger.add_scalar('Train/Loss', mean_loss, i)
        for key, value in logger_info.items():
            logger.add_scalar('Train/' + key, value, i)
        prog_bar.set_description('loss: {:.4f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    current_model_path = os.path.join(save_dir, 'current_model.pt')
    if debug:
        print('Minimal t during training was %f'%t_min)
        plt.hist(ts, bins=100)
        plt.title('Histogram of sampled ts')
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
            posterior = get_posterior(y, device = 'cpu')

            for _ in range(n_repeats):
                x_pred = get_grid(model.to(device), y.to(device), xdim, ydim, num_samples=n_samples_x)
                x_true = posterior.sample((n_samples_x,)).to(device)

                # calculate MSE of score on test set
                t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1).to(device)
                g_0 = model.base_sde.g(t0, x_true)
                inflated_ys = torch.ones_like(x_true)*y
                score_predict = model.a(x_true.to(device), t0.to(device), inflated_ys.to(device)) / g_0
                score_true = score_posterior(x_true, inflated_ys)
                mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

                # generate histograms
                hist_true, _ = np.histogramdd(x_true.data.cpu().numpy(), bins=(nbins, nbins),
                                              range=((-4, 4), (-4, 4)))
                hist_diffusion, _ = np.histogramdd(x_pred, bins=(nbins, nbins),
                                                   range=((-4, 4), (-4, 4)))

                hist_true_sum += hist_true
                hist_diffusion_sum += hist_diffusion

                # calculate negaitve log likelihood of the samples
                nll_sum_true -= torch.mean(posterior.log_prob(x_true.to('cpu')))
                nll_sum_diffusion -= torch.mean(posterior.log_prob(torch.from_numpy(x_pred)))

            # only plot samples of the last repeat otherwise it gets too much and plot only for some fixed y
            if i in plot_ys:
                fig, ax = conditional_pairplot(posterior, condition=y.to('cpu'), limits=[[-4, 4], [-4, 4]])
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
        return np.mean(kl2_vals),np.mean(nll_true),np.mean(nll_diffusion),np.mean(mse_score_vals)
        
"""
def evaluate(model, xs,ys, save_dir, n_samples = 2000, n_plots = 10):

    if isinstance(model, KillSignal):
        return torch.inf,torch.inf,torch.inf,torch.inf
    else:
        model.eval()
        with torch.no_grad():

            mse_score = 0
            nll_sample = 0
            nll_true = 0
            kl_div = 0
            # calculate MSE of score on test set
            t0 = torch.zeros(xs.shape[0], requires_grad=False).view(-1, 1).to(device)
            g_0 = model.base_sde.g(t0, xs)
            score_predict = model.a(xs, t0, ys)/g_0
            score_true = score_posterior(xs, ys)
            mse_score += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

            plot_ys = [3,5,22,39,51,53,60,71,81,97]
            prog_bar = tqdm(total=len(xs))
            for i,y in enumerate(ys):

                posterior = get_posterior(y, device = 'cpu')
                x_pred = get_grid(model.to(device), y.to(device), xdim=2, ydim=2, num_samples=n_samples)
                x_true = posterior.sample((n_samples,))
                if i in plot_ys:
                # log_plot(prior)
                # fig, ax = conditional_pairplot(likelihood, condition=xs[0], limits=[[-3, 3], [-3, 3]])
                # fig.suptitle('Likelihood at x=(%.2f,%.2f)'%(xs[0,0],xs[0,1]))
                # fig.show()
                    fig, ax = conditional_pairplot(posterior, condition=y.to('cpu'), limits=[[-3, 3], [-3, 3]])
                    fig.suptitle('Posterior at y=(%.2f,%.2f)' % (y[0], y[1]))
                    fname = os.path.join(save_dir, 'posterior-true%d.png'%i)
                    plt.savefig(fname)
                    plt.close()
                    fig, ax = pairplot([x_true], limits=[[-3, 3], [-3, 3]])
                    fig.suptitle('N=%d samples from the posterior at y=(%.2f,%.2f)' % (n_samples, y[0], y[1]))
                    fname = os.path.join(save_dir, 'posterior-true-samples%d.png'%i)
                    plt.savefig(fname)
                    plt.close()
                # x_pred = sample(model, y=ys[0], dt = .005, n_samples=n_samples)
                    fig, ax = pairplot([x_pred], limits=[[-3, 3], [-3, 3]])
                    fig.suptitle('N=%d samples from the posterior at y=(%.2f,%.2f)' % (n_samples, y[0], y[1]))
                    fname = os.path.join(save_dir, 'posterior-diffusion%d.png'%i)
                    plt.savefig(fname)
                    plt.close()

            # calculate negative log likelihood and KL-Div of samples on test set
                nll_sample += -torch.mean(posterior.log_prob(torch.from_numpy(x_pred)))
            #calculate nll of true samples from posterior for reference
                nll_true += -torch.mean(posterior.log_prob(x_true))
                kl_div += nn.functional.kl_div(torch.from_numpy(x_pred), x_true).mean()
                prog_bar.set_description('NLL samples: %.4f NLL true %.4f'%(nll_sample,nll_true))
                prog_bar.update()

            mse_score /= xs.shape[0]
            nll_sample /= xs.shape[0]
            nll_true /= xs.shape[0]
            kl_div += xs.shape[0]

        df = pd.DataFrame(
                {'KL': np.array([kl_div.cpu()]), 'NLL_true': np.array([nll_true.cpu()]), 'NLL_diffusion': np.array([nll_sample.cpu()]), 'MSE': np.array([mse_score.cpu()])})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('MSE: %.4f, NLL of samples: %.4f, NLL of true samples: %.4f, KL Div: %.4f'%(mse_score,nll_sample,nll_true, kl_div))
        return mse_score,nll_sample,nll_true,kl_div
"""
if __name__ == '__main__':

    #create data
    xs,ys = generate_dataset(n_samples=100000)

    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=.9, random_state = 7)
    best_score = torch.inf
    #best_params={}
    best_loss = None
    params = {'loss_fn': ['PINNLoss4'],
    'lr': [1e-4],
    'lam': [.1,.001],
    'lam2':[.01,10.],
    'pde_loss': ['FPE'],
    'metric': ['L1'],
    'ic_metric':['L2'],
    'hidden_layers':['3layer']}
    src_dir = 'new_folder_structure'
    resume_training = False
    
    for param_configuration in utils.product_dict(**params):
        skip = False
        lr = param_configuration.pop('lr')
        num_layers = param_configuration.pop('hidden_layers')
        if num_layers == '2layer':
            hidden_layers = [512,512]
        elif num_layers == '3layer':
            hidden_layers = [512,512,512]
        elif num_layers== '4layer':
            hidden_layers = [512,512,512,512]
        else:
            hidden_layers = [512,512,512,512,512]
        loss_fn = param_configuration.pop('loss_fn')
        if loss_fn == 'PINNLoss2':
            loss_fn = PINNLoss2(initial_condition = score_posterior,boundary_condition=score_prior,**param_configuration)
        elif loss_fn == 'PINNLoss4':
            loss_fn = PINNLoss4(initial_condition= score_posterior,**param_configuration)
        elif loss_fn == 'PINNLoss3':
            loss_fn = PINNLoss3(initial_condition = score_posterior, **param_configuration)
        elif loss_fn == 'ErmonLoss':
            if 'lam2' in param_configuration.keys():
                _ = param_configuration.pop('lam2')
            if 'ic_metric' in param_configuration.keys():
                _ = param_configuration.pop('ic_metric')
            loss_fn = ErmonLoss(**param_configuration)
        else:
            raise ValueError('No correct loss fn specified')
        if param_configuration['metric'] == 'L1' and param_configuration['pde_loss'] == 'CFM':
            skip = True
        if not skip:
            model = create_diffusion_model2(xdim,ydim,hidden_layers)
            optimizer = Adam(model.a.parameters(), lr=lr)

            if loss_fn.name == 'ErmonLoss':
                train_dir = os.path.join(src_dir,param_configuration['pde_loss'], loss_fn.name, num_layers, param_configuration['metric'], 'lam:{}'.format(param_configuration['lam']))
            else:
                train_dir = os.path.join(src_dir,param_configuration['pde_loss'], loss_fn.name, num_layers, param_configuration['metric'],param_configuration['ic_metric'], 'lam:{}'.format(param_configuration['lam']),'lam2:{}'.format(param_configuration['lam2']))
            log_dir = os.path.join(train_dir, 'logs')
            
            if os.path.exists(log_dir) and not resume_training:
                shutil.rmtree(log_dir)
                
            out_dir = os.path.join(train_dir, 'results')
            if os.path.exists(out_dir) and not resume_training:
                shutil.rmtree(out_dir)
                
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print('-----------------')
            print(vars(loss_fn))
            model = train(model,x_train,y_train, optimizer, loss_fn, train_dir, log_dir, num_epochs=5000)
    #we need to wrap the reverse SDE into an own class to use the integration method from torchsde
    #reverse_process = SDE(reverse_process.a, reverse_process.base_sde, xdim, ydim, sde_type='stratonovich')
            kl_div,nll_true,nll_diffusion,mse = evaluate(model,y_test[:100], out_dir, n_samples_x = 30000, n_repeats = 10)
            print('KL: %.3f, NLL: %.3f, NLL-pred: %.3f, MSE: %.7f'%(kl_div,nll_true,nll_diffusion,mse))
            if kl_div < best_score:
                best_score = kl_div
                best_params = param_configuration
	   
    print('Best score:', best_score)
    print('Best params:')
    print(best_params) 
