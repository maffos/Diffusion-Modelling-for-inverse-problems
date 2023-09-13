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

def evaluate(model, xs,ys, save_dir, n_samples = 2000, n_plots = 10):

    model.eval()
    with torch.no_grad():

        mse_score = 0
        nll_sample = 0
        nll_true = 0
        kl_div = 0
        # calculate MSE of score on test set
        t0 = torch.zeros(xs.shape[0], requires_grad=False).view(-1, 1)
        g_0 = model.base_sde.g(t0, xs)
        score_predict = model.a(xs, t0, ys)/g_0
        score_true = score_posterior(xs, ys)
        mse_score += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

        plot_ys = np.random.choice(ys.shape[0], size=n_plots,replace=False)
        prog_bar = tqdm(total=len(xs))
        for i,y in enumerate(ys):
            posterior = get_posterior(y)
            x_pred = get_grid(model.to(device), y.to(device), xdim=2, ydim=2, num_samples=n_samples)

            if i in plot_ys:
                # log_plot(prior)
                # fig, ax = conditional_pairplot(likelihood, condition=xs[0], limits=[[-3, 3], [-3, 3]])
                # fig.suptitle('Likelihood at x=(%.2f,%.2f)'%(xs[0,0],xs[0,1]))
                # fig.show()
                fig, ax = conditional_pairplot(posterior, condition=y, limits=[[-3, 3], [-3, 3]])
                fig.suptitle('Posterior at y=(%.2f,%.2f)' % (y[0], y[1]))
                fname = os.path.join(save_dir, 'posterior-true%d.png'%i)
                plt.savefig(fname)
                plt.close()
                # x_pred = sample(model, y=ys[0], dt = .005, n_samples=n_samples)
                fig, ax = pairplot([x_pred], limits=[[-3, 3], [-3, 3]])
                fig.suptitle('N=%d samples from the posterior at y=(%.2f,%.2f)' % (n_samples, y[0], y[1]))
                fname = os.path.join(save_dir, 'posterior-diffusion%d.png'%i)
                plt.savefig(fname)
                plt.close()

            x_true = posterior.sample((n_samples,))
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
            {'KL': np.array([kl_div]), 'NLL_true': np.array([nll_true]), 'NLL_diffusion': np.array([nll_sample]), 'MSE': np.array([mse_score])})
        df.to_csv(os.path.join(save_dir, 'results.csv'))
        print('MSE: %.4f, NLL of samples: %.4f, NLL of true samples: %.4f, KL Div: %.4f'%(mse_score,nll_sample,nll_true, kl_div))

if __name__ == '__main__':

    #create data
    xs,ys = generate_dataset(n_samples=100000)
    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=.8, random_state = 7)

    #define parameters
    src_dir = 'test'
    hidden_layers = [512,512,512]
    resume_training = False
    pde_loss = 'CFM'
    lam = .1
    lr = 1e-4

    #define models
    model = create_diffusion_model2(xdim,ydim, hidden_layers=hidden_layers)
    #loss_fn =PINNLoss2(initial_condition=score_posterior, boundary_condition=lambda x: -x, pde_loss=pde_loss, lam=lam)
    loss_fn = PINNLoss3(initial_condition=score_posterior, lam=.1,lam2=1., pde_loss = pde_loss)
    #loss_fn = ScoreFlowMatchingLoss(lam=.1)
    #loss_fn = PINNLoss3(initial_condition = score_posterior, lam = .1, lam2 = 1)
    #loss_fn = ErmonLoss(lam=0.1, pde_loss = 'FPE')
    optimizer = Adam(model.a.parameters(), lr = 1e-4)

    train_dir = os.path.join(src_dir,loss_fn.name, 'lam=0.1')
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


    model = train(model,x_train,y_train, optimizer, loss_fn, train_dir, log_dir, num_epochs=200, resume_training = resume_training)
    #we need to wrap the reverse SDE into an own class to use the integration method from torchsde
    #reverse_process = SDE(reverse_process.a, reverse_process.base_sde, xdim, ydim, sde_type='stratonovich')
    evaluate(model, x_test[:100], y_test[:100], out_dir, n_samples = 20000, n_plots=10)

    #model = create_diffusion_model2(xdim,ydim, hidden_layers=[512,512])
    #check_diffusion(model, n_samples=20000,num_plots=3)