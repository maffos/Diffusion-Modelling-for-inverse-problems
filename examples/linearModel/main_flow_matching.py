import matplotlib.pyplot as plt
from sbi.analysis import pairplot, conditional_pairplot
import os
import shutil
import utils
from losses import *
from include.sdeflow_light.lib.sdes import VariancePreservingSDE
from nets import MLP
from sklearn.model_selection import train_test_split
from torchdiffeq import odeint
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

class CNF(nn.Module):

    def __init__(self, net,xdim=2,ydim=2):
        super(CNF, self).__init__()
        self.net = net
        self.xdim=xdim
        self.ydim=ydim

    def forward(self, t, x):
        y = x[-1,:self.ydim]
        x = x[:-1,:self.xdim]
        y_batched = y.repeat(len(x),1)
        t_batched = t.repeat(len(x),1)
        x_new = self.net(x,t_batched,y_batched)
        pad_x = torch.zeros(len(x),np.abs(self.ydim-self.xdim))
        x_new = torch.cat([x_new,pad_x],dim=1)
        pad_y = torch.zeros(np.abs(self.ydim-self.xdim))
        y = torch.cat([y,pad_y])
        x_new = torch.cat([x_new,y.view(1,-1)],dim=0)

        return x_new
def train(model, forward_sde, xs,ys, optim,loss_fn,out_dir,log_dir,batch_size = 1000,num_epochs=1000):
    model.train()
    logger = SummaryWriter(log_dir)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):

        train_loader = utils.get_dataloader_noise(xs, ys, scale, batch_size)
        mean_loss = 0

        for x, y in train_loader():

            #the std of p(x_t|x_0) where t=0 becomes nan. Therefore we subtract by 1e-5 to avoid t=1, since we sample at 1-t
            t = torch.rand([batch_size,1]).to(x)-1e-5
            t[torch.where(t<0)]+=1e-5
            loss = loss_fn(model, forward_sde, x, t, y).mean()
            mean_loss += loss.data.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        mean_loss /= (xs.shape[0] // batch_size)
        logger.add_scalar('Train/Loss', mean_loss, i)

        prog_bar.set_description('loss: {:.4f}'.format(mean_loss))
        prog_bar.update()


    prog_bar.close()

    current_model_path = os.path.join(out_dir, 'current_model.pt')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(model.state_dict(), current_model_path)
    return model

def sample(model,y,num_samples,dt=.01):
    x0 = torch.randn(num_samples,xdim).to(device)
    t_ = torch.linspace(0,1,int(1/dt))
    x = torch.cat([x0,y.view(1,-1)],dim=0)
    x_t = odeint(model,x,t_)

    return x_t

def evaluate(model,xs,ys, save_dir, n_samples = 2000, n_plots = 10):

    model.eval()
    ode_net = CNF(model)
    with torch.no_grad():

        nll_sample = 0
        nll_true = 0
        kl_div = 0

        plot_ys = np.random.choice(ys.shape[0], size=n_plots, replace=False)
        prog_bar = tqdm(total=len(xs))
        for i, y in enumerate(ys):
            posterior = get_posterior(y)
            trajectory = sample(ode_net.to(device),y,num_samples=n_samples)
            x_pred = trajectory[-1,:n_samples,:xdim]

            if i in plot_ys:
                fig, ax = conditional_pairplot(posterior, condition=y, limits=[[-3, 3], [-3, 3]])
                fig.suptitle('Posterior at y=(%.2f,%.2f)' % (y[0], y[1]))
                fname = os.path.join(save_dir, 'posterior-true%d.png' % i)
                plt.savefig(fname)
                plt.close()
                fig, ax = pairplot([x_pred], limits=[[-3, 3], [-3, 3]])
                fig.suptitle('N=%d samples from the posterior at y=(%.2f,%.2f)' % (n_samples, y[0], y[1]))
                fname = os.path.join(save_dir, 'posterior-diffusion%d.png' % i)
                plt.savefig(fname)
                plt.close()

            x_true = posterior.sample((n_samples,))
            nll_sample += -torch.mean(posterior.log_prob(x_pred))
            # calculate nll of true samples from posterior for reference
            nll_true += -torch.mean(posterior.log_prob(x_true))
            kl_div += nn.functional.kl_div(x_pred, x_true).mean()
            prog_bar.set_description('NLL samples: %.4f NLL true %.4f' % (nll_sample, nll_true))
            prog_bar.update()

        nll_sample /= xs.shape[0]
        nll_true /= xs.shape[0]
        kl_div += xs.shape[0]

        df = pd.DataFrame(
            {'KL': np.array([kl_div]), 'NLL_true': np.array([nll_true]), 'NLL_diffusion': np.array([nll_sample])})
        df.to_csv(os.path.join(save_dir, 'results.csv'))
        print(' NLL of samples: %.4f, NLL of true samples: %.4f, KL Div: %.4f' % ( nll_sample, nll_true, kl_div))

if __name__ == '__main__':

    #create data
    xs,ys = generate_dataset(n_samples=100000)
    train_dir = 'examples/linearModel/FlowMatching'
    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=.9, random_state = 7)
    net_params = {'input_dim': xdim + ydim + 1,
                  'output_dim': xdim,
                  'hidden_layers': [512,512],
                  'activation': nn.Tanh()}
    forward_process = VariancePreservingSDE()
    model = MLP(**net_params).to(device)
    optimizer = Adam(model.parameters(), lr = 1e-4)
    loss_fn = DFMLoss()
    out_dir = os.path.join(train_dir, 'results')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(train_dir, 'logs')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)


    model = train(model,forward_process,x_train,y_train, optimizer,loss_fn, train_dir, log_dir, num_epochs=500)
    evaluate(model, x_test[:100], y_test[:100], out_dir, n_samples = 10000, n_plots=10)