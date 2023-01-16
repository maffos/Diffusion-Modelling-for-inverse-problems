import torch
import numpy as np
import loss as ls
from scipy.stats import multivariate_normal
import utils
from libraries.sdeflow_light.lib import sdes
from torch import nn
from torch.optim import Adam
import nets
from sklearn.model_selection import train_test_split
import torchsde
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from tqdm import tqdm

def generate_dataset(n_samples,Sigma, random_state = 7):

    random_gen = torch.random.manual_seed(random_state)
    x = torch.randn(n_samples,xdim, generator = random_gen)
    y = f(x)
    noise = torch.randn(n_samples, ydim)
    noise = (Sigma@noise.T).T
    y+=noise
    #x = torch.from_numpy(x)
    #y = torch.from_numpy(y)
    return x.float(),y.float()

#toy function as forward problem
def f(x):
    return (A@x.T).T+b

def log_likelihood(y,x):

    mean = A@x+b
    return MultivariateNormal(mean,Sigma).log_prob(y)

def log_Z(y):
    mean = A@mu+b
    cov = Sigma+A@Lam@A.T

    return MultivariateNormal(mean,cov).log_prob(y)

def log_posterior(x, y):
    y_res = y-(A@mu+b)
    mean = Lam@A.T@Sigma_y_inv@y_res
    cov = Lam-Lam@A.T@Sigma_y_inv@A@Lam

    log_p1 = torch.log(multivariate_normal.pdf(x,mean,cov))
    log_p2 = log_likelihood(y,x)+prior.log_prob(x)-log_Z(y)

    assert torch.allclose(log_p1, log_p2), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)
    return log_p1

def score_q(x,y):
    y_res = y-(x@A.T+b)
    score_prior = -x
    score_likelihood = (y_res@Sigma_inv.T)@A
    return score_prior+score_likelihood

#PDE of the score. In the case of an Ohrnstein-Uhlenbeck process, the hessian of f is 0
def pde_loss(u,u_x,u_xx,u_t, f, grad_f, sigma, Hessian_f = 0.):

    fx_u = grad_f*u
    f_ux = f*u_x
    u_ux = u*u_x
    return (u_t - Hessian_f - fx_u -  f_ux- .5*sigma**2*(u_xx+2*u_ux))

def initial_condition_loss(u,x,y):

    return torch.mean(torch.sum(u-score_q(x,y))**2)

def dsm_loss(a,std,g,target, xdim):

    return ((a * std + target) ** 2).view(xdim, -1).sum(1, keepdim=False) / 2
def PINN_loss(model, x,y):

    if model.debias:
        t_ = model.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        t_.requires_grad = True
    else:
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)], requires_grad=True).to(x) * model.T
        t_.requires_grad = True
    t0 = torch.zeros_like(t_)
    x_t, target, std, g = model.base_sde.sample(t_, x, return_noise=True)
    u = model.a(x_t, t_, y)
    u_0 = model.a(x,t0,y)
    #a = u*g

    u_x = torch.autograd.grad(u.sum(), x_t, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_t,retain_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t_,retain_graph=True)[0]

    MSE_u = initial_condition_loss(u_0, x,y)+dsm_loss(u,std,g,target,x.size(0))
    grad_f = torch.ones_like(x_t)*-0.5*model.base_sde.beta(t_)
    MSE_pde = torch.mean(pde_loss(u,u_x,u_xx, u_t, model.base_sde.f(t_,x_t), grad_f, model.base_sde.g(t_,x_t))**2, dim = 1)
    loss = torch.mean(MSE_u+MSE_pde)

    return loss

def train(model,xs,ys, optim, num_epochs, batch_size=100):

    model.train()
    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):

        train_loader = utils.get_dataloader(xs, ys, batch_size)
        mean_loss = 0
        for x,y in train_loader():

            x = torch.ones_like(x, requires_grad=True)*x
            loss = PINN_loss(model,x,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            mean_loss += loss.data.item()

        mean_loss /= (xs.shape[0]//batch_size)
        prog_bar.set_description('loss: {:.4f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    return model

class SDE(nn.Module):

    def __init__(self, net, forward_sde, xdim, ydim, sde_type, T=1, t0=0):
        super(SDE, self).__init__()
        self.net = net
        self.forward_sde = forward_sde
        self.T = T
        self.t0 = t0
        self.xdim = xdim
        self.ydim = ydim
        self.sde_type = sde_type

    #x and y are passed as one tensor to be compatible with sdeint
    def f(self, t, x_t,y, lmbd = 0.):
        #unpack x and y from the inputs to pass them to the net
        #x_t = inputs[:,:xdim]
        #y = inputs[:,-ydim:]
        f_x = (1. - 0.5 * lmbd) * self.forward_sde.g(self.T - t, x_t) * self.net(x_t, self.T - t.squeeze(), y) - \
            self.forward_sde.f(self.T - t, x_t)
        #return torch.concat(f_x,y, dim=1)
        return f_x
    def g(self, t, x_t, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T - t, x_t)

def sample(model, y, x_T=None, dt=0.01, t0=0.,t1=1., n_samples = 500):
    """

    :param model: (Object) Object of class that implements functions f and g as drift and diffusion coefficient.
    :param y: (Tensor) of shape (1,ydim)
    :param x_T: (Tensor) of shape (n_samples, xdim)
    :param dt: (float) Timestep to integrate
    :param t0: (scalar)
    :param t1:
    :param n_samples:
    :return:
    """
    model.eval()
    with torch.no_grad():
        x_T = torch.randn((n_samples, model.xdim)) if x_T is None else x_T
        t = torch.Tensor([t0,t1])
        #concatenate x and y to use with sdeint
        y = y.repeat(n_samples,1)
        #xy = torch.concat(x_T,y,dim=1)
        x_pred = torchsde.sdeint(model, x_T.flatten(start_dim = 1), t, extra_solver_state = y, dt=dt)

    return x_pred

def evaluate(model, xs,ys):
    for x_true,y in zip(xs,ys):
        x_pred = sample(model, y)
        utils.make_image(x_pred, x_true, num_epochs = 1000, show_plot=True, savefig=False)

def plot_prior_pdf():
    x1 = np.linspace(-3,3,100)
    x2 = np.linspace(-3,3,100)
    logp = np.arange(100*100).reshape(100,100)
    for i,x in enumerate(x1):
        for j,y in enumerate(x2):
            logp[i,j] = prior.log_prob(torch.Tensor([x,y]))

    h = plt.contourf(x1, x2, logp)
    plt.axis('scaled')
    plt.colorbar()
    plt.show()

"""
def plot_likelihood():
    yy = np.linspace(-3, 3, 100)
    xx = np.linspace(-3, 3, 100)
    logp_yx1 = np.arange(100 * 100).reshape(100, 100)
    logp_yx2 = np.arange(100 * 100).reshape(100, 100)
    logp_y2x1 = np.arange(100 * 100).reshape(100, 100)
    logp_y2x2 = np.arange(100 * 100).reshape(100, 100)

    for i,y  in enumerate(yy):
        for j,x in enumerate(xx):
            logp = log_likelihood(torch.Tensor([x, y]))
            logp_y1x1[i, j] = log_likelihood(torch.Tensor([x, y]))
            logp_y1x2[i, j] = log_likelihood(torch.Tensor([x, y]))
            logp_y2x1[i, j] = log_likelihood(torch.Tensor([x, y]))
            logp_y2x2[i, j] = log_likelihood(torch.Tensor([x, y]))
"""

if __name__ == '__main__':

    #define parameters of the inverse problem
    epsilon = 1e-6
    xdim = 2
    ydim = 2
    A = torch.randn(ydim,xdim)
    b = torch.randn(ydim)
    scale = 2
    Sigma = scale*torch.eye(ydim)
    Lam = torch.eye(xdim)
    Sigma_inv = torch.linalg.inv(Sigma+epsilon*torch.eye(ydim))
    Sigma_y_inv = torch.linalg.inv(Sigma+A@Lam@A.T+epsilon*torch.eye(ydim))
    mu = torch.zeros(xdim)
    prior = MultivariateNormal(torch.Tensor(mu),torch.Tensor(Lam))

    #create data
    xs,ys = generate_dataset(n_samples=1000, Sigma = Sigma)
    #plot_prior_pdf()

    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=.8, random_state = 7)
    n_samples = 500
    embed_dim = 2
    net_params = {'input_dim': xdim+ydim,
                  'output_dim': xdim,
                  'hidden_layers': [64,128,256,128,64],
                  'embed_dim': embed_dim}

    forward_process = sdes.VariancePreservingSDE()
    """
    score_net = nn.Sequential(nn.Linear(xdim+ydim+embed_dim, 64),
                                nn.ReLU(),
                                nn.Linear(64, 128),
                                nn.ReLU(),
                                nn.Linear(128, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, xdim))
    """
    score_net = nets.TemporalMLP(**net_params)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias = True)
    optimizer = Adam(reverse_process.parameters(), lr = 1e-4)

    reverse_process = train(reverse_process,x_train,y_train, optimizer, num_epochs=1000)
    reverse_process = SDE(reverse_process.a, reverse_process.base_sde, sde_type = 'stratonovich')
    evaluate(reverse_process, x_test, y_test)