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

def generate_dataset(n_samples,Sigma):
    x = np.random.randn(n_samples,xdim)
    y = f(x)
    noise = np.random.randn(n_samples, ydim)
    noise = (Sigma@noise.T).T
    y+=noise
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
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
    y_res = y-(A@x+b)
    score_prior = -x
    score_likelihood = A.T@Sigma_inv@y_res
    return score_prior+score_likelihood

#PDE of the score. In the case of an Ohrnstein-Uhlenbeck process, the hessian of f is 0
def pde_loss(u,u_x,u_xx,u_t, f, grad_f, sigma, Hessian_f = 0):

    return (u_t - Hessian_f - grad_f@u - f@u_x - .5*sigma**2*(u_xx+2*u@u_x))

def initial_condition_loss(u,x,y):

    return torch.mean(torch.sum(u-score_q(x,y))**2)

def dsm_loss(a,std,g,target, xdim):

    return ((a * std / g + target) ** 2).view(xdim, -1).sum(1, keepdim=False) / 2
#code taken from https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def PINN_loss(model, x,y):
    """
    pinn loss
    """
    if model.debias:
        t_ = model.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
    else:
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * model.T
    #todo: x_t should be 3D afaik
    x_t, target, std, g = model.base_sde.sample(t_, x, return_noise=True)
    a = model.a(x_t, t_, y)

    u = a/g
    u_x = torch.autograd.grad(u, x_t, create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_t, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t_)[0]
    MSE_u = initial_condition_loss(u, x,y)+dsm_loss(a,std,g,target,x.size(0))
    grad_f = -0.5*model.base_sde.beta(t_)
    MSE_pde = torch.mean(pde_loss(u,u_x,u_xx, u_t, model.base_sde.f(t_,x_t), grad_f, model.base_sde.g(t_,x_t)))

    return MSE_u+MSE_pde

def train(model,xs,ys, optim, num_epochs):

    model.train()
    for i in range(num_epochs):
        train_loader = utils.get_dataloader(xs, ys, batch_size=100)

        for x,y in train_loader():

            loss = PINN_loss(model,x,y)
            optim.zero_grad()
            loss.backward()
            optim.step()

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
    A = np.random.randn(ydim,xdim)
    b = np.random.randn(ydim)
    scale = 2
    Sigma = scale*np.eye(ydim)
    Lam = np.eye(xdim)
    Sigma_inv = np.linalg.inv(Sigma+epsilon*np.eye(ydim))
    Sigma_y_inv = np.linalg.inv(Sigma+A@Lam@A.T+epsilon*np.eye(ydim))
    mu = np.zeros(xdim)
    prior = MultivariateNormal(torch.Tensor(mu),torch.Tensor(Lam))

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