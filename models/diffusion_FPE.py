import torch
import numpy as np
import loss as ls
from scipy.stats import multivariate_normal
from libraries.sdeflow_light.lib import sdes
import models
#toy function as forward problem
def f(x):
    return A@x+b

def log_posterior(x, y):
    Lam = np.eye(2,2)
    Sigma_y_inv = np.linalg.inv(Sigma+A@Lam@A.T)
    y_res = y-b
    mean = Lam@A.T@Sigma_y_inv@y_res
    cov = Lam-Lam@A.T@Sigma_y_inv@A@Lam

    return torch.log(multivariate_normal.pdf(x,mean,cov))

def generate_dataset(n_samples):
    x = np.random.normal(size=(n_samples,2))
    y = (f(x.T)).T
    return x,y

#PDE of the score. In the case of an Ohrnstein-Uhlenbeck process, the hessian of f is 0
def pde_score(net, x, t, f, grad_f, sigma, Hessian_f = 0):

    u = net(x,t)
    u_x = torch.autograd.grad(u, x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, create_graph=True)[0]

    return (u_t - Hessian_f - grad_f(x)@u - f(x) @ u_x - .5*sigma**2*(u_xx+2*u*u_x))

def train(model,xs,ys):

if __name__ == '__main__':
    scale = 2
    epsilon = .01
    A = np.randn(2, 2)
    #make the matrix A psd and multiply scale and add some epsilon for nicer numerical behaviour
    A = scale*(A@A.T+epsilon)
    b = np.randn(2)
    Sigma = np.randn(2,2)
    Sigma = scale*Sigma@Sigma.T+epsilon

    xs,ys = generate_dataset(n_samples=1000)

    forward_process = sdes.VariancePreservingSDE()