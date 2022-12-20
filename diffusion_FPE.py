import torch
import numpy as np
import loss as ls
from scipy.stats import multivariate_normal
from libraries.sdeflow_light.lib import sdes
#toy function as forward problem
def f(x):
    return A@x+b

def log_posterior(x, y):
    Lam = np.eye(2,2)
    Sigma_y_inv = np.linalg.inv(Sigma+A@Lam@A.T)
    y_res = y-b
    mean = Lam@A.T@Sigma_y_inv@y_res
    cov = Lam-Lam@A.T@Sigma_y_inv@A@Lam

    return multivariate_normal.pdf(x,mean,cov)

def generate_dataset(n_samples):
    x = np.random.normal(size=(n_samples,2))
    y = (f(x.T)).T
    return x,y

#fokker-planck-equation
def fpe(p):

if __name__ == '__main__':
    scale = 2
    epsilon = .01
    A = np.randn(2, 2)
    #make the matrix A psd and multiply scale and add some epsilon for nicer numerical behaviour
    A = scale*A@A.T+epsilon
    b = np.randn(2)
    Sigma = np.randn(2,2)
    Sigma = scale*Sigma@Sigma.T+epsilon

    xs,ys = generate_dataset(n_samples=1000)

    forward_process = sdes.VariancePreservingSDE()