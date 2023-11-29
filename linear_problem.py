import torch
from torch.distributions import MultivariateNormal
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearForwardProblem():

    def __init__(self):
        # define parameters of the inverse problem
        self.epsilon = 1e-6
        self.xdim = 2
        self.ydim = 2

        # f is a shear by factor 0.5 in x-direction and tranlsation by (0.3, 0.5).
        self.A = torch.Tensor([[1, 0.5], [0, 1]])
        self.b = torch.Tensor([0.3, 0.5])
        self.scale = .3
        self.Sigma = self.scale * torch.eye(self.ydim)
        self.Lam = torch.eye(self.xdim)
        self.Sigma_inv = 1 / self.scale * torch.eye(self.ydim)
        self.Sigma_y_inv = torch.linalg.inv(self.Sigma + self.A @ self.Lam @ self.A.T + self.epsilon * torch.eye(self.ydim))
        self.mu = torch.zeros(self.xdim)

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])

    """
    def get_params(self):
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

        return epsilon,xdim,ydim,A,b,scale,Sigma,Lam,Sigma_inv,Sigma_y_inv,mu
    """

    #affine function as forward problem
    def forward(self,x):
        return (self.A@x.T).T+self.b

    def get_likelihood(self,x):

        mean = self.A.to(x)@x+self.b.to(x)
        return MultivariateNormal(mean,self.Sigma)

    def get_evidence(self):
        mean = self.A@self.mu+self.b
        cov = self.Sigma+self.A@self.Lam@self.A.T

        return MultivariateNormal(mean,cov)

    def get_posterior(self,y, device = device):
        y_res = y-(self.A@self.mu+self.b)
        mean = self.Lam@self.A.T@self.Sigma_y_inv@y_res
        cov = self.Lam-self.Lam@self.A.T@self.Sigma_y_inv@self.A@self.Lam

        return MultivariateNormal(mean.to(device),cov.to(device))

    def log_posterior(self,xs, ys):
        y_res = ys - (self.A @ self.mu + self.b)
        mean = y_res @ (self.A.T @ self.Sigma_y_inv)
        x_res = xs - mean
        log_probs = .5 * x_res @ self.cov_inv
        log_probs = log_probs[:, None, :] @ x_res[:, :, None]

        return log_probs.view(-1, 1)

    #analytical score of the posterior
    def score_posterior(self,x,y):
        y_res = y-(x@self.A.T+self.b)
        score_prior = -x
        score_likelihood = (y_res@self.Sigma_inv.T)@self.A
        return score_prior+score_likelihood