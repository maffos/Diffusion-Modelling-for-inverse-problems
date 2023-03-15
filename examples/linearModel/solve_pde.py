from include.sdeflow_light.lib import sdes
from pde import PDEBase, VectorField, CartesianGrid, ScalarField
import numpy as np
import torch
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

#affine function as forward problem
def f(x):
    return x@A.T+b

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
    y_res = y-f(x)
    score_prior = -x
    score_likelihood = y_res@Sigma_inv@A.T
    return score_prior+score_likelihood

class ScoreFPE(PDEBase):

    def __init__(self,x, sde, s_max=50):
        self.x = x
        self.sde = sde
        self.bc_x = [{'value': -s_max}, {'value': s_max}]
        self.bc_y = [{'value': -s_max}, {'value':s_max}]
    def evolution_rate(self, state, t=0):

        div = state.divergence(bc = [self.bc_x, self.bc_y])
        sq_norm = state.dot(state)
        s_x = state.dot(self.x)
        rhs = div+sq_norm+s_x
        rhs = rhs.gradient(bc = [self.bc_x, self.bc_y])
        beta = self.sde.beta(t)
        rhs += 0.5*beta

        return rhs

if __name__ == '__main__':

    n_samples = 30000
    x_probe =torch.randn(2)
    y = f(x_probe)
    posterior = get_posterior(y)
    nx = 500
    minx = -20
    maxx = 20
    x1 = torch.linspace(minx, maxx, steps=nx)
    x2 = torch.linspace(minx, maxx, steps=nx)
    x_points = torch.cartesian_prod(x1, x2)
    initial_condition = score_posterior(x_points,y).data.numpy().reshape(2,nx,nx)
    #inspect values on the boundary
    print('left boundary')
    print(np.mean(initial_condition[:,0,:]))
    print('lower boundary')
    print(np.mean(initial_condition[:,:,0]))
    print('upper boundary')
    print(np.mean(initial_condition[:,:,-1]))
    print('right boundary')
    print(np.mean(initial_condition[:,-1,:]))

    grid = CartesianGrid([[minx,maxx],[minx,maxx]], nx) # generate grid
    state = VectorField(grid, data=initial_condition)
    x_field = VectorField(grid, data=x_points.data.numpy().reshape(2,nx,nx))
    state.plot(method = 'streamplot', title = 'Initial Condition')
    forwardSDE = sdes.VariancePreservingSDE()
    eq = ScoreFPE(x_field,forwardSDE)  # define the pde
    result = eq.solve(state, t_range=1, dt=0.01)
    result.plot()
