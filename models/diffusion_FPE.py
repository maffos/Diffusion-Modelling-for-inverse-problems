import torch
import numpy as np
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
from sbi.analysis import pairplot, conditional_pairplot

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

def check_posterior(x,y,posterior, prior, likelihood, evidence):


    log_p1 = posterior.log_prob(x)
    log_p2 = prior.log_prob(x)+likelihood.log_prob(y)-evidence.log_prob(y)

    assert torch.allclose(log_p1, log_p2, atol = 1e-5), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)


#toy function as forward problem
def f(x):
    return (A@x.T).T+b

def get_likelihood(x, Sigma, A, b):

    mean = A@x+b
    return MultivariateNormal(mean,Sigma)

def get_evidence(A, mu, b, Sigma):
    mean = A@mu+b
    cov = Sigma+A@Lam@A.T

    return MultivariateNormal(mean,cov)

def get_posterior(y, A, b, mu, Lam, Sigma_y_inv):
    y_res = y-(A@mu+b)
    mean = Lam@A.T@Sigma_y_inv@y_res
    cov = Lam-Lam@A.T@Sigma_y_inv@A@Lam

    return MultivariateNormal(mean,cov)

#analytical score of the posterior
def score_posterior(x,y):
    y_res = y-(x@A.T+b)
    score_prior = -x
    score_likelihood = (y_res@Sigma_inv.T)@A
    return score_prior+score_likelihood


#PDE of the score. In the case of an Ohrnstein-Uhlenbeck process, the hessian of f is 0
def pde_loss(u,u_x1, u_x2,u_xx1, u_xx2, u_t1, u_t2, f, grad_f, sigma, H_f=0.):

    """
    :param u:       (Tensor:(batch_size,xdim))  Output of the NN, e.g. approximation of the score.
    :param u_x1:    (Tensor:(batch_size,xdim))  Partial derivative of the first dimension of the score wrt. x, e.g. first row of the hessian.
    :param u_x2:    (Tensor:(batch_size,xdim))  Second row of the Hessian.
    :param u_xx1:   (Tensor:(batch.size,xdim)   Contains the partial derivatives of the score wrt. dx1^3 and dx1^2dx2.
    :param u_xx2:   (Tensor:(batch_size,xdim)   Contains the partial derivatives of the score wrt. dx1dx2^2 and dx2^3.
    :param u_t1:    (Tensor: (batch_size,1))    Partial derivative of the first dimension of the score wrt. t.
    :param u_t2:    (Tensor: (batch_size,1))    Partial derivative of the second dimension of the score wrt. t.
    :param f:       (Tensor: (batch_size,xdim)) Drift evaluated at (x_t,t).
    :param grad_f:  (Tensor:(batch_size, xdim)) Gradient of the drift.
    :param sigma:   (Tensor:(batch_size,xdim))  Diffusion coefficient evaluated at (t).
    :param H_f:     (Tensor:(batch_size,xdim))  Hessian of the drift. Default 0
    :return:        (Tensor:(batch_size,))      L2 norm of the deviation between the temporal derivative and the evaluation of the pde.

    So far we calculate the loss in each dimension individually. The sum of third order terms is written out explicitly.
    """

    batch_size = u.shape[0]
    fx_u = grad_f*u
    f_ux1 = (u_x1[:,None,:]@f[:,:,None]).view(batch_size,1)
    f_ux2 = (u_x2[:,None,:]@f[:,:,None]).view(batch_size,1)
    u_ux1 = torch.sum(u_x1[:,:,None]@u[:,None,:],dim=(1,2)).view(batch_size,1)
    u_ux2 = torch.sum(u_x2[:,:,None]@u[:,None,:], dim=(1,2)).view(batch_size,1)
    third_order_term1 = (u_xx1[:,0]+u_xx2[:,0]+2*u_xx1[:,1]).view(batch_size,1)
    third_order_term2 = (2 * u_xx2[:, 0] + u_xx1[:, 1] + u_xx2[:, 1]).view(batch_size,1)
    l1 = (-u_t1 - H_f - (fx_u[:,0]).view(batch_size,1) - f_ux1 + .5*(sigma[:,0]).view(batch_size,1)**2*(third_order_term1+2*u_ux1))
    l2 = (-u_t2 - H_f - (fx_u[:,1]).view(batch_size,1) - f_ux2 + .5*sigma[:,1].view(batch_size,1)**2*(third_order_term2+2*u_ux2))
    loss = torch.cat([l1,l2], dim=1)
    loss = torch.linalg.vector_norm(loss, dim=1)

    """
    These are some quantities I calculated for debugging
    l = torch.linalg.vector_norm(loss, dim=1)
    l_mean = l.mean()
    f_ux1_mean = f_ux1.mean()
    f_ux2_mean = f_ux2.mean()
    u_ux1_mean = u_ux1.mean()
    u_ux2_mean = u_ux2.mean()
    third_order_mean1 = third_order_term1.mean()
    third_order_mean2 = third_order_term2.mean()
    """
    return loss
def initial_condition_loss_(u,x,y):

    return torch.linalg.vector_norm(u-score_posterior(x,y), dim=1)

def dsm_loss(a,std,g,target,xdim):

    return ((a * std + target) ** 2).view(xdim, -1).sum(1, keepdim=False) / 2
def PINN_loss(model,x,y):

    if model.debias:
        t_ = model.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        t_.requires_grad = True
    else:
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)], requires_grad=True).to(x) * model.T
    t0 = torch.zeros_like(t_)
    x_t, target, std, g = model.base_sde.sample(t_, x, return_noise=True)
    a = model.a(x_t, t_,y)
    g_0 = model.base_sde.g(torch.zeros(x.shape[0],1), x)
    u_0 = model.a(x,t0,y)/g_0
    u = a/g
    grad_f = -model.base_sde.g(t_,x_t)

    u_x1 = torch.autograd.grad(u[:,0].sum(), x_t, create_graph=True, retain_graph=True)[0]
    u_x2 = torch.autograd.grad(u[:,1].sum(), x_t, create_graph=True, retain_graph=True)[0]
    u_xx1 = torch.autograd.grad(u_x1[:,0].sum(), x_t,retain_graph=True)[0]
    u_xx2 = torch.autograd.grad(u_x2[:,1].sum(), x_t, retain_graph=True)[0]
    u_t1 = torch.autograd.grad(u[:, 0].sum(), t_, retain_graph=True)[0]
    u_t2 = torch.autograd.grad(u[:, 1].sum(), t_, retain_graph=True)[0]

    #u_xx12 = torch.autograd.grad(u_x1[:,1].sum(), x_t,retain_graph=True)[0]
    #u_xx21 = torch.autograd.grad(u_x2[:,0].sum(), x_t,retain_graph=True)[0]

    initial_condition_loss = initial_condition_loss_(u_0, x,y)
    x_collocation_loss = dsm_loss(u,std,g,target,x.size(0))
    MSE_u = initial_condition_loss+x_collocation_loss
    MSE_pde = pde_loss(u,u_x1,u_x2,u_xx1,u_xx2,u_t1,u_t2, model.base_sde.f(t_,x_t), grad_f, model.base_sde.g(t_,x_t))

    """
    This was an idea to penalize if the Hessian and third order matrix are not symmetric. 
    lam = 0.1
    second_order_reg = torch.linalg.vector_norm(u_x1[:,1]-u_x2[:,0])
    third_order_reg = torch.linalg.vector_norm(u_xx11[:,1]-u_xx12[:,0])+torch.linalg.vector_norm(u_xx12[:,1]-u_xx22[:,0])+torch.linalg.vector_norm(u_xx21[:,0]-u_xx11[:,1])+torch.linalg.vector_norm(u_xx21[:,1]-u_xx22[:,0])
    """

    loss = torch.mean(MSE_u+MSE_pde)
    #loss = model.dsm(x).mean()

    #these are just things I calculate for debugging
    IC_norm = torch.linalg.norm(initial_condition_loss)
    col_norm = torch.linalg.norm(x_collocation_loss)
    pde_norm = torch.linalg.norm(MSE_pde)
    u_x1_norm = torch.linalg.vector_norm(u_x1, dim=1).mean()
    u_x2_norm = torch.linalg.vector_norm(u_x2, dim=1).mean()
    u_xx1_norm = torch.linalg.vector_norm(u_xx1, dim=1).mean()
    u_xx2_norm = torch.linalg.vector_norm(u_xx2, dim=1).mean()
    u_t1_norm = u_t1.mean()
    u_t2_norm = u_t2.mean()

    #if model.debias is set to True, very low values for t may be sampled, at which the gradient will explode.
    if u_t1_norm == torch.Tensor([torch.inf]) or u_t2_norm == torch.Tensor([torch.inf]):
        print('u_t1: ', u_t1_norm)
        print('u_t2: ', u_t2_norm)
        print('u_x1: ', u_x1_norm)
        print('u_x2: ', u_x2_norm)
        print('u_xx1: ', u_xx1_norm)
        print('u_xx2: ', u_xx2_norm)
        print('t: ', t_)
        raise ValueError()
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

    def __init__(self, net, forward_sde, xdim, ydim, sde_type, noise_type='diagonal', T=1, t0=0):
        super(SDE, self).__init__()
        self.net = net
        self.forward_sde = forward_sde
        self.T = T
        self.t0 = t0
        self.xdim = xdim
        self.ydim = ydim
        self.sde_type = sde_type
        self.noise_type = noise_type

    #x and y are passed as one tensor to be compatible with sdeint
    def f(self, t, inputs, lmbd = 0.):
        #unpack x and y from the inputs to pass them to the net
        x_t = inputs[:,:xdim]
        y = inputs[:,-ydim:]
        if t.ndim <= 1:
            t = torch.full((x_t.shape[0], 1), t)
        f_x = (1. - 0.5 * lmbd) * self.forward_sde.g(self.T - t, x_t) * self.net(x_t, self.T - t, y) - \
            self.forward_sde.f(self.T - t, x_t)
        return torch.cat([f_x,y],dim=1)
    def g(self, t, x_t, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.forward_sde.g(self.T - t, x_t)

def sample(model, y, x_T=None, dt=0.01, t0=0.,t1=1., n_samples = 500):
    """

    :param model: (Object) Object of class that implements functions f and g as drift and diffusion coefficient.
    :param y: (Tensor) of shape (1,ydim)
    :param x_T: (Tensor) of shape (n_samples, xdim)
    :param dt: (float) Timestep to integrate
    :param t0: (scalar) Start time. Default 0.
    :param t1: (scalar) End Time. Default 1.
    :param n_samples: (int) Number samples to draw.
    :return:    (Tensor:(n_samples,xdim)) Samples from the posterior.
    """
    model.eval()
    with torch.no_grad():
        x_T = torch.randn((n_samples, model.xdim)) if x_T is None else x_T
        t = torch.linspace(t0,t1,20)
        #concatenate x and y to use with sdeint
        y = y.repeat(n_samples,1)
        xy = torch.concat([x_T,y],dim=1)
        x_pred = torchsde.sdeint(model, xy, t, dt=dt)[-1,:,:]

    return x_pred[:,:xdim]

def evaluate(model, xs,ys, n_samples = 1000):

    #for x_true,y in zip(xs,ys):
    #    x_pred = sample(model, y)
    #    utils.make_image(x_pred, x_true, num_epochs = 1000, show_plot=True, savefig=False)

    # some example distributions to plot
    prior = MultivariateNormal(torch.Tensor(mu), torch.Tensor(Lam))
    likelihood = get_likelihood(xs[0], Sigma, A, b)
    evidence = get_evidence(A, mu, b, Sigma)
    posterior = get_posterior(ys[0], A, b, mu, Lam, Sigma_y_inv)

    check_posterior(xs[0], ys[0], posterior, prior, likelihood, evidence)

    log_plot(prior)
    fig, ax = conditional_pairplot(likelihood, condition=xs[0], limits=[[-3, 3], [-3, 3]])
    temp = xs[0]
    fig.suptitle('Likelihood at x=(%.2f,%.2f)'%(xs[0,0],xs[0,1]))
    fig.show()
    fig, ax = conditional_pairplot(posterior, condition=ys[0], limits=[[-3, 3], [-3, 3]])
    fig.suptitle('Posterior at y=(%.2f,%.2f)'%(ys[0,0],ys[0,1]))
    fig.show()
    x_pred = sample(model, y=ys[0], n_samples=n_samples)
    utils.make_image(x_pred.detach().data.numpy(), xs[0].detach().data.numpy().reshape(1, 2), num_epochs=500, show_plot=True, savefig=False)
    fig, ax = pairplot([x_pred])
    fig.suptitle('N=%d samples from the posterior at y=(%.2f,%.2f)'%(n_samples,ys[0,0],ys[0,1]))
    fig.show()

def log_plot(prior):
    x1 = np.linspace(-3,3,100)
    x2 = np.linspace(-3,3,100)
    logp = np.arange(100*100).reshape(100,100)
    for i,x in enumerate(x1):
        for j,y in enumerate(x2):
            logp[i,j] = prior.log_prob(torch.Tensor([x,y]))

    h = plt.contourf(x1, x2, logp)
    plt.axis('scaled')
    plt.colorbar()
    plt.title('Prior Distribution')
    plt.show()


if __name__ == '__main__':

    #define parameters of the inverse problem
    epsilon = 1e-6
    xdim = 2
    ydim = 2
    A = torch.randn(ydim,xdim)
    b = torch.randn(ydim)
    scale = .3
    Sigma = scale*torch.eye(ydim)
    Lam = torch.eye(xdim)
    Sigma_inv = torch.linalg.inv(Sigma+epsilon*torch.eye(ydim))
    Sigma_y_inv = torch.linalg.inv(Sigma+A@Lam@A.T+epsilon*torch.eye(ydim))
    mu = torch.zeros(xdim)

    #create data
    xs,ys = generate_dataset(n_samples=1000, Sigma = Sigma)

    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=.8, random_state = 7)
    embed_dim = 2
    net_params = {'input_dim': xdim+ydim,
                  'output_dim': xdim,
                  'hidden_layers': [64,128,256,128,64],
                  'embed_dim': embed_dim}

    forward_process = sdes.VariancePreservingSDE()
    score_net = nets.TemporalMLP(**net_params)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=False)
    optimizer = Adam(reverse_process.a.parameters(), lr = 1e-4)

    reverse_process = train(reverse_process,x_train,y_train, optimizer, num_epochs=500)
    #we need to wrap the reverse SDE into an own class to use the integration metod from torchsde
    reverse_process = SDE(reverse_process.a, reverse_process.base_sde, xdim, ydim, sde_type='stratonovich')
    evaluate(reverse_process, x_test, y_test)