import torch
import numpy as np
import os
import utils
import nets
from libraries.sdeflow_light.lib import sdes, plotting
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import torchsde
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from sbi.analysis import pairplot, conditional_pairplot
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_dataset(n_samples, random_state = 7):

    random_gen = torch.random.manual_seed(random_state)
    x = torch.randn(n_samples,xdim, generator = random_gen)
    y = f(x)
    #x = torch.from_numpy(x)
    #y = torch.from_numpy(y)
    return x.float(),y.float()

def check_posterior(x,y,posterior, prior, likelihood, evidence):


    log_p1 = posterior.log_prob(x)
    log_p2 = prior.log_prob(x)+likelihood.log_prob(y)-evidence.log_prob(y)

    print(log_p2, log_p1)
    #assert torch.allclose(log_p1, log_p2, atol = 1e-5), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)

def get_grid(sde, cond1,dim, num_samples = 2000, num_steps=200, transform=None,
             mean=0, std=1, clip=True):

    cond = torch.zeros(num_samples,dim)
    cond += cond1
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, dim)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1) * sde.T
    ones = torch.ones(num_samples, 1)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, cond)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    y0 = y0.data.cpu().numpy()
    return y0

#toy function as forward problem
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
    score_likelihood = (y_res@Sigma_inv.T)@A
    return score_prior+score_likelihood

def sample_t(model,x):

    if model.debias:
        t_ = model.base_sde.sample_debiasing_t([x.size(0), ] + [1 for _ in range(x.ndim - 1)])
        t_.requires_grad = True
    else:
        t_ = 1e-5+torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)], requires_grad=True).to(x) * model.T

    return t_
def initial_condition_loss_(u,x,y):
    return torch.sum(torch.abs(u-score_posterior(x,y)), dim=1)

def boundary_condition_loss_(u,x):
    return torch.sum(torch.abs(u+x), dim=1)

def dsm_loss_fn(s,std,target,xdim):

    return ((s * std + target) ** 2).view(xdim, -1).sum(1, keepdim=False) / 2

#the difference between this and ermon_loss is that ermon_loss does not include the initial and boundary condition.
#doesn't work yet
def PINN_loss(model,x,y, lam1 = 1., lam2 = 1., lam3 = 1.) :

    t_ = sample_t(model,x)
    batch_size,xdim = x.shape
    x_t, target, std, g = model.base_sde.sample(t_, x, return_noise=True)
    t0 = torch.zeros_like(t_)
    T = torch.ones_like(t_)
    g_0 = model.base_sde.g(t0, x_t)
    x_T, target_T, std_T, g_T = model.base_sde.sample(T, x, return_noise=True)
    beta = model.base_sde.beta(t_)

    s_0 = model.a(x, t0, y)/g_0
    s_T = model.a(x_T, T, y)/g_T
    s = model.a(x_t, t_,y)/g


    initial_condition_loss = initial_condition_loss_(s_0, x, y)
    x_collocation_loss = dsm_loss_fn(s, std, target, x.size(0))
    boundary_condition_loss = boundary_condition_loss_(s_T, x_T)

    #print('Initial condition loss: ', initial_condition_loss.mean().item())
    #print('DSM loss: ', x_collocation_loss.mean().item())
    #print('Boundary condition: ', boundary_condition_loss.mean().item())
    MSE_u = lam2*initial_condition_loss + lam3*boundary_condition_loss + x_collocation_loss

    s_x1 = torch.autograd.grad(s[:, 0].sum(), x_t, create_graph=True, retain_graph=True)[0]
    s_x2 = torch.autograd.grad(s[:,1].sum(), x_t, create_graph = True, retain_graph = True)[0]
    s_t1 = torch.autograd.grad(s[:,0].sum(),t_, retain_graph=True)[0]
    s_t2 = torch.autograd.grad(s[:,1].sum(),t_, retain_graph=True)[0]
    s_t = torch.cat([s_t1,s_t2],dim=1)
    divx_s = s_x1[:,0]+s_x2[:,1]
    pde_loss = torch.autograd.grad(divx_s.sum()+torch.sum(s**2), x_t, retain_graph = True)[0]
    pde_loss =torch.sum((s_t-.5*beta*(s+pde_loss))**2, dim=1).view(batch_size,1)
    #print('PDE Loss: ', pde_loss.mean().item())
    loss = torch.mean(MSE_u+lam1*pde_loss)
    return loss

#calculates PINN loss without x-collocations term, e.g. no dsm loss included
def PINN_loss2(model,x,y, lam1 = 1., lam2 = 1., lam3=1.):

    t_= (sample_t(model,x)).to(x.device)
    batch_size, xdim = x.shape
    x_t, target, std, g = model.base_sde.sample(t_, x, return_noise=True)
    x_t = x_t.to(x.device)
    t0 = torch.zeros_like(t_).to(x.device)
    T = torch.ones_like(t_).to(x.device)
    g_0 = model.base_sde.g(t0, x_t)
    x_T, target_T, std_T, g_T = model.base_sde.sample(T, x, return_noise=True)
    x_T = x_T.to(x.device)
    beta = model.base_sde.beta(t_)

    s_0 = model.a(x, t0, y) / g_0
    s = model.a(x_t, t_, y) / g
    s_T = model.a(x_T, T, y)/g_T

    initial_condition_loss = initial_condition_loss_(s_0, x, y)
    boundary_condition_loss = boundary_condition_loss_(s_T, x_T)
    MSE_u = lam2 * initial_condition_loss + lam3 * boundary_condition_loss

    s_x1 = torch.autograd.grad(s[:, 0].sum(), x_t, create_graph=True, retain_graph=True)[0]
    s_x2 = torch.autograd.grad(s[:, 1].sum(), x_t, create_graph=True, retain_graph=True)[0]
    s_t1 = torch.autograd.grad(s[:, 0].sum(), t_, retain_graph=True)[0]
    s_t2 = torch.autograd.grad(s[:, 1].sum(), t_, retain_graph=True)[0]
    s_t = torch.cat([s_t1, s_t2], dim=1)
    divx_s = s_x1[:, 0] + s_x2[:, 1]
    pde_loss = torch.autograd.grad(divx_s.sum() + torch.sum(s ** 2), x_t, retain_graph=True)[0]
    pde_loss = torch.sum((s_t - .5 * beta * (s + pde_loss)) ** 2, dim=1).view(batch_size, 1)

    loss = torch.mean(MSE_u + lam1 * pde_loss)
    return loss

#calculates PINN loss without boundary condition term
def PINN_loss3(model,x,y, lam1 = 1., lam2 = 1.) :

    t_ = sample_t(model,x)
    batch_size,xdim = x.shape
    x_t, target, std, g = model.base_sde.sample(t_, x, return_noise=True)
    t0 = torch.zeros_like(t_)
    g_0 = model.base_sde.g(t0, x_t)
    beta = model.base_sde.beta(t_)

    s_0 = model.a(x, t0, y)/g_0
    s = model.a(x_t, t_,y)/g

    initial_condition_loss = initial_condition_loss_(s_0, x, y)
    x_collocation_loss = dsm_loss_fn(s, std, target, x.size(0))
    MSE_u = lam2*initial_condition_loss + x_collocation_loss

    s_x1 = torch.autograd.grad(s[:, 0].sum(), x_t, create_graph=True, retain_graph=True)[0]
    s_x2 = torch.autograd.grad(s[:,1].sum(), x_t, create_graph = True, retain_graph = True)[0]
    s_t1 = torch.autograd.grad(s[:,0].sum(),t_, retain_graph=True)[0]
    s_t2 = torch.autograd.grad(s[:,1].sum(),t_, retain_graph=True)[0]
    s_t = torch.cat([s_t1,s_t2],dim=1)
    divx_s = s_x1[:,0]+s_x2[:,1]
    pde_loss = torch.autograd.grad(divx_s.sum()+torch.sum(s**2), x_t, retain_graph = True)[0]
    pde_loss =torch.sum((s_t-.5*beta*(s+pde_loss))**2, dim=1).view(batch_size,1)

    loss = torch.mean(MSE_u+lam1*pde_loss)
    return loss

#the loss is growing but samples look okay
def ermon_loss(model,x,y,lam = 1.):

    t_ = sample_t(model,x)
    batch_size,xdim = x.shape
    x_t, target, std, g = model.base_sde.sample(t_, x, return_noise=True)
    s = model.a(x_t, t_,y)/g
    beta = model.base_sde.beta(t_)

    s_x1 = torch.autograd.grad(s[:, 0].sum(), x_t, create_graph=True, retain_graph=True)[0]
    s_x2 = torch.autograd.grad(s[:,1].sum(), x_t, create_graph = True, retain_graph = True)[0]
    s_t1 = torch.autograd.grad(s[:,0].sum(),t_, retain_graph=True)[0]
    s_t2 = torch.autograd.grad(s[:,1].sum(),t_, retain_graph=True)[0]
    s_t = torch.cat([s_t1,s_t2],dim=1)

    divx_s = s_x1[:,0]+s_x2[:,1]
    pde_loss = torch.autograd.grad(divx_s.sum()+torch.sum(s**2), x_t, retain_graph = True)[0]
    #pde_loss =torch.abs(s_t-.5*beta*torch.sum(s+pde_loss,dim=1).view(batch_size,1))
    pde_loss = torch.sum((s_t-.5*beta*(s+pde_loss))**2, dim=1).view(batch_size,1)
    loss = dsm_loss_fn(s,std,target,xdim)+lam*pde_loss
    return loss.mean()

def train(model,xs,ys, optim, loss_fn, save_dir, log_dir, num_epochs, batch_size=1000, **kwargs):

    model.train()
    writer = SummaryWriter(log_dir)

    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):

        train_loader = utils.get_dataloader_noise(xs, ys,scale,batch_size)
        mean_loss = 0
        for x,y in train_loader():

            x = torch.ones_like(x, requires_grad=True)*x
            x = x.to(device)
            y = y.to(device)

            #loss = loss_fn(model,x,y,**kwargs)
            loss = model.dsm(x,y).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            mean_loss += loss.data.item()

        mean_loss /= (xs.shape[0]//batch_size)
        writer.add_scalar('Loss/train', mean_loss, i)
        prog_bar.set_description('loss: {:.4f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    current_model_path = os.path.join(save_dir, 'current_model.pt')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.a.state_dict(), current_model_path)
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

def sample(model, y, x_T=None, dt=0.005, t0=0.,t1=1., n_samples = 500):
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
        t = torch.linspace(t0,t1,int((t1-t0)/dt))
        #concatenate x and y to use with sdeint
        y = y.repeat(n_samples,1)
        xy = torch.concat([x_T,y],dim=1)
        x_pred = torchsde.sdeint(model, xy, t, dt=dt)[-1,:,:]

    return x_pred[:,:xdim]

def evaluate(model, xs,ys, n_samples = 2000):

    model.eval()
    with torch.no_grad():
        # some example distributions to plot
        prior = MultivariateNormal(torch.Tensor(mu), torch.Tensor(Lam))
        likelihood = get_likelihood(xs[0])
        evidence = get_evidence()
        posterior = get_posterior(ys[0])

        check_posterior(xs[0], ys[0], posterior, prior, likelihood, evidence)

        #log_plot(prior)
        #fig, ax = conditional_pairplot(likelihood, condition=xs[0], limits=[[-3, 3], [-3, 3]])
        #fig.suptitle('Likelihood at x=(%.2f,%.2f)'%(xs[0,0],xs[0,1]))
        #fig.show()
        fig, ax = conditional_pairplot(posterior, condition=ys[0], limits=[[-3, 3], [-3, 3]])
        fig.suptitle('Posterior at y=(%.2f,%.2f)'%(ys[0,0],ys[0,1]))
        fname = os.path.join(save_dir, 'posterior-true.png')
        plt.savefig(fname)
        fig.show()
        #x_pred = sample(model, y=ys[0], dt = .005, n_samples=n_samples)
        x_pred = get_grid(model.to(device),ys[0].to(device),xdim=2,ydim=2, num_samples=n_samples)
        #utils.make_image(x_pred, xs[0].detach().data.numpy().reshape(1, 2), num_epochs=500, show_plot=True, savefig=False)
        fig, ax = pairplot([x_pred], limits=[[-3, 3], [-3, 3]])
        fig.suptitle('N=%d samples from the posterior at y=(%.2f,%.2f)'%(n_samples,ys[0,0],ys[0,1]))
        fname = os.path.join(save_dir, 'posterior-predict.png')
        plt.savefig(fname)
        fig.show()

        mse_score = 0
        nll_sample = 0
        nll_true = 0

        # calculate MSE of score on test set
        t0 = torch.zeros(xs.shape[0], requires_grad=False).view(-1, 1)
        score_predict = model.a(xs, t0, ys)
        score_true = score_posterior(xs, ys)
        mse_score += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

        prog_bar = tqdm(total=len(xs))
        for x_true,y in zip(xs,ys):

            # calculate negative log likelihood of samples on test set
            x_predict = get_grid(reverse_process,y,xdim=2,ydim=2, num_samples=int(n_samples*.1))
            posterior = get_posterior(y)
            nll_sample += -torch.sum(posterior.log_prob(torch.Tensor(x_predict)))
            #calculate nll of true samples from posterior for reference
            nll_true += -n_samples*posterior.log_prob(x_true)
            prog_bar.set_description('NLL samples: %.4f NLL true %.4f'%(nll_sample,nll_true))
            prog_bar.update()

        mse_score /= xs.shape[0]
        nll_sample /= xs.shape[0]
        nll_true /= xs.shape[0]
        print('MSE: %.4f, NLL of samples: %.4f, NLL of true samples: %.4f'%(mse_score,nll_sample,nll_true))
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

if __name__ == '__main__':

    #define parameters of the inverse problem
    epsilon = 1e-6
    xdim = 2
    ydim = 2
    #f is a shear by factor 0.5 in x-direction and tranlsation by (0.3, 0.5).
    A = torch.Tensor([[1,0.5],[0,1]])
    b = torch.Tensor([0.3,0.5])
    scale = .3 #measurement noise
    Sigma = scale*torch.eye(ydim)
    Lam = torch.eye(xdim)
    Sigma_inv = 1/scale*torch.eye(ydim)
    Sigma_y_inv = torch.linalg.inv(Sigma+A@Lam@A.T+epsilon*torch.eye(ydim))
    mu = torch.zeros(xdim)

    #create data
    xs,ys = generate_dataset(n_samples=100000)

    x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=.8, random_state = 7)
    embed_dim = 2
    net_params = {'input_dim': xdim+ydim,
                  'output_dim': xdim,
                  'hidden_layers': [256,256],
                  'embed_dim': embed_dim}
    log_dir = 'logs/diffusion_FPE/'
    save_dir = 'models/inverse_models/diffusion_FPE/Pinn_loss'
    forward_process = sdes.VariancePreservingSDE()
    score_net = nets.TemporalMLP_small(**net_params).to(device)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=False)
    loss_fn = reverse_process.dsm
    optimizer = Adam(reverse_process.a.parameters(), lr = 1e-4)

    reverse_process = train(reverse_process,x_train,y_train, optimizer, loss_fn, save_dir, log_dir, num_epochs=500)
    #we need to wrap the reverse SDE into an own class to use the integration method from torchsde
    #reverse_process = SDE(reverse_process.a, reverse_process.base_sde, xdim, ydim, sde_type='stratonovich')
    evaluate(reverse_process, x_test, y_test)