from torch.optim import Adam
import ot
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import os
import pickle
import time

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock

batch_size = 1024
num_samples_per_epoch = 16384
num_epochs_INN = 100 #this is adjusted for runtime
DATASET_SIZE = num_samples_per_epoch*num_epochs_INN
DIMENSION=100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_image(true_samples, pred_samples,inds=None):

    cmap = plt.cm.tab20
    range_param = 1.2
    if inds is None:
        no_params = min(5, true_samples.shape[1])
        inds=range(no_params)
    else:
        no_params=len(inds)
    fig, axes = plt.subplots(figsize=[12,12], nrows=no_params, ncols=no_params, gridspec_kw={'wspace':0., 'hspace':0.});

    for j, ij in enumerate(inds):
        for k, ik in enumerate(inds):
            axes[j,k].get_xaxis().set_ticks([])
            axes[j,k].get_yaxis().set_ticks([])
            # if k == 0: axes[j,k].set_ylabel(j)
            # if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), alpha=0.3, range=(-range_param,range_param))
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), histtype="step", range=(-range_param,range_param))

                axes[j,k].hist(true_samples[:,ij], bins=50, color=cmap(2), alpha=0.3, range=(-range_param,range_param))
                axes[j,k].hist(true_samples[:,ij], bins=50, color=cmap(2), histtype="step", range=(-range_param,range_param))
            else:
                val, x, y = np.histogram2d(pred_samples[:,ij], pred_samples[:,ik], bins=25, range = [[-range_param, range_param], [-range_param, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

                val, x, y = np.histogram2d(true_samples[:,ij], true_samples[:,ik], bins=25, range = [[-range_param, range_param], [-range_param, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(2)])
    plt.show()

class GaussianMixture():

    def __init__(self, mixture_params):
        self.mixture_params = mixture_params
    # draw num_samples samples from the distributions given by the mixture_params
    # returns those samples
    def sample(self, num_samples):
        n = len(mixture_params)
        sigmas=torch.stack([torch.sqrt(sigma) for w,mu,sigma in mixture_params])
        probs=np.array([w for w, mu, sigma in mixture_params])
        zs = np.random.choice(n, (num_samples,), p=probs/probs.sum())
        mus = torch.stack([mu for w, mu, sigma in mixture_params])[zs]
        sigmas_samples = sigmas[zs]
        multinomial_samples = torch.randn(num_samples, mus.shape[1], device=device)
        if len(sigmas_samples.shape)==1:
            sigmas_samples=sigmas_samples.unsqueeze(-1)
        out_samples = mus + multinomial_samples*sigmas_samples
        return out_samples

    def log_prob(self, samples):
        # gets the log of the prior of some samples given its mixture parameters
        exponents = torch.zeros((samples.shape[0], len(self.mixture_params)), device=device)
        dimension = samples.shape[1]
        for k, (w, mu, sigma) in enumerate(self.mixture_params):
            log_gauss_prefactor = (-dimension / 2) * (np.log(2 * np.pi) + torch.log(sigma))
            tmp = -0.5 * torch.sum((samples - mu[None, :]) ** 2, dim=1) / sigma
            exponents[:, k] = tmp + np.log(w) + log_gauss_prefactor

        max_exponent = torch.max(exponents, dim=1)[0].detach()
        exponents_ = exponents - max_exponent.unsqueeze(-1)
        exp_sum = torch.log(torch.sum(torch.exp(exponents_), dim=1)) + max_exponent
        return exp_sum

# gets mean and covariance of the Gaussian posterior with linear forward model
# mean, sigma are the parameters of the prior distribution
def get_single_gaussian_posterior(mean, sigma, forward_mat, b_sq, y):
    ATA = forward_mat**2/b_sq
    cov_gauss = 1/(ATA+1/sigma)

    mean_gauss = cov_gauss*forward_mat*y/b_sq+cov_gauss*mean/sigma
    return mean_gauss, cov_gauss

# returns the mixture parameters of the posterior given the mixture parameters of the
# prior, the forward model and the likelihood (for a specific y)
def get_mixture_posterior(x_gauss_mixture_params, forward_mat, b_sq, y):
    out_mixtures = []
    nenner = 0
    ws=torch.zeros(len(x_gauss_mixture_params),device=device)
    mus_new=[]
    sigmas_new=[]
    log_zaehler=torch.zeros(len(x_gauss_mixture_params),device=device)
    for k,(w, mu, sigma) in enumerate(x_gauss_mixture_params):
        mu_new, sigma_new = get_single_gaussian_posterior(mu, sigma, forward_mat, b_sq, y)
        mus_new.append(mu_new)
        sigmas_new.append(sigma_new)
        ws[k]=w
        log_zaehler[k]=torch.log(torch.tensor(w,device=device,dtype=torch.float))+(0.5*torch.sum(mu_new**2/sigma_new)-0.5*torch.sum(mu**2)/sigma)
    const=torch.max(log_zaehler)
    log_nenner=torch.log(torch.sum(torch.exp(log_zaehler-const)))+const
    for k in range(len(x_gauss_mixture_params)):
        out_mixtures.append((torch.exp(log_zaehler[k]-log_nenner).detach().cpu().numpy(),mus_new[k],sigmas_new[k]))
    return out_mixtures

# creates forward map
# scale controls how illposed the problem is

def create_forward_model(scale,dimension):
    s = torch.ones(dimension, device = device)
    for i in range(dimension):
        s[i] = scale/(i+1)
    return s

# evaluates forward_map
def forward_pass(x, forward_map):
    return x*forward_map

#returns the (negative) log posterior given a y, the mixture params of the prior, the likelihood model b and y
def get_log_posterior(mixture_model, samples, forward_map, b, y):
    p = -mixture_model.log_prob(samples)
    p2 = 0.5 * torch.sum((y-forward_pass(samples, forward_map))**2 * (1/b**2), dim=1)
    return (p+p2).view(len(samples))



# creates a data loader returning (x,y) pairs of the joint distribution
def get_epoch_data_loader(mixture_model, num_samples_per_epoch, batch_size, forward_map, b):
    x = mixture_model.sample(num_samples_per_epoch)
    y = forward_pass(x, forward_map)
    y += torch.randn_like(y) * b
    def epoch_data_loader():
        for i in range(0, num_samples_per_epoch, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader

# creates a conditional INN object using the FrEIA package with num_layers many layers,
# hidden neurons given by sub_net_size, dimension and dimension_condition specifying the dim of x/y respectively
# returns a nn.module object
def create_INN(num_layers, sub_net_size,dimension=5,dimension_condition=5):
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))
    nodes = [InputNode(dimension, name='input')]
    cond = ConditionNode(dimension_condition, name='condition')
    for k in range(num_layers):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.4},
                          conditions = cond,
                          name=F'coupling_{k}'))
    nodes.append(OutputNode(nodes[-1], name='output'))

    model = ReversibleGraphNet(nodes + [cond], verbose=False).to(device)
    return model

def MLELoss(z,log_det_J,log_p_z):
    return (-log_p_z-log_det_J).mean()

# trains an epoch of the INN
# given optimizer, the model and the data_loader
# training is done via maximum likelihood loss
# returns mean loss

def train_inn_epoch(optimizer, model, epoch_data_loader, prior, loss_fn, **kwargs):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        z, jac_inv = model(x, c = y, rev = True)
        log_p_z = prior.log_prob(z)
        loss = 0
        l5 = 0.5 * torch.sum(z**2, dim=1) - jac_inv
        loss += (torch.sum(l5) / cur_batch_size)
        #loss = loss_fn(z,jac_inv,log_p_z,**kwargs)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss

# train from scratch or just use pretrained model
retrain=True

# trains and evaluates both the INN and SNF and returns the Wasserstein distance on the mixture example
# parameters are the mixture params (parameters of the mixture model in the prior), b (likelihood parameter)
# a set of testing_ys and the forward model (forward_map)
#
# prints and returns the Wasserstein distance of INN
def train_and_eval(mixture_model, b, testing_ys, forward_map,training_run, **kwargs):

    forward_model=lambda x: forward_pass(x, forward_map)
    log_posterior=lambda samples,y:get_log_posterior(mixture_model, samples,forward_map,b,y)
    INN = create_INN(8,128,dimension=DIMENSION,dimension_condition=DIMENSION)
    if retrain:

        optimizer_inn = Adam(INN.parameters(), lr = 1e-4)

        prog_bar = tqdm(total=num_epochs_INN)

        for i in range(num_epochs_INN):
            data_loader = get_epoch_data_loader(mixture_model, num_samples_per_epoch, batch_size, forward_map, b)
            loss = train_inn_epoch(optimizer_inn, INN, data_loader,**kwargs)
            prog_bar.set_description('loss: {:.4f}, b: {}, n_mix: {}'.format(loss, b, len(mixture_model.mixture_params)))
            prog_bar.update()
        prog_bar.close()

        torch.save(INN.state_dict(),'tmp/inn_'+str(training_run)+'.pt')
    else:
        INN.load_state_dict(torch.load('tmp/inn_'+str(training_run)+'.pt'))

    testing_x_per_y = 5000
    testing_x_per_y_more = 10000

    testing_num_y = len(testing_ys)
    weights2 = np.ones((testing_x_per_y,)) / testing_x_per_y

    weights2 = weights2.astype(np.float64)
    weights2_large = np.ones((testing_x_per_y_more,)) / testing_x_per_y_more

    weights2_large = weights2_large.astype(np.float64)
    w2=[]


    w2_large=[]
    tic=time.time()
    for i, y in enumerate(testing_ys):
        true_posterior_params = get_mixture_posterior(mixture_params, forward_map, b**2, y)
        true_posterior = GaussianMixture(true_posterior_params)
        true_posterior_samples = true_posterior.sample(testing_x_per_y).cpu().numpy()
        inflated_ys = y[None, :].repeat(testing_x_per_y, 1)
        inp_samps=torch.randn(testing_x_per_y, DIMENSION, device=device)
        samples_INN = INN(inp_samps, c = inflated_ys)[0].detach().cpu().numpy()

        if ((i <10) and (training_run ==0)):
            make_image(true_posterior_samples, samples_INN,inds=[0,49,99])

        M2 =ot.dist(samples_INN, true_posterior_samples, metric='euclidean')

        #w2.append(ot.emd2(weights1, weights2, M2, numItermax=1000000))
        #some random operation like taking the mean over all distances to circumvent the weird earth mover distance including the SNF samples
        w2.append(np.mean(M2))
        true_posterior_samples = true_posterior.sample(testing_x_per_y_more).cpu().numpy()

        inflated_ys = y[None, :].repeat(testing_x_per_y_more, 1)
        inp_samps=torch.randn(testing_x_per_y_more, DIMENSION, device=device)
        samples_INN = INN(inp_samps, c = inflated_ys)[0].detach().cpu().numpy()
        M2 =ot.dist(samples_INN, true_posterior_samples, metric='euclidean')

        #w2_large.append(ot.emd2(weights1_large, weights2_large, M2, numItermax=1000000))
        w2_large.append(np.mean(M2))
        toc=time.time()-tic
    w2_mean=np.mean(w2)
    w2_std=np.std(w2)

    w2_mean_large=np.mean(w2_large)
    print('W INN:', w2_mean,'+-',w2_std)

    print('W INN large:', w2_mean_large)

    return w2_mean,w2_mean_large

#numbers of testing_ys
testing_num_y = 100
# likelihood parameter
b = 0.1
# forward_model
forward_map = create_forward_model(scale = 0.1,dimension=DIMENSION)
# number of mixtures
n_mixtures=12
np.random.seed(0)
torch.manual_seed(0)
results_array = np.zeros((5,2))
prior_params = [torch.zeros(DIMENSION), torch.eye(DIMENSION)]
prior = torch.distributions.MultivariateNormal(*prior_params)
for r in range(5):

    mixture_params=[]
    # create mixture params (weights, means, covariances)
    for i in range(n_mixtures):
        mixture_params.append((1./n_mixtures,torch.tensor(np.random.uniform(size=DIMENSION)*2-1, device = device,dtype=torch.float),torch.tensor(0.0001,device=device,dtype=torch.float)))
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    with open('tmp/models_mixture' +str(r), 'wb') as fp:
        pickle.dump(mixture_params, fp)
    # draws testing_ys
    mixture_model = GaussianMixture(mixture_params)
    testing_xs = mixture_model.sample(testing_num_y)
    testing_ys = forward_pass(testing_xs, forward_map) + b * torch.randn(testing_num_y, DIMENSION, device=device)

    results = train_and_eval(mixture_model, b,testing_ys,forward_map,training_run = r, loss_fn = MLELoss, prior = prior)
    results_array[r,0] = results[0]
    results_array[r,1] = results[1]


print('MEANS OF INN')
print(np.mean(results_array[:,0]))

print('STD OF INN')
print(np.std(results_array[:,0]))

print('MEAN OF INN LARGE')
print(np.mean(results_array[:,1]))

print('STD OF INN LARGE')
print(np.std(results_array[:,1]))