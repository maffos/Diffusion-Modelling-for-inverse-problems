from torch.optim import Adam
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import os
import pickle
import time
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
from sklearn.model_selection import train_test_split
import utils
import loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sigma = 0.05
conv_lambda = 0.05
output_dir = 'plots/lamb=%.2f/sigma=%.2f'%(conv_lambda,sigma)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def create_INN(num_layers, sub_net_size,dimension,dimension_condition):
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

    # trains an epoch of the INN
# given optimizer, the model and the data_loader
# returns mean loss

def train_inn_epoch(optimizer, model, epoch_data_loader, backward_training = True, **loss_params):
    model.train()
    mean_loss = 0
    relu = torch.nn.ReLU()
    for k, (x, y_noise, y_true) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        z, jac_inv = model(x, c = y_noise, rev = True)
        #log_p_z = latent.log_prob(z)
        #l5 = 0.5 * torch.sum(z**2, dim=1) - jac_inv
        #mle_loss = (torch.sum(l5) / cur_batch_size)
        #mle_loss = (-log_p_z+jac_inv).mean()
        if backward_training:
            z_samples = torch.randn(cur_batch_size, DIMENSION, device=device)
            x, log_det_J = model(z_samples, c=y_noise)
            loss = utils.ForwardBackwardKLLoss(x,z,jac_inv, log_det_J, y_true, y_noise, **loss_params)
        else:
            loss = loss.MLLoss(z,-jac_inv, **loss_params)
        #loss_kl = -torch.mean(jac) + torch.sum((y - MLP(x)) ** 2) / (cur_batch_size * 2 * sigma ** 2)
        #loss_kl = -torch.mean(jac) + torch.sum((y_noise - y_true) ** 2) / (cur_batch_size * 2 * sigma ** 2) #anmerkung: vorzeichen unklar. eventuell plus und minus vertauschen
        #loss_relu = 100 * torch.sum(relu(x - 1) + relu(-x))
        #likelihood = MultivariateNormal(y_true, torch.eye(COND_DIM)*sigma)
        #loss_kl = (-log_likelihood(y_noise, y_true) - jac).mean()
        #loss = mle_loss*(1-conv_lambda) + (loss_kl+loss_relu)*conv_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)

    return mean_loss
    
def train_MLP_epoch(optimizer, model, epoch_data_loader):
    model.train()
    mean_loss = 0
    mse = nn.MSELoss()
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        y_pred = model(x)
        loss = mse(y,y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss

def make_image(pred_samples,x_true, num_epochs, inds=None,):

    cmap = plt.cm.tab20
    range_param = 1.2
    if inds is None:
        no_params = pred_samples.shape[1]
        inds=range(no_params)
    else:
        no_params=len(inds)
    fig, axes = plt.subplots(figsize=[12,12], nrows=no_params, ncols=no_params, gridspec_kw={'wspace':0., 'hspace':0.});
    fig.suptitle('Epochs=%d'%num_epochs)
    for j, ij in enumerate(inds):
        for k, ik in enumerate(inds):
            axes[j,k].get_xaxis().set_ticks([])
            axes[j,k].get_yaxis().set_ticks([])
            # if k == 0: axes[j,k].set_ylabel(j)
            # if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), alpha=0.3, range=(-0,range_param))
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), histtype="step", range=(-0,range_param))
                axes[j,k].axvline(x_true[:,j])

            else:
                val, x, y = np.histogram2d(pred_samples[:,ij], pred_samples[:,ik], bins=25, range = [[-0, range_param], [-0, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

    plt.savefig(os.path.join(output_dir, 'posterior_epochs=%d.png'%num_epochs))

def get_epoch_dataloader(x_train, y_train):
    perm = torch.randperm(len(x_train))
    x = x_train[perm]
    y = y_train[perm]
    #y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    return epoch_data_loader

def get_epoch_dataloader_noise(x_train, y_train):
    perm = torch.randperm(len(x_train))
    x = x_train[perm]
    y = y_train[perm]
    y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size], y_train[i:i+batch_size]

    return epoch_data_loader

def load_data(filename,age):

    data = np.load(filename, allow_pickle=True)["data"].item()
    x_labels = data['parameters']
    data = data[age]
    xs = data[0]

    # normalize x
    xs = (xs - xs.min(axis=0)) / (xs.max(axis=0) - xs.min(axis=0))
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(data[1]).float()

    return xs,ys,x_labels

# trains and evaluates both the INN and SNF and returns the Wasserstein distance on the mixture example
# parameters are the mixture params (parameters of the mixture model in the prior), b (likelihood parameter)
# a set of testing_ys and the forward model (forward_map)

num_epochs = 6400
DIMENSION = 6
COND_DIM = 469
#the distribution of the latent space is defined. Usually this is a standard normal. This is not necessarily the same as the prior, which in most of our simulations is uniform.
latent_prior = MultivariateNormal(torch.zeros(DIMENSION), torch.eye(DIMENSION))
INN = create_INN(8,128,dimension=DIMENSION,dimension_condition=COND_DIM)

"""
MLP = nn.Sequential(nn.Linear(DIMENSION, 256),
                      nn.ReLU(),
                      nn.Linear(256, 256),
                      nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                      nn.Linear(256, COND_DIM))
"""
optimizer_inn = Adam(INN.parameters(), lr = 1e-4)
#optimizer_mlp = Adam(MLP.parameters(),lr = 1e-4)

prog_bar = tqdm(total=num_epochs)
filename = 'data/ppg/AbdAorta_PPG.npz'
age = '25'
data = np.load(filename, allow_pickle=True)["data"].item()
xs,ys,x_labels = load_data(filename,age)

X_train, X_test, y_train, y_test = train_test_split(xs,ys, train_size=.8)
#choose an arbitrary y to evaluate the posterior
y_ex = y_test[-1].clone()
y_ex = y_ex + torch.randn_like(y_ex) * sigma
y_ex = y_ex.repeat(1000, 1)

#MLP_error = []
epochs = []
test_loader = get_epoch_dataloader(X_test, y_test)

"""
#train MLP surrogate
for i in range(num_epochs):
    data_loader = get_epoch_dataloader(X_train,y_train)

    #print the absolute error every 200 epochs
    if i%800 == 0:
        MLP.eval()
        err = []
        for k, (x, y) in enumerate(test_loader()):
            diff = torch.abs(y - MLP(x))
            # print(diff.shape)
            err.append(diff.detach().data.numpy())

        err = np.concatenate(err, axis=0)
        print('--------------')
        print('Num Epochs = ', i)
        print('Mean absolute approximation error of the forward problem:', np.mean(err))
        MLP_error.append(np.mean(err))
        epochs.append(i)

    loss = train_MLP_epoch(optimizer_mlp, MLP, data_loader)
    prog_bar.set_description('loss: {:.4f}'.format(loss))
    prog_bar.update()
prog_bar.close()

#last evaluation after training is done
MLP.eval()
err = []
for k, (x, y) in enumerate(test_loader()):
    diff = torch.abs(y - MLP(x))
    # print(diff.shape)
    err.append(diff.detach().data.numpy())

err = np.concatenate(err, axis=0)
print('--------------')
print('Num Epochs = ', num_epochs)
print('Mean absolute approximation error of the forward problem:', np.mean(err))
MLP_error.append(np.mean(err))
epochs.append(num_epochs)

#plot the mean absolute error against number of epochs
fig = plt.figure()
plt.plot(epochs, MLP_error)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Mean absolute Error of the forward problem')
plt.savefig(os.path.join(output_dir, 'mean_error.png'))
"""

#train INN
prog_bar = tqdm(total=num_epochs)

for i in range(num_epochs):
    data_loader = get_epoch_dataloader_noise(X_train,y_train)

    #evaluate the posterior every 200 Epochs
    if i%200 == 0:
        INN.eval()
        samples = INN(torch.randn(1000, DIMENSION, device=device), c=y_ex)[0]

        make_image(samples.detach().data.numpy(), X_test[-1].detach().data.numpy().reshape(1, DIMENSION), i)

    loss = train_inn_epoch(optimizer_inn, INN, data_loader, backward_training = False, latent_prior = latent_prior)
    prog_bar.set_description('loss: {:.4f}'.format(loss))
    prog_bar.update()

#last evaluation after training is finished
INN.eval()
samples = INN(torch.randn(1000, DIMENSION, device=device), c=y_ex)[0]
make_image(samples.detach().data.numpy(), X_test[-1].detach().data.numpy().reshape(1, DIMENSION), num_epochs)

prog_bar.close()

