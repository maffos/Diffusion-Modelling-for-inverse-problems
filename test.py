from torch.optim import Adam
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sigma = 0.05
conv_lambda = 0.01
forward = False

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

def train_inn_epoch(optimizer, model, epoch_data_loader):
    mean_loss = 0
    relu = torch.nn.ReLU()
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)
        z, jac_inv = model(x, c = y, rev = True)
        #log_p_z = prior.log_prob(z)
        loss = 0
        l5 = 0.5 * torch.sum(z**2, dim=1) - jac_inv
        loss += (torch.sum(l5) / cur_batch_size)
        z = torch.randn(cur_batch_size, DIMENSION, device=device)
        x, jac = model(z, c=y)
        loss_kl = -torch.mean(jac) + torch.sum((y - MLP(x)) ** 2) / (cur_batch_size * 2 * sigma ** 2)
        loss_relu = 100 * torch.sum(relu(x - 1) + relu(-x))
        print(loss_relu)
        loss = loss*(1-conv_lambda) + (loss_kl + loss_relu)*conv_lambda


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)

    return mean_loss
def train_MLP_epoch(optimizer, model, epoch_data_loader):
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

def make_image(pred_samples,x_true, inds=None):

    cmap = plt.cm.tab20
    range_param = 1.2
    if inds is None:
        no_params = pred_samples.shape[1]
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
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), alpha=0.3, range=(-0,range_param))
                axes[j,k].hist(pred_samples[:,ij], bins=50, color=cmap(0), histtype="step", range=(-0,range_param))
                axes[j,k].axvline(x_true[:,j])

            else:
                val, x, y = np.histogram2d(pred_samples[:,ij], pred_samples[:,ik], bins=25, range = [[-0, range_param], [-0, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

    plt.savefig('posterior.png')
    plt.show()

def get_epoch_dataloader(x_train, y_train):
    perm = torch.randperm(len(x_train))
    x = x_train[perm].clone()
    y = y_train[perm].clone()
    #y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size].clone(), y[i:i + batch_size].clone()

    return epoch_data_loader

def get_epoch_dataloader_noise(x_train, y_train):
    perm = torch.randperm(len(x_train))
    x = x_train[perm].clone()
    y = y_train[perm].clone()
    y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size].clone(), y[i:i + batch_size].clone()

    return epoch_data_loader

# trains and evaluates both the INN and SNF and returns the Wasserstein distance on the mixture example
# parameters are the mixture params (parameters of the mixture model in the prior), b (likelihood parameter)
# a set of testing_ys and the forward model (forward_map)
#
# prints and returns the Wasserstein distance of INN
num_epochs_INN = 400
DIMENSION = 6
COND_DIM = 469
INN = create_INN(8,128,dimension=DIMENSION,dimension_condition=COND_DIM)
MLP = nn.Sequential(nn.Linear(DIMENSION, 256),
                      nn.ReLU(),
                      nn.Linear(256, 256),
                      nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                      nn.Linear(256, COND_DIM))
optimizer_inn = Adam(INN.parameters(), lr = 1e-4)
optimizer_mlp = Adam(MLP.parameters(),lr = 1e-4)
num_epochs_MLP = 400

prog_bar = tqdm(total=num_epochs_MLP)
filename = 'data/ppg/AbdAorta_PPG.npz'
data = np.load(filename, allow_pickle=True)["data"].item()
print(data['25'][1].shape)
xs = torch.from_numpy(data['25'][0]).float()
for i in range(xs.shape[1]):
    xs[:,i]= (xs[:,i]-xs[:,i].min())/(xs[:,i].max()-xs[:,i].min())
print(xs)
ys = torch.from_numpy(data['25'][1]).float()
for i in range(ys.shape[1]):
    if (ys[:,i].max()-ys[:,i].min())>0:
        ys[:,i]= (ys[:,i]-ys[:,i].min())/(ys[:,i].max()-ys[:,i].min())

print(ys)

for i in range(num_epochs_MLP):
    data_loader = get_epoch_dataloader(xs,ys)
    loss = train_MLP_epoch(optimizer_mlp, MLP, data_loader)
    prog_bar.set_description('loss: {:.4f}'.format(loss))
    prog_bar.update()
prog_bar.close()

prog_bar = tqdm(total=num_epochs_INN)
for i in range(num_epochs_INN):
    data_loader = get_epoch_dataloader_noise(xs,ys)
    loss = train_inn_epoch(optimizer_inn, INN, data_loader)
    prog_bar.set_description('loss: {:.4f}'.format(loss))
    prog_bar.update()
prog_bar.close()

y_ex = ys[-1].clone()
y_ex = y_ex + torch.randn_like(y_ex)*sigma
y_ex = y_ex.repeat(1000,1)

samples = INN(torch.randn(1000, DIMENSION, device = device), c = y_ex)[0]

make_image(samples.detach().data.numpy(), xs[-1].detach().data.numpy().reshape(1,DIMENSION))
