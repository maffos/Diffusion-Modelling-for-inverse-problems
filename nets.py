import torch
from torch import nn
import numpy as np
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
import os
import utils
    
class MLP(nn.Sequential):

    def __init__(self, input_dim, output_dim, hidden_layers, activation):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        super().__init__(nn.Linear(input_dim, hidden_layers[0]), activation)
        self.act = activation
        for i in range(len(hidden_layers)-1):
            self.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.append(self.act)
        self.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x,y,t):
        input = torch.cat([x, y, t.view(len(x),1)], dim=1)
        assert input.ndim == 2, 'Input Tensor is expected to be 2D with shape (batch_size, ydim+ydim+embeddim)'
        return super().forward(input)

class MLP2(nn.Sequential):

    def __init__(self, input_dim, output_dim, hidden_layers, activation):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        super().__init__(nn.Linear(input_dim, hidden_layers[0]), activation)
        self.act = activation
        for i in range(len(hidden_layers)-1):
            self.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.append(self.act)
        self.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self,x,t):
        #inputs = torch.cat([input] + list(more_inputs), dim = 1)
        #inputs = torch.flatten(inputs, start_dim= 1)
        input = torch.cat([x,t.view(len(x),1)], dim=1)
        assert input.ndim == 2, 'Input Tensor is expected to be 2D with shape (batch_size, ydim+ydim+embeddim)'
        return super().forward(input)


class TemporalMLP(nn.Module):
    """
    Embeds the time variable with a gaussian fourier projection. Hidden layers are hard coded.
    Initial experiments showed no promising results so not considered in any further work.
    """

    def __init__(self, input_dim, output_dim, embed_dim, hidden_layers, activation='tanh'):

        super(TemporalMLP, self).__init__()
        self.input_dim = input_dim+embed_dim
        self.output_dim = output_dim
        self.embed = utils.GaussianFourierProjection(embed_dim)

        #build the net
        if activation in ['tanh', 'Tanh']:
            self.act = nn.Tanh()
        elif activation in ['relu', 'Relu', 'ReLU']:
            self.act = nn.ReLU()
        else:
            raise ValueError('$s as activation function is not supported. Please choose either relu or tanh.'%activation)
        self.fc1 = nn.Linear(self.input_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.fc5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.fc6 = nn.Linear(hidden_layers[4], self.output_dim)

    def forward(self,x,t,y):

        t_embed = self.embed(t)
        input = torch.cat([x,t_embed,y], dim=1)
        assert input.ndim == 2, 'Input Tensor is expected to be 2D with shape (batch_size, ydim+ydim+embeddim)'
        x = self.fc1(input)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        x = self.act(x)
        x = self.fc5(x)
        x = self.act(x)
        x = self.fc6(x)

        return x

class TemporalMLP_small(nn.Module):

    def __init__(self, input_dim, output_dim, embed_dim, hidden_layers, activation='sigmoid'):

        super(TemporalMLP_small, self).__init__()
        self.input_dim = input_dim + embed_dim
        self.output_dim = output_dim
        self.embed = utils.GaussianFourierProjection(embed_dim)

        # build the net
        if activation in ['tanh', 'Tanh']:
            self.act = nn.Tanh()
        elif activation in ['Sigmoid', 'sigmoid']:
            self.act = nn.Sigmoid()
        elif activation in ['relu', 'Relu', 'ReLU']:
            self.act = nn.ReLU()
        else:
            raise ValueError(
                '$s as activation function is not supported. Please choose one of relu, tanh or sigmoid.' % activation)
        self.fc1 = nn.Linear(self.input_dim, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])

        self.fc6 = nn.Linear(hidden_layers[1], self.output_dim)

    def forward(self, x, t, y):

        t_embed = self.embed(t)
        input = torch.cat([x, t_embed, y], dim=1)
        assert input.ndim == 2, 'Input Tensor is expected to be 2D with shape (batch_size, ydim+ydim+embeddim)'
        x = self.fc1(input)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc6(x)

        return x

class PosteriorScore(torch.nn.Module):
    """
    Calculates the score of a posterior distribution by adding the prior and the likelihood score.
    Both of which are represented by neural nets.
    """

    def __init__(self, prior_net, likelihood_net, forward_process):
        super(PosteriorScore, self).__init__()
        self.prior_net = prior_net
        self.likelihood_net = likelihood_net
        self.forward_sde = forward_process

    def forward(self, x,y,t):
        posterior_score = self.prior_net(x, t) + self.likelihood_net(x,y,t)
        return self.forward_sde.g(t, x) * posterior_score
