import torch
from torch import nn
import numpy as np
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
import os
import utils

class INN(GraphINN):

    def __init__(self,num_layers, sub_net_size,dimension,dimension_condition):

        self.sub_net_size = sub_net_size
        self.num_layers = num_layers
        self.input_dimension = dimension
        self.condition_dimension = dimension_condition
        self.nodes = [InputNode(dimension, name='input')]
        self.condition = ConditionNode(dimension_condition, name='condition')

        for k in range(num_layers):
            self.nodes.append(Node(self.nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':self.subnet_fn, 'clamp':1.4},
                              conditions = self.condition,
                              name=F'coupling_{k}'))
        self.nodes.append(OutputNode(self.nodes[-1], name='output'))

        super().__init__(self.nodes + [self.condition], verbose=False)

    def subnet_fn(self, c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, self.sub_net_size), nn.ReLU(),
                             nn.Linear(self.sub_net_size, self.sub_net_size), nn.ReLU(),
                             nn.Linear(self.sub_net_size,  c_out))
    
class MLP(nn.Sequential):

    def __init__(self, input_dimension, output_dimension, hidden_layers):

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers
        super().__init__(nn.Linear(input_dimension, hidden_layers[0]), nn.ReLU())
        for i in range(len(hidden_layers)-1):
            self.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.append(nn.ReLU())
        self.append(nn.Linear(hidden_layers[-1], output_dimension))

    def forward(self, input, *more_inputs):
        inputs = torch.cat([input, *more_inputs], dim = 1)
        inputs = torch.flatten(inputs, start_dim= 1)
        return super().forward(inputs)



"""
class TemporalMLP(MLP):

    def __init__(self, input_dim,output_dim,hidden_layers, embed_dim):

        super().__init__(input_dim+embed_dim,output_dim, hidden_layers)
        self.embed = utils.GaussianFourierProjection(embed_dim)

    def forward(self, x,t,*more_inputs):
        t_embed = self.embed(t.squeeze())
        return super().forward(x,t_embed,*more_inputs)
        

"""
class TemporalMLP(nn.Module):

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
