import torch
from torch import nn
import numpy as np
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
import os

class INN(GraphINN):

    def __init__(self,num_layers, sub_net_size,dimension,dimension_condition, training_dir):

        self.sub_net_size = sub_net_size
        self.num_layers = num_layers
        self.input_dimension = dimension
        self.condition_dimension = dimension_condition
        self.nodes = [InputNode(dimension, name='input')]
        self.condition = ConditionNode(dimension_condition, name='condition')
        self.training_dir = training_dir
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

    def __init__(self, input_dimension, output_dimension, hidden_layer_size):

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layer_size = hidden_layer_size

        super().__init__(nn.Linear(input_dimension, hidden_layer_size),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_size, hidden_layer_size),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_size, hidden_layer_size),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_size, hidden_layer_size),
                      nn.ReLU(),
                      nn.Linear(hidden_layer_size, output_dimension))

    def eval_pass(self, data_loader, loss_fn):

        with torch.no_grad():
            self.eval()
            mean_loss = 0
            for k, (x, y) in enumerate(data_loader()):
                loss = loss_fn(y, self.forward(x))
                mean_loss * k / (k + 1) + loss.data.item() / (k + 1)

        return mean_loss

    def train_pass(self, data_loader, loss_fn, optimizer):
        self.train()
        mean_loss = 0
        for k, (x, y) in enumerate(data_loader()):
            y_pred = self.forward(x)
            loss = loss_fn(y, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)

        return mean_loss