import torch
from torch import nn
import numpy as np
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN, ConditionNode
from FrEIA.modules import GLOWCouplingBlock
import os

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
