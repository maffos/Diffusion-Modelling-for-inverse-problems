import torch
from torch import nn
from include.sdeflow_light.lib.sdes import VariancePreservingSDE

class CFMModel(nn.Module):
    def __init__(self):
        self.forward_sde = VariancePreservingSDE()
    def forward(self):
