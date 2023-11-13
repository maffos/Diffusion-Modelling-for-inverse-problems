import torch
from torch import nn as nn
from torch.func import jvp

if __name__ == '__main__':
    x = torch.tensor([0.0, 2.0, 8.0], requires_grad=True)
    y = torch.tensor([5.0 , 1.0 , 7.0], requires_grad = True)
    v = torch.ones_like(x)
    v2 = torch.rand_like(x)

    #z = x@y+ x*5
    model = nn.Linear(3, 3)
    model.weight = torch.nn.Parameter(torch.tensor([[1, 2, -1], [0, -1, 0], [2, 1, 1]], dtype=torch.float))
    model.bias = torch.nn.Parameter(torch.tensor([1, -1, 1], dtype=torch.float))
    x = torch.randn(5,3, requires_grad = True)
    out = model(x)
    #vJ = torch.autograd.grad(z,x,v)
    #out,Jv = jvp(model, (x,), (v,))
    vJ = torch.autograd.grad(out, x,torch.rand_like(out))

    #print(Jv)
    print(out)
    print('------------')
    print(vJ)
