from utils import batch_gradient
import torch

x = torch.randn(5,2,requires_grad = True)
t = torch.randn(5,1,requires_grad = True)
A = torch.tensor([[1.,0.5],[0.,1.]])
y = (A@x.T).T*t

print('X: \n', x)
print('t: \n', t)
print('y: \n', y)

dy_dt = batch_gradient(y,t)

print('dy_dt: \n', dy_dt)
print('Ax: \n', (A@x.T).T)

by_hand = torch.zeros_like(x)
for i in range(x.shape[0]):
    for j in range(A.shape[0]):
        for k in range(x.shape[1]):
            by_hand[i,j] +=A[j,k]*x[i,k]

print('by hand: \n', by_hand)

torch.testing.assert_close((A@x.T).T, by_hand)