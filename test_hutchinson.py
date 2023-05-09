import torch
import matplotlib.pyplot as plt

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div
    
def rademacher_like(s):

    v = torch.distributions.bernoulli.Bernoulli(torch.ones_like(s)*.5).sample()
    v[torch.where(v==0)]=-1
    return v
    
def div_estimator(s,x,num_samples=1, dist='rademacher'):

    div = torch.zeros(s.shape[0],1)
    for _ in range(num_samples):
        #v = torch.randn_like(s)
        v = rademacher_like(s)
        vjp = torch.autograd.grad(s,x,grad_outputs = v, create_graph=True, retain_graph=True)[0]
        div += (vjp[:,None,:]@v[:,:,None]).view(-1,1)
    
    div /= num_samples
    return div
      
x = torch.tensor([[1., 2., 3.],[0.,2.,1.]], requires_grad=True)
w = torch.tensor([[1., 2., 3.], [0., 1., -1.],[1.,0.,1.]])
b = torch.tensor([1., 2.,3.])
y = torch.matmul(x, w.t()) + b # y = x @ wT + b => y1 = x1 + 2*x2 + 3*x3 + 1 = 15, y2 = x2 - x3 + 2 = 1
print(y.shape)
div = divergence(y,x)
print(div.shape)
n_iter = 100
divs = torch.zeros(n_iter,2,1)
for i in range(1,n_iter+1):
    divs[i-1]+=div_estimator(y,x,i)
    

plt.plot(divs[:,0,0], linestyle='dotted', alpha=.6,label = 'estimator-batch1')
plt.plot(divs[:,1,0], linestyle='dotted', alpha=.6,label = 'estimator-batch2')
plt.plot(torch.ones(n_iter)*div[0,0], label='true')
plt.legend()
plt.show()
