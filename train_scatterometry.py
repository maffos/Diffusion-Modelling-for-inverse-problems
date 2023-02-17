import torch
from torch import nn
from torch.optim import Adam
from models.SNF import *
from models.diffusion import *
from tqdm import tqdm
import os
from sbi.analysis import pairplot, conditional_pairplot
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load forward model
forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256, 256), nn.ReLU(),
                              nn.Linear(256,  23)).to(device)

forward_model.load_state_dict(torch.load('models/surrogate/scatterometry_surrogate.pt', map_location = torch.device(device)))
for param in forward_model.parameters():
    param.requires_grad=False

def get_epoch_data_loader(batch_size, forward_model,a, b,lambd_bd):
    x = torch.tensor(inverse_cdf_prior(np.random.uniform(size=(8*batch_size,3)),lambd_bd),dtype=torch.float,device=device)
    y = forward_model(x)
    y += torch.randn_like(y) * b + torch.randn_like(y)*a*y
    def epoch_data_loader():
        for i in range(0, 8*batch_size, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader


# returns (negative) log_posterior evaluation for the scatterometry model
# likelihood is determined by the error model
# uniform prior is approximated via boundary loss for a.e. differentiability
def get_log_posterior(samples, forward_model, a, b, ys,lambd_bd):
    relu=torch.nn.ReLU()
    forward_samps=forward_model(samples)
    prefactor = ((a*forward_samps)**2+b**2)
    p = .5*torch.sum(torch.log(prefactor), dim = 1)
    p2 = 0.5*torch.sum((ys-forward_samps)**2/prefactor, dim = 1)
    p3 = lambd_bd*torch.sum(relu(samples-1)+relu(-1-samples), dim = 1)
    return p+p2+p3


# returns samples from the boundary loss approximation prior
# lambd_bd controlling the strength of boundary loss
def inverse_cdf_prior(x,lambd_bd):
    x*=(2*lambd_bd+2)/lambd_bd
    y=np.zeros_like(x)
    left=x<1/lambd_bd
    y[left]=np.log(x[left]*lambd_bd)-1
    middle=np.logical_and(x>=1/lambd_bd,x < 2+1/lambd_bd)
    y[middle]=x[middle]-1/lambd_bd-1
    right=x>=2+1/lambd_bd
    y[right]=-np.log(((2+2/lambd_bd)-x[right])*lambd_bd)+1
    return y

def train(forward_model, num_epochs_SNF, num_epochs_diffusion, batch_size, lambd_bd, save_dir):
    # define networks
    log_posterior=lambda samples, ys:get_log_posterior(samples,forward_model,a,b,ys,lambd_bd)
    snf = create_snf(4,64,log_posterior,metr_steps_per_block=10,dimension=3,dimension_condition=23,noise_std=0.4)
    #diffusion_model = create_diffusion_model(xdim=3,ydim=23, embed_dim=2,hidden_layers=[512,512])
    diffusion_model = create_diffusion_model2(xdim=3,ydim=23,hidden_layers=[1024,1024])
    optimizer = Adam(snf.parameters(), lr = 1e-3)

    prog_bar = tqdm(total=num_epochs_SNF)
    for i in range(num_epochs_SNF):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_SNF_epoch(optimizer, snf, data_loader,forward_model, a, b,None)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_diffusion = Adam(diffusion_model.parameters(), lr = 1e-4)
    prog_bar = tqdm(total=num_epochs_diffusion)
    for i in range(num_epochs_diffusion):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_diffusion_epoch(optimizer_diffusion, diffusion_model, data_loader)
        prog_bar.set_description('determ diffusion loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    chkpnt_file_snf = os.path.join(save_dir, 'snf.pt')
    chkpnt_file_diff = os.path.join(save_dir, 'diffusion.pt')
    torch.save(snf.state_dict(),chkpnt_file_snf)
    torch.save(diffusion_model.a.state_dict(),chkpnt_file_diff)

    return snf, diffusion_model

def evaluate(snf,diffusion_model, save_dir, n_samples = 2000):

    snf.eval()
    diffusion_model.eval()

    xs = torch.rand(n_samples, xdim, device=device) * 2 - 1
    ys = forward_model(xs)
    ys = ys + b * torch.randn_like(ys) + ys * a * torch.randn_like(ys)
    y= ys[0]
    inflated_ys = y[None, :].repeat(n_samples, 1)
    with torch.no_grad():

        x_pred_diffusion = get_grid(diffusion_model,y,xdim,ydim, num_samples=n_samples)
        fig, ax = pairplot([x_pred_diffusion])
        fig.suptitle('N=%d samples from the posterior with diffusion.'%(n_samples))
        fname = os.path.join(save_dir, 'posterior-predict-diffusion.png')
        plt.savefig(fname)
        plt.show()

        x_pred_snf = snf.forward(torch.randn(n_samples, xdim, device=device), inflated_ys)[0].detach().cpu().numpy()
        fig, ax = pairplot([x_pred_snf])
        fig.suptitle('N=%d samples from the posterior with snf' % (n_samples))
        fname = os.path.join(save_dir, 'posterior-predict-snf.png')
        plt.savefig(fname)
        plt.show()

if __name__ == '__main__':

    # load forward model
    xdim=3
    ydim=23

    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)

    forward_model.load_state_dict(torch.load('models/surrogate/scatterometry_surrogate.pt', map_location=torch.device(device)))
    for param in forward_model.parameters():
        param.requires_grad = False

    a = 0.2
    b = 0.01
    n_epochs_snf = 50
    n_epochs_diffusion = 5000

    plot_dir='plots/scatterometry'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    save_dir = 'models/scatterometry/no_embed'
    snf,diffusion_model = train(forward_model, n_epochs_snf, n_epochs_diffusion, batch_size=1000, lambd_bd=1000, save_dir = save_dir)
    evaluate(snf,diffusion_model, plot_dir, n_samples=1000)