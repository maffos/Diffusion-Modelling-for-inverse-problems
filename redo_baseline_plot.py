import matplotlib as plt
from utils import plot_density,get_linear_params,get_dataloader_noise,generate_dataset
from examples.scatterometry.utils_scatterometry import get_epoch_data_loader,get_forward_model_params,get_dataset,get_log_posterior
from examples.linearModel.main_diffusion import f
from train_baselines_linear import log_posterior
from models.SNF import *
from models.INN import *
from sklearn.model_selection import train_test_split
import os
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(snf,INN, num_epochs_SNF, num_epochs_INN, batch_size, save_dir, log_dir, linear=False,**kwargs):
    # define networks
    optimizer = Adam(snf.parameters(), lr=1e-4)
    logger = SummaryWriter(log_dir)
    prog_bar = tqdm(total=num_epochs_SNF)
    for i in range(num_epochs_SNF):

        if linear:
            data_loader = get_dataloader_noise(kwargs['xs'],kwargs['ys'],kwargs['scale'],batch_size)
        else:
            data_loader = get_epoch_data_loader(batch_size, kwargs['forward_model'], kwargs['a'], kwargs['b'], kwargs['lambd_bd'])

        loss = train_SNF_epoch(optimizer, snf, data_loader)
        logger.add_scalar('Train/SNF-Loss', loss, i)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_inn = Adam(INN.parameters(), lr=1e-3)

    prog_bar = tqdm(total=num_epochs_INN)
    for i in range(num_epochs_INN):
        if linear:
            data_loader = get_dataloader_noise(xs, ys, scale, batch_size)
        else:
            data_loader = get_epoch_data_loader(batch_size, forward_model, a, b, lambd_bd)
        loss = train_inn_epoch(optimizer_inn, INN, data_loader)
        logger.add_scalar('Train/INN-Loss', loss, i)
        prog_bar.set_description('INN loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    chkpnt_file_snf = os.path.join(save_dir, 'snf.pt')
    chkpnt_file_inn = os.path.join(save_dir, 'INN.pt')
    torch.save(snf.state_dict(), chkpnt_file_snf)
    torch.save(INN.state_dict(), chkpnt_file_inn)

    return snf, INN

def replot(snf,INN,ys,plot_ys,n_samples_x,nbins,figsize,labelsize,out_dir,linear = False):
    snf.eval()
    INN.eval()
    with torch.no_grad():
        prog_bar = tqdm(total=len(plot_ys))
        for i in plot_ys:
            y = ys[i]
            inflated_ys = y[None, :].repeat(n_samples_x, 1)
            x_pred_snf = snf.forward(torch.randn(n_samples_x, xdim, device=device), inflated_ys)[
                0].detach().cpu().numpy()
            x_pred_inn = INN(torch.randn(n_samples_x, xdim, device=device), c=inflated_ys)[0].detach().cpu().numpy()

            if linear:
                plot_density(x_pred_snf, nbins, limits=(-3.5, 3.5), xticks=[-3,3], size=figsize, labelsize=labelsize,
                                   fname=os.path.join(out_dir, 'posterior-snf-xlim3-%d.svg' % i), show_mode=True)
                #plot_density(x_pred_snf, nbins, limits=(-4, 4),xticks=[-4,4], size=figsize, labelsize=labelsize,
                #            fname=os.path.join(out_dir, 'posterior-snf-xlim4-%d.svg' % i))
                plot_density(x_pred_inn, nbins, limits=(-3.5, 3.5), xticks=[-3,3], size=figsize,
                             labelsize=labelsize,
                             fname=os.path.join(out_dir, 'posterior-inn-xlim3-%d.svg' % i), show_mode=True)
                plot_density(x_pred_inn, nbins, limits=(-4, 4), xticks=[-4,4], size=figsize,
                             labelsize=labelsize,
                             fname=os.path.join(out_dir, 'posterior-inn-xlim4-%d.svg' % i), show_mode=True)

            else:
                plot_density(x_pred_snf, nbins, limits=(-1.2, 1.2),xticks = [-1,0,1], size=figsize,labelsize=labelsize,
                             fname=os.path.join(out_dir, 'posterior-snf-%d.svg' % i))
                plot_density(x_pred_inn, nbins, limits=(-1.2, 1.2), xticks=[-1,0, 1], size=figsize,
                            labelsize=labelsize,
                            fname=os.path.join(out_dir, 'posterior-inn-%d.svg' % i))

            plot_density(x_pred_snf, nbins,size=figsize,labelsize=labelsize,
                             fname=os.path.join(out_dir, 'posterior-snf_no_limits-%d.svg' % i))
            plot_density(x_pred_inn, nbins, size=figsize, labelsize=labelsize,
                         fname=os.path.join(out_dir, 'posterior-inn_no_limits-%d.svg' % i))
            prog_bar.update()

if __name__ == '__main__':

    nbins = 75
    figsize = (12,12)
    labelsize = 30

    surrogate_dir = 'examples/scatterometry'
    forward_model, a, b, lambd_bd, xdim, ydim = get_forward_model_params(surrogate_dir)
    n_epochs_SNF = 500
    n_epochs_INN = 3000
    batch_size = 1000
    n_samples_y = 100
    n_samples_x = 30000

    save_dir = 'examples/scatterometry/results/baselines3'
    log_dir = os.path.join(save_dir, 'logs')
    plot_dir = 'plots/scatterometry/baselines'

    '''
    log_posterior_scatterometry = lambda x,y: get_log_posterior(x,forward_model,a,b,y,lambd_bd)
    snf = create_snf(4, 64, log_posterior_scatterometry, metr_steps_per_block=10, dimension=xdim, dimension_condition=ydim,
                     noise_std=0.4)
    inn = create_INN(4, 64, dimension=xdim, dimension_condition=ydim)
    snf,inn = train(snf,inn, n_epochs_SNF, n_epochs_INN, batch_size, save_dir,log_dir=log_dir, linear = False, forward_model=forward_model, a=a, b=b, lambd_bd=lambd_bd)

    x_test,y_test = get_dataset(forward_model,a,b,size = n_samples_y)
    plot_ys = [0, 5, 6, 20, 23, 42, 50, 77, 81, 93]
    replot(snf,inn,y_test,plot_ys,n_samples_x,nbins,figsize,labelsize,plot_dir,linear = False)
    '''

    #linear baselines
    n_epochs_SNF = 100
    n_epochs_INN = 500
    save_dir = ('examples/linearModel/results/baselines3')
    log_dir = os.path.join(save_dir, 'logs')
    plot_dir = 'plots/linear/baselines'

    epsilon, xdim, ydim, A, b, scale, Sigma, Lam, Sigma_inv, Sigma_y_inv, mu = get_linear_params()
    snf = create_snf(4, 64, log_posterior, metr_steps_per_block=10, dimension=xdim, dimension_condition=ydim,
                     noise_std=0.4)
    inn = create_INN(4, 64, dimension=xdim, dimension_condition=ydim)
    xs, ys = generate_dataset(xdim, f, n_samples=100000)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=.9, random_state=7)
    snf,inn = train(snf, inn, n_epochs_SNF, n_epochs_INN, batch_size, save_dir, log_dir=log_dir, linear = True, xs=x_train,ys=y_train, scale = scale)

    plot_ys = [3, 5, 22, 39, 51, 53, 60, 71, 81, 97]
    y_test = y_test[:n_samples_y]

    replot(snf, inn, y_test, plot_ys, n_samples_x, nbins, figsize, labelsize, plot_dir, linear=True)


