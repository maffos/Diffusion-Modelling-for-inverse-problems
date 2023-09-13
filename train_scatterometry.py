
import matplotlib.pyplot as plt
from sbi.analysis import pairplot, conditional_pairplot

from models.SNF import *
from models.diffusion import *
from models.INN import *
from losses import *
from examples.scatterometry.utils_scatterometry import *
from tqdm import tqdm
import os
import scipy
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# mcmc parameters for "discovering" the ground truth
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000
RANDOM_STATE = 13
lambd_bd = 1000
num_epochs_INN = 5000



    
#is added to KL Divergence to improve numerical stability
reg = 1e-10

def train(forward_model, num_epochs_SNF, num_epochs_diffusion, num_epochs_INN,batch_size, lambd_bd, save_dir,log_dir):
    # define networks
    log_posterior=lambda samples, ys:get_log_posterior(samples,forward_model,a,b,ys,lambd_bd)
    snf = create_snf(4,64,log_posterior,metr_steps_per_block=10,dimension=3,dimension_condition=23,noise_std=0.4)
    diffusion_model = create_diffusion_model2(xdim=3,ydim=23,hidden_layers=[512,512,512])
    INN = create_INN(4, 64, dimension=3, dimension_condition=23)
    optimizer = Adam(snf.parameters(), lr = 1e-4)

    logger = SummaryWriter(log_dir)

    loss_fn_diffusion = DSMLoss()
    prog_bar = tqdm(total=num_epochs_SNF)
    for i in range(num_epochs_SNF):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss = train_SNF_epoch(optimizer, snf, data_loader,forward_model, a, b,None)
        logger.add_scalar('Train/SNF-Loss', loss, i)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_diffusion = Adam(diffusion_model.parameters(), lr = 1e-4)
    prog_bar = tqdm(total=num_epochs_diffusion)
    for i in range(num_epochs_diffusion):
        data_loader=get_epoch_data_loader(batch_size,forward_model,a,b,lambd_bd)
        loss,logger_info,t = train_diffusion_epoch(optimizer_diffusion, loss_fn_diffusion, diffusion_model, data_loader)
        logger.add_scalar('Train/diffusion-Loss', loss, i)
        prog_bar.set_description('diffusion loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_inn = Adam(INN.parameters(), lr=1e-3)

    prog_bar = tqdm(total=num_epochs_INN)
    for i in range(num_epochs_INN):
        data_loader = get_epoch_data_loader(batch_size, forward_model, a, b, lambd_bd)
        loss = train_inn_epoch(optimizer_inn, INN, data_loader)
        logger.add_scalar('Train/INN-Loss', loss, i)
        prog_bar.set_description('INN loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    chkpnt_file_snf = os.path.join(save_dir, 'snf.pt')
    chkpnt_file_diff = os.path.join(save_dir, 'diffusion.pt')
    chkpnt_file_inn = os.path.join(save_dir, 'INN.pt')
    torch.save(snf.state_dict(),chkpnt_file_snf)
    torch.save(diffusion_model.a.state_dict(),chkpnt_file_diff)
    torch.save(INN.state_dict(),chkpnt_file_inn)

    return snf, diffusion_model,INN

def evaluate(snf,diffusion_model, INN, out_dir, plot_dir, n_samples_y = 200, n_samples_x=5000,n_repeats=10, n_plots = 10):

    snf.eval()
    INN.eval()
    diffusion_model.eval()
    xs,ys = get_dataset(forward_model,a,b,size=n_samples_y)
    with torch.no_grad():

        nll_diffusion = []
        nll_mcmc = []
        nll_snf = []
        nll_inn = []
        kl1_sum = 0.
        kl2_sum = 0.
        kl3_sum = 0.
        kl1_vals = []
        kl2_vals = []
        kl3_vals = []
        nbins = 75
        #I just hardcode some ys that I use for plotting for reproducability
        plot_y = [0,5,6,20,23,42,50,77,81,93]
        prog_bar = tqdm(total=n_samples_y)
        for i, y in enumerate(ys):
            # testing
            hist_mcmc_sum = np.zeros((nbins, nbins, nbins))
            hist_snf_sum = np.zeros((nbins, nbins, nbins))
            hist_diffusion_sum = np.zeros((nbins, nbins, nbins))
            hist_inn_sum = np.zeros((nbins, nbins, nbins))
            nll_sum_mcmc = 0
            nll_sum_snf = 0
            nll_sum_diffusion = 0
            nll_sum_inn = 0.
            inflated_ys = y[None, :].repeat(n_samples_x, 1)

            mcmc_energy = lambda x: get_log_posterior(x, forward_model, a, b, inflated_ys, lambd_bd)
            
            for _ in range(n_repeats):
                x_pred_diffusion = get_grid(diffusion_model,y,xdim,ydim, num_samples=n_samples_x)
                x_pred_snf = snf.forward(torch.randn(n_samples_x, xdim, device=device), inflated_ys)[0].detach().cpu().numpy()
                x_pred_inn = INN(torch.randn(n_samples_x, xdim, device=device), c = inflated_ys)[0].detach().cpu().numpy()
                x_true = anneal_to_energy(torch.rand(n_samples_x, xdim, device=device) * 2 - 1, mcmc_energy, METR_STEPS,
                                 noise_std=NOISE_STD_MCMC)[0].detach().cpu().numpy()
        
                # generate histograms
                hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                              range=((-1, 1), (-1, 1), (-1, 1)))
                hist_snf, _ = np.histogramdd(x_pred_snf, bins=(nbins, nbins, nbins), range=((-1, 1), (-1, 1), (-1, 1)))
                hist_diffusion, _ = np.histogramdd(x_pred_diffusion, bins=(nbins, nbins, nbins), range=((-1, 1), (-1, 1), (-1, 1)))
                hist_inn, _ = np.histogramdd(x_pred_inn, bins=(nbins, nbins, nbins),
                                                   range=((-1, 1), (-1, 1), (-1, 1)))
                hist_mcmc_sum += hist_mcmc
                hist_snf_sum += hist_snf
                hist_diffusion_sum += hist_diffusion
                hist_inn_sum += hist_inn

                #calculate negaitve log likelihood of the samples
                nll_sum_snf += mcmc_energy(torch.from_numpy(x_pred_snf).to(device)).sum() / n_samples_x
                nll_sum_mcmc += mcmc_energy(torch.from_numpy(x_true).to(device)).sum() / n_samples_x
                nll_sum_diffusion += mcmc_energy(torch.from_numpy(x_pred_diffusion).to(device)).sum() / n_samples_x
                nll_sum_inn += mcmc_energy(torch.from_numpy(x_pred_inn).to(device)).sum() / n_samples_x
            if i in plot_y:
                # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
                fig, ax = pairplot([x_true])
                fig.suptitle('MCMC')
                fname = os.path.join(plot_dir, 'posterior-mcmc-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred_snf])
                fig.suptitle('SNF')
                fname = os.path.join(plot_dir, 'posterior-snf-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred_diffusion])
                fig.suptitle('DSM-Loss')
                fname = os.path.join(plot_dir, 'posterior-dsm-loss-%d.png' % i)
                plt.savefig(fname)
                plt.close()

                fig, ax = pairplot([x_pred_inn])
                fig.suptitle('GLOW')
                fname = os.path.join(plot_dir, 'posterior-inn-%d.png' % i)
                plt.savefig(fname)
                plt.close()

            hist_mcmc = hist_mcmc_sum / hist_mcmc_sum.sum()
            hist_snf = hist_snf_sum / hist_snf_sum.sum()
            hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
            hist_inn = hist_inn_sum / hist_inn_sum.sum()
            hist_mcmc += reg
            hist_snf += reg
            hist_diffusion += reg
            hist_inn += reg
            hist_mcmc /= hist_mcmc.sum()
            hist_snf /= hist_snf.sum()
            hist_diffusion /= hist_diffusion.sum()
            hist_inn /= hist_inn.sum()
        
            kl1 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_snf))
            kl2 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_diffusion))
            kl3 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_inn))
            kl1_sum += kl1
            kl2_sum += kl2
            kl3_sum += kl3
            kl1_vals.append(kl1)
            kl2_vals.append(kl2)
            kl3_vals.append(kl3)
            nll_mcmc.append(nll_sum_mcmc.item() / n_repeats)
            nll_snf.append(nll_sum_snf.item() / n_repeats)
            nll_inn.append(nll_sum_inn.item()/n_repeats)
            nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
            prog_bar.set_description('KL_SNF: {:.3f}, KL_diffusion: {:.3f}'.format(np.mean(kl1_vals),np.mean(kl2_vals)))
            prog_bar.update()

        prog_bar.close()
        kl1_vals = np.array(kl1_vals)
        kl2_vals = np.array(kl2_vals)
        kl3_vals = np.array(kl3_vals)
        kl1_var = np.sum((kl1_vals - kl1_sum / n_samples_y) ** 2) / n_samples_y
        kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
        kl3_var = np.sum((kl3_vals - kl3_sum / n_samples_y) ** 2) / n_samples_y
        nll_snf = np.array(nll_snf)
        nll_mcmc = np.array(nll_mcmc)
        nll_diffusion = np.array(nll_diffusion)
        nll_inn = np.array(nll_inn)
        df = pd.DataFrame(
            {'KL1': kl1_vals, 'KL2': kl2_vals, 'KL3': kl3_vals, 'NLL_mcmc': nll_mcmc, 'NLL_snf': nll_snf, 'NLL_diffusion': nll_diffusion, 'NLL_inn':nll_inn})
        df.to_csv(os.path.join(out_dir, 'results.csv'))
        print('KL1:', kl1_sum / n_samples_y, '+-', kl1_var)
        print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
        print('KL3:', kl3_sum / n_samples_y, '+-', kl3_var)

if __name__ == '__main__':

    # load forward model
    xdim=3
    ydim=23

    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)

    forward_model.load_state_dict(torch.load('examples/scatterometry/surrogate.pt', map_location=torch.device(device)))
    for param in forward_model.parameters():
        param.requires_grad = False

    a = 0.2
    b = 0.01
    n_epochs_snf = 200
    n_epochs_INN = 200
    n_epochs_diffusion = 20000

    train_dir = 'test/'
    log_dir = os.path.join(train_dir,'logs')
    out_dir = 'results/scatterometry'
    plot_dir = os.path.join(out_dir,'plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    snf,diffusion_model,INN = train(forward_model, n_epochs_snf, n_epochs_diffusion, n_epochs_INN,batch_size=1000, lambd_bd=1000, save_dir = train_dir,log_dir=log_dir)
    evaluate(snf,diffusion_model, INN, out_dir, n_samples_y=100, n_samples_x = 30000, n_plots = 10)
