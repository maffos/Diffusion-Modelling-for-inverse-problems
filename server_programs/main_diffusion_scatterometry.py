import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from models.diffusion import *
from models.SNF import anneal_to_energy, energy_grad
from utils_scatterometry import *
from utils import product_dict
from losses import *
import scipy
import pandas as pd
from tqdm import tqdm
import os
import shutil
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# mcmc parameters for "discovering" the ground truth
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000

def train_epoch(optimizer, loss_fn, model, epoch_data_loader, t_min):
    mean_loss = 0
    logger_info = {}
    model.train()
    for k, (x, y) in enumerate(epoch_data_loader()):

        t = sample_t(model,x)
        loss = loss_fn(model,x,t,y)
        if isinstance(loss, tuple):
            loss_info = loss[1]
            loss = loss[0]
            for key,value in loss_info.items():
                try:
                    logger_info[key] = logger_info[key] * k / (k + 1) + value.item() / (k + 1)
                except:
                    logger_info[key] = 0
                    logger_info[key] = logger_info[key] * k / (k + 1) + value.item() / (k + 1)

        if torch.min(t) < t_min:
            t_min = torch.min(t)
        if torch.isnan(loss):
            for key, value in loss_info.items():
                print(key + ':' + str(value))
            raise ValueError(
                'Loss is nan, min sampled t was %f. Minimal t during training was %f' % (torch.min(t), t_min))
        #loss = model.dsm(x,y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss, logger_info, t_min

def train(model, optimizer, loss_fn, forward_model, a,b,lambd_bd, num_epochs, batch_size, save_dir, log_dir):

    logger = SummaryWriter(log_dir)
    #track the smallest t during training for debugging
    t_min = torch.inf
    prog_bar = tqdm(total=num_epochs)
    for i in range(num_epochs):
        data_loader = get_epoch_data_loader(batch_size, forward_model, a, b, lambd_bd)
        loss,logger_info, t_min = train_epoch(optimizer, loss_fn, model, data_loader, t_min)
        prog_bar.set_description('Loss:{:.3f}'.format(loss))
        logger.add_scalar('Train/Loss', loss, i)
        for key,value in logger_info.items():
            logger.add_scalar('Train/'+key, value, i)
        prog_bar.update()
    prog_bar.close()

    print('Minimal t during training was %f'%t_min)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    chkpnt_file = os.path.join(save_dir, 'diffusion.pt')
    torch.save(model.a.state_dict(), chkpnt_file)

    return model

def evaluate(model,ys,forward_model, a,b,lambd_bd, out_dir, n_samples_x=5000,n_repeats=10, epsilon=1e-10):
    n_samples_y = ys.shape[0]
    model.eval()
    nll_diffusion = []
    nll_mcmc = []
    kl2_sum = 0.
    kl2_vals = []
    mse_score_vals = []
    nbins = 75
    # randomly select some y's to plot the posterior (otherwise we would get ~2000 plots)
    plot_y = [0,5,6,20,23,42,50,77,81,93]  
    prog_bar = tqdm(total=n_samples_y)
    for i, y in enumerate(ys):
        # testing
        hist_mcmc_sum = np.zeros((nbins, nbins, nbins))
        hist_diffusion_sum = np.zeros((nbins, nbins, nbins))
        nll_sum_mcmc = 0
        nll_sum_diffusion = 0
        mse_score_sum = 0
        inflated_ys = y[None, :].repeat(n_samples_x, 1)
        mcmc_energy = lambda x: get_log_posterior(x, forward_model, a, b, inflated_ys, lambd_bd)

        for _ in range(n_repeats):
            x_pred = get_grid(model, y, xdim, ydim, num_samples=n_samples_x)
            x_true = anneal_to_energy(torch.rand(n_samples_x, xdim, device=device) * 2 - 1, mcmc_energy, METR_STEPS,noise_std=NOISE_STD_MCMC)[0].detach().cpu().numpy()
            x_true_tensor = torch.from_numpy(x_true).to(device)
            # calculate MSE of score on test set
            t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1).to(device)
            g_0 = model.base_sde.g(t0, x_true_tensor)
            score_predict = model.a(x_true_tensor, t0.to(device), inflated_ys.to(device)) / g_0             
            score_true = score_posterior(x_true_tensor,inflated_ys)
            #score_true = -energy_grad(x_true_tensor,mcmc_energy)
            mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))
            # generate histograms
            hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                          range=((-1, 1), (-1, 1), (-1, 1)))
            hist_diffusion, _ = np.histogramdd(x_pred, bins=(nbins, nbins, nbins),
                                               range=((-1, 1), (-1, 1), (-1, 1)))

            hist_mcmc_sum += hist_mcmc
            hist_diffusion_sum += hist_diffusion
             # calculate negaitve log likelihood of the samples
            nll_sum_mcmc += mcmc_energy(x_true_tensor).sum() / n_samples_x
            nll_sum_diffusion += mcmc_energy(torch.from_numpy(x_pred).to(device)).sum() /n_samples_x
         # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
        if i in plot_y:
            fig, ax = pairplot([x_true], limits = [[-1,1],[-1,1],[-1,1]])
            fig.suptitle('MCMC')
            fname = os.path.join(out_dir, 'posterior-mcmc-%d.png' % i)
            plt.savefig(fname)
            plt.close()

            fig, ax = pairplot([x_pred], limits = [[-1,1],[-1,1],[-1,1]])
            fig.suptitle('PINN-Loss')
            fname = os.path.join(out_dir, 'posterior-diffusion-limits%d.png' % i)
            plt.savefig(fname)
            plt.close()
            fig, ax = pairplot([x_pred])
            fig.suptitle('PINN-Loss')
            fname = os.path.join(out_dir, 'posterior-diffusion-nolimits%d.png' % i)
            plt.savefig(fname)
            plt.close()

        hist_mcmc = hist_mcmc_sum / hist_mcmc_sum.sum()
        hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
        hist_mcmc += epsilon
        hist_diffusion += epsilon
        hist_mcmc /= hist_mcmc.sum()
        hist_diffusion /= hist_diffusion.sum()
        kl2 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_diffusion))
        kl2_sum += kl2
        kl2_vals.append(kl2)
        nll_mcmc.append(nll_sum_mcmc.item() / n_repeats)
        nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
        mse_score_vals.append(mse_score_sum.item()/n_repeats)
        prog_bar.set_description(
                'KL_diffusion: {:.3f}'.format(np.mean(kl2_vals)))
        prog_bar.update()

    prog_bar.close()
    kl2_vals = np.array(kl2_vals)
    kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
    nll_mcmc = np.array(nll_mcmc)
    nll_diffusion = np.array(nll_diffusion)
    df = pd.DataFrame({'KL2': kl2_vals, 'NLL_mcmc': nll_mcmc,'NLL_diffusion': nll_diffusion,'MSE':np.array(mse_score_vals)})
    df.to_csv(os.path.join(out_dir, 'results.csv'))
    print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
        
if __name__ == '__main__':

    forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)
    surrogate_fn = 'models/surrogate/scatterometry_surrogate.pt'
    forward_model.load_state_dict(
        torch.load(surrogate_fn, map_location=torch.device(device)))
    for param in forward_model.parameters():
        param.requires_grad = False

    a = 0.2
    b = 0.01
    lambd_bd = 1000
    xdim = 3
    ydim = 23

    n_samples_y = 100
    n_samples_x = 30000
    x_test,y_test = get_dataset(forward_model,a,b,size=n_samples_y)
    n_epochs = 20000


    score_posterior = lambda x,y: -energy_grad(x, lambda x:  get_log_posterior(x,forward_model,a,b,y,lambd_bd))[0]
    score_prior = lambda x: -x

    params = {'loss_fn': ['PINNLoss4'],
    'lr': [1e-4],
    'lam': [1.,.1,.001],
    'lam2':[1.,.1],
    'pde_loss': ['FPE'],
    'metric': ['L1'],
    'ic_metric':['L1'],
    'hidden_layers':['3layer']}
    
    resume_training = False
    src_dir = 'new_folder_strucutre_scatterometry'
    for param_configuration in product_dict(**params):
        skip = False
        lr = param_configuration.pop('lr')
        num_layers = param_configuration.pop('hidden_layers')
        if num_layers == '2layer':
            hidden_layers = [512,512]
        elif num_layers == '3layer':
            hidden_layers = [512,512,512]
        elif num_layers== '4layer':
            hidden_layers = [512,512,512,512]
        else:
            hidden_layers = [512,512,512,512,512]
        loss_fn = param_configuration.pop('loss_fn')
        if loss_fn == 'PINNLoss2':
            loss_fn = PINNLoss2(initial_condition = score_posterior,boundary_condition=score_prior,**param_configuration)
        elif loss_fn == 'PINNLoss4':
            loss_fn = PINNLoss4(initial_condition= score_posterior,**param_configuration)
        elif loss_fn == 'PINNLoss3':
            loss_fn = PINNLoss3(initial_condition = score_posterior, **param_configuration)
        elif loss_fn == 'ErmonLoss':
            if 'lam2' in param_configuration.keys():
                _ = param_configuration.pop('lam2')
            if 'ic_metric' in param_configuration.keys():
                _ = param_configuration.pop('ic_metric')
            loss_fn = ErmonLoss(**param_configuration)
        else:
            raise ValueError('No correct loss fn specified')
        if param_configuration['metric'] == 'L1' and param_configuration['pde_loss'] == 'CFM':
            skip = True
        if not skip:
            model = create_diffusion_model2(xdim,ydim,hidden_layers)
            optimizer = Adam(model.a.parameters(), lr=lr)

            if loss_fn.name == 'ErmonLoss':
                train_dir = os.path.join(src_dir,param_configuration['pde_loss'], loss_fn.name, num_layers, param_configuration['metric'], 'lam:{}'.format(param_configuration['lam']))
            else:
                train_dir = os.path.join(src_dir,param_configuration['pde_loss'], loss_fn.name, num_layers, param_configuration['metric'],param_configuration['ic_metric'], 'lam:{}'.format(param_configuration['lam']),'lam2:{}'.format(param_configuration['lam2']))
            log_dir = os.path.join(train_dir, 'logs')
            
            if os.path.exists(log_dir) and not resume_training:
                shutil.rmtree(log_dir)
                
            out_dir = os.path.join(train_dir, 'results')
            if os.path.exists(out_dir) and not resume_training:
                shutil.rmtree(out_dir)
                
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print('-----------------')
            print(vars(loss_fn))
            model = train(model, optimizer, loss_fn,forward_model, a,b,lambd_bd, n_epochs, batch_size=1000,save_dir=train_dir, log_dir = log_dir)
            evaluate(model, y_test, forward_model, a,b,lambd_bd, out_dir, n_samples_x=n_samples_x)
