import utils
from datasets import generate_dataset_scatterometry, get_dataloader_scatterometry, get_gt_samples_scatterometry
from models.SNF import *
from models.diffusion import *
from models.INN import *
from losses import *
from utils_scatterometry import *
from tqdm import tqdm
import os
import yaml
import scipy
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(snf,diffusion_model,INN,forward_model, forward_model_params,num_epochs_SNF, num_epochs_diffusion, num_epochs_INN,batch_size, save_dir,log_dir, lr, lr_INN):

    optimizer_snf = Adam(snf.parameters(), lr = lr)
    logger = SummaryWriter(log_dir)

    loss_fn_diffusion = DSMLoss()

    prog_bar = tqdm(total=num_epochs_SNF)
    for i in range(num_epochs_SNF):
        data_loader=get_dataloader_scatterometry(batch_size,forward_model, forward_model_params['a'],forward_model_params['b'],forward_model_params['lambd_bd'])
        loss = train_SNF_epoch(optimizer_snf, snf, data_loader,forward_model,  forward_model_params['a'],forward_model_params['b'])
        logger.add_scalar('Train/SNF-Loss', loss, i)
        prog_bar.set_description('SNF loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_diffusion = Adam(diffusion_model.sde.parameters(), lr = lr)
    prog_bar = tqdm(total=num_epochs_diffusion)
    for i in range(num_epochs_diffusion):
        data_loader=get_dataloader_scatterometry(batch_size,forward_model,forward_model_params['a'],forward_model_params['b'],forward_model_params['lambd_bd'])
        loss,logger_info = diffusion_model.train_epoch(optimizer_diffusion, loss_fn_diffusion, data_loader)
        logger.add_scalar('Train/diffusion-Loss', loss, i)
        prog_bar.set_description('diffusion loss:{:.3f}'.format(loss))
        prog_bar.update()
    prog_bar.close()

    optimizer_inn = Adam(INN.parameters(), lr=lr_INN)

    prog_bar = tqdm(total=num_epochs_INN)
    for i in range(num_epochs_INN):
        data_loader = get_dataloader_scatterometry(batch_size, forward_model,forward_model_params['a'],forward_model_params['b'],forward_model_params['lambd_bd'])
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
    torch.save(diffusion_model.sde.a.state_dict(),chkpnt_file_diff)
    torch.save(INN.state_dict(),chkpnt_file_inn)

    return snf, diffusion_model,INN

def evaluate(ys, snf, diffusion_model, INN, forward_model, out_dir, plot_ys, score_posterior, gt_path, n_samples_x=30000, n_repeats=10, epsilon = 1e-10, xlim = (-1.2,1.2),nbins = 75, figsize = (12,12), labelsize = 30):

    n_samples_y = len(ys)
    nll_diffusion = []
    nll_mcmc = []
    nll_snf = []
    nll_inn = []
    kl1_sum = 0.
    kl2_sum = 0.
    kl3_sum = 0.
    kl1_vals = []
    kl1_reverse_vals = []
    kl2_vals = []
    kl2_reverse_vals = []
    kl3_vals = []
    kl3_reverse_vals = []
    mse_score_vals = []

    prog_bar = tqdm(total=n_samples_y)
    for i, y in enumerate(ys):
        hist_mcmc_sum = np.zeros((nbins, nbins, nbins))
        hist_snf_sum = np.zeros((nbins, nbins, nbins))
        hist_diffusion_sum = np.zeros((nbins, nbins, nbins))
        hist_inn_sum = np.zeros((nbins, nbins, nbins))
        nll_sum_mcmc = 0
        nll_sum_snf = 0
        nll_sum_diffusion = 0
        nll_sum_inn = 0.
        mse_score_sum = 0
        inflated_ys = y[None, :].repeat(n_samples_x, 1)

        mcmc_energy = lambda x: get_log_posterior(x, forward_model, forward_model_params['a'], forward_model_params['b'], inflated_ys, forward_model_params['lambd_bd'])

        for j in range(n_repeats):
            x_pred_diffusion = diffusion_model(y,num_samples=n_samples_x)
            x_pred_snf = snf.forward(torch.randn(n_samples_x, forward_model_params['xdim'], device=device), inflated_ys)[0].detach().cpu().numpy()
            x_pred_inn = INN(torch.randn(n_samples_x, forward_model_params['xdim'], device=device), c = inflated_ys)[0].detach().cpu().numpy()
            x_true = get_gt_samples_scatterometry(gt_path, i,j)
            x_true_tensor = torch.from_numpy(x_true).to(device)

            # calculate MSE of score on test set
            t0 = torch.zeros(x_true.shape[0], requires_grad=False).view(-1, 1).to(device)
            g_0 = diffusion_model.sde.base_sde.g(t0, x_true_tensor)
            score_predict = diffusion_model.sde.a(x_true_tensor, inflated_ys.to(device), t0.to(device)) / g_0
            score_true = score_posterior(x_true_tensor, inflated_ys)
            mse_score_sum += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

            # generate histograms
            hist_mcmc, _ = np.histogramdd(x_true, bins=(nbins, nbins, nbins),
                                          range=(xlim, xlim, xlim))
            hist_snf, _ = np.histogramdd(x_pred_snf, bins=(nbins, nbins, nbins), range=(xlim, xlim, xlim))
            hist_diffusion, _ = np.histogramdd(x_pred_diffusion, bins=(nbins, nbins, nbins), range=(xlim, xlim, xlim))
            hist_inn, _ = np.histogramdd(x_pred_inn, bins=(nbins, nbins, nbins),
                                               range=(xlim, xlim, xlim))
            hist_mcmc_sum += hist_mcmc
            hist_snf_sum += hist_snf
            hist_diffusion_sum += hist_diffusion
            hist_inn_sum += hist_inn

            #calculate negaitve log likelihood of the samples
            nll_sum_snf += mcmc_energy(torch.from_numpy(x_pred_snf).to(device)).sum() / n_samples_x
            nll_sum_mcmc += mcmc_energy(torch.from_numpy(x_true).to(device)).sum() / n_samples_x
            nll_sum_diffusion += mcmc_energy(torch.from_numpy(x_pred_diffusion).to(device)).sum() / n_samples_x
            nll_sum_inn += mcmc_energy(torch.from_numpy(x_pred_inn).to(device)).sum() / n_samples_x
        if i in plot_ys:
            utils.plot_density(x_true, nbins, limits=xlim, xticks=(-1,0,1), size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-true-%d.svg' % i), show_mean=False)

            utils.plot_density(x_pred_diffusion, nbins, limits=xlim, xticks=(-1,0,1), size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-diffusion-%d.svg' % i), show_mean=False)

            utils.plot_density(x_pred_snf, nbins, limits=xlim, xticks=(-1,0,1), size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-snf-%d.svg' % i), show_mean=False)

            utils.plot_density(x_pred_inn, nbins, limits=xlim, xticks=(-1,0,1), size=figsize,
                               labelsize=labelsize,
                               fname=os.path.join(out_dir, 'posterior-inn-%d.svg' % i), show_mean=False)


        hist_mcmc = hist_mcmc_sum / hist_mcmc_sum.sum()
        hist_snf = hist_snf_sum / hist_snf_sum.sum()
        hist_diffusion = hist_diffusion_sum / hist_diffusion_sum.sum()
        hist_inn = hist_inn_sum / hist_inn_sum.sum()
        hist_mcmc += epsilon
        hist_snf += epsilon
        hist_diffusion += epsilon
        hist_inn += epsilon
        hist_mcmc /= hist_mcmc.sum()
        hist_snf /= hist_snf.sum()
        hist_diffusion /= hist_diffusion.sum()
        hist_inn /= hist_inn.sum()

        kl1 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_snf))
        kl1_reverse = np.sum(scipy.special.rel_entr(hist_snf,hist_mcmc))
        kl2 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_diffusion))
        kl2_reverse = np.sum(scipy.special.rel_entr(hist_diffusion,hist_mcmc))
        kl3 = np.sum(scipy.special.rel_entr(hist_mcmc, hist_inn))
        kl3_reverse = np.sum(scipy.special.rel_entr(hist_inn,hist_mcmc))
        kl1_sum += kl1
        kl2_sum += kl2
        kl3_sum += kl3
        kl1_vals.append(kl1)
        kl1_reverse_vals.append(kl1_reverse)
        kl2_vals.append(kl2)
        kl2_reverse_vals.append(kl2_reverse)
        kl3_vals.append(kl3)
        kl3_reverse_vals.append(kl3_reverse)
        nll_mcmc.append(nll_sum_mcmc.item() / n_repeats)
        nll_snf.append(nll_sum_snf.item() / n_repeats)
        nll_inn.append(nll_sum_inn.item()/n_repeats)
        nll_diffusion.append(nll_sum_diffusion.item() / n_repeats)
        mse_score_vals.append(mse_score_sum.item() / n_repeats)

        prog_bar.set_description('KL_SNF: {:.3f}, KL_diffusion: {:.3f}'.format(np.mean(kl1_vals),np.mean(kl2_vals)))
        prog_bar.update()

    prog_bar.close()
    kl1_vals = np.array(kl1_vals)
    kl1_reverse_vals = np.array(kl1_reverse_vals)
    kl2_vals = np.array(kl2_vals)
    kl2_reverse_vals = np.array(kl2_reverse_vals)
    kl3_vals = np.array(kl3_vals)
    kl3_reverse_vals = np.array(kl3_reverse_vals)
    kl1_var = np.sum((kl1_vals - kl1_sum / n_samples_y) ** 2) / n_samples_y
    kl2_var = np.sum((kl2_vals - kl2_sum / n_samples_y) ** 2) / n_samples_y
    kl3_var = np.sum((kl3_vals - kl3_sum / n_samples_y) ** 2) / n_samples_y
    nll_snf = np.array(nll_snf)
    nll_mcmc = np.array(nll_mcmc)
    nll_diffusion = np.array(nll_diffusion)
    nll_inn = np.array(nll_inn)
    df = pd.DataFrame(
        {'KL_SNF': kl1_vals, 'KL_SNF_reverse': kl1_reverse_vals, 'KL_diffusion': kl2_vals,
         'KL_diffusion_reverse': kl2_reverse_vals,'KL_INN': kl3_vals, 'KL_INN_reverse':kl3_reverse_vals,
         'NLL_mcmc': nll_mcmc, 'NLL_snf': nll_snf, 'NLL_diffusion': nll_diffusion, 'NLL_inn':nll_inn,
         'MSE':np.array(mse_score_vals)})
    df.to_csv(os.path.join(out_dir, 'results.csv'))
    print('KL1:', kl1_sum / n_samples_y, '+-', kl1_var)
    print('KL2:', kl2_sum / n_samples_y, '+-', kl2_var)
    print('KL3:', kl3_sum / n_samples_y, '+-', kl3_var)
    

if __name__ == '__main__':

    # load config params
    config_dir = 'config/'
    config = yaml.safe_load(open(os.path.join(config_dir, "config_baselines_scatterometry.yml")))

    surrogate_dir = 'trained_models/scatterometry'
    gt_dir = 'data/gt_samples_scatterometry'

    # load the forward model
    forward_model, forward_model_params = load_forward_model(surrogate_dir)

    # define function to calcualate the score of the posterior and negative log posterior
    score_posterior = lambda x, y: -energy_grad(x,
                                                lambda x: get_log_posterior(x, forward_model, forward_model_params['a'],
                                                                            forward_model_params['b'], y,
                                                                            forward_model_params['lambd_bd']))[0]
    log_posterior = lambda samples, ys: get_log_posterior(samples, forward_model, forward_model_params['a'], forward_model_params['b'], ys, forward_model_params['lambd_bd'])

    log_dir = utils.set_directories(config['train_dir'], config['out_dir'])

    snf = create_snf(config['num_layers_INN'], config['size_hidden_layers_INN'], log_posterior,
                     metr_steps_per_block=config['metr_steps_per_block'], dimension=forward_model_params['xdim'], dimension_condition=forward_model_params['ydim'],
                     noise_std=config['noise_std'])

    if config['model'] == 'CDE':
        diffusion_model = CDE(xdim=forward_model_params['xdim'], ydim=forward_model_params['ydim'], hidden_layers=config['hidden_layers'])
    elif config['model'] == 'CDiffE':
        diffusion_model = CDiffE(xdim=forward_model_params['xdim'], ydim=forward_model_params['ydim'], hidden_layers=config['hidden_layers'])

    INN = create_INN(config['num_layers_INN'], config['size_hidden_layers_INN'], dimension=forward_model_params['xdim'], dimension_condition=forward_model_params['ydim'])

    # generate test set
    x_test, y_test = generate_dataset_scatterometry(forward_model, forward_model_params['a'], forward_model_params['b'],
                                                    size=config['n_samples_y'])

    snf,diffusion_model,INN = train(snf,diffusion_model,INN, forward_model, forward_model_params,
                                    config['n_epochs_SNF'], config['n_epochs_dsm'],config['n_epochs_INN'],
                                    batch_size=config['batch_size'], save_dir=config['train_dir'],
                                    log_dir=log_dir, lr = config['lr'], lr_INN = config['lr_INN'])
    evaluate(y_test,snf, diffusion_model, INN, forward_model, config['out_dir'], config['plot_ys'], score_posterior, gt_dir, n_samples_x=config['n_samples_x'])
