import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import generate_dataset
from utils import plot_density
from examples.scatterometry.utils_scatterometry import get_dataset, get_forward_model_params
from models.diffusion import create_diffusion_model2,get_grid
from examples.linearModel.main_diffusion import f
import torch
import os
from tqdm import tqdm
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

models_to_plot = [
    'linearModel/results/CFM/PINNLoss4/3layer/L2/L1/lam:1.0/lam2:0.001/current_model.pt',
    'linearModel/results/FPE/PINNLoss4/3layer/L2/L2/lam:0.001/lam2:1.0/current_model.pt',
    'linearModel/results/CFM/PINNLoss4/3layer/L2/L2/lam:0.0001/lam2:0.0001/current_model.pt',
    'linearModel/results/CFM/ErmonLoss/3layer/L2/lam:1.0/current_model.pt',
    'linearModel/results/FPE/ErmonLoss/3layer/L1/lam:0.01/lr:0.0001/current_model.pt',
    'linearModel/results/FPE/ErmonLoss/3layer/L2/lam:0.0001/current_model.pt',
    'linearModel/results/FPE/PINNLoss4/3layer/L1/L1/lam:0.01/lam2:0.01/lr:0.0001/current_model.pt',
    'linearModel/results/FPE/PINNLoss4/3layer/L1/L2/lam:0.001/lam2:1.0/current_model.pt',
    'linearModel/results/FPE/PINNLoss4/3layer/L2/L1/lam:0.0001/lam2:0.0001/current_model.pt',
    'linearModel/results/baselines/diffusion.pt',
    'scatterometry/results/CFM/PINNLoss4/3layer/L2/L1/lam:0.1/lam2:0.1/diffusion.pt',
    'scatterometry/results/FPE/PINNLoss4/3layer/L1/L1/lam:0.001/lam2:0.1/diffusion.pt',
    'scatterometry/results/FPE/PINNLoss4/3layer/L1/L2/lam:0.0001/lam2:0.0001/diffusion.pt',
    'scatterometry/results/CFM/PINNLoss4/3layer/L2/L2/lam:0.001/lam2:0.0001/diffusion.pt',
    'scatterometry/results/CFM/ErmonLoss/3layer/L2/lam:0.0001/diffusion.pt',
    'scatterometry/results/FPE/ErmonLoss/3layer/L1/lam:0.0001/diffusion.pt',
    'scatterometry/results/FPE/ErmonLoss/3layer/L2/lam:0.0001/diffusion.pt',
    'scatterometry/results/FPE/PINNLoss4/3layer/L2/L1/lam:0.0001/lam2:0.001/diffusion.pt',
    'scatterometry/results/FPE/PINNLoss4/3layer/L2/L2/lam:0.0001/lam2:0.0001/diffusion.pt',
    'scatterometry/results/FPE/PINNLoss4/3layer/L2/L2/lam:0.01/lam2:1.0/diffusion.pt',
    'scatterometry/results/FPE/PINNLoss4/3layer/L2/L2/lam:0.01/lam2:0.0001/diffusion.pt',
    'scatterometry/results/FPE/PINNLoss4/3layer/L1/L2/lam:1.0/lam2:1.0/diffusion.pt',
    'scatterometry/results/baselines/diffusion.pt',
]
def replot(model,xdim,ydim,ys,plot_ys,out_dir,n_samples_x,linear = False, nbins = 75, figsize = (12,12),labelsize = 30, plot_outlier = False):

    model.eval()
    with (torch.no_grad()):
        prog_bar = tqdm(total=len(plot_ys))
        for i in plot_ys:
            y = ys[i]
            x_pred = get_grid(model, y, xdim, ydim, num_samples=n_samples_x)
            mean = np.mean(x_pred,axis=0)
            std = np.std(x_pred, axis=0)
            fname = os.path.join(out_dir, 'mean_std.csv')
            df = pandas.DataFrame({'mean': mean, 'std': std})
            df.to_csv(fname)
            if linear:
                plot_density(x_pred, nbins, limits=(-3.5, 3.5), xticks=[-3.5, 3.5], size=figsize, labelsize=labelsize,
                                   fname=os.path.join(out_dir, 'posterior-diffusion-xlim3-%d.svg' % i), show_mode=True)
                #plot_density(x_pred, nbins, limits=(-4, 4),xticks=[-4, 4], size=figsize, labelsize=labelsize,
                #            fname=os.path.join(out_dir, 'posterior-diffusion-xlim4-%d.svg' % i), show_mode=True)

            else:
                plot_density(x_pred, nbins, limits=(-1.2, 1.2),xticks = [-1,0,1], size=figsize,labelsize=labelsize,
                             fname=os.path.join(out_dir, 'posterior-diffusion-%d.svg' % i))

            if plot_outlier:
                plot_density(x_pred, nbins,size=figsize,labelsize=labelsize,
                             fname=os.path.join(out_dir, 'posterior-diffusion_no_limits-%d.svg' % i))
            prog_bar.update()

if __name__ == '__main__':

    n_samples_x = 30000
    n_samples_y = 100
    out_src = 'plots'
    for model_path in models_to_plot:
        model_path = os.path.join('examples',model_path)
        params = model_path.split('/')
        sub_path = '/'.join(params[3:-1])
        if params[1] == 'linearModel':
            skip = False
            xdim=2
            ydim=2
            # create data
            out_dir = os.path.join(out_src, 'linear',sub_path)
            xs, ys = generate_dataset(xdim,f,n_samples=100000)
            x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=.9, random_state=7)
            plot_ys = [3, 5, 22, 39, 51, 53, 60, 71, 81, 97]
            y_test = y_test[:n_samples_y]
            linear = True

        elif params[1] == 'scatterometry':
            skip = False
            out_dir = os.path.join(out_src,'scatterometry', sub_path)
            forward_model, a, b, lambd_bd, xdim, ydim = get_forward_model_params('examples/scatterometry')
            x_test,y_test = get_dataset(forward_model,a,b,size=n_samples_y)
            plot_ys = [0,5,6,20,23,42,50,77,81,93]
            linear = False

        if not skip:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print(out_dir)
            model = create_diffusion_model2(xdim, ydim, hidden_layers=[512, 512, 512])
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            model.a.load_state_dict(checkpoint)
            replot(model, xdim, ydim, y_test, plot_ys, out_dir, n_samples_x, linear)
