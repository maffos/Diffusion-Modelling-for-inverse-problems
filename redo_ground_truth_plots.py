import matplotlib as plt
from utils import plot_density,get_linear_params,generate_dataset
from examples.scatterometry.utils_scatterometry import get_forward_model_params,get_dataset,get_gt_samples
from examples.linearModel.main_diffusion import get_posterior,f
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def replot(ys,plot_ys, n_samples_x, figsize, labelsize, nbins, out_dir, linear = False, **kwargs):

    prog_bar = tqdm(total=len(plot_ys))
    for i in plot_ys:
        y = ys[i]

        if linear:
            posterior = get_posterior(y)
            x_true = posterior.sample((n_samples_x,)).numpy()

            plot_density(x_true, nbins, limits=(-3.5, 3.5), xticks=[-3.5, 3.5], size=figsize, labelsize=labelsize,
                                   fname=os.path.join(out_dir, 'posterior-true-xlim3-%d.svg' % i), show_mode=True)
            #plot_density(x_true, nbins, limits=(-4, 4), xticks=[-4, 4], size=figsize,
            #            labelsize=labelsize,
            #            fname=os.path.join(out_dir, 'posterior-true-xlim4-%d.svg' % i), show_mode = True)
        else:
            x_true = get_gt_samples(kwargs['gt_path'], i, 9)
            plot_density(x_true, nbins, limits=(-1.2, 1.2), xticks=[-1, 0, 1], size=figsize,
                         labelsize=labelsize,
                         fname=os.path.join(out_dir, 'posterior-mcmc-%d.svg' % i))

        prog_bar.update()


if __name__ == '__main__':

    epsilon, xdim, ydim, A, b, scale, Sigma, Lam, Sigma_inv, Sigma_y_inv, mu = get_linear_params()
    nbins = 75
    figsize = (12, 12)
    labelsize = 30
    n_samples_x = 30000
    n_samples_y = 100
    surrogate_dir = 'examples/scatterometry'
    plot_dir = 'plots/scatterometry/mcmc'
    gt_dir = 'examples/scatterometry/gt_samples'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    forward_model, a, b, lambd_bd, xdim, ydim = get_forward_model_params(surrogate_dir)
    plot_ys = [0, 5, 6, 20, 23, 42, 50, 77, 81, 93]
    x_test, y_test = get_dataset(forward_model,a,b,size = n_samples_y)
    replot(y_test, plot_ys, n_samples_x,figsize = (12,12),labelsize = 30, nbins = 75, out_dir = plot_dir, linear=False, gt_path = gt_dir)

    plot_dir = 'plots/linear/ground_truth'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    epsilon, xdim, ydim, A, b, scale, Sigma, Lam, Sigma_inv, Sigma_y_inv, mu = get_linear_params()
    xs,ys = generate_dataset(xdim, f, n_samples=100000)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=.9, random_state=7)
    plot_ys = [3, 5, 22, 39, 51, 53, 60, 71, 81, 97]
    replot(y_test[:n_samples_y],plot_ys, n_samples_x,figsize = (12,12),labelsize = 30, nbins = 75, out_dir = plot_dir, linear=True)
