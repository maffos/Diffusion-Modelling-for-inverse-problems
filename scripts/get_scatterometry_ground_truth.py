import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from models.diffusion import *
from models.SNF import anneal_to_energy, energy_grad
from examples.scatterometry.utils_scatterometry import *
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import torch

forward_model = nn.Sequential(nn.Linear(3, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, 23)).to(device)
src_dir = 'examples/scatterometry/'
forward_model.load_state_dict(
    torch.load(os.path.join(src_dir, 'surrogate.pt'), map_location=torch.device(device)))
for param in forward_model.parameters():
    param.requires_grad = False

a = 0.2
b = 0.01
lambd_bd = 1000
xdim = 3
ydim = 23

n_samples_y = 100
n_samples_x = 30000
n_epochs = 20000

# mcmc parameters for "discovering" the ground truth
NOISE_STD_MCMC = 0.5
METR_STEPS = 1000
RANDOM_STATE = 13
# hardcoded ys to plot the posterior for reproducibility (otherwise we would get ~2000 plots)
plot_y = [0, 5, 6, 20, 23, 42, 50, 77, 81, 93]
n_repeats = 10

def get_gt_samples(i,j):
    x_true = anneal_to_energy(torch.rand(n_samples_x, xdim, device=device) * 2 - 1, mcmc_energy, METR_STEPS,
                              noise_std=NOISE_STD_MCMC)[0].detach().cpu().numpy()
    out_dir = os.path.join(src_dir, str(i))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = os.path.join(out_dir,'%d.npy'%j)
    with open(filename, 'wb') as f:
        np.save(f, x_true)

    # only plot samples of the last repeat otherwise it gets too much and plot only for some sandomly selected y
    if i in plot_y and j == n_repeats-1:
        fig, ax = pairplot([x_true])
        fig.suptitle('MCMC')
        fname = os.path.join(out_dir, 'posterior-mcmc-nolimits%d.png' % i)
        plt.savefig(fname)
        plt.close()

if __name__ == '__main__':

    x_test, y_test = get_dataset(forward_model, a, b, size=n_samples_y)
    src_dir = os.path.join(src_dir, 'gt_samples')

    prog_bar = tqdm(total=n_samples_y)
    for i, y in enumerate(y_test):
        inflated_ys = y[None, :].repeat(n_samples_x, 1)
        mcmc_energy = lambda x: get_log_posterior(x, forward_model, a, b, inflated_ys, lambd_bd)
        Parallel(n_jobs=10)(delayed(get_gt_samples)(i,j) for j in range(n_repeats))
        prog_bar.update()


