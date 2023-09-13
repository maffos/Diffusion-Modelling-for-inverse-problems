import numpy as np
import matplotlib.pyplot as plt
import utils
from models.diffusion import *
from losses import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.distributions import MultivariateNormal
def sample_gmm(num_samples=1):
    """
    Sample from the Gaussian Mixture Distribution:
    1/5 * N([-5,5], I) + 4/5 * N([5,5], I)

    Args:
    - num_samples (int): Number of samples to draw.

    Returns:
    - samples (np.array): Array of shape (num_samples, 2) containing the samples.
    """
    samples = []

    for _ in range(num_samples):
        # Decide which Gaussian to sample from based on the weights
        choice = np.random.choice([0, 1], p=[.2, .8])

        if choice == 0:
            mean = [-5, -5]
        else:
            mean = [5, 5]

        # Sample from the chosen Gaussian with identity covariance
        sample = np.random.multivariate_normal(mean, np.eye(2))
        samples.append(sample)

    return np.array(samples)


def plot_samples(samples):
    """
    Plot the samples on a 2D plane.

    Args:
    - samples (np.array): Array of shape (num_samples, 2) containing the samples.
    """
    plt.scatter(samples[:, 0], samples[:, 1], marker='o', color='blue', alpha=0.5)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Samples from GMD')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    # Test the function
    samples_true = sample_gmm(5000,)
    plot_samples(samples_true)

    #define parameters
    src_dir = 'test'
    hidden_layers = [512,512,512]
    resume_training = False
    pde_loss = 'FPE'
    lam = .1
    lr = 1e-4
    xdim,ydim = 2,2
    score_posterior =
    #define models
    model = create_diffusion_model2(xdim,ydim, hidden_layers=hidden_layers)
    #loss_fn =PINNLoss2(initial_condition=score_posterior, boundary_condition=lambda x: -x, pde_loss=pde_loss, lam=lam)
    loss_fn = PINNLoss3(initial_condition=score_posterior, lam=.1,lam2=1., pde_loss = pde_loss)
    #loss_fn = ScoreFlowMatchingLoss(lam=.1)
    #loss_fn = PINNLoss3(initial_condition = score_posterior, lam = .1, lam2 = 1)
    #loss_fn = ErmonLoss(lam=0.1, pde_loss = 'FPE')
    optimizer = Adam(model.a.parameters(), lr = 1e-4)

    train_dir = os.path.join(src_dir,loss_fn.name, 'lam=0.1')
    if resume_training:
        model.a.load_state_dict(torch.load(os.path.join(train_dir,'current_model.pt'),map_location=torch.device(device)))
        out_dir = os.path.join(train_dir, 'results_resume')
    else:
        out_dir = os.path.join(train_dir, 'results')

    if os.path.exists(out_dir) and not resume_training:
        shutil.rmtree(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_dir = os.path.join(train_dir, 'logs')

    if os.path.exists(log_dir) and not resume_training:
        shutil.rmtree(log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    model = train(model,x_train,y_train, optimizer, loss_fn, train_dir, log_dir, num_epochs=200, resume_training = resume_training)
    #we need to wrap the reverse SDE into an own class to use the integration method from torchsde
    #reverse_process = SDE(reverse_process.a, reverse_process.base_sde, xdim, ydim, sde_type='stratonovich')
    evaluate(model, x_test[:100], y_test[:100], out_dir, n_samples = 20000, n_plots=10)
