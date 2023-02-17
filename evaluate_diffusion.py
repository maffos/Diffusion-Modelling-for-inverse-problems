import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from sbi.analysis import pairplot, conditional_pairplot
import os
from libraries.sdeflow_light.lib import sdes, plotting
import sys
sys.path.append("/home/matthias/Uni/SoSe22/Master/Inverse-Modelling-of-Hemodynamics/")
from tqdm import tqdm
import nets
def generate_dataset(n_samples, random_state = 1):

    random_gen = torch.random.manual_seed(random_state)
    x = torch.randn(n_samples,xdim, generator = random_gen)
    y = f(x)
    #x = torch.from_numpy(x)
    #y = torch.from_numpy(y)
    return x.float(),y.float()

def check_posterior(x,y,posterior, prior, likelihood, evidence):


    log_p1 = posterior.log_prob(x)
    log_p2 = prior.log_prob(x)+likelihood.log_prob(y)-evidence.log_prob(y)

    print(log_p2, log_p1)
    #assert torch.allclose(log_p1, log_p2, atol = 1e-5), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)

def get_grid(sde, cond1,dim, n=4, num_samples = 2000, num_steps=200, transform=None,
             mean=0, std=1, clip=True):

    cond = torch.zeros(num_samples,dim)
    cond += cond1
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, dim)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1) * sde.T
    ones = torch.ones(num_samples, 1)

    with torch.no_grad():
        for i in range(num_steps):
            mu = sde.mu(ones * ts[i], y0, cond)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    y0 = y0.data.cpu().numpy()
    return y0

#toy function as forward problem
def f(x):
    return (A@x.T).T+b

def get_likelihood(x):

    mean = A@x+b
    return MultivariateNormal(mean,Sigma)

def get_evidence():
    mean = A@mu+b
    cov = Sigma+A@Lam@A.T

    return MultivariateNormal(mean,cov)

def get_posterior(y):
    y_res = y-(A@mu+b)
    mean = Lam@A.T@Sigma_y_inv@y_res
    cov = Lam-Lam@A.T@Sigma_y_inv@A@Lam

    return MultivariateNormal(mean,cov)

#analytical score of the posterior
def score_posterior(x,y):
    y_res = y-(x@A.T+b)
    score_prior = -x
    score_likelihood = (y_res@Sigma_inv.T)@A
    return score_prior+score_likelihood

def evaluate(model, xs,ys, n_samples = 2000):

    model.eval()
    with torch.no_grad():
        # some example distributions to plot
        prior = MultivariateNormal(torch.Tensor(mu), torch.Tensor(Lam))
        likelihood = get_likelihood(xs[0])
        evidence = get_evidence()
        posterior = get_posterior(ys[0])

        check_posterior(xs[0], ys[0], posterior, prior, likelihood, evidence)

        #log_plot(prior)
        #fig, ax = conditional_pairplot(likelihood, condition=xs[0], limits=[[-3, 3], [-3, 3]])
        #fig.suptitle('Likelihood at x=(%.2f,%.2f)'%(xs[0,0],xs[0,1]))
        #fig.show()
        fig, ax = conditional_pairplot(posterior, condition=ys[0], limits=[[-3, 3], [-3, 3]])
        fig.suptitle('Posterior at y=(%.2f,%.2f)'%(ys[0,0],ys[0,1]))
        fname = os.path.join(out_dir, 'posterior-true.png')
        plt.savefig(fname)
        fig.show()
        #x_pred = sample(model, y=ys[0], dt = .005, n_samples=n_samples)
        x_pred = get_grid(model,ys[0],2, num_samples=n_samples)
        #utils.make_image(x_pred, xs[0].detach().data.numpy().reshape(1, 2), num_epochs=500, show_plot=True, savefig=False)
        fig, ax = pairplot([x_pred], limits=[[-3, 3], [-3, 3]])
        fig.suptitle('N=%d samples from the posterior at y=(%.2f,%.2f)'%(n_samples,ys[0,0],ys[0,1]))
        fname = os.path.join(out_dir, 'posterior-predict.png')
        plt.savefig(fname)
        fig.show()

        mse_score = 0
        nll_sample = 0
        nll_true = 0

        # calculate MSE of score on test set
        t0 = torch.zeros(xs.shape[0], requires_grad=False).view(-1, 1)
        score_predict = model.a(xs, t0, ys)
        score_true = score_posterior(xs, ys)
        mse_score += torch.mean(torch.sum((score_predict - score_true) ** 2, dim=1))

        prog_bar = tqdm(total=len(xs))
        for x_true,y in zip(xs,ys):

            # calculate negative log likelihood of samples on test set
            x_predict = get_grid(reverse_process,y,2, num_samples=100)
            posterior = get_posterior(y)
            nll_sample += -torch.mean(posterior.log_prob(torch.Tensor(x_predict)))
            #calculate nll of true samples from posterior for reference
            nll_true += posterior.log_prob(x_true)
            prog_bar.set_description('NLL sample: %.4f, NLL true: %.4f'%(nll_sample,nll_true))
            prog_bar.update()

        prog_bar.close()
        mse_score /= xs.shape[0]
        nll_sample /= xs.shape[0]
        nll_true /= xs.shape[0]
        print('MSE: %.4f, NLL of samples: %.4f, NLL of true samples: %.4f'%(mse_score,nll_sample,nll_true))
        
if __name__ == '__main__':
    
    # define parameters of the inverse problem
    epsilon = 1e-6
    xdim = 2
    ydim = 2
    rand_gen = torch.manual_seed(0)
    # f is a shear by factor 0.5 in x-direction and tranlsation by (0.3, 0.5).
    A = torch.Tensor([[1, 0.5], [0, 1]])
    b = torch.Tensor([0.3, 0.5])
    scale = .3
    Sigma = scale * torch.eye(ydim)
    Lam = torch.eye(xdim)
    Sigma_inv = 1 / scale * torch.eye(ydim)
    Sigma_y_inv = torch.linalg.inv(Sigma + A @ Lam @ A.T + epsilon * torch.eye(ydim))
    mu = torch.zeros(xdim)

    # create data
    xs, ys = generate_dataset(n_samples=1000)
    random_gen = torch.random.manual_seed(0)
    perm = torch.randperm(len(xs), generator=random_gen)
    idx = int(.8 * len(xs))
    xs = xs[perm]
    ys = ys[perm]
    x_train = xs[:idx]
    x_test = xs[idx:]
    y_train = ys[:idx]
    y_test = ys[idx:]
    embed_dim = 2
    net_params = {'input_dim': xdim + ydim,
                  'output_dim': xdim,
                  'hidden_layers': [1024, 1024],
                  'embed_dim': embed_dim}
    out_dir = 'models/inverse_models/diffusion_FPE/dsm_loss'
    chkpnt_path = 'models/inverse_models/diffusion_FPE/dsm_loss/current_model.pt'
    forward_process = sdes.VariancePreservingSDE()
    score_net = nets.TemporalMLP_small(**net_params)
    checkpoint = torch.load(chkpnt_path, map_location=torch.device('cpu'))
    score_net.load_state_dict(checkpoint)
    reverse_process = sdes.PluginReverseSDE(forward_process, score_net, T=1, debias=False)
    evaluate(reverse_process, xs,ys)