def check_posterior(x,y,posterior, prior, likelihood, evidence):


    log_p1 = posterior.log_prob(x)
    log_p2 = prior.log_prob(x)+likelihood.log_prob(y)-evidence.log_prob(y)

    print(log_p2, log_p1)
    #assert torch.allclose(log_p1, log_p2, atol = 1e-5), "2 ways of calculating the posterior should be the same but are {} and {}".format(log_p1, log_p2)

def check_diffusion(model, n_samples, num_plots):
    for i in range(num_plots):
        x = torch.randn(2)
        y = f(x)
        posterior = get_posterior(y)
        x_0 = posterior.sample((n_samples,))
        T = torch.ones((x_0.shape[0],1))
        x_T, target_T, std_T, g_T = model.base_sde.sample(T, x_0, return_noise=True)
        x_prior = torch.randn((n_samples,2))
        kl_div = nn.functional.kl_div(x_T, x_prior)
        fig, ax = pairplot([x_0], condition=y, limits=[[-3, 3], [-3, 3]])
        fig.suptitle('Samples from the Posterior at y=(%.2f,%.2f)' % (y[0], y[1]))
        fname = 'posterior-true%d.png' % i
        plt.savefig(fname)
        plt.show()
        plt.close()
        heatmap, xedges, yedges = np.histogram2d(x_T[:,0].data.numpy(), x_T[:,1].data.numpy(), bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title('Samples from the prior by running the SDE')
        fname = 'prior-diffusion%d.png'%i
        plt.savefig(fname)
        plt.show()
        plt.close()
        heatmap, xedges, yedges = np.histogram2d(x_prior[:,0].data.numpy(), x_prior[:,1].data.numpy(), bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title('Samples from the true prior')
        fname = 'prior-true%d.png'%i
        plt.savefig(fname)
        plt.show()
        plt.close()
        print('KL Divergence = %.4f'%kl_div)