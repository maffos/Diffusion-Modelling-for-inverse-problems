import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def rademacher_like(s):

    v = torch.distributions.bernoulli.Bernoulli(torch.ones_like(s)*.5).sample()
    v[torch.where(v==0)]=-1
    return v

#function copied from https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True, retain_graph=True)[0][..., i:i+1]
    return div

def batch_gradient(y,x):
    grad = torch.zeros_like(y)
    for i in range(y.shape[1]):
        dy_dx = torch.autograd.grad(y[:,i].sum(),x, retain_graph=True, create_graph=True)[0]
        dy_dx = dy_dx.view(-1)
        grad[:,i] += dy_dx
    return grad

def div_estimator(s,x,num_samples=1, rademacher = True):

    div = torch.zeros(s.shape[0],1)
    for _ in range(num_samples):
        if rademacher:
            v = rademacher_like(s)
        else:
            v = torch.randn_like(s)
        vjp = torch.autograd.grad(s,x,grad_outputs = v, create_graph=True, retain_graph=True)[0]
        div += (vjp[:,None,:]@v[:,:,None]).view(-1,1)

    div /= num_samples
    return div

class DSMLoss(nn.Module):
    
    def __init__(self):

        super(DSMLoss,self).__init__()
        self.name = 'DSMLoss'
                 
    def forward(self,s, std,target):

        batch_size = s.shape[0]
        return ((s * std + target) ** 2).view(batch_size, -1).sum(1, keepdim=False) / 2

class CDiffELoss(nn.Module):
    def __init__(self):
        self.name = 'CDiffE'

    def forward(self, model):
        pass
        
class ScoreFPELoss(nn.Module):
    """
       Implements the Loss based on the ScoreFPE.

        Attributes:
            name (str): A name for the loss function, set to 'FPELoss'.
            metric (str): The metric used for loss computation. Defaults to 'l1', can be 'L1' or 'L2'.

        Methods:
            forward(s, x_t, t, beta, divergence_method='exact'): Computes the ScoreFPE loss.

        Args:
            metric (str, optional): The metric used for computing the loss. Can be 'L1' or 'L2'.
                                    Defaults to 'L1'.

        """
    def __init__(self, metric = 'L1'):

        super(ScoreFPELoss,self).__init__()
        self.name = 'FPELoss'
        self.metric = metric

    def forward(self, s, x_t,t, beta, divergence_method = 'exact'):

        assert s.shape == x_t.shape, 's and x_t need to have the same shape, but {} and {} was given, repsectively.'.format(s.shape, x_t.shape)
        batch_size = x_t.shape[0]
        if divergence_method == 'exact':
            divx_s = divergence(s,x_t)
        elif divergence_method in ['hutchinson', 'approx', 'approximate']:
            divx_s = div_estimator(s,x_t)
        else:
            raise ValueError('No valid value for divergence method specified. Need to be one of "exact","hutchinson","approx" or "approximate", but {} was given'.format)

        ds_dt = batch_gradient(s,t)
        grad_x = torch.autograd.grad(divx_s + torch.sum(s ** 2,dim=1).view(-1,1) + (x_t[:, None, :] @ s[:, :, None]).view(-1,1),
        x_t, grad_outputs=torch.ones_like(divx_s), retain_graph=True)[0]

        if self.metric == 'L1':
            loss = torch.mean(torch.abs(ds_dt - .5 * beta * grad_x), dim=1).view(batch_size, 1)
        elif self.metric == 'L2':
            loss = torch.mean((ds_dt-0.5*beta*grad_x)**2, dim = 1).view(batch_size,1)
        else:
            raise ValueError('No valid metric specified. Metric should be one of "L1" or "L2" but was {}'.format(self.metric))
        return loss

class ConditionalScoreFPELoss(nn.Module):

    """
    Implements the Loss based on the cScoreFPE.

    Methods:
            forward(s, x_t, t, beta, divergence_method='exact'): Computes the cScoreFPE loss.

        Args:
            metric (str, optional): The metric used for computing the loss. Can be 'L1' or 'L2'.
                                    Defaults to 'L2'.
    """
    def __init__(self, metric = 'L2'):
        super(ConditionalScoreFPELoss, self).__init__()
        self.name = 'cScoreFPELoss'
        self.metric = metric
    def forward(self,s,t,alpha,beta,target,std):
        ds_dt = batch_gradient(s,t)
        u = .5 * target * beta * alpha ** 2
        if self.metric == 'L2':
            loss = torch.sum((std**3 * ds_dt-u) ** 2, dim=1)
        elif self.metric == 'L1':
            loss = torch.sum(torch.abs(std**3*ds_dt - u), dim=1)

        return loss

class DSM_PDELoss(nn.Module):

    """
    Loss as used by Lai et al. (2023)
    """

    def __init__(self, lam=1., pde_loss='FPE'):

        super(DSM_PDELoss,self).__init__()
        self.lam = lam
        self.dsm_loss = DSMLoss()
        if pde_loss == 'FPE':
            self.pde_loss = ScoreFPELoss()
        else:
            self.pde_loss = ConditionalScoreFPELoss()
        self.name = 'DSM_PDELoss'

    def forward(self,model,x,t,y):

        z = torch.concat([x,y], dim = 1)
        z_t, target, std, g = model.base_sde.sample(t, z, return_noise=True)
        s = model.a(z_t, t)/g
        beta = model.base_sde.beta(t)

        MSE_u = self.dsm_loss(s,std,target)
        if self.pde_loss.name == 'cScoreFPELoss':
            MSE_pde = self.lam * self.pde_loss(model,s,t,beta,target,std)
        else:
            MSE_pde = self.lam*self.pde_loss(s,z_t,t,beta)
        loss = MSE_u+MSE_pde
        return loss.mean(), {'DSM-Loss': MSE_u.mean(), 'PDE-Loss': MSE_pde.mean()}



class PINNLoss(nn.Module):
    '''
        Implements the Physics-Informed-Neural-Network (PINNs) approach by Raissi et al. (2019).

        Attributes:
            lam (float): A scaling factor for the PDE loss component. Default value is 1.0.
            lam2 (float): A scaling factor for the initial condition loss component. Default value is 1.0.
            initial_condition (callable): A function representing the initial condition to be used in the loss calculation.
            pde_loss (str): A string specifying the type of PDE loss to be used. Options are 'FPE' or 'cScoreFPE', defaulting to 'FPE'.
            ic_metric (str): A string specifying the metric used for the initial condition loss. Options are 'L1' and 'L2'.
            dsm_loss (DSMLoss): An instance of a DSMLoss class used to compute the data-driven component of the loss.
            name (str): A string representing the name of the loss class. Set to 'PINNLoss'.

        Methods:
            forward(model, x, t, y):
                Computes the overall loss for the given input.

                Parameters:
                    model (nn.Module): The model for which the loss is being computed. An instance of the BaseClassDiffusionModel.
                    x (torch.Tensor): The parameter tensor.
                    t (torch.Tensor): The temporal input tensor.
                    y (torch.Tensor): The measurement tensor.

                Returns:
                    torch.Tensor: The computed loss value.
                    dict: A dictionary containing individual components of the loss ('PDE-Loss', 'Initial Condition', 'DSM-Loss').

        Example:
            >>> loss_function = PINNLoss(initial_condition=my_initial_condition_function)
            >>> loss, loss_components = loss_function(model, x, t, y)
    '''

    def __init__(self,initial_condition, lam = 1., lam2 = 1., pde_loss = 'FPE',ic_metric = 'L1', **kwargs):

        super(PINNLoss, self).__init__()
        self.lam = lam
        self.lam2 = lam2
        self.initial_condition = initial_condition
        if pde_loss == 'cScoreFPE':
            self.pde_loss = ConditionalScoreFPELoss(**kwargs)
        else:
            self.pde_loss = ScoreFPELoss(**kwargs)
        self.dsm_loss = DSMLoss()
        self.name = 'PINNLoss'
        self.ic_metric = ic_metric

    def forward(self,model,x,y,diffused_samples,t,target,std,g):

        batch_size,xdim = x.shape
        if diffused_samples.shape[1] == xdim:
            cond_input = y
        else:
            cond_input = torch.Tensor([]) # empty tensor because the condition is already diffused
        t_0 = torch.zeros_like(t)
        g_0 = model.base_sde.g(t_0, diffused_samples)
        s_0 = model.a(x,y,t_0)/g_0
        score = model.a(diffused_samples,cond_input,t)/g
        beta = model.base_sde.beta(t)

        if self.ic_metric == 'L2':
            initial_condition_loss = self.lam2 * torch.mean((s_0[:,:xdim] - self.initial_condition(x, y)) ** 2, dim=1).view(batch_size, 1)
        elif self.ic_metric == 'L1':
            initial_condition_loss = self.lam2 * torch.mean(torch.abs(s_0[:,:xdim] - self.initial_condition(x, y)), dim=1).view(batch_size, 1)

        dsm_loss = self.dsm_loss(score,std,target)

        if self.pde_loss.name == 'cScoreFPELoss':
            alpha = model.base_sde.mean_weight(t)
            MSE_pde = self.lam * self.pde_loss(score,t,alpha,beta,target,std)
        else:
            MSE_pde = self.lam*self.pde_loss(score,diffused_samples,t,beta)
        MSE_u = dsm_loss+initial_condition_loss
        loss = torch.mean(MSE_u+ MSE_pde)

        return loss, {'PDE-Loss': MSE_pde.mean(), 'Initial Condition': initial_condition_loss.mean(), 'DSM-Loss':dsm_loss.mean()}


class PINNLoss2(nn.Module):
    """
    Ssimilar to the PINNLoss but without the data-driven DSM-loss term.
    """

    def __init__(self,initial_condition, lam = 1., lam2 = 1., pde_loss = 'FPE'):

        super(PINNLoss2, self).__init__()
        self.lam = lam
        self.lam2 = lam2
        self.initial_condition = initial_condition
        if pde_loss == 'FPE':
            self.pde_loss = ScoreFPELoss()
        else:
            self.pde_loss = ConditionalScoreFPELoss()
        self.eval_metric = DSMLoss()
        self.name = 'PINNLoss2'

    def forward(self,model,x,y,x_t,s_0,s,t,beta,target,std):


        #initial_condition_loss = self.lam2 * torch.mean((s_0 - self.initial_condition(x, y)) ** 2, dim=1).view(
        #    batch_size, 1)
        batch_size,xdim = x.shape
        initial_condition_loss = self.lam2 * torch.mean(torch.abs(s_0[:,:xdim] - self.initial_condition(x, y)), dim=1).view(
            batch_size, 1)
        #dsm_loss = self.dsm_loss(s, std, target)
        if self.pde_loss.name == 'cScoreFPELoss':
            alpha = model.base_sde.mean_weight(t)
            MSE_pde = self.lam * self.pde_loss(s,t,alpha,beta,target,std)
        else:
            MSE_pde = self.lam*self.pde_loss(s, x_t, t, beta)
        loss = torch.mean(initial_condition_loss + MSE_pde)
        eval_metric = self.eval_metric(s,std,target)
        return loss, {'PDE-Loss': MSE_pde.mean(), 'Initial Condition': initial_condition_loss.mean(), 'DSM-Loss': eval_metric.mean()}

class PosteriorLoss(nn.Module):

    #todo: Generalize for other inverse problems than scatterometry.
    '''
    Implements the work in Chung & Kim et al. (2023) using Neural Networks, where the posterior score is split into the prior p_t(x_t) and likelihood
    term p_t(y|x_t). Both scores are approximated by two neural networks which are jointly trained using the PosteriorLoss(). This class only works on the scatterometry dataset so far.
    Not used in the Master thesis.

    Attributes:
        name (str): Name of the loss class, set to 'PosteriorLoss'.
        dsm_loss (DSMLoss): An instance of the DSMLoss class used to compute loss term for the prior score model.
        forward_model (nn.Module): A model of the forward problem.
        a (float): Parameter of the forward model to calculate the likelihood term.
        b (float): Parameter of the forward model to calculate the likelihood term.
        lam (float): A scaling factor for the likelihood loss component.

    Methods:
        likelihood_target(self, x_0, y, x_t, s, sigma):
            Calculates the likelihood score of p(y|x_0).

            Parameters:
                x_0 (torch.Tensor): The mean of p(x_0|x_t) as described by Chung&Kim (2023).
                y (torch.Tensor): The measurement tensor to calculate the conditional score of p(x|y).
                x_t (torch.Tensor): The transformed state tensor.
                s (torch.Tensor): The score tensor.
                sigma (float): The standard deviation parameter.

            Returns:
                torch.Tensor: The calculated likelihood score.

        forward(self, model, x, t, y):
            Computes the overall loss for the given input.

            Parameters:
                model (nn.Module): The neural network model for which the loss is being computed. Instance of nets.PosteriorDrift().
                x (torch.Tensor): The input tensor.
                t (torch.Tensor): The temporal input tensor.
                y (torch.Tensor): The measurement tensor.

            Returns:
                torch.Tensor: The computed loss value.
                dict: A dictionary containing individual components of the loss ('PriorLoss', 'LikelihoodLoss').

    Example:
        >>> loss_function = PosteriorLoss(forward_model=my_forward_model, a=1.0, b=1.0, lam=0.5)
        >>> loss, loss_components = loss_function(model, x, t, y)
    '''
    def __init__(self, forward_model,a,b, lam):
        super(PosteriorLoss, self).__init__()
        self.name = 'PosteriorLoss'
        self.dsm_loss = DSMLoss()
        self.forward_model = forward_model
        self.a = a
        self.b = b
        self.lam = lam

    def likelihood_target(self,x_0,y,x_t, s, sigma):
        f_x = self.forward_model(x_0)
        prefactor = ((self.a * f_x) ** 2 + self.b ** 2)

        #define vectors for the vector-Jacobian products
        v1 = f_x/prefactor
        v2 = (y-f_x)/prefactor
        v3 = (y-f_x)**2*f_x/prefactor

        #compute vector-jacobian products
        vjp1 = torch.autograd.grad(f_x, x_0, v1, retain_graph = True)[0]
        vjp2 = torch.autograd.grad(f_x, x_0, v2, retain_graph = True)[0]
        vjp3 = torch.autograd.grad(f_x, x_0, v3, retain_graph = True)[0]
        
        #compute vector-Hessian products
        vhp1 = torch.autograd.grad(s, x_t, vjp1, retain_graph = True)[0]
        vhp2 = torch.autograd.grad(s, x_t, vjp2, retain_graph = True)[0]
        vhp3 = torch.autograd.grad(s, x_t, vjp3, retain_graph = True)[0]
        
        #calculate the likelihood score of p(y|x_0)
        score = -self.a**2*(sigma**2 * vhp1+ vjp1) + sigma**2 * vhp2+ vjp2 + self.a**2*(sigma**2*vhp3+vjp3)

        return score
    def forward(self, model, x,y,t):

        x_t, target, std, g = model.base_sde.sample(t, x, return_noise=True)
        s_prior = model.a.prior_net(x_t, t)
        s_likelihood = model.a.likelihood_net(x_t,y,t)
        alpha = model.base_sde.mean_weight(t)
        prior_loss = self.dsm_loss(s_prior, std,target)

        #calculate the loss for the likelihood model
        x_0 = 1/model.base_sde.mean_weight(t)*(x_t+std**2*s_prior) #calculates mean of p(x_0|x_t) as per Chung & Kim et al. (2023)
        likelihood_loss = torch.sum((alpha*s_likelihood - self.likelihood_target(x_0,y,x_t,s_prior, std))**2,dim=1)

        loss = torch.mean(prior_loss+self.lam*likelihood_loss)

        return loss, {'PriorLoss': prior_loss.mean(), 'LikelihoodLoss': self.lam*likelihood_loss.mean()}



