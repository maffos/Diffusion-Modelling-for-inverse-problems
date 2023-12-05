import utils
from torch.optim import Adam
import os
import numpy as np
def grid_search(y_test, config, forward_model,forward_model_params,score_posterior,train,evaluate, train_args,eval_args):

    already_visited = []
    params = config['params']
    best_kl = np.infty
    best_params_kl = {}
    best_nlpd = np.infty
    best_params_nlpd = {}
    best_fisher = np.infty
    best_params_fisher = {}

    for param_configuration in utils.product_dict(**params):
        skip = False
        model,loss_fn = utils.get_model_from_args(param_configuration, forward_model_params,score_posterior,forward_model, config)

        if param_configuration['pde_metric'] == 'L1' and param_configuration['pde_loss'] == 'cScoreFPE':
            skip = True
        if loss_fn.name == 'DSM_PDELoss':
            if (param_configuration['lam'],param_configuration['pde_metric']) in already_visited:
                skip = True
            else:
                already_visited.append((param_configuration['lam'],param_configuration['pde_metric']))

        if not skip:
            optimizer = Adam(model.sde.a.parameters(), lr=config['lr'])

            if loss_fn.name == 'DSM_PDELoss':
                train_dir = os.path.join(config['src_dir'], param_configuration['pde_loss'], loss_fn.name,
                                         param_configuration['pde_metric'], 'lam:{}'.format(param_configuration['lam']))
            else:
                train_dir = os.path.join(config['src_dir'], param_configuration['pde_loss'], loss_fn.name,
                                         param_configuration['pde_metric'], param_configuration['ic_metric'],
                                         'lam:{}'.format(param_configuration['lam']),
                                         'lam2:{}'.format(param_configuration['lam2']))
            out_dir = os.path.join(train_dir, 'results')
            log_dir = utils.set_directories(train_dir,out_dir)
            print('-----------------')
            print(param_configuration)
            model = train(model, optimizer, loss_fn, forward_model_params,train_dir, log_dir, config['n_epochs'], config['batch_size'], **train_args)
            kl,nlpd,fisher = evaluate(model, y_test, forward_model, out_dir, config['plot_ys'], config['n_samples_x'], **eval_args)
            if (kl < best_kl):
                best_params_kl = param_configuration
                best_kl = kl
            if nlpd < best_nlpd:
                best_params_nlpd = param_configuration
                best_nlpd = nlpd
            if fisher < best_fisher:
                best_params_fisher = param_configuration
                best_fisher = fisher

            print('---------------------------------')
            print('Best KL: ', best_kl)
            print(best_params_kl)
            print('-------------------')
            print('Best NLPD: ', best_nlpd)
            print(best_params_nlpd)
            print('-------------------')
            print('Best Fisher divergence: ', best_fisher)
            print(best_params_fisher)
            print('-------------------')