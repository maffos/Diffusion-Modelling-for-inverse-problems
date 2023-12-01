from main_diffusion_scatterometry import train,evaluate
from model_selection import grid_search
from datasets import generate_dataset_scatterometry
from utils_scatterometry import load_forward_model,get_log_posterior
from models.SNF import energy_grad
import yaml
import os

config_dir = 'config/'
surrogate_dir = 'trained_models/scatterometry'
gt_dir = 'data/gt_samples_scatterometry'

# load config params
config = yaml.safe_load(open(os.path.join(config_dir, "config_gridsearch.yml")))

# load the forward model
forward_model, forward_model_params = load_forward_model(surrogate_dir)

#generate test set
x_test, y_test = generate_dataset_scatterometry(forward_model,forward_model_params['a'],forward_model_params['b'],size=config['n_samples_y'])


# define the score of the posterior
score_posterior = lambda x, y: -energy_grad(x,
                                            lambda x: get_log_posterior(x, forward_model, forward_model_params['a'],
                                                                        forward_model_params['b'], y,
                                                                        forward_model_params['lambd_bd']))[0]

train_args = {'forward_model': forward_model}
eval_args = {'a': forward_model_params['a'],
             'b': forward_model_params['b'],
             'lambd_bd': forward_model_params['lambd_bd'],
             'gt_dir': config['gt_dir'],
             'n_samples_x': config['n_samples_x']}

grid_search(y_test, config, forward_model,forward_model_params,score_posterior,train,evaluate, train_args,eval_args)
