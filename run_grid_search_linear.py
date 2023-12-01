from main_diffusion_linear import train,evaluate
from model_selection import grid_search
from datasets import generate_dataset_linear
from linear_problem import LinearForwardProblem
import yaml
import os
from sklearn.model_selection import train_test_split

config_dir = 'config/'

# load config params
config = yaml.safe_load(open(os.path.join(config_dir, "config_gridsearch_linear.yml")))

# load the forward model
f = LinearForwardProblem()

#generate test set
#create data
xs,ys = generate_dataset_linear(f.xdim, f, config['dataset_size'])
x_train,x_test,y_train,y_test = train_test_split(xs,ys,train_size=config['train_size'], random_state = config['random_state'])


train_args = {'xs': x_train, 'ys': y_train}
eval_args = {'n_samples_x': config['n_samples_x']}

grid_search(y_test, config, f,vars(f),f.score_posterior,train,evaluate, train_args,eval_args)
