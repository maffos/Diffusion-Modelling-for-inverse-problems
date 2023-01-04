from models.surrogate_MLP import train
import loss as ls
import os
import nets
from torch.optim import Adam
import utils
from torch import nn
from models.surrogate_MLP import test
from sklearn.model_selection import train_test_split
import numpy as np

def grid_search(model_fn, dataset, init_params, optimizer_fn, param_grid):

    results = {'scores': [], 'params': []}
    x_train,x_test,y_train,y_test = dataset
    for params in utils.product_dict(**param_grid):
        model = model_fn(*init_params)
        lr = params.pop('lr')
        optimizer = optimizer_fn(model.parameters(),lr=lr)
        #save_dir = 'models/surrogate/MLP/MSE_FFTmse/modes{}-{}/{}/'.format(params['flow'], params['fhigh'], params['lmbd'])
        #log_dir = 'runs/MSE_FFT_modes_{}-{}_lmbd{}/'.format(params['flow'], params['fhigh'], params['lmbd'])
        save_dir = 'models/surrogate/MLP/MSE_MRE/lmbd{}'.format(params['lmbd'])
        log_dir = 'runs/MSE_MRE_lmbd{}/'.format(params['lmbd'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model = train(model, [x_train,x_test,y_train,y_test], optimizer, num_epochs, batch_size, save_dir, loss_fn, eval_metric, log_dir, checkpoint_file = checkpoint_file, resume_training=True, **params)
        test_score = test(model, x_test, y_test, eval_metric)
        results['scores'].append(test_score)
        results['params'].append(params)

    best_idx = np.argmin(results['scores'])
    print('-----------------------')
    print(results)
    print('-----------------------')
    print('Best parameter configuration was: {}.'.format(results['params'][best_idx]))
    print('Best score was %f' % results['scores'][best_idx])

if __name__ == '__main__':
    num_epochs = 5
    hidden_size = [512,512,512,512]
    batch_size = 100
    x_dim = 6
    y_dim = 468
    random_state = 7
    train_size = .8
    data_filename = 'data/uniform_age_25/npz/AbdAorta_PPG.npz'
    checkpoint_file = 'models/surrogate/MLP/MSE/current_model.pt'
    loss_fn = ls.MultipleLoss(nn.MSELoss(), ls.mean_relative_error)
    eval_metric = ls.mean_relative_error
    """
    params = {'lr': [1e-4],
              'flow': [0,5,8],
              'fhigh': [12,20,468],
              'lmbd': [.1,.25,.4]}
    """
    params = {'lr': [1e-4],
              'lmbd': [0.1,0.25,0.4]}
    # todo: add hyper-parameters like learning rate, optimizer, batch size... to be read from a configuration file
    optimizer_fn = Adam
    model_fn = nets.MLP
    init_params = [x_dim, y_dim, hidden_size]
    # load the dataset
    xs, ys, labels = utils.load_dataset(data_filename)
    # split the data in training and test set
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state=random_state)
    grid_search(model_fn, [x_train, x_test, y_train,y_test], init_params, optimizer_fn, params)