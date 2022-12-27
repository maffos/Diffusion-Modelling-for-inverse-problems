from train_surrogate import train
import loss as ls
import os
import models
from torch.optim import Adam

def grid_search(model_fn, init_params, optimizer_fn, lr, params):

    for i in range(len(params)):
        model = model_fn(*init_params)
        optimizer = optimizer_fn(model.parameters(),lr=lr)
        save_dir = 'models/surrogate/MLP/MSE_FFTmse/modes{}-{}/'.format(params[i]['flow'], params[i]['fhigh'])
        log_dir = 'runs/MSE_FFT_modes_{}-{}/'.format(params[i]['flow'], params[i]['fhigh'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train(model, optimizer, data_filename, num_epochs, batch_size, save_dir, loss_fn, eval_metric, log_dir,
              random_state, train_size, checkpoint_file, resume_training=True, **params[i])

if __name__ == '__main__':
    num_epochs = 5000
    lr = 1e-4
    hidden_size = 512
    batch_size = 100
    x_dim = 6
    y_dim = 468
    random_state = 7
    train_size = .8
    data_filename = 'data/uniform_age_25/npz/AbdAorta_PPG.npz'
    checkpoint_file = 'models/surrogate/MLP/MSE/current_model.pt'
    loss_fn = ls.FftMseLoss()
    eval_metric = ls.mean_relative_error
    params = [{'flow': 0,'fhigh': 5}, {'flow': 0,'fhigh': 10}, {'flow': 3,'fhigh': 8},{'flow': 3,'fhigh': 12},{'flow': 5,'fhigh': 12}]

    # todo: add hyper-parameters like learning rate, optimizer, batch size... to be read from a configuration file
    optimizer_fn = Adam
    model_fn = models.MLP
    init_params = [x_dim, y_dim, hidden_size]
    grid_search(model_fn, init_params, optimizer_fn, lr, params)