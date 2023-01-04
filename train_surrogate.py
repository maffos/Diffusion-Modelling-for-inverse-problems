from models.surrogate_MLP import train
import loss as ls
from torch import nn
import os
import nets
from torch.optim import Adam

if __name__ == '__main__':
    num_epochs = 10000
    lr = 1e-4
    hidden_size = 512
    batch_size = 100
    x_dim = 6
    y_dim = 468
    random_state = 7
    train_size = .8
    data_filename = 'data/uniform_age_25/npz/AbdAorta_PPG.npz'
    save_dir = 'models/surrogate/MLP/testbed'
    checkpoint_file = 'models/surrogate/MLP/MSE/current_model.pt'
    log_dir = 'runs/testbed'
    loss_fn = ls.MultipleLoss(nn.MSELoss(), ls.FftMseLoss())
    eval_metric = ls.mean_relative_error
    loss2_params = {'flow': 0,
              'fhigh': 3,
              'lmbd': .1}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = nets.MLP(x_dim, y_dim, hidden_size)
    # todo: add hyper-parameters like learning rate, optimizer, batch size... to be read from a configuration file
    optimizer = 'Adam'
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('Given optimizer is currently not supported')

    model = train(model, data_filename, optimizer, num_epochs, batch_size, save_dir, loss_fn, eval_metric, log_dir,
          random_state, train_size, checkpoint_file, resume_training=True, **loss2_params)