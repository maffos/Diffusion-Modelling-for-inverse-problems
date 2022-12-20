import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

import utils
import loss as ls
import models


def eval_pass(data_loader, loss_fn):
    with torch.no_grad():
        model.eval()
        mean_loss = 0
        for k, (x, y) in enumerate(data_loader()):
            loss = loss_fn(y, model(x))
            mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss


def train_pass(data_loader, loss_fn):
    # todo: check if optimizer gets updated when sent to model class or if it explixitly needs to be passed back
    model.train()
    mean_loss = 0
    for k, (x, y) in enumerate(data_loader()):
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)

    return mean_loss
def train(model,
          optimizer,
          dataset,
          num_epochs,
          batch_size,
          save_path,
          eval_metric,
          log_dir,
          random_state,
          train_size,
          validate_every_ith_epoch = 100,
          store_every_ith_epoch = 50):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load the dataset
    xs, ys, labels = utils.load_dataset(dataset)
    # split the data in training and test set
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state=random_state)

    eval_loader = utils.get_dataloader(x_test, y_test, batch_size)
    # keep track of the validation score of the best model
    best_model_score = torch.inf

    writer = SummaryWriter(log_dir)
    prog_bar = tqdm(total=num_epochs)

    for i in range(num_epochs):

        train_loader = utils.get_dataloader(x_train, y_train, batch_size)
        mse = nn.MSELoss()

        mean_loss = train_pass(train_loader, mse)

        #helper function so that we don't need to write down the dict twice. Needs to be defined within the training loop
        def _state_dict():
            return {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
                'best_model_score': best_model_score,
                'data_filename': data_filename,
                'random_state': random_state,
                'train_size': train_size,
                'batch_size': batch_size,
                'log_dir': log_dir
            }

        #store current states
        if i % store_every_ith_epoch == 0:
            current_model_path = os.path.join(save_path, 'current_model.pt')
            torch.save(_state_dict(), current_model_path)

        writer.add_scalar('Loss/train', mean_loss, i)

        # perform model evaluation each ith epoch to store the current best model
        if i % validate_every_ith_epoch == 0:
            model.eval()
            score = eval_pass(eval_loader, eval_metric)

            # store current model as best if it's best
            if score < best_model_score:
                best_model_score = score
                best_model_path = os.path.join(save_path, 'best_model.pt')
                torch.save(_state_dict(), best_model_path)

            writer.add_scalar('Loss/test', mean_loss, i)

        prog_bar.set_description('loss: {:.4f}'.format(mean_loss))
        prog_bar.update()


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
    save_dir = 'models/surrogate/MLP'
    log_dir = 'runs/exp1'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = models.MLP(x_dim,y_dim,hidden_size)
    #todo: add resume training
    #todo: add hyper-parameters like learning rate, optimizer, batch size... to be read from a configuration file
    optimizer = 'Adam'
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr = lr)
    else:
        raise ValueError('Given optimizer is currently not supported')

    train(model, optimizer, data_filename, num_epochs, batch_size, save_dir, ls.mean_relative_error, log_dir, random_state, train_size)

    
