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


def train(model,
          optimizer,
          data_filename,
          num_epochs,
          batch_size,
          save_path,
          eval_metric,
          train_size = .75,
          val_size = .1,
          validation_epoch = 100):
    
    #data preparation
    xs,ys,labels = utils.load_dataset(data_filename)

    #split the data in training, validation and test set
    x_train, x_test, y_train, y_test = train_test_split(xs,ys, train_size=train_size)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size = val_size/(1-train_size))

    eval_loader = utils.get_dataloader(x_val,y_val, batch_size)
    # keep track of the validation score of the best model
    best_model_score = torch.inf

    writer = SummaryWriter()
    prog_bar = tqdm(total=num_epochs)

    for i in range(num_epochs):

        #perform model evaluation each ith epoch to store the current best model
        if i % validation_epoch == 0:
            model.eval()
            mean_loss = model.eval_pass(eval_loader, eval_metric)

            # store current model as best if it's best
            if mean_loss < best_model_score:
                best_model_score = mean_loss
                best_model_fname = os.path.join(save_path, 'best_model_epoch%d.pt'%i)
                torch.save(model.state_dict(), best_model_fname)
            writer.add_scalar('Loss/eval', mean_loss, i)

        train_loader = utils.get_dataloader(x_train, y_train, batch_size)
        mse = nn.MSELoss()

        mean_loss = model.train_pass(train_loader, mse, optimizer)

        #store current model
        current_model_fname = os.path.join(save_path, 'current_model_epoch%d.pt'%i)
        torch.save(model.state_dict(), current_model_fname)

        writer.add_scalar('Loss/train', mean_loss, i)
        prog_bar.set_description('loss: {:.4f}'.format(mean_loss))
        prog_bar.update()
        
if __name__ == '__main__':
    num_epochs = 10000
    lr = 1e-4
    x_dim = 6
    y_dim = 469
    hidden_size = 512
    batch_size = 100

    data_filename = 'data/uniform_age_25/npz/AbdAorta_PPG.npz'
    save_dir = 'models/surrogate/MLP'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = models.MLP(x_dim,y_dim,hidden_size)
    optimizer = 'Adam'
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr = lr)
    else:
        raise ValueError('Given optimizer is currently not supported')

    train(model, optimizer, data_filename, num_epochs, batch_size, save_dir, ls.mean_relative_error)

    
