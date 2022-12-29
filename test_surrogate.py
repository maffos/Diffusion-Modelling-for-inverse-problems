import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
import models
import utils
import loss as ls

def eval_pass(model, data_loader, loss_fn):
    with torch.no_grad():
        model.eval()
        mean_loss = 0
        for k, (x, y) in enumerate(data_loader()):
            loss = loss_fn(y, model(x))
            mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss

def test(model, xs, ys, metric):

    with torch.no_grad():
        model.eval()
        score = metric(ys, model(xs))

    print('Score on the test set is ', score)
    return score

if __name__ == '__main__':

    #these parameters need to be the same as for the training so it's best to write them in a configuration file and laod
    model_params = {'hidden_layer_size': 512,'input_dimension' : 6,'output_dimension': 468}
    PATH = 'models/surrogate/MLP/current_model.pt'

    model = models.MLP(**model_params)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    metric = ls.mean_relative_error
    # load the dataset
    xs, ys, labels = utils.load_dataset(checkpoint['data_filename'])
    # split the data in training and test set
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state=random_state)
    score = test(model, x_test, y_test, metric)