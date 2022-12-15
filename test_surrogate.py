import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
import models
import utils
import loss as ls
def test(model, metric, data_filename, train_size, random_state, batch_size, log_dir, epoch):

    # load the dataset
    xs, ys, labels = utils.load_dataset(data_filename)
    writer = SummaryWriter(log_dir)
    # recover the same test set that was held out during training
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state = random_state)

    data_loader = utils.get_dataloader(x_test,y_test, batch_size)
    score = model.eval_pass(data_loader, metric)
    writer.add_scalar('Loss/test', score, epoch)
    print('Score on the test set is ', score)

if __name__ == '__main__':

    #these parameters need to be the same as for the training so it's best to write them in a configuration file and laod
    model_params = {'hidden_layer_size': 512,'input_dimension' : 6,'output_dimension': 468}
    PATH = 'models/surrogate/MLP/current_model.pt'

    model = models.MLP(**model_params)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    metric = ls.mean_relative_error
    test(model, metric, checkpoint['data_filename'], checkpoint['train_size'], checkpoint['random_state'], checkpoint['batch_size'], checkpoint['log_dir'], checkpoint['epoch'])