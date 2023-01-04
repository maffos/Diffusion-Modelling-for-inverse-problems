import torch
from sklearn.model_selection import train_test_split
import nets
import utils
import loss as ls
from models.surrogate_MLP import test

if __name__ == '__main__':

    #these parameters need to be the same as for the training so it's best to write them in a configuration file and laod
    model_params = {'hidden_layers': [512,512,512,512],'input_dimension' : 6,'output_dimension': 468}
    PATH = 'models/surrogate/MLP/MSE/current_model.pt'

    model = nets.MLP(**model_params)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    train_size = checkpoint['train_size']
    random_state = checkpoint['random_state']
    metric = ls.mean_relative_error
    # load the dataset
    xs, ys, labels = utils.load_dataset(checkpoint['data_filename'])
    # split the data in training and test set
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=train_size, random_state=random_state)
    score = test(model, x_test, y_test, metric)