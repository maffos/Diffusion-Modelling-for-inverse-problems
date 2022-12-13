import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import utils
import loss
from sklearn.model_selection import train_test_split


def train(model,optimizer, data_filename,num_epochs, save_path, train_size = .8):
    
    #data preparation
    xs,ys,labels = utils.load_dataset(data_filename)
    x_train, x_test, y_train, y_test = train_test_split(xs,ys, train_size=train_size)
    eval_loader = utils.get_dataloader(x_test,y_test)
    
    """
    //todo:
    Implement validation method to log validation results and store the best model every 100 epochs or so
    """
    
    writer = SummaryWriter()
    prog_bar = tqdm(total=num_epochs)
    model.train()
    
    for i in range(num_epochs):
        epoch_data_loader = utils.get_dataloader(x_train, y_train)
        mean_loss = 0
        mse = nn.MSELoss()
        
        for k, (x, y) in enumerate(epoch_data_loader()):
            cur_batch_size = len(x)
            y_pred = model(x)
            loss = mse(y,y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
        writer.add_scalar('Loss/train', mean_loss, i)
        prog_bar.set_description('loss: {:.4f}'.format(mean_loss))
        prog_bar.update()
        current_model_dir = os.path,join(save_path, 'current_model')
        if not os.path.exists(current_model_dir):
            os.makedirs(current_model_dir)
        torch.save(model.state_dict(), current_model_dir)
        
if __name__ == '__main__':
    num_epochs = 10000
    lr = 1e-4
    model = 
    optimizer = 'Adam'
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr = lr)
    else:
        raise ValueError('Given optimizer is currently not supported')
    train()

    
