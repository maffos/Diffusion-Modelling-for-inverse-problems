import numpy as np
import torch
import torch.utils.data
import os

__all__ = ['load_dataset', 'generate_dataset']



def get_dataset_names():
    return ['gaussian_ring']
    
def load_gaussian_ring_dataset(labels, tot_dataset_size):

    verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

    label_maps = {
              'all':  [0, 1, 2, 3, 4, 5, 6, 7],
              'some': [0, 0, 0, 0, 1, 1, 2, 3],
              'none': [0, 0, 0, 0, 0, 0, 0, 0],
             }


    # print('Generating artifical data for setup "%s"' % (labels))

    np.random.seed(0)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.2)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.

    shuffling = np.random.permutation(N)
    pos = torch.tensor(pos[shuffling], dtype=torch.float)
    labels = torch.tensor(labels[shuffling], dtype=torch.float)

    return pos, labels
    
def generate_dataset(name, **kwargs):
    if name not in get_dataset_names():
        raise ValueError(f"'{name}' is not one of the example datasets.")
        
    elif name == 'gaussian_ring':
        return load_gaussian_ring_dataset(**kwargs)
    
    else:
    
        raise ValueError('This should not be displayed, update the list of known Datasets.')
          
def load_dataset(filename):
    
    try:
        data = np.load(filename, allow_pickle=True)["data"].item()
    except:
        raise ValueError('You need to specify an existing dataset as filename.')
        
    x_labels = data['parameters']
    xs = data['x_train']

    #drop the age column
    xs = xs[:,1:]
    x_labels = x_labels[1:]
        
    # normalize x
    xs = (xs - xs.min(axis=0)) / (xs.max(axis=0) - xs.min(axis=0))
    xs = torch.from_numpy(xs).float()

    #y is already normalized. But throw away first entry as it is always 0
    ys = torch.from_numpy(data['y_train'][:,1:]).float()

    return xs,ys,x_labels    
    
def get_dataloader(x, y, batch_size):
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]
    def data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    return data_loader
    
def get_epoch_dataloader_noise(x_train, y_train, sigma):
    perm = torch.randperm(len(x_train))
    x = x_train[perm]
    y = y_train[perm]
    y = y + sigma*torch.randn_like(y)
    batch_size = 100
    def epoch_data_loader():
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size], y_train[i:i+batch_size]

    return epoch_data_loader