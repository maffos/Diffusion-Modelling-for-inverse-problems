from torch.utils.data import Dataset
import os
import numpy as np

class PPGDataset(Dataset):

    def __init__(self, src_dir, location):
        self.src_dir = src_dir
        self.location = location
        self.filename = os.path.join(src_dir, location, '_PPG.npz')

        try:
            data = np.load(self.filename, allow_pickle=True)["data"].item()
        except:
            raise ValueError('You need to specify an existing dataset as filename.')

        x_labels = data['parameters']
        xs = data['x_train']

        #drop the age column
        xs = xs[:,1:]
        self.x_labels = x_labels[1:]

        # normalize x
        self.xs = (xs - xs.min(axis=0)) / (xs.max(axis=0) - xs.min(axis=0))

        #y is already normalized. But throw away first entry as it is always 0
        self.ys = data['y_train'][:,1:]