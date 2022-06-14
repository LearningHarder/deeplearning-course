import config

import torch 
import torch.nn as nn 
import numpy as np


class dataloader(torch.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = data[1]
        image = np.array(data[0])
        if self.transforms:
            image = self.transforms(image=image)['image']

        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)

        return {
            'patches' : image,
            'label' : torch.tensor(label, dtype=torch.long),
        }