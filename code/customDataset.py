import torch
from torch.utils.data import Dataset
import os


class customDataset(Dataset):
    def __init__(self, path, Lack=None):
        super(customDataset, self).__init__()
        self.databuffer = os.listdir(path)
        self._len_ = len(self.databuffer)
        self.path = path + '/{}'
        self.Lack = Lack

    def __getitem__(self, idx):
        data = torch.load(self.path.format(self.databuffer[idx]))
        if self.Lack is None:
            return data[0].to(torch.float32), data[1].to(torch.float32), data[2], data[3]
        elif self.Lack == 'PSSM':
            return (data[0][:, 20:]).to(torch.float32), data[1].to(torch.float32), data[2], data[3]
        elif self.Lack == 'Atchley':
            return (torch.cat([data[0][:, :20], data[0][:, 25:]], dim=-1)).to(torch.float32), data[1].to(torch.float32), \
                   data[2], data[3]
        elif self.Lack == 'AtomModel':
            return data[0][:, :25].to(torch.float32), data[1].to(torch.float32), data[2], data[3]

    def __len__(self):
        return self._len_
