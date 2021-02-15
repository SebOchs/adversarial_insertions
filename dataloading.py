import numpy as np
import torch
from torch.utils.data import Dataset


# data loaders for a given data set based on model
class MyBertDataset(Dataset):

    def __init__(self, filename):
        self.data = np.load(filename, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ids, seg, att, lab = self.data[index]
        return torch.tensor(ids).long(), torch.tensor(seg).long(), torch.tensor(att).long(), lab




class MyT5Dataset(Dataset):

    def __init__(self, filename):
        self.data = np.load(filename, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # lab attention is not helpful right now, but maybe in the future
        seq, attn, label, lab_attn = self.data[index]
        return torch.tensor(seq).long(), torch.tensor(attn).long(), torch.tensor(label).long(), \
               torch.tensor(lab_attn).long()