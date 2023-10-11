import torch
from torch.utils.data import DataLoader, Dataset
import warnings

class ClientDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class. Use to create client datasets that are subsets
    of the complete dataset.
    :param dataset: complete dataset
    :param idxs: list of indices assigned to a particular client
    """
    warnings.filterwarnings("ignore")
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)