import os
import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset


class AudioDataset(Dataset):
  def __init__(self, annotations_path, dataset_path):
    self.annotations = pd.read_csv(annotations_path)
    self.path = dataset_path

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, idx):
    path = os.path.join(self.path, self.annotations.iloc[idx, 0])

    item = np.load(path, allow_pickle=True)
    signal, sr = th.from_numpy(item[0]), item[1]
    label = self.annotations.iloc[idx, 1]

    return signal, label
