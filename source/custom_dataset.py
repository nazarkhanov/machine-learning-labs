import os
import numpy as np
import pandas as pd
import torch as th
import torchaudio as tha
from torch.utils.data import Dataset
from source.extract_features import preprocess_data


class AudioDataset(Dataset):
  def __init__(self, annotations_path, dataset_path, config_path, transform=None):
    self.annotations = pd.read_csv(annotations_path)
    self.dataset_path = dataset_path
    self.config_path = config_path
    self.transform = transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, idx):
    path = os.path.join(self.dataset_path, self.annotations.iloc[idx, 0])

    if path.endswith(".npy"):
      item = np.load(path, allow_pickle=True)
      signal, sr = th.from_numpy(item[0]), item[1]
    else:
      signal, sr = self._prepropcess(path)

    if self.transform:
      signal = self.transform(signal)

    label = self.annotations.iloc[idx, 1]
    return signal, label

  def _prepropcess(self, path):
    return preprocess_data(path, self.config_path, formalize=True)
