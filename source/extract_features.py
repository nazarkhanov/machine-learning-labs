import os, errno
import argparse
import yaml
import numpy as np
from torch import nn
from torchaudio import load
from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC
from tqdm import tqdm


def load_config(config_path):
  with open(config_path, "r") as stream:
    config = yaml.safe_load(stream)

  return config


def main(dataset_path, output_path, config_path):
  os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  config = load_config(config_path)
  dataset = os.listdir(dataset_path)

  for item in tqdm(dataset):
    signal, sr = load(os.path.join(dataset_path, item))

    signal = signal[:,:config["preprocess"]["formalize"]] # cut
    signal = nn.ZeroPad2d((0, config["preprocess"]["formalize"] - signal.size()[1]))(signal) # pad

    path = os.path.join(output_path, item.replace(".wav", ".npy"))

    if config["preprocess"]["transform"] == "spectrogram":
      feature = Spectrogram(**config["preprocess"]["params"])(signal)
    elif config["preprocess"]["transform"] == "mel_spectrogram":
      feature = MelSpectrogram(**config["preprocess"]["params"])(signal)
    else:
      feature = MFCC(**config["preprocess"]["params"])(signal)

    feature = feature.cpu().detach().numpy()[0]
    np.save(path, np.array([feature, sr]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Script for extracting features from raw files")
  parser.add_argument("dataset_path", type=str, help="[input] path to folder with raw files")
  parser.add_argument("output_path", type=str, help="[output] path to file to store feature files")
  parser.add_argument("config_path", type=str, help="[config] path to file with parameters")

  args = parser.parse_args()
  main(args.dataset_path, args.output_path, args.config_path)
