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


global_config_path = None
global_config_params = None
global_transform = None


def preprocess_init(config_path):
  global global_config_path, global_config_params, global_transform

  if global_config_path != config_path:
    global_config_path = config_path
    global_config_params = load_config(config_path)

    if global_config_params["preprocess"]["transform"] == "spectrogram":
      global_transform = Spectrogram(**global_config_params["preprocess"]["params"])
    elif global_config_params["preprocess"]["transform"] == "mel_spectrogram":
      global_transform = MelSpectrogram(**global_config_params["preprocess"]["params"])
    else:
      global_transform = MFCC(**global_config_params["preprocess"]["params"])

  return global_config_params, global_transform


def preprocess_data(file_path, config_path, formalize=False):
  config, transform = preprocess_init(config_path)

  signal, sr = load(file_path)

  signal = signal[:, :config["preprocess"]["formalize"]]  # cut
  signal = nn.ZeroPad2d((0, config["preprocess"]["formalize"] - signal.size()[1]))(signal)  # pad

  signal = transform(signal)

  return signal, sr


def main(dataset_path, output_path, config_path):
  os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  dataset = os.listdir(dataset_path)

  for item in tqdm(dataset):
    path = os.path.join(dataset_path, item)
    signal, sr = preprocess_data(path, config_path)

    feature = signal.cpu().detach().numpy()[0]

    path = os.path.join(output_path, item.replace(".wav", ".npy"))
    np.save(path, np.array([feature, sr]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Script for extracting features from raw files")
  parser.add_argument("dataset_path", type=str, help="[input] path to folder with raw files")
  parser.add_argument("output_path", type=str, help="[output] path to file to store feature files")
  parser.add_argument("config_path", type=str, help="[config] path to file with parameters")

  args = parser.parse_args()
  main(args.dataset_path, args.output_path, args.config_path)
