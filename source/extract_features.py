import os, errno
import argparse
import numpy as np
from torch import nn
from torchaudio import load
from torchaudio.transforms import Spectrogram, MelSpectrogram, MFCC
from tqdm import tqdm


N_FFT = 1024
WIN_LENGTH = 512
HOP_LENGTH = 512

MIN_LENGTH = 8000
MAX_LENGTH = 8000


def main(dataset_path, output_path, feature_type):
  os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  dataset = os.listdir(dataset_path)

  spectrogram = Spectrogram(
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
  )

  mel_spectrogram = MelSpectrogram(
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
  )

  mfcc_transform = MFCC(
    melkwargs={
      "n_fft": N_FFT,
      "hop_length": HOP_LENGTH,
    },
  )

  for item in tqdm(dataset):
    signal, sr = load(os.path.join(dataset_path, item))

    signal = signal[:,:MAX_LENGTH]
    signal = nn.ZeroPad2d((0, MIN_LENGTH - signal.size()[1]))(signal)

    path = os.path.join(output_path, item.replace(".wav", ".npy"))

    if feature_type == "spectrogram":
      feature = spectrogram(signal)
    elif feature_type == "mel_spectrogram":
      feature = mel_spectrogram(signal)
    else:
      feature = mfcc_transform(signal)

    feature = feature.cpu().detach().numpy()[0]
    np.save(path, np.array([feature, sr]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Script for extracting features from raw files")
  parser.add_argument("dataset_path", type=str, help="[input] path to folder with raw files")
  parser.add_argument("output_path", type=str, help="[output] path to file to store feature files")
  parser.add_argument("feature_type", type=str, help="[feature] type of feature to extract", nargs='?', default="mfcc")

  args = parser.parse_args()
  main(args.dataset_path, args.output_path, args.feature_type)
