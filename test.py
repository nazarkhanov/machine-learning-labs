import argparse
from source.custom_dataset import AudioDataset
from torch.utils.data import DataLoader
from source.extract_features import load_config


def main(annotations_path, dataset_path, config_path, batches_count):
  train_dataset = AudioDataset(annotations_path=annotations_path, dataset_path=dataset_path, config_path=config_path)
  print(f"There are {len(train_dataset)} samples in the dataset.")

  config = load_config(config_path)
  train_dataloader = DataLoader(train_dataset, **config.get("dataloader", { "batches_count": batches_count }))

  batch = iter(train_dataloader)
  print(f"Size of first batch is {len(list(batch))}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test script")
  parser.add_argument("annotations_path", type=str, help="[annotations] path to file with dataset annotations")
  parser.add_argument("dataset_path", type=str, help="[dataset] path to folder with raw files")
  parser.add_argument("config_path", type=str, help="[config] path to file with parameters")
  parser.add_argument("batches_count", type=str, help="[batches] count", nargs='?', default=12)

  args = parser.parse_args()
  main(args.annotations_path, args.dataset_path, args.config_path, args.batches_count)
