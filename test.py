from source.custom_dataset import AudioDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
  BATCHES_COUNT = 100

  dataset = AudioDataset("local/annotations.csv", "local/features/")
  print(f"There are {len(dataset)} samples in the dataset.")

  dataloader = DataLoader(dataset, batch_size=BATCHES_COUNT)

  batch = iter(dataloader)
  print(f"Size of first batch is {len(list(batch))}")
