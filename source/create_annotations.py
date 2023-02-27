import os, errno
import argparse
import pandas as pd


def main(input, output):
  os.makedirs(os.path.dirname(input), exist_ok=True)
  os.makedirs(os.path.dirname(output), exist_ok=True)

  dataset = os.listdir(input)

  annotations = pd.DataFrame(dataset, columns=["fname"])
  annotations["class"] = annotations.apply(lambda row: row.str.split("_").str[0], axis=1)

  annotations.to_csv(output, index=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Script for creating annotations file")
  parser.add_argument("dataset_path", type=str, help="[input] path to folder with raw files")
  parser.add_argument("output_path", type=str, help="[output] path to file to store annotations info")

  args = parser.parse_args()
  main(args.dataset_path, args.output_path)
