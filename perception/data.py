"""CSV-backed image dataset shared by training and evaluation."""

import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PerceptionDataset(Dataset):
    def __init__(self, split_directory):
        self.split_directory = os.path.abspath(split_directory)
        labels_path = os.path.join(self.split_directory, "labels.csv")
        with open(labels_path, "r", newline="") as labels_file:
            self.rows = list(csv.DictReader(labels_file))
        if not self.rows:
            raise ValueError("No labeled images found in {}".format(labels_path))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = os.path.join(self.split_directory, row["image"])
        with Image.open(image_path) as image_file:
            grayscale_image = image_file.convert("L").resize((128, 128), Image.BILINEAR)
            frame_array = np.asarray(grayscale_image, dtype=np.float32) / 255.0
        batch_of_input_frames = torch.from_numpy(frame_array).unsqueeze(0)
        class_label = int(row["label"])
        return batch_of_input_frames, class_label
