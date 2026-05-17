from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


class WasteImageDataset(Dataset):
    def __init__(self, dataframe, class_to_idx: dict, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = Path(row["image_path"])

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]

        return image, torch.tensor(label, dtype=torch.long)