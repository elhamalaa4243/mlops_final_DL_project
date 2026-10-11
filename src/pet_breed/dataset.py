## Create the dataset loader
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PetBreedDataset(Dataset):
    def __init__(self, manifest_path, split, transform=None):
        manifest = pd.read_csv(manifest_path)
        self.data = manifest[
            manifest["split"] == split
        ].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])

        return image, label