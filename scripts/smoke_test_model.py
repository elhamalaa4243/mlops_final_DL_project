
####Test the complete data
##load images, create a batch, and produce predictions
import torch
from torch.utils.data import DataLoader

from pet_breed.dataset import PetBreedDataset
from pet_breed.model import create_model
from pet_breed.transforms import get_eval_transforms


def main():
    dataset = PetBreedDataset(
        manifest_path="data/processed/manifest.csv",
        split="train",
        transform=get_eval_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    model = create_model(num_classes=37, pretrained=False)
    model.eval()

    with torch.inference_mode():
        outputs = model(images)

    print("Model output shape:", outputs.shape)

    assert outputs.shape == (4, 37)

    print("Smoke test passed!")


if __name__ == "__main__":
    main()