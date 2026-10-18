"""Export the frozen PyTorch pet-breed classifier to ONNX."""

from pathlib import Path

import onnx
import torch

from pet_breed.inference import PetBreedPredictor


CHECKPOINT = Path("models/frozen/resnet18_v1.pt")
CLASSES = Path("data/processed/classes.json")
OUTPUT = Path("models/onnx/resnet18_v1.onnx")


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    if not CLASSES.is_file():
        raise FileNotFoundError(CLASSES)

    predictor = PetBreedPredictor(
        checkpoint_path=CHECKPOINT,
        classes_path=CLASSES,
        device="cpu",
    )

    model = predictor.model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(OUTPUT),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },dynamo=False,
        )

    exported_model = onnx.load(str(OUTPUT))
    onnx.checker.check_model(exported_model)

    print(f"ONNX export successful: {OUTPUT}")
    print(f"Model size: {OUTPUT.stat().st_size / (1024**2):.2f} MiB")
    print("ONNX graph validation: PASS")


if __name__ == "__main__":
    main()
